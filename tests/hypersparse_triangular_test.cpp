#include "../include/lp_solver/linalg/detail/sparse_lu_engine.hpp"
#include "../include/lp_solver/linalg/detail/sparse_triangular.hpp"
#include "../include/lp_solver/util/indexed_vector.hpp"
#include "../include/lp_solver/util/packed_matrix.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void expect(bool cond, const char* msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

lp_solver::linalg::detail::CscMatrixView makeCsc(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_ind,
    const std::vector<double>& values
) {
    return lp_solver::linalg::detail::CscMatrixView{
        n, col_ptr.data(), row_ind.data(), values.data()
    };
}

std::vector<double> denseLowerUnitSolve(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_ind,
    const std::vector<double>& values,
    std::vector<double> b
) {
    for (int j = 0; j < n; ++j) {
        const double xj = b[static_cast<size_t>(j)];
        for (int p = col_ptr[static_cast<size_t>(j)]; p < col_ptr[static_cast<size_t>(j + 1)]; ++p) {
            const int i = row_ind[static_cast<size_t>(p)];
            if (i > j) {
                b[static_cast<size_t>(i)] -= values[static_cast<size_t>(p)] * xj;
            }
        }
    }
    return b;
}

std::vector<double> denseUpperSolve(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_ind,
    const std::vector<double>& values,
    std::vector<double> b,
    bool implicit_unit_diagonal
) {
    for (int j = n - 1; j >= 0; --j) {
        double diag = 0.0;
        for (int p = col_ptr[static_cast<size_t>(j)]; p < col_ptr[static_cast<size_t>(j + 1)]; ++p) {
            if (row_ind[static_cast<size_t>(p)] == j) {
                diag = values[static_cast<size_t>(p)];
                break;
            }
        }
        if (std::abs(diag) <= 1e-19) {
            diag = implicit_unit_diagonal ? 1.0 : 0.0;
        }
        double xj = b[static_cast<size_t>(j)] / diag;
        b[static_cast<size_t>(j)] = xj;
        for (int p = col_ptr[static_cast<size_t>(j)]; p < col_ptr[static_cast<size_t>(j + 1)]; ++p) {
            const int i = row_ind[static_cast<size_t>(p)];
            if (i < j) {
                b[static_cast<size_t>(i)] -= values[static_cast<size_t>(p)] * xj;
            }
        }
    }
    return b;
}

void testManualReachExample() {
    // Manual Section 4 example: 5x5 unit lower L, b = e_0 * 7.
    const int n = 5;
    const std::vector<int> Lp{0, 1, 2, 3, 3, 3};
    const std::vector<int> Li{2, 3, 4};
    const std::vector<double> Lx{2.0, 4.0, 5.0};
    const auto L = makeCsc(n, Lp, Li, Lx);

    lp_solver::util::IndexedVector x(n);
    x.set(0, 7.0);

    std::vector<int> mark(n, 0);
    std::vector<int> pattern;
    lp_solver::linalg::detail::gpLowerUnitSolve(L, x, mark, pattern);

    const auto dense = denseLowerUnitSolve(n, Lp, Li, Lx, std::vector<double>{7, 0, 0, 0, 0});
    for (int i = 0; i < n; ++i) {
        expect(std::abs(x[i] - dense[static_cast<size_t>(i)]) < 1e-12, "manual example component mismatch");
    }
    expect(std::abs(x[2] + 14.0) < 1e-12, "expected x[2] = -14");
    expect(std::abs(x[4] - 70.0) < 1e-12, "expected x[4] = 70");
    expect(x.numNonZeros() == 3, "only reachable nodes should be nonzero");
}

void testHypersparseMatchesDenseOnRandomPattern() {
    const int n = 40;
    std::vector<int> Lp(n + 1, 0);
    std::vector<int> Li;
    std::vector<double> Lx;
    for (int j = 0; j < n; ++j) {
        for (int k = 1; k <= 3 && j + k < n; ++k) {
            Li.push_back(j + k);
            Lx.push_back(0.1 * static_cast<double>((j + k) % 5 + 1));
        }
        Lp[static_cast<size_t>(j + 1)] = static_cast<int>(Li.size());
    }
    const auto L = makeCsc(n, Lp, Li, Lx);

    std::vector<double> b_dense(static_cast<size_t>(n), 0.0);
    b_dense[2] = 3.0;
    b_dense[17] = -2.5;

    auto dense = denseLowerUnitSolve(n, Lp, Li, Lx, b_dense);

    lp_solver::util::IndexedVector x(n);
    x.set(2, 3.0);
    x.set(17, -2.5);
    std::vector<int> mark(n, 0);
    std::vector<int> pattern;
    lp_solver::linalg::detail::gpLowerUnitSolve(L, x, mark, pattern);

    for (int i = 0; i < n; ++i) {
        expect(std::abs(x[i] - dense[static_cast<size_t>(i)]) < 1e-10, "lower unit mismatch vs dense");
    }
}

void testUpperHypersparseMatchesDense() {
    const int n = 12;
    std::vector<int> Up(n + 1, 0);
    std::vector<int> Ui;
    std::vector<double> Ux;
    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < 2 && j - k >= 0; ++k) {
            Ui.push_back(j - k);
            Ux.push_back((j == j - k) ? (2.0 + 0.1 * j) : 0.05 * (j + 1));
        }
        Up[static_cast<size_t>(j + 1)] = static_cast<int>(Ui.size());
    }
    const auto U = makeCsc(n, Up, Ui, Ux);

    std::vector<double> b_dense(static_cast<size_t>(n), 0.0);
    b_dense[static_cast<size_t>(n - 1)] = 4.0;
    b_dense[5] = -1.0;

    auto dense = denseUpperSolve(n, Up, Ui, Ux, b_dense, false);

    lp_solver::util::IndexedVector x(n);
    x.set(n - 1, 4.0);
    x.set(5, -1.0);
    std::vector<int> mark(n, 0);
    std::vector<int> pattern;
    lp_solver::linalg::detail::gpUpperSolve(U, x, mark, pattern, false);

    for (int i = 0; i < n; ++i) {
        expect(std::abs(x[i] - dense[static_cast<size_t>(i)]) < 1e-10, "upper solve mismatch vs dense");
    }
}

void testSparseLuEngineHypersparseFtran() {
    // A = L * U, 2x2, identity permutations: L unit lower [[1,0],[3,1]], U [[2,1],[0,4]].
    const int n = 2;
    const std::vector<int> Lp{0, 1, 1};
    const std::vector<int> Li{1};
    const std::vector<double> Lx{3.0};
    const std::vector<int> Up{0, 1, 3};
    const std::vector<int> Ui{0, 0, 1};
    const std::vector<double> Ux{2.0, 1.0, 4.0};
    const std::vector<int> perm{0, 1};

    lp_solver::util::PackedMatrix::Builder builder(n, n);
    builder.appendColumn(std::vector<int>{0, 1}, std::vector<double>{2.0, 6.0});
    builder.appendColumn(std::vector<int>{0, 1}, std::vector<double>{1.0, 7.0});
    const auto A = std::move(builder).build();

    lp_solver::linalg::detail::SparseLuEngine engine;
    engine.adoptFactorData(n, Lp, Li, Lx, Up, Ui, Ux, perm, perm);

    std::vector<double> e0{1.0, 0.0};

    lp_solver::util::IndexedVector rhs(n);
    rhs.set(0, 1.0);
    engine.ftran(rhs);

    const auto residual = A.multiply(rhs.rawValues());
    double err = 0.0;
    for (int i = 0; i < n; ++i) {
        err = std::max(err, std::abs(residual[static_cast<size_t>(i)] - e0[static_cast<size_t>(i)]));
    }
    expect(err < 1e-10, "hypersparse SparseLuEngine ftran residual too large");
}

}  // namespace

int main() {
    try {
        testManualReachExample();
        testHypersparseMatchesDenseOnRandomPattern();
        testUpperHypersparseMatchesDense();
        testSparseLuEngineHypersparseFtran();
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
    return 0;
}
