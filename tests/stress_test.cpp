#include "../include/lp_solver/linalg/i_basis_factor.hpp"
#include "../include/lp_solver/model/problem_data.hpp"
#include "../include/lp_solver/model/solver_state.hpp"
#include "../include/lp_solver/simplex/dual_simplex.hpp"
#include "../include/lp_solver/util/indexed_vector.hpp"
#include "../include/lp_solver/util/packed_matrix.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void expect(bool cond, const char* msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

lp_solver::util::PackedMatrix buildLargeIdentityWithExtraCols(int m, int n) {
    lp_solver::util::PackedMatrix::Builder builder(m, n);

    for (int j = 0; j < m; ++j) {
        builder.appendColumn(std::vector<int>{j}, std::vector<double>{1.0});
    }

    for (int j = m; j < n; ++j) {
        const int r1 = j % m;
        const int r2 = (j * 37 + 11) % m;
        if (r1 == r2) {
            builder.appendColumn(std::vector<int>{r1}, std::vector<double>{0.5});
        } else {
            builder.appendColumn(std::vector<int>{r1, r2}, std::vector<double>{0.5, -0.25});
        }
    }

    return std::move(builder).build();
}

void testIndexedVectorStress() {
    constexpr int kN = 100'000;
    lp_solver::util::IndexedVector v(kN);

    for (int i = 0; i < kN; i += 3) {
        v.add(i, 1.0);
    }
    expect(v.numNonZeros() == 33334, "unexpected nnz after first fill");

    for (int i = 0; i < kN; i += 6) {
        v.add(i, -1.0);
    }
    expect(v.numNonZeros() == 16667, "unexpected nnz after cancellations");
}

void testPackedMatrixStress() {
    constexpr int kRows = 3'000;
    constexpr int kCols = 6'000;

    auto A = buildLargeIdentityWithExtraCols(kRows, kCols);
    expect(A.numRows() == kRows, "row count mismatch");
    expect(A.numCols() == kCols, "col count mismatch");
    expect(A.numNonZeros() >= kRows, "nnz too small");

    const auto col2048 = A.column(2048);
    expect(col2048.numNonZeros() == 1, "identity column should have one nonzero");
    expect(col2048[2048] == 1.0, "identity diagonal entry mismatch");
}

void testDualSimplexStressFeasibleBasis() {
    constexpr int kRows = 1'500;
    constexpr int kCols = 3'000;

    lp_solver::model::ProblemData prob{
        buildLargeIdentityWithExtraCols(kRows, kCols),
        std::vector<double>(kCols, 1.0),
        std::vector<double>(kRows, 0.0),
        std::vector<double>(kCols, 0.0),
        std::vector<double>(kCols, 1.0e20)
    };

    lp_solver::model::SolverState state;
    state.basic_indices.resize(kRows);
    state.x_basic.assign(kRows, 1.0);
    for (int i = 0; i < kRows; ++i) {
        state.basic_indices[i] = i;
    }

    auto factor = lp_solver::linalg::makeFactor(lp_solver::linalg::FactorBackend::Eigen);
    lp_solver::simplex::DualSimplex solver(
        std::move(factor),
        std::unique_ptr<lp_solver::simplex::IRowPivot>{},
        nullptr
    );

    const auto status = solver.solve(prob, state, lp_solver::simplex::SolverConfig{});
    expect(status == lp_solver::simplex::DualSimplex::Status::Optimal, "solver did not return Optimal");
}

}  // namespace

int main() {
    try {
        testIndexedVectorStress();
        testPackedMatrixStress();
        testDualSimplexStressFeasibleBasis();
    } catch (const std::exception&) {
        return 1;
    }
    return 0;
}
