#include "../../include/lp_solver/linalg/detail/sparse_triangular.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace lp_solver::linalg::detail {

namespace {
constexpr double kZeroTol = 1e-19;

void dfsMarkLowerReach(
    int j,
    const CscMatrixView& L,
    std::vector<std::uint8_t>& marked
) {
    if (marked[static_cast<size_t>(j)]) {
        return;
    }
    marked[static_cast<size_t>(j)] = 1;
    for (int p = L.col_ptr[j]; p < L.col_ptr[j + 1]; ++p) {
        const int i = L.row_ind[p];
        if (i > j) {
            dfsMarkLowerReach(i, L, marked);
        }
    }
}

void dfsMarkLowerDiagReach(
    int j,
    const CscMatrixView& L,
    std::vector<std::uint8_t>& marked
) {
    if (marked[static_cast<size_t>(j)]) {
        return;
    }
    marked[static_cast<size_t>(j)] = 1;
    for (int p = L.col_ptr[j]; p < L.col_ptr[j + 1]; ++p) {
        const int i = L.row_ind[p];
        if (i > j) {
            dfsMarkLowerDiagReach(i, L, marked);
        }
    }
}
}  // namespace

void gpLowerUnitSolve(
    const CscMatrixView& L,
    util::IndexedVector& x,
    std::vector<int>& mark,
    std::vector<int>& stack
) {
    (void)mark;
    (void)stack;
    const int n = L.n;
    if (n <= 0) {
        return;
    }
    if (x.capacity() < n) {
        throw std::logic_error("gpLowerUnitSolve: rhs capacity mismatch");
    }
    std::vector<std::uint8_t> marked(static_cast<size_t>(n), 0);
    for (int k = 0; k < n; ++k) {
        if (std::abs(x[k]) > kZeroTol) {
            dfsMarkLowerReach(k, L, marked);
        }
    }
    for (int j = 0; j < n; ++j) {
        if (!marked[static_cast<size_t>(j)]) {
            continue;
        }
        for (int p = L.col_ptr[j]; p < L.col_ptr[j + 1]; ++p) {
            const int i = L.row_ind[p];
            if (i > j) {
                x.add(i, -L.values[p] * x[j]);
            }
        }
    }
}

void gpUpperSolve(
    const CscMatrixView& U,
    util::IndexedVector& x,
    std::vector<int>& mark,
    std::vector<int>& stack,
    bool implicit_unit_diagonal
) {
    (void)mark;
    (void)stack;
    const int n = U.n;
    if (n <= 0) {
        return;
    }
    if (x.capacity() < n) {
        throw std::logic_error("gpUpperSolve: rhs capacity mismatch");
    }
    for (int j = n - 1; j >= 0; --j) {
        double diag = 0.0;
        for (int p = U.col_ptr[j]; p < U.col_ptr[j + 1]; ++p) {
            if (U.row_ind[p] == j) {
                diag = U.values[p];
                break;
            }
        }
        double xj = x[j];
        if (std::abs(diag) <= kZeroTol) {
            if (implicit_unit_diagonal) {
                diag = 1.0;
            } else {
                throw std::runtime_error("gpUpperSolve: missing or singular diagonal");
            }
        }
        xj /= diag;
        x.set(j, xj);
        for (int p = U.col_ptr[j]; p < U.col_ptr[j + 1]; ++p) {
            const int i = U.row_ind[p];
            const double v = U.values[p];
            if (i < j) {
                x.add(i, -v * xj);
            }
        }
    }
}

void gpLowerDiagSolve(
    const CscMatrixView& L,
    util::IndexedVector& x,
    std::vector<int>& mark,
    std::vector<int>& stack
) {
    (void)mark;
    (void)stack;
    const int n = L.n;
    if (n <= 0) {
        return;
    }
    if (x.capacity() < n) {
        throw std::logic_error("gpLowerDiagSolve: rhs capacity mismatch");
    }
    for (int j = 0; j < n; ++j) {
        double diag = 0.0;
        for (int p = L.col_ptr[j]; p < L.col_ptr[j + 1]; ++p) {
            if (L.row_ind[p] == j) {
                diag = L.values[p];
                break;
            }
        }
        if (std::abs(diag) <= kZeroTol) {
            throw std::runtime_error("gpLowerDiagSolve: missing or singular diagonal");
        }
        const double xj = x[j] / diag;
        x.set(j, xj);
        for (int p = L.col_ptr[j]; p < L.col_ptr[j + 1]; ++p) {
            const int i = L.row_ind[p];
            const double v = L.values[p];
            if (i > j) {
                x.add(i, -v * xj);
            }
        }
    }
}

}  // namespace lp_solver::linalg::detail
