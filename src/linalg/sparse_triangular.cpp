#include "../../include/lp_solver/linalg/detail/sparse_triangular.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace lp_solver::linalg::detail {

namespace {
constexpr double kZeroTol = 1e-19;

void clearReachPattern(std::vector<int>& mark, std::vector<int>& pattern) {
    for (int j : pattern) {
        mark[static_cast<size_t>(j)] = 0;
    }
    pattern.clear();
}

void reachLowerUnit(
    int j,
    const CscMatrixView& L,
    std::vector<int>& mark,
    std::vector<int>& pattern
) {
    if (mark[static_cast<size_t>(j)] != 0) {
        return;
    }
    mark[static_cast<size_t>(j)] = 1;
    for (int p = L.col_ptr[j]; p < L.col_ptr[j + 1]; ++p) {
        const int i = L.row_ind[p];
        if (i > j) {
            reachLowerUnit(i, L, mark, pattern);
        }
    }
    pattern.push_back(j);
}

void reachUpper(
    int j,
    const CscMatrixView& U,
    std::vector<int>& mark,
    std::vector<int>& pattern
) {
    if (mark[static_cast<size_t>(j)] != 0) {
        return;
    }
    mark[static_cast<size_t>(j)] = 1;
    for (int p = U.col_ptr[j]; p < U.col_ptr[j + 1]; ++p) {
        const int i = U.row_ind[p];
        if (i < j) {
            reachUpper(i, U, mark, pattern);
        }
    }
    pattern.push_back(j);
}

}  // namespace

void gpLowerUnitSolve(
    const CscMatrixView& L,
    util::IndexedVector& x,
    std::vector<int>& mark,
    std::vector<int>& stack
) {
    const int n = L.n;
    if (n <= 0) {
        return;
    }
    if (x.capacity() < n) {
        throw std::logic_error("gpLowerUnitSolve: rhs capacity mismatch");
    }
    if (static_cast<int>(mark.size()) < n) {
        mark.assign(static_cast<size_t>(n), 0);
    }

    stack.clear();
    for (int k : x.nonZeroIndices()) {
        if (std::abs(x[k]) > kZeroTol && mark[static_cast<size_t>(k)] == 0) {
            reachLowerUnit(k, L, mark, stack);
        }
    }

    for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
        const int j = *it;
        const double xj = x[j];
        if (std::abs(xj) <= kZeroTol) {
            continue;
        }
        for (int p = L.col_ptr[j]; p < L.col_ptr[j + 1]; ++p) {
            const int i = L.row_ind[p];
            if (i > j) {
                x.add(i, -L.values[p] * xj);
            }
        }
    }

    clearReachPattern(mark, stack);
}

void gpUpperSolve(
    const CscMatrixView& U,
    util::IndexedVector& x,
    std::vector<int>& mark,
    std::vector<int>& stack,
    bool implicit_unit_diagonal
) {
    const int n = U.n;
    if (n <= 0) {
        return;
    }
    if (x.capacity() < n) {
        throw std::logic_error("gpUpperSolve: rhs capacity mismatch");
    }
    if (static_cast<int>(mark.size()) < n) {
        mark.assign(static_cast<size_t>(n), 0);
    }

    stack.clear();
    for (int k : x.nonZeroIndices()) {
        if (std::abs(x[k]) > kZeroTol && mark[static_cast<size_t>(k)] == 0) {
            reachUpper(k, U, mark, stack);
        }
    }

    for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
        const int j = *it;
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
                clearReachPattern(mark, stack);
                throw std::runtime_error("gpUpperSolve: missing or singular diagonal");
            }
        }
        xj /= diag;
        x.set(j, xj);
        for (int p = U.col_ptr[j]; p < U.col_ptr[j + 1]; ++p) {
            const int i = U.row_ind[p];
            if (i < j) {
                x.add(i, -U.values[p] * xj);
            }
        }
    }

    clearReachPattern(mark, stack);
}

void gpLowerDiagSolve(
    const CscMatrixView& L,
    util::IndexedVector& x,
    std::vector<int>& mark,
    std::vector<int>& stack
) {
    const int n = L.n;
    if (n <= 0) {
        return;
    }
    if (x.capacity() < n) {
        throw std::logic_error("gpLowerDiagSolve: rhs capacity mismatch");
    }
    if (static_cast<int>(mark.size()) < n) {
        mark.assign(static_cast<size_t>(n), 0);
    }

    stack.clear();
    for (int k : x.nonZeroIndices()) {
        if (std::abs(x[k]) > kZeroTol && mark[static_cast<size_t>(k)] == 0) {
            reachLowerUnit(k, L, mark, stack);
        }
    }

    for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
        const int j = *it;
        double diag = 0.0;
        for (int p = L.col_ptr[j]; p < L.col_ptr[j + 1]; ++p) {
            if (L.row_ind[p] == j) {
                diag = L.values[p];
                break;
            }
        }
        if (std::abs(diag) <= kZeroTol) {
            clearReachPattern(mark, stack);
            throw std::runtime_error("gpLowerDiagSolve: missing or singular diagonal");
        }
        const double xj = x[j] / diag;
        x.set(j, xj);
        for (int p = L.col_ptr[j]; p < L.col_ptr[j + 1]; ++p) {
            const int i = L.row_ind[p];
            if (i > j) {
                x.add(i, -L.values[p] * xj);
            }
        }
    }

    clearReachPattern(mark, stack);
}

}  // namespace lp_solver::linalg::detail
