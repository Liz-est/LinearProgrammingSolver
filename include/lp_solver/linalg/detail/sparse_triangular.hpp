#pragma once

#include <cstdint>
#include <vector>

#include "../../util/indexed_vector.hpp"

namespace lp_solver::linalg::detail {

/// Zero-copy view of a square matrix in CSC (column-major sparse) format.
struct CscMatrixView {
    int n{0};
    const int* col_ptr{nullptr};
    const int* row_ind{nullptr};
    const double* values{nullptr};
};

/// Solve L x = b in place, where L is **unit** lower triangular (diagonal not stored),
/// CSC format: column j holds entries L(i,j) with i > j only.
void gpLowerUnitSolve(const CscMatrixView& L, util::IndexedVector& x, std::vector<int>& mark, std::vector<int>& stack);

/// Solve U x = b in place, U **upper** triangular in CSC (each column j contains row indices i <= j).
/// If `implicit_unit_diagonal` is true, missing U(j,j) is treated as 1 (for L^T when L is unit lower).
void gpUpperSolve(
    const CscMatrixView& U,
    util::IndexedVector& x,
    std::vector<int>& mark,
    std::vector<int>& stack,
    bool implicit_unit_diagonal = false
);

/// Solve L x = b, L **lower** triangular with explicit diagonal (non-unit), CSC.
void gpLowerDiagSolve(const CscMatrixView& L, util::IndexedVector& x, std::vector<int>& mark, std::vector<int>& stack);

}  // namespace lp_solver::linalg::detail
