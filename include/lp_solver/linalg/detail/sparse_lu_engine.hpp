#pragma once

#include <vector>

#include "../../util/indexed_vector.hpp"
#include "../../util/packed_matrix.hpp"

namespace lp_solver::linalg::detail {

/// Sparse LU (Eigen::SparseLU) with hypersparse lower-unit and structured upper/lower triangular solves.
class SparseLuEngine {
public:
    [[nodiscard]] bool factorize(const util::PackedMatrix& basis_matrix);

    /// Install precomputed sparse LU factors (same layout as after `factorize`): unit lower L,
    /// upper U with diagonal, row/column permutations matching Eigen SparseLU / UMFPACK extract order.
    void adoptFactorData(
        int n,
        std::vector<int> Lp,
        std::vector<int> Li,
        std::vector<double> Lx,
        std::vector<int> Up,
        std::vector<int> Ui,
        std::vector<double> Ux,
        std::vector<int> perm_r,
        std::vector<int> perm_c
    );

    void ftran(util::IndexedVector& rhs) const;
    void btran(util::IndexedVector& rhs) const;

    [[nodiscard]] int dimension() const { return n_; }
    [[nodiscard]] bool ok() const { return factor_ok_; }

private:
    void applyRowPermForward(const std::vector<int>& perm, util::IndexedVector& x) const;
    void applyRowPermInverseTranspose(const std::vector<int>& perm, util::IndexedVector& x) const;
    void applyColPermForward(const std::vector<int>& perm, util::IndexedVector& x) const;
    void applyColPermInverseTranspose(const std::vector<int>& perm, util::IndexedVector& x) const;

    int n_{0};
    bool factor_ok_{false};

    std::vector<int> Lp_;
    std::vector<int> Li_;
    std::vector<double> Lx_;

    std::vector<int> Up_;
    std::vector<int> Ui_;
    std::vector<double> Ux_;

    std::vector<int> perm_r_;
    std::vector<int> perm_c_;

    std::vector<int> Lt_p_;
    std::vector<int> Lt_i_;
    std::vector<double> Lt_x_;

    std::vector<int> Ut_p_;
    std::vector<int> Ut_i_;
    std::vector<double> Ut_x_;

    mutable std::vector<int> gp_mark_;
    mutable std::vector<int> gp_stack_;
};

}  // namespace lp_solver::linalg::detail
