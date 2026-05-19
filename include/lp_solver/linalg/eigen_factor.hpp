#pragma once

#include "detail/sparse_lu_engine.hpp"
#include "eta_file.hpp"
#include "i_basis_factor.hpp"

namespace lp_solver::linalg {

/// Basis factorization using **Eigen::SparseLU** on the sparse basis matrix and
/// **Gilbert–Peierls-style** triangular solves on the extracted L/U factors.
class EigenFactor final : public IBasisFactor {
public:
    [[nodiscard]] bool factorize(const ::lp_solver::util::PackedMatrix& basis_matrix) override;
    void ftran(::lp_solver::util::IndexedVector& rhs) const override;
    void btran(::lp_solver::util::IndexedVector& rhs) const override;
    void updateEta(int pivot_row, const ::lp_solver::util::IndexedVector& ftran_col) override;
    [[nodiscard]] int etaFileLength() const override;

private:
    detail::SparseLuEngine engine_;
    EtaFile eta_file_;
};

}  // namespace lp_solver::linalg
