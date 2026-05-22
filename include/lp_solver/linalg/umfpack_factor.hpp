#pragma once

#include <vector>

#include "detail/sparse_lu_engine.hpp"
#include "eta_file.hpp"
#include "i_basis_factor.hpp"

namespace lp_solver {
namespace linalg {

/// Basis factorization: **SuiteSparse UMFPACK** when available (extracted L/U + Gilbert–Peierls
/// triangular solves), otherwise **Eigen::SparseLU** (same pipeline as `EigenFactor`).
class UmfpackFactor final : public IBasisFactor {
public:
    ~UmfpackFactor() override;
    [[nodiscard]] bool factorize(const ::lp_solver::util::PackedMatrix& basis_matrix) override;
    void ftran(::lp_solver::util::IndexedVector& rhs) const override;
    void btran(::lp_solver::util::IndexedVector& rhs) const override;
    [[nodiscard]] bool updateEta(int pivot_row, const ::lp_solver::util::IndexedVector& ftran_col) override;
    [[nodiscard]] int etaFileLength() const override;

private:
    int dimension_{0};
    detail::SparseLuEngine engine_;
    EtaFile eta_file_;

#ifdef LP_SOLVER_HAVE_UMFPACK
    void clearUmfpackState();
    [[nodiscard]] bool factorizeWithUmfpackDirect(const ::lp_solver::util::PackedMatrix& basis_matrix);
    void umfpackSolve(int sys, ::lp_solver::util::IndexedVector& rhs) const;

    bool use_umfpack_direct_{false};
    std::vector<int> Ap_;
    std::vector<int> Ai_;
    std::vector<double> Ax_;
    void* umf_numeric_{nullptr};
#endif
};

}  // namespace linalg
}  // namespace lp_solver
