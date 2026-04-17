#pragma once

#include <vector>

#include "detail/sparse_lu_engine.hpp"
#include "i_basis_factor.hpp"

namespace lp_solver {
namespace linalg {

/// Basis factorization: **SuiteSparse UMFPACK** when available (extracted L/U + Gilbert–Peierls
/// triangular solves), otherwise **Eigen::SparseLU** (same pipeline as `EigenFactor`).
class UmfpackFactor final : public IBasisFactor {
public:
    [[nodiscard]] bool factorize(const ::lp_solver::util::PackedMatrix& basis_matrix) override;
    void ftran(::lp_solver::util::IndexedVector& rhs) const override;
    void btran(::lp_solver::util::IndexedVector& rhs) const override;
    void updateEta(int pivot_row, const ::lp_solver::util::IndexedVector& ftran_col) override;
    [[nodiscard]] int etaFileLength() const override;

private:
    struct EtaUpdate {
        int pivot_row{0};
        std::vector<double> d;
    };

    static void applyEtaForward(std::vector<double>& v, const std::vector<EtaUpdate>& etas);
    static void applyEtaBackward(std::vector<double>& v, const std::vector<EtaUpdate>& etas);

    detail::SparseLuEngine engine_;
    std::vector<EtaUpdate> eta_updates_;
};

}  // namespace linalg
}  // namespace lp_solver
