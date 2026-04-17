#pragma once

#include <vector>

#include "i_basis_factor.hpp"

namespace lp_solver::linalg {

class EigenFactor final : public IBasisFactor {
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

    bool is_factorized_{false};
    int dimension_{0};
    std::vector<std::vector<double>> lu_;
    std::vector<int> pivot_;
    std::vector<EtaUpdate> eta_updates_;
};

}  // namespace lp_solver::linalg
