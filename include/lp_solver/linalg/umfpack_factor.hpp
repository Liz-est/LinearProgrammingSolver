#pragma once

#include <vector>

#include "i_basis_factor.hpp"

namespace lp_solver {
namespace linalg {

class UmfpackFactor final : public IBasisFactor {
public:
    [[nodiscard]] bool factorize(const util::PackedMatrix& basis_matrix) override;
    void ftran(util::IndexedVector& rhs) const override;
    void btran(util::IndexedVector& rhs) const override;
    void updateEta(int pivot_row, const util::IndexedVector& ftran_col) override;
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

}  // namespace linalg
}  // namespace lp_solver
