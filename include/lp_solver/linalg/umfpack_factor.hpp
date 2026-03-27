#pragma once

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
    bool is_factorized_{false};
    int eta_length_{0};
};

}  // namespace linalg
}  // namespace lp_solver
