#pragma once

#include "i_basis_factor.hpp"

namespace lp_solver::linalg {

class EigenFactor final : public IBasisFactor {
public:
    [[nodiscard]] bool factorize(const util::PackedMatrix& basis_matrix) override;
    void ftran(util::IndexedVector& rhs) const override;
    void btran(util::IndexedVector& rhs) const override;

private:
    bool is_factorized_{false};
};

}  // namespace lp_solver::linalg
