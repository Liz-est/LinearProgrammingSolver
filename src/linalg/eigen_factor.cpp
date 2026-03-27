#include "../../include/lp_solver/linalg/eigen_factor.hpp"

namespace lp_solver::linalg {

bool EigenFactor::factorize(const util::PackedMatrix& basis_matrix) {
    (void)basis_matrix;
    is_factorized_ = true;
    return true;
}

void EigenFactor::ftran(util::IndexedVector& rhs) const {
    (void)rhs;
}

void EigenFactor::btran(util::IndexedVector& rhs) const {
    (void)rhs;
}

}  // namespace lp_solver::linalg
