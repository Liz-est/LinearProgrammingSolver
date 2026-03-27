#include "../../include/lp_solver/linalg/umfpack_factor.hpp"

namespace lp_solver::linalg {

bool UmfpackFactor::factorize(const util::PackedMatrix& basis_matrix) {
    (void)basis_matrix;
    is_factorized_ = true;
    eta_length_ = 0;
    return true;
}

void UmfpackFactor::ftran(util::IndexedVector& rhs) const {
    (void)rhs;
}

void UmfpackFactor::btran(util::IndexedVector& rhs) const {
    (void)rhs;
}

void UmfpackFactor::updateEta(int pivot_row, const util::IndexedVector& ftran_col) {
    (void)pivot_row;
    (void)ftran_col;
    ++eta_length_;
}

int UmfpackFactor::etaFileLength() const { return eta_length_; }

}  // namespace lp_solver::linalg
