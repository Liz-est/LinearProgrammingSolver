#include "../../include/lp_solver/linalg/eigen_factor.hpp"

#include <stdexcept>

namespace lp_solver::linalg {

bool EigenFactor::factorize(const util::PackedMatrix& basis_matrix) {
    eta_file_.clear();
    return engine_.factorize(basis_matrix);
}

void EigenFactor::ftran(util::IndexedVector& rhs) const {
    if (!engine_.ok()) {
        throw std::logic_error("EigenFactor::ftran called before successful factorization");
    }
    engine_.ftran(rhs);
    eta_file_.applyForward(rhs);
}

void EigenFactor::btran(util::IndexedVector& rhs) const {
    if (!engine_.ok()) {
        throw std::logic_error("EigenFactor::btran called before successful factorization");
    }
    eta_file_.applyBackward(rhs);
    engine_.btran(rhs);
}

void EigenFactor::updateEta(int pivot_row, const util::IndexedVector& ftran_col) {
    eta_file_.append(pivot_row, ftran_col);
}

int EigenFactor::etaFileLength() const { return eta_file_.length(); }

}  // namespace lp_solver::linalg
