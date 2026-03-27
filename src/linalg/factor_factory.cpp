#include "../../include/lp_solver/linalg/eigen_factor.hpp"
#include "../../include/lp_solver/linalg/i_basis_factor.hpp"
#include "../../include/lp_solver/linalg/umfpack_factor.hpp"

#include <stdexcept>

namespace lp_solver::linalg {

std::unique_ptr<IBasisFactor> makeFactor(FactorBackend backend) {
    switch (backend) {
        case FactorBackend::Eigen:
            return std::make_unique<EigenFactor>();
        case FactorBackend::Umfpack:
            return std::make_unique<UmfpackFactor>();
        default:
            throw std::invalid_argument("Unknown FactorBackend");
    }
}

}  // namespace lp_solver::linalg
