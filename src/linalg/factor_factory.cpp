#include "../../include/lp_solver/linalg/eigen_factor.hpp"
#include "../../include/lp_solver/linalg/i_basis_factor.hpp"
#include "../../include/lp_solver/linalg/umfpack_factor.hpp"

#include <stdexcept>

namespace lp_solver::linalg {

FactorBackend defaultFactorBackend() {
#if LP_SOLVER_HAVE_UMFPACK
    return FactorBackend::Umfpack;
#else
    return FactorBackend::Eigen;
#endif
}

std::unique_ptr<IBasisFactor> makeFactor(FactorBackend backend) {
    if (backend == FactorBackend::Default) {
        backend = defaultFactorBackend();
    }
    switch (backend) {
        case FactorBackend::Eigen:
            return std::make_unique<EigenFactor>();
        case FactorBackend::Umfpack:
            return std::make_unique<UmfpackFactor>();
        case FactorBackend::Default:
            break;
    }
    throw std::invalid_argument("Unknown FactorBackend");
}

std::unique_ptr<IBasisFactor> makeDefaultFactor() { return makeFactor(FactorBackend::Default); }

}  // namespace lp_solver::linalg
