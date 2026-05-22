#pragma once

#include <memory>

#include "../util/indexed_vector.hpp"
#include "../util/packed_matrix.hpp"

namespace lp_solver::linalg {

class IBasisFactor {
public:
    virtual ~IBasisFactor() = default;

    [[nodiscard]] virtual bool factorize(const util::PackedMatrix& basis_matrix) = 0;
    virtual void ftran(util::IndexedVector& rhs) const = 0;
    virtual void btran(util::IndexedVector& rhs) const = 0;

    [[nodiscard]] virtual bool updateEta(int pivot_row, const util::IndexedVector& ftran_col) {
        (void)pivot_row;
        (void)ftran_col;
        return false;
    }

    [[nodiscard]] virtual int etaFileLength() const { return 0; }
};

enum class FactorBackend {
    /// Prefer SuiteSparse UMFPACK when built with `LP_SOLVER_HAVE_UMFPACK`, otherwise Eigen SparseLU.
    Default,
    Eigen,
    Umfpack
};

/// Resolved backend used by `FactorBackend::Default` for this build.
[[nodiscard]] FactorBackend defaultFactorBackend();

[[nodiscard]] std::unique_ptr<IBasisFactor> makeFactor(FactorBackend backend);

/// `makeFactor(FactorBackend::Default)`.
[[nodiscard]] std::unique_ptr<IBasisFactor> makeDefaultFactor();

}  // namespace lp_solver::linalg
