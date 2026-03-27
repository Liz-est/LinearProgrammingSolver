#pragma once

#include <memory>

#include "../linalg/i_basis_factor.hpp"
#include "../model/problem_data.hpp"
#include "../model/solver_state.hpp"
#include "i_row_pivot.hpp"
#include "i_solver_observer.hpp"
#include "../util/indexed_vector.hpp"

namespace lp_solver::simplex {

class IRowPivot;
class ISolverObserver;

struct SolverConfig {
    int max_iterations{10'000};
    int refactor_frequency{100};
    double dual_feasibility_tol{1e-7};
    double primal_feasibility_tol{1e-7};
};

class DualSimplex {
public:
    enum class Status { Optimal, Infeasible, Unbounded, IterationLimit };

    DualSimplex(
        std::unique_ptr<linalg::IBasisFactor> factor,
        std::unique_ptr<IRowPivot> row_pivot,
        ISolverObserver* observer = nullptr
    );

    [[nodiscard]] Status solve(
        const model::ProblemData& prob,
        model::SolverState& state,
        const SolverConfig& cfg = {}
    );

private:
    [[nodiscard]] int chuzr(const model::SolverState& state) const;
    void btran(util::IndexedVector& ep) const;
    [[nodiscard]] int chuzc(const util::IndexedVector& pivot_row, const model::SolverState& state) const;
    void ftran(util::IndexedVector& aq) const;
    void pivot(
        int leaving_row,
        int entering_col,
        const util::IndexedVector& ftran_col,
        const model::ProblemData& prob,
        model::SolverState& state
    );
    void updateDuals(
        int leaving_row,
        int entering_col,
        const util::IndexedVector& btran_row,
        model::SolverState& state
    ) const;

    std::unique_ptr<linalg::IBasisFactor> factor_;
    std::unique_ptr<IRowPivot> row_pivot_;
    ISolverObserver* observer_;
};

}  // namespace lp_solver::simplex
