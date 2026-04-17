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
    double harris_tolerance{1e-7};
    double steepest_edge_weight_floor{1.0};
    bool use_harris_two_pass{true};
    bool use_dual_steepest_edge{true};
    bool use_presolve{true};
    bool enable_big_m_phase_one{true};
    double big_m_scale{1000.0};
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
    [[nodiscard]] int chuzr(const model::SolverState& state, const SolverConfig& cfg) const;
    void btran(util::IndexedVector& ep) const;
    [[nodiscard]] int chuzc(
        const model::ProblemData& prob,
        const model::SolverState& state,
        int leaving_row,
        const util::IndexedVector& rho,
        const SolverConfig& cfg,
        util::IndexedVector& pivot_row_out
    ) const;
    void ftran(util::IndexedVector& aq) const;
    [[nodiscard]] bool initializeBasisAndReducedCosts(
        const model::ProblemData& prob,
        model::SolverState& state,
        const SolverConfig& cfg
    );
    /// Textbook Big-M Phase I (manual): add bounding row, artificial in basis, one forcing pivot.
    /// On success, if an artificial column was added, sets \p artificial_column_index to its index; else -1.
    [[nodiscard]] bool runBigMPhaseOne(
        model::ProblemData& prob,
        model::SolverState& state,
        const SolverConfig& cfg,
        int& artificial_column_index
    );
    void computePrimalBasic(const model::ProblemData& prob, model::SolverState& state) const;
    void computeDualAndReducedCosts(const model::ProblemData& prob, model::SolverState& state) const;
    [[nodiscard]] double computeObjective(const model::ProblemData& prob, const model::SolverState& state) const;
    void pivot(
        int leaving_row,
        int entering_col,
        const util::IndexedVector& ftran_col,
        const model::ProblemData& prob,
        model::SolverState& state
    );
    void updateDuals(
        const model::ProblemData& prob,
        int leaving_row,
        int entering_col,
        const util::IndexedVector& btran_row,
        const util::IndexedVector& ftran_col,
        model::SolverState& state
    ) const;
    void initializeDseWeights(model::SolverState& state) const;

    std::unique_ptr<linalg::IBasisFactor> factor_;
    std::unique_ptr<IRowPivot> row_pivot_;
    ISolverObserver* observer_;
};

}  // namespace lp_solver::simplex
