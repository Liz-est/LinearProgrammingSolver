#include "../../include/lp_solver/simplex/dual_simplex.hpp"
#include "../../include/lp_solver/model/solver_state.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lp_solver {
namespace simplex {

namespace {
int chooseMostInfeasibleRow(const model::SolverState& state, double primal_feasibility_tol) {
    int best = -1;
    double most_negative = -primal_feasibility_tol;
    for (int i = 0; i < static_cast<int>(state.x_basic.size()); ++i) {
        if (state.x_basic[i] < most_negative) {
            most_negative = state.x_basic[i];
            best = i;
        }
    }
    return best;
}

util::PackedMatrix buildBasisMatrix(
    const model::ProblemData& prob,
    const model::SolverState& state
) {
    const int m = prob.numRows();
    util::PackedMatrix::Builder builder(m, m);
    for (int i = 0; i < m; ++i) {
        const int col = state.basic_indices[i];
        const auto col_vec = prob.A.column(col);
        builder.appendColumn(col_vec.nonZeroIndices(), col_vec.nonZeroValues());
    }
    return std::move(builder).build();
}

double computeObjective(const model::ProblemData& prob, const model::SolverState& state) {
    double obj = 0.0;
    const int m = static_cast<int>(state.basic_indices.size());
    const int xsz = static_cast<int>(state.x_basic.size());
    for (int i = 0; i < m && i < xsz; ++i) {
        const int col = state.basic_indices[i];
        if (col >= 0 && col < static_cast<int>(prob.c.size())) {
            obj += prob.c[col] * state.x_basic[i];
        }
    }
    return obj;
}
}  // namespace

DualSimplex::DualSimplex(
    std::unique_ptr<linalg::IBasisFactor> factor,
    std::unique_ptr<IRowPivot> row_pivot,
    ISolverObserver* observer
)
    : factor_(std::move(factor)), row_pivot_(std::move(row_pivot)), observer_(observer) {}

DualSimplex::Status DualSimplex::solve(
    const model::ProblemData& prob,
    model::SolverState& state,
    const SolverConfig& cfg
) {
    if (!factor_) {
        throw std::invalid_argument("DualSimplex requires a non-null basis factor");
    }
    const int m = prob.numRows();
    const int n = prob.numCols();
    if (m <= 0 || n <= 0) {
        if (observer_ != nullptr) {
            observer_->onTermination(state, "Empty problem dimensions");
        }
        return Status::Infeasible;
    }
    if (static_cast<int>(state.basic_indices.size()) != m) {
        if (observer_ != nullptr) {
            observer_->onTermination(state, "basic_indices size mismatch");
        }
        return Status::Infeasible;
    }
    if (state.x_basic.size() != static_cast<size_t>(m)) {
        state.x_basic.assign(m, 0.0);
    }
    if (state.nonbasic_indices.empty()) {
        std::vector<bool> in_basis(n, false);
        for (int b : state.basic_indices) {
            if (b >= 0 && b < n) {
                in_basis[b] = true;
            }
        }
        state.nonbasic_indices.clear();
        for (int j = 0; j < n; ++j) {
            if (!in_basis[j]) {
                state.nonbasic_indices.push_back(j);
            }
        }
    }
    if (state.reduced_costs.size() != state.nonbasic_indices.size()) {
        state.reduced_costs.assign(state.nonbasic_indices.size(), 0.0);
    }
    if (state.dual_pi.size() != static_cast<size_t>(m)) {
        state.dual_pi.assign(m, 0.0);
    }

    if (!factor_->factorize(buildBasisMatrix(prob, state))) {
        if (observer_ != nullptr) {
            observer_->onTermination(state, "Initial basis factorization failed");
        }
        return Status::Infeasible;
    }

    for (; state.iteration < cfg.max_iterations; ++state.iteration) {
        if (observer_ != nullptr) {
            observer_->onIterationBegin(state);
        }

        int leaving_row = chuzr(state);
        if (leaving_row < 0) {
            leaving_row = chooseMostInfeasibleRow(state, cfg.primal_feasibility_tol);
        }
        if (leaving_row < 0) {
            state.objective = computeObjective(prob, state);
            if (observer_ != nullptr) {
                observer_->onIterationEnd(state);
                observer_->onTermination(state, "Primal feasible basis reached");
            }
            return Status::Optimal;
        }

        util::IndexedVector pivot_row(prob.numCols());
        int entering_col = chuzc(pivot_row, state);
        if (entering_col < 0) {
            if (observer_ != nullptr) {
                observer_->onIterationEnd(state);
                observer_->onTermination(state, "No entering column candidate");
            }
            return Status::Infeasible;
        }

        util::IndexedVector aq = prob.A.column(entering_col);
        ftran(aq);

        double ratio = std::abs(state.x_basic[leaving_row]);
        if (observer_ != nullptr) {
            observer_->onPivot(leaving_row, entering_col, ratio);
        }

        pivot(leaving_row, entering_col, aq, prob, state);
        updateDuals(leaving_row, entering_col, pivot_row, state);

        if (cfg.refactor_frequency > 0 && factor_->etaFileLength() >= cfg.refactor_frequency) {
            if (!factor_->factorize(buildBasisMatrix(prob, state))) {
                if (observer_ != nullptr) {
                    observer_->onTermination(state, "Periodic refactorization failed");
                }
                return Status::Infeasible;
            }
        }

        if (observer_ != nullptr) {
            observer_->onIterationEnd(state);
        }
    }

    if (observer_ != nullptr) {
        observer_->onTermination(state, "Reached max iterations");
    }
    return Status::IterationLimit;
}

int DualSimplex::chuzr(const model::SolverState& state) const {
    if (row_pivot_) {
        const int row = row_pivot_->chooseRow(state);
        if (row >= 0 && row < static_cast<int>(state.basic_indices.size())) {
            return row;
        }
    }
    return -1;
}

void DualSimplex::btran(util::IndexedVector& ep) const { factor_->btran(ep); }

int DualSimplex::chuzc(const util::IndexedVector& pivot_row, const model::SolverState& state) const {
    (void)pivot_row;
    if (state.nonbasic_indices.empty()) {
        return -1;
    }
    return state.nonbasic_indices.front();
}

void DualSimplex::ftran(util::IndexedVector& aq) const { factor_->ftran(aq); }

void DualSimplex::pivot(
    int leaving_row,
    int entering_col,
    const util::IndexedVector& ftran_col,
    const model::ProblemData& prob,
    model::SolverState& state
) {
    if (leaving_row < 0 || leaving_row >= static_cast<int>(state.basic_indices.size())) {
        return;
    }
    (void)prob;

    const int leaving_col = state.basic_indices[leaving_row];
    state.basic_indices[leaving_row] = entering_col;
    for (int& nb : state.nonbasic_indices) {
        if (nb == entering_col) {
            nb = leaving_col;
            break;
        }
    }

    if (leaving_row < static_cast<int>(state.x_basic.size())) {
        state.x_basic[leaving_row] = 0.0;
    }

    factor_->updateEta(leaving_row, ftran_col);
}

void DualSimplex::updateDuals(
    int leaving_row,
    int entering_col,
    const util::IndexedVector& btran_row,
    model::SolverState& state
) const {
    (void)leaving_row;
    (void)btran_row;
    for (size_t i = 0; i < state.nonbasic_indices.size(); ++i) {
        if (state.nonbasic_indices[i] == entering_col && i < state.reduced_costs.size()) {
            state.reduced_costs[i] = 0.0;
        }
    }
}

}  // namespace simplex
}  // namespace lp_solver
