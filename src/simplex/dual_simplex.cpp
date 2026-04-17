#include "../../include/lp_solver/simplex/dual_simplex.hpp"
#include "../../include/lp_solver/presolve/presolver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lp_solver {
namespace simplex {

namespace {
constexpr double kTiny = 1e-12;

int chooseMostInfeasibleRow(const model::SolverState& state, double tol) {
    int row = -1;
    double min_value = -tol;
    for (int i = 0; i < static_cast<int>(state.x_basic.size()); ++i) {
        if (state.x_basic[i] < min_value) {
            min_value = state.x_basic[i];
            row = i;
        }
    }
    return row;
}

util::PackedMatrix buildBasisMatrix(const model::ProblemData& prob, const model::SolverState& state) {
    const int m = prob.numRows();
    util::PackedMatrix::Builder builder(m, m);
    for (int i = 0; i < m; ++i) {
        const int col = state.basic_indices[i];
        const auto col_vec = prob.A.column(col);
        builder.appendColumn(col_vec.nonZeroIndices(), col_vec.nonZeroValues());
    }
    return std::move(builder).build();
}

double dotColumn(const util::PackedMatrix& A, int col, const std::vector<double>& y) {
    const auto vec = A.column(col);
    double out = 0.0;
    for (int idx : vec.nonZeroIndices()) {
        out += y[idx] * vec[idx];
    }
    return out;
}

std::vector<double> basicCosts(const model::ProblemData& prob, const model::SolverState& state) {
    std::vector<double> cb(prob.numRows(), 0.0);
    for (int i = 0; i < prob.numRows(); ++i) {
        const int col = state.basic_indices[i];
        if (col >= 0 && col < static_cast<int>(prob.c.size())) {
            cb[i] = prob.c[col];
        }
    }
    return cb;
}

/// Adds row: sum_{j in nonbasic_cols} x_j + x_a = M, new slack/artificial column x_a >= 0 (standard form column).
[[nodiscard]] model::ProblemData expandWithBoundingConstraint(
    const model::ProblemData& prob,
    const std::vector<int>& nonbasic_cols,
    double M
) {
    const int m = prob.numRows();
    const int n = prob.numCols();
    std::vector<bool> is_nonbasic(n, false);
    for (int j : nonbasic_cols) {
        if (j >= 0 && j < n) {
            is_nonbasic[j] = true;
        }
    }

    util::PackedMatrix::Builder builder(m + 1, n + 1);
    for (int j = 0; j < n; ++j) {
        std::vector<int> rows;
        std::vector<double> vals;
        const auto col = prob.A.column(j);
        for (int r : col.nonZeroIndices()) {
            rows.push_back(r);
            vals.push_back(col[r]);
        }
        if (is_nonbasic[j]) {
            rows.push_back(m);
            vals.push_back(1.0);
        }
        builder.appendColumn(rows, vals);
    }
    builder.appendColumn(std::vector<int>{m}, std::vector<double>{1.0});

    model::ProblemData out;
    out.A = std::move(builder).build();
    out.b = prob.b;
    out.b.push_back(M);
    out.c = prob.c;
    out.c.push_back(0.0);

    out.lower_bounds = prob.lower_bounds;
    while (static_cast<int>(out.lower_bounds.size()) < n) {
        out.lower_bounds.push_back(0.0);
    }
    out.lower_bounds.push_back(0.0);

    out.upper_bounds = prob.upper_bounds;
    while (static_cast<int>(out.upper_bounds.size()) < n) {
        out.upper_bounds.push_back(1.0e30);
    }
    out.upper_bounds.push_back(1.0e30);
    return out;
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

    model::ProblemData work_prob = prob;
    presolve::Presolver presolver;
    presolve::Presolver::ReductionResult reduction;
    bool has_reduction = false;
    double objective_offset = 0.0;
    if (cfg.use_presolve) {
        auto reduced = presolver.run(prob);
        if (reduced.status == presolve::Presolver::Status::Infeasible) {
            if (observer_ != nullptr) {
                observer_->onTermination(state, "Presolve detected infeasibility");
            }
            return Status::Infeasible;
        }
        if (reduced.status == presolve::Presolver::Status::Unbounded) {
            if (observer_ != nullptr) {
                observer_->onTermination(state, "Presolve detected unboundedness");
            }
            return Status::Unbounded;
        }
        has_reduction = true;
        reduction = std::move(reduced);
        work_prob = reduction.reduced_problem;
        objective_offset = reduction.objective_offset;
    }

    const int m = work_prob.numRows();
    const int n = work_prob.numCols();
    if (m <= 0 || n <= 0 || m > n) {
        if (observer_ != nullptr) {
            observer_->onTermination(state, "Invalid reduced dimensions");
        }
        return Status::Infeasible;
    }

    if (state.basic_indices.size() != static_cast<size_t>(m)) {
        state.basic_indices.resize(m);
        for (int i = 0; i < m; ++i) {
            state.basic_indices[i] = i;
        }
        state.nonbasic_indices.clear();
    }

    std::vector<bool> in_basis(n, false);
    for (int b : state.basic_indices) {
        if (b < 0 || b >= n || in_basis[b]) {
            if (observer_ != nullptr) {
                observer_->onTermination(state, "Invalid basic indices");
            }
            return Status::Infeasible;
        }
        in_basis[b] = true;
    }
    state.nonbasic_indices.clear();
    for (int j = 0; j < n; ++j) {
        if (!in_basis[j]) {
            state.nonbasic_indices.push_back(j);
        }
    }

    if (!initializeBasisAndReducedCosts(work_prob, state, cfg)) {
        if (observer_ != nullptr) {
            observer_->onTermination(state, "Initial basis factorization failed");
        }
        return Status::Infeasible;
    }
    int artificial_column_index = -1;
    if (cfg.enable_big_m_phase_one &&
        !runBigMPhaseOne(work_prob, state, cfg, artificial_column_index)) {
        if (observer_ != nullptr) {
            observer_->onTermination(state, "Big-M phase failed");
        }
        return Status::Infeasible;
    }

    for (; state.iteration < cfg.max_iterations; ++state.iteration) {
        if (observer_ != nullptr) {
            observer_->onIterationBegin(state);
        }

        int leaving_row = chuzr(state, cfg);
        if (leaving_row < 0) {
            leaving_row = chooseMostInfeasibleRow(state, cfg.primal_feasibility_tol);
        }
        if (leaving_row < 0) {
            state.objective = computeObjective(work_prob, state) + objective_offset;
            if (observer_ != nullptr) {
                observer_->onIterationEnd(state);
                observer_->onTermination(state, "Primal feasible basis reached");
            }
            break;
        }

               const int m_rows = work_prob.numRows();
        const int n_cols = work_prob.numCols();

        util::IndexedVector rho(m_rows);
        rho.set(leaving_row, 1.0);
        btran(rho);

        util::IndexedVector pivot_row(n_cols);
        const int entering_col = chuzc(work_prob, state, leaving_row, rho, cfg, pivot_row);
        if (entering_col < 0) {
            if (observer_ != nullptr) {
                observer_->onIterationEnd(state);
                observer_->onTermination(state, "No entering column satisfies ratio test");
            }
            return Status::Infeasible;
        }

        util::IndexedVector aq = work_prob.A.column(entering_col);
        ftran(aq);
        if (std::abs(aq[leaving_row]) <= kTiny) {
            if (observer_ != nullptr) {
                observer_->onTermination(state, "Near-zero pivot element encountered");
            }
            return Status::Infeasible;
        }

        const double ratio = state.reduced_costs.empty() ? 0.0 : std::abs(state.reduced_costs.front());
        if (observer_ != nullptr) {
            observer_->onPivot(leaving_row, entering_col, ratio);
        }

        pivot(leaving_row, entering_col, aq, work_prob, state);
        updateDuals(work_prob, leaving_row, entering_col, rho, aq, state);

        if (cfg.refactor_frequency > 0 && factor_->etaFileLength() >= cfg.refactor_frequency) {
            if (!factor_->factorize(buildBasisMatrix(work_prob, state))) {
                if (observer_ != nullptr) {
                    observer_->onTermination(state, "Periodic refactorization failed");
                }
                return Status::Infeasible;
            }
            initializeDseWeights(state);
            computePrimalBasic(work_prob, state);
            computeDualAndReducedCosts(work_prob, state);
        }

        if (observer_ != nullptr) {
            observer_->onIterationEnd(state);
        }
    }

    if (state.iteration >= cfg.max_iterations) {
        if (observer_ != nullptr) {
            observer_->onTermination(state, "Reached max iterations");
        }
        return Status::IterationLimit;
    }

    const int n_primal_core =
        (artificial_column_index >= 0) ? work_prob.numCols() - 1 : work_prob.numCols();
    std::vector<double> reduced_primal(n_primal_core, 0.0);
    for (int i = 0; i < work_prob.numRows(); ++i) {
        const int col = state.basic_indices[i];
        if (col >= 0 && col < n_primal_core) {
            reduced_primal[col] = state.x_basic[i];
        }
    }
    state.dual_solution = state.dual_pi;

    if (has_reduction) {
        state.primal_solution = presolver.postsolvePrimal(reduction, reduced_primal);
    } else {
        state.primal_solution = std::move(reduced_primal);
    }

    return Status::Optimal;
}

int DualSimplex::chuzr(const model::SolverState& state, const SolverConfig& cfg) const {
    if (cfg.use_dual_steepest_edge && state.dse_weights.size() == state.x_basic.size()) {
        int best = -1;
        double best_score = -1.0;
        for (int i = 0; i < static_cast<int>(state.x_basic.size()); ++i) {
            if (state.x_basic[i] >= -cfg.primal_feasibility_tol) {
                continue;
            }
            const double wi = std::max(state.dse_weights[i], cfg.steepest_edge_weight_floor);
            const double score = (state.x_basic[i] * state.x_basic[i]) / wi;
            if (score > best_score) {
                best_score = score;
                best = i;
            }
        }
        if (best >= 0) {
            return best;
        }
    }
    if (row_pivot_) {
        const int row = row_pivot_->chooseRow(state);
        if (row >= 0 && row < static_cast<int>(state.basic_indices.size())) {
            return row;
        }
    }
    return -1;
}

void DualSimplex::btran(util::IndexedVector& ep) const { factor_->btran(ep); }

int DualSimplex::chuzc(
    const model::ProblemData& prob,
    const model::SolverState& state,
    int leaving_row,
    const util::IndexedVector& rho,
    const SolverConfig& cfg,
    util::IndexedVector& pivot_row_out
) const {
    (void)leaving_row;
    if (state.nonbasic_indices.empty() || state.reduced_costs.size() != state.nonbasic_indices.size()) {
        return -1;
    }

    int entering = -1;
    double best_ratio = std::numeric_limits<double>::infinity();
    double delta = std::numeric_limits<double>::infinity();
    std::vector<double> alpha_values(state.nonbasic_indices.size(), 0.0);

    for (size_t k = 0; k < state.nonbasic_indices.size(); ++k) {
        const int col = state.nonbasic_indices[k];
        const double alpha = dotColumn(prob.A, col, rho.rawValues());
        alpha_values[k] = alpha;
        if (std::abs(alpha) > kTiny) {
            pivot_row_out.set(col, alpha);
        }
        if (alpha < -cfg.primal_feasibility_tol) {
            const double cand = (state.reduced_costs[k] + cfg.harris_tolerance) / (-alpha);
            delta = std::min(delta, cand);
            const double ratio = state.reduced_costs[k] / (-alpha);
            if (ratio < best_ratio) {
                best_ratio = ratio;
                entering = col;
            }
        }
    }

    if (!cfg.use_harris_two_pass || !std::isfinite(delta)) {
        return entering;
    }

    int stable_col = -1;
    double max_pivot = -1.0;
    for (size_t k = 0; k < state.nonbasic_indices.size(); ++k) {
        const double alpha = alpha_values[k];
        if (alpha >= -cfg.primal_feasibility_tol) {
            continue;
        }
        const double ratio = state.reduced_costs[k] / (-alpha);
        if (ratio <= delta + cfg.harris_tolerance) {
            const double abs_pivot = std::abs(alpha);
            if (abs_pivot > max_pivot) {
                max_pivot = abs_pivot;
                stable_col = state.nonbasic_indices[k];
            }
        }
    }

    return stable_col >= 0 ? stable_col : entering;
}

void DualSimplex::ftran(util::IndexedVector& aq) const { factor_->ftran(aq); }

bool DualSimplex::initializeBasisAndReducedCosts(
    const model::ProblemData& prob,
    model::SolverState& state,
    const SolverConfig& cfg
) {
    (void)cfg;
    state.x_basic.assign(prob.numRows(), 0.0);
    state.dual_pi.assign(prob.numRows(), 0.0);
    state.reduced_costs.assign(state.nonbasic_indices.size(), 0.0);

    if (!factor_->factorize(buildBasisMatrix(prob, state))) {
        return false;
    }
    computePrimalBasic(prob, state);
    computeDualAndReducedCosts(prob, state);
    initializeDseWeights(state);
    return true;
}

bool DualSimplex::runBigMPhaseOne(
    model::ProblemData& prob,
    model::SolverState& state,
    const SolverConfig& cfg,
    int& artificial_column_index
) {
    artificial_column_index = -1;

    double min_rc = 0.0;
    for (double rc : state.reduced_costs) {
        min_rc = std::min(min_rc, rc);
    }
    if (min_rc >= -cfg.dual_feasibility_tol) {
        return true;
    }

    const std::vector<int> nonbasic_at_detection = state.nonbasic_indices;
    if (nonbasic_at_detection.empty()) {
        return false;
    }

    const double max_c = [&]() {
        double m = 1.0;
        for (double v : prob.c) {
            m = std::max(m, std::abs(v));
        }
        return m;
    }();
    const double big_m = std::max(1.0, cfg.big_m_scale * max_c);

    const int n_old = prob.numCols();
    const int m_old = prob.numRows();
    const int artificial_col = n_old;

    prob = expandWithBoundingConstraint(prob, nonbasic_at_detection, big_m);
    artificial_column_index = artificial_col;

    state.basic_indices.resize(m_old + 1);
    state.basic_indices[m_old] = artificial_col;

    std::vector<bool> in_basis(prob.numCols(), false);
    for (int i = 0; i < prob.numRows(); ++i) {
        const int b = state.basic_indices[i];
        if (b < 0 || b >= prob.numCols() || in_basis[b]) {
            return false;
        }
        in_basis[b] = true;
    }
    state.nonbasic_indices.clear();
    for (int j = 0; j < prob.numCols(); ++j) {
        if (!in_basis[j]) {
            state.nonbasic_indices.push_back(j);
        }
    }

    if (!initializeBasisAndReducedCosts(prob, state, cfg)) {
        return false;
    }

    int entering = -1;
    double best_rc = 0.0;
    for (size_t k = 0; k < state.nonbasic_indices.size(); ++k) {
        const int col = state.nonbasic_indices[k];
        if (col == artificial_col) {
            continue;
        }
        if (state.reduced_costs[k] < best_rc) {
            best_rc = state.reduced_costs[k];
            entering = col;
        }
    }
    if (entering < 0) {
        return false;
    }

    int leaving_row = -1;
    for (int i = 0; i < prob.numRows(); ++i) {
        if (state.basic_indices[i] == artificial_col) {
            leaving_row = i;
            break;
        }
    }
    if (leaving_row < 0) {
        return false;
    }

    util::IndexedVector col_vec = prob.A.column(entering);
    ftran(col_vec);
    if (std::abs(col_vec[leaving_row]) <= kTiny) {
        return false;
    }

    if (observer_ != nullptr) {
        observer_->onPivot(leaving_row, entering, std::abs(best_rc));
    }

    pivot(leaving_row, entering, col_vec, prob, state);

    if (!factor_->factorize(buildBasisMatrix(prob, state))) {
        return false;
    }
    computePrimalBasic(prob, state);
    computeDualAndReducedCosts(prob, state);
    initializeDseWeights(state);

    return true;
}

void DualSimplex::computePrimalBasic(const model::ProblemData& prob, model::SolverState& state) const {
    util::IndexedVector rhs(prob.numRows());
    for (int i = 0; i < prob.numRows(); ++i) {
        if (std::abs(prob.b[i]) > kTiny) {
            rhs.set(i, prob.b[i]);
        }
    }
    ftran(rhs);
    for (int i = 0; i < prob.numRows(); ++i) {
        state.x_basic[i] = rhs[i];
    }
}

void DualSimplex::computeDualAndReducedCosts(const model::ProblemData& prob, model::SolverState& state) const {
    util::IndexedVector cb(prob.numRows());
    const auto cB = basicCosts(prob, state);
    for (int i = 0; i < prob.numRows(); ++i) {
        if (std::abs(cB[i]) > kTiny) {
            cb.set(i, cB[i]);
        }
    }
    btran(cb);

    if (state.dual_pi.size() < static_cast<size_t>(prob.numRows())) {
        throw std::runtime_error("computeDualAndReducedCosts: dual_pi size mismatch");
    }

    for (int i = 0; i < prob.numRows(); ++i) {
        state.dual_pi[i] = cb[i];
    }
    for (size_t k = 0; k < state.nonbasic_indices.size(); ++k) {
        const int col = state.nonbasic_indices[k];
        if (col < 0 || col >= prob.numCols()) {
            throw std::runtime_error("computeDualAndReducedCosts: nonbasic col out of range");
        }
        if (k >= state.reduced_costs.size()) {
            throw std::runtime_error("computeDualAndReducedCosts: reduced_costs size mismatch");
        }
        state.reduced_costs[k] = prob.c[col] - dotColumn(prob.A, col, state.dual_pi);
    }
}

double DualSimplex::computeObjective(const model::ProblemData& prob, const model::SolverState& state) const {
    double obj = 0.0;
    for (int i = 0; i < prob.numRows(); ++i) {
        const int col = state.basic_indices[i];
        obj += prob.c[col] * state.x_basic[i];
    }
    return obj;
}

void DualSimplex::pivot(
    int leaving_row,
    int entering_col,
    const util::IndexedVector& ftran_col,
    const model::ProblemData& prob,
    model::SolverState& state
) {
    (void)prob;
    if (leaving_row < 0 || leaving_row >= static_cast<int>(state.basic_indices.size())) {
        return;
    }

    const int leaving_col = state.basic_indices[leaving_row];
    state.basic_indices[leaving_row] = entering_col;
    for (int& nb : state.nonbasic_indices) {
        if (nb == entering_col) {
            nb = leaving_col;
            break;
        }
    }

    const double pivot_element = ftran_col[leaving_row];
    if (std::abs(pivot_element) > kTiny && leaving_row < static_cast<int>(state.x_basic.size())) {
        const double theta = state.x_basic[leaving_row] / pivot_element;
        for (int i = 0; i < static_cast<int>(state.x_basic.size()); ++i) {
            state.x_basic[i] -= ftran_col[i] * theta;
        }
        state.x_basic[leaving_row] = theta;
    }

    factor_->updateEta(leaving_row, ftran_col);
}

void DualSimplex::updateDuals(
    const model::ProblemData& prob,
    int leaving_row,
    int entering_col,
    const util::IndexedVector& btran_row,
    const util::IndexedVector& ftran_col,
    model::SolverState& state
) const {
    (void)entering_col;
    computeDualAndReducedCosts(prob, state);

    if (state.dse_weights.size() != static_cast<size_t>(prob.numRows())) {
        return;
    }
    const double dp = ftran_col[leaving_row];
    if (std::abs(dp) <= kTiny) {
        return;
    }

    const double w_p_old = state.dse_weights[leaving_row];
    util::IndexedVector v_rhs(prob.numRows());
    for (int i = 0; i < prob.numRows(); ++i) {
        const double val = btran_row[i];
        if (std::abs(val) > kTiny) {
            v_rhs.set(i, val);
        }
    }
    ftran(v_rhs);

    for (int i = 0; i < prob.numRows(); ++i) {
        if (i == leaving_row) {
            state.dse_weights[i] = std::max(1.0, w_p_old / (dp * dp));
            continue;
        }
        const double di = ftran_col[i];
        if (std::abs(di) <= kTiny) {
            continue;
        }
        const double ratio = di / dp;
        const double vi = v_rhs[i];
        const double next = state.dse_weights[i] - 2.0 * ratio * vi + ratio * ratio * w_p_old;
        state.dse_weights[i] = std::max(1.0, next);
    }
}

void DualSimplex::initializeDseWeights(model::SolverState& state) const {
    if (state.dse_weights.size() != state.basic_indices.size()) {
        state.dse_weights.assign(state.basic_indices.size(), 1.0);
    } else {
        std::fill(state.dse_weights.begin(), state.dse_weights.end(), 1.0);
    }
}

}  // namespace simplex
}  // namespace lp_solver
