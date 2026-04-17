#include "../include/lp_solver/linalg/i_basis_factor.hpp"
#include "../include/lp_solver/linalg/umfpack_factor.hpp"
#include "../include/lp_solver/model/problem_data.hpp"
#include "../include/lp_solver/model/solver_state.hpp"
#include "../include/lp_solver/presolve/presolver.hpp"
#include "../include/lp_solver/simplex/dual_simplex.hpp"
#include "../include/lp_solver/util/indexed_vector.hpp"
#include "../include/lp_solver/util/packed_matrix.hpp"

#include <stdexcept>
#include <iostream>
#include <utility>
#include <vector>

namespace {

void expect(bool cond, const char* msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

lp_solver::util::PackedMatrix buildSmallMatrix() {
    lp_solver::util::PackedMatrix::Builder builder(2, 3);
    builder.appendColumn(std::vector<int>{0}, std::vector<double>{1.0});
    builder.appendColumn(std::vector<int>{1}, std::vector<double>{1.0});
    builder.appendColumn(std::vector<int>{0, 1}, std::vector<double>{1.0, -1.0});
    return std::move(builder).build();
}

void testEtaLengthAndRefactorReset() {
    lp_solver::util::PackedMatrix::Builder b(2, 2);
    b.appendColumn(std::vector<int>{0}, std::vector<double>{1.0});
    b.appendColumn(std::vector<int>{1}, std::vector<double>{1.0});
    auto I = std::move(b).build();

    lp_solver::linalg::UmfpackFactor factor;
    expect(factor.factorize(I), "factorize should succeed");
    lp_solver::util::IndexedVector d(2);
    d.set(0, 0.5);
    d.set(1, 1.0);
    factor.updateEta(1, d);
    expect(factor.etaFileLength() == 1, "eta file length should increase");
    expect(factor.factorize(I), "refactorize should succeed");
    expect(factor.etaFileLength() == 0, "eta file length should reset after factorize");
}

void testBigMEntryPathRuns() {
    lp_solver::model::ProblemData prob{
        buildSmallMatrix(),
        std::vector<double>{0.0, 0.0, -1.0},
        std::vector<double>{1.0, 1.0},
        std::vector<double>(3, 0.0),
        std::vector<double>(3, 1.0e30)
    };

    lp_solver::model::SolverState state;
    state.basic_indices = {0, 1};

    auto factor = lp_solver::linalg::makeFactor(lp_solver::linalg::FactorBackend::Umfpack);
    lp_solver::simplex::DualSimplex solver(
        std::move(factor),
        std::unique_ptr<lp_solver::simplex::IRowPivot>{},
        nullptr
    );

    lp_solver::simplex::SolverConfig cfg;
    cfg.max_iterations = 50;
    cfg.use_presolve = false;
    cfg.enable_big_m_phase_one = true;
    const auto status = solver.solve(prob, state, cfg);
    expect(status == lp_solver::simplex::DualSimplex::Status::Optimal ||
               status == lp_solver::simplex::DualSimplex::Status::IterationLimit,
           "big-M path should not immediately fail");
}

void testPresolveAndPostsolve() {
    lp_solver::util::PackedMatrix::Builder builder(2, 3);
    builder.appendColumn(std::vector<int>{0}, std::vector<double>{1.0});         // singleton row variable
    builder.appendColumn(std::vector<int>{}, std::vector<double>{});              // empty column
    builder.appendColumn(std::vector<int>{1}, std::vector<double>{1.0});
    lp_solver::model::ProblemData prob{
        std::move(builder).build(),
        std::vector<double>{2.0, 1.0, 3.0},
        std::vector<double>{4.0, 2.0},
        std::vector<double>(3, 0.0),
        std::vector<double>(3, 1.0e30)
    };

    lp_solver::presolve::Presolver presolver;
    const auto reduced = presolver.run(prob);
    expect(reduced.status == lp_solver::presolve::Presolver::Status::Reduced, "presolve should reduce");
    const std::vector<double> core_solution(reduced.reduced_problem.numCols(), 1.5);
    const auto restored = presolver.postsolvePrimal(reduced, core_solution);
    expect(static_cast<int>(restored.size()) == prob.numCols(), "postsolve size mismatch");
}

}  // namespace

int main() {
    try {
        testEtaLengthAndRefactorReset();
        testBigMEntryPathRuns();
        testPresolveAndPostsolve();
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
    return 0;
}
