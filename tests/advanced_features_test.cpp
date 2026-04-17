#include "../include/lp_solver/linalg/i_basis_factor.hpp"
#include "../include/lp_solver/linalg/eigen_factor.hpp"
#include "../include/lp_solver/linalg/umfpack_factor.hpp"
#include "../include/lp_solver/model/problem_data.hpp"
#include "../include/lp_solver/model/solver_state.hpp"
#include "../include/lp_solver/presolve/presolver.hpp"
#include "../include/lp_solver/simplex/dual_simplex.hpp"
#include "../include/lp_solver/util/indexed_vector.hpp"
#include "../include/lp_solver/util/packed_matrix.hpp"

#include <stdexcept>
#include <iostream>
#include <cmath>
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

void testSparseFactorBackendsAgreeOnSolve() {
    lp_solver::util::PackedMatrix::Builder b(4, 4);
    b.appendColumn(std::vector<int>{0, 1}, std::vector<double>{4.0, 1.0});
    b.appendColumn(std::vector<int>{0, 1, 2}, std::vector<double>{1.0, 3.0, 1.0});
    b.appendColumn(std::vector<int>{1, 2, 3}, std::vector<double>{1.0, 2.5, 0.5});
    b.appendColumn(std::vector<int>{2, 3}, std::vector<double>{1.0, 2.0});
    const auto A = std::move(b).build();

    lp_solver::linalg::EigenFactor eigen_factor;
    lp_solver::linalg::UmfpackFactor umf_factor;
    expect(eigen_factor.factorize(A), "EigenFactor factorize should succeed");
    expect(umf_factor.factorize(A), "UmfpackFactor factorize should succeed");

    lp_solver::util::IndexedVector rhs_f_eig(4);
    rhs_f_eig.set(0, 2.0);
    rhs_f_eig.set(3, -1.0);
    auto rhs_f_umf = rhs_f_eig;
    eigen_factor.ftran(rhs_f_eig);
    umf_factor.ftran(rhs_f_umf);
    for (int i = 0; i < 4; ++i) {
        expect(std::abs(rhs_f_eig[i] - rhs_f_umf[i]) < 1e-8, "ftran mismatch between EigenFactor and UmfpackFactor");
    }

    lp_solver::util::IndexedVector rhs_b_eig(4);
    rhs_b_eig.set(1, 1.5);
    rhs_b_eig.set(2, -0.25);
    auto rhs_b_umf = rhs_b_eig;
    eigen_factor.btran(rhs_b_eig);
    umf_factor.btran(rhs_b_umf);
    for (int i = 0; i < 4; ++i) {
        expect(std::abs(rhs_b_eig[i] - rhs_b_umf[i]) < 1e-8, "btran mismatch between EigenFactor and UmfpackFactor");
    }
}

}  // namespace

int main() {
    try {
        testEtaLengthAndRefactorReset();
        testBigMEntryPathRuns();
        testPresolveAndPostsolve();
        testSparseFactorBackendsAgreeOnSolve();
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
    return 0;
}
