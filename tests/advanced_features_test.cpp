#include "../include/lp_solver/linalg/eta_file.hpp"
#include "../include/lp_solver/linalg/i_basis_factor.hpp"
#include "../include/lp_solver/linalg/eigen_factor.hpp"
#include "../include/lp_solver/linalg/umfpack_factor.hpp"
#include "../include/lp_solver/model/problem_data.hpp"
#include "../include/lp_solver/model/solver_state.hpp"
#include "../include/lp_solver/presolve/presolver.hpp"
#include "../include/lp_solver/simplex/detail/dse_weight_update.hpp"
#include "../include/lp_solver/simplex/dual_simplex.hpp"
#include "../include/lp_solver/util/indexed_vector.hpp"
#include "../include/lp_solver/util/packed_matrix.hpp"

#include <stdexcept>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
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

std::vector<double> toDense(const lp_solver::util::IndexedVector& v) {
    return v.rawValues();
}

lp_solver::util::IndexedVector toIndexed(const std::vector<double>& v) {
    lp_solver::util::IndexedVector out(static_cast<int>(v.size()));
    for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        if (std::abs(v[static_cast<size_t>(i)]) > 1e-14) {
            out.set(i, v[static_cast<size_t>(i)]);
        }
    }
    return out;
}

double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
    expect(a.size() == b.size(), "size mismatch in maxAbsDiff");
    double d = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, std::abs(a[i] - b[i]));
    }
    return d;
}

lp_solver::util::PackedMatrix buildRandomWellConditionedSparse(int n, std::mt19937& rng) {
    std::uniform_real_distribution<double> val_dist(-0.03, 0.03);
    std::uniform_int_distribution<int> count_dist(1, std::max(1, n / 2));
    std::uniform_int_distribution<int> row_dist(0, n - 1);

    lp_solver::util::PackedMatrix::Builder b(n, n);
    for (int col = 0; col < n; ++col) {
        std::vector<int> rows{col};
        std::vector<double> vals{8.0 + static_cast<double>(col % 3)};

        const int extra = count_dist(rng);
        for (int k = 0; k < extra; ++k) {
            const int r = row_dist(rng);
            if (r == col) {
                continue;
            }
            rows.push_back(r);
            vals.push_back(val_dist(rng));
        }

        std::vector<std::pair<int, double>> entries;
        entries.reserve(rows.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            entries.emplace_back(rows[i], vals[i]);
        }
        std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });

        std::vector<int> merged_rows;
        std::vector<double> merged_vals;
        for (const auto& [r, v] : entries) {
            if (!merged_rows.empty() && merged_rows.back() == r) {
                merged_vals.back() += v;
            } else {
                merged_rows.push_back(r);
                merged_vals.push_back(v);
            }
        }
        b.appendColumn(merged_rows, merged_vals);
    }
    return std::move(b).build();
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

void testPresolveAndPostsolveDual() {
    lp_solver::util::PackedMatrix::Builder builder(2, 3);
    builder.appendColumn(std::vector<int>{0}, std::vector<double>{1.0});
    builder.appendColumn(std::vector<int>{}, std::vector<double>{});
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
    expect(reduced.original_num_rows == 2, "original row count");

    expect(reduced.reduced_problem.numRows() == 0, "both rows are singleton-eliminated");
    expect(reduced.reduced_problem.numCols() == 0, "all columns fixed or empty");

    const std::vector<double> core_primal;
    const std::vector<double> core_dual;

    const auto full_x = presolver.postsolvePrimal(reduced, core_primal);
    const auto full_pi = presolver.postsolveDual(reduced, core_dual);

    expect(static_cast<int>(full_x.size()) == prob.numCols(), "primal postsolve size");
    expect(static_cast<int>(full_pi.size()) == prob.numRows(), "dual postsolve size");
    expect(std::abs(full_x[0] - 4.0) < 1e-10, "singleton fixed primal x0");
    expect(std::abs(full_x[1]) < 1e-10, "empty column primal x1");
    expect(std::abs(full_x[2] - 2.0) < 1e-10, "second singleton fixed primal x2");
    expect(std::abs(full_pi[0] - 2.0) < 1e-10, "first singleton recovered dual pi0");
    expect(std::abs(full_pi[1] - 3.0) < 1e-10, "second singleton recovered dual pi1");

    // Complementary slackness for fixed positive column 0: c0 - A^T pi = 0 on column 0.
    const auto col0 = prob.A.column(0);
    double rc0 = prob.c[0];
    for (int r : col0.nonZeroIndices()) {
        rc0 -= full_pi[static_cast<size_t>(r)] * col0[r];
    }
    expect(std::abs(rc0) < 1e-10, "recovered dual satisfies reduced cost for fixed column");
}

void testPresolveSolvePostsolveDualEndToEnd() {
    // One singleton row (x0 = 4); second row couples x1 + x2 = 5 so the core stays 1x2.
    lp_solver::util::PackedMatrix::Builder builder(2, 3);
    builder.appendColumn(std::vector<int>{0}, std::vector<double>{1.0});
    builder.appendColumn(std::vector<int>{1}, std::vector<double>{1.0});
    builder.appendColumn(std::vector<int>{1}, std::vector<double>{1.0});
    lp_solver::model::ProblemData prob{
        std::move(builder).build(),
        std::vector<double>{1.0, 2.0, 3.0},
        std::vector<double>{4.0, 5.0},
        std::vector<double>(3, 0.0),
        std::vector<double>(3, 1.0e30)
    };

    lp_solver::model::SolverState state;
    state.basic_indices = {0, 1};

    auto factor = lp_solver::linalg::makeFactor(lp_solver::linalg::FactorBackend::Eigen);
    lp_solver::simplex::DualSimplex solver(std::move(factor), nullptr, nullptr);

    lp_solver::simplex::SolverConfig cfg;
    cfg.max_iterations = 100;
    cfg.use_presolve = true;
    cfg.enable_big_m_phase_one = false;

    const auto status = solver.solve(prob, state, cfg);
    expect(status == lp_solver::simplex::DualSimplex::Status::Optimal, "solve should be optimal");

    expect(static_cast<int>(state.primal_solution.size()) == prob.numCols(), "full primal size");
    expect(static_cast<int>(state.dual_solution.size()) == prob.numRows(), "full dual size");
    expect(std::abs(state.primal_solution[0] - 4.0) < 1e-7, "optimal x0");
    expect(std::abs(state.primal_solution[1] - 5.0) < 1e-7, "optimal x1");
    expect(std::abs(state.primal_solution[2]) < 1e-7, "optimal x2");
    expect(std::abs(state.dual_solution[0] - 1.0) < 1e-6, "optimal pi0 from postsolve");
    expect(std::abs(state.dual_solution[1] - 2.0) < 1e-6, "optimal pi1 on surviving row");

    const auto ax = prob.A.multiply(state.primal_solution);
    expect(maxAbsDiff(ax, prob.b) < 1e-6, "primal feasibility on original problem");
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

void testDefaultFactorBackendSelection() {
#if LP_SOLVER_HAVE_UMFPACK
    expect(
        lp_solver::linalg::defaultFactorBackend() == lp_solver::linalg::FactorBackend::Umfpack,
        "default backend should be UMFPACK when SuiteSparse is linked"
    );
#else
    expect(
        lp_solver::linalg::defaultFactorBackend() == lp_solver::linalg::FactorBackend::Eigen,
        "default backend should be Eigen when UMFPACK is unavailable"
    );
#endif

    lp_solver::util::PackedMatrix::Builder b(3, 3);
    b.appendColumn(std::vector<int>{0}, std::vector<double>{1.0});
    b.appendColumn(std::vector<int>{1}, std::vector<double>{1.0});
    b.appendColumn(std::vector<int>{2}, std::vector<double>{1.0});
    const auto I = std::move(b).build();

    auto from_default = lp_solver::linalg::makeDefaultFactor();
    auto from_enum = lp_solver::linalg::makeFactor(lp_solver::linalg::FactorBackend::Default);
    auto from_resolved = lp_solver::linalg::makeFactor(lp_solver::linalg::defaultFactorBackend());

    expect(from_default->factorize(I), "makeDefaultFactor should factorize");
    expect(from_enum->factorize(I), "makeFactor(Default) should factorize");
    expect(from_resolved->factorize(I), "makeFactor(resolved default) should factorize");

#if LP_SOLVER_HAVE_UMFPACK
    expect(
        dynamic_cast<lp_solver::linalg::UmfpackFactor*>(from_default.get()) != nullptr,
        "default factor should be UmfpackFactor"
    );
#else
    expect(
        dynamic_cast<lp_solver::linalg::EigenFactor*>(from_default.get()) != nullptr,
        "default factor should be EigenFactor"
    );
#endif

    lp_solver::util::IndexedVector rhs(3);
    rhs.set(0, 1.0);
    rhs.set(2, -0.5);
    auto rhs_enum = rhs;
    auto rhs_resolved = rhs;

    from_default->ftran(rhs);
    from_enum->ftran(rhs_enum);
    from_resolved->ftran(rhs_resolved);
    expect(maxAbsDiff(toDense(rhs), toDense(rhs_enum)) < 1e-12, "Default enum matches makeDefaultFactor ftran");
    expect(maxAbsDiff(toDense(rhs), toDense(rhs_resolved)) < 1e-12, "resolved backend matches default ftran");
}

void testDualSimplexWithDefaultFactor() {
    lp_solver::model::ProblemData prob{
        buildSmallMatrix(),
        std::vector<double>{0.0, 0.0, 1.0},
        std::vector<double>{1.0, 1.0},
        std::vector<double>(3, 0.0),
        std::vector<double>(3, 1.0e30)
    };

    lp_solver::model::SolverState state;
    state.basic_indices = {0, 1};
    state.x_basic = {1.0, 1.0};

    lp_solver::simplex::SolverConfig cfg;
    cfg.use_presolve = false;
    cfg.enable_big_m_phase_one = false;

    lp_solver::simplex::DualSimplex solver(lp_solver::linalg::makeDefaultFactor(), nullptr, nullptr);
    const auto status = solver.solve(prob, state, cfg);
    expect(status == lp_solver::simplex::DualSimplex::Status::Optimal, "DualSimplex with default factor should be optimal");
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

void testEigenFactorRandomResiduals() {
    std::mt19937 rng(20260417);
    std::uniform_real_distribution<double> rhs_dist(-2.0, 2.0);

    for (int trial = 0; trial < 12; ++trial) {
        const int n = 6 + (trial % 5);
        const auto A = buildRandomWellConditionedSparse(n, rng);

        lp_solver::linalg::EigenFactor factor;
        expect(factor.factorize(A), "EigenFactor factorize should succeed on random sparse matrix");

        std::vector<double> b(static_cast<size_t>(n), 0.0);
        for (int i = 0; i < n; ++i) {
            b[static_cast<size_t>(i)] = rhs_dist(rng);
        }

        auto x_idx = toIndexed(b);
        factor.ftran(x_idx);
        const auto x = toDense(x_idx);
        const auto ax = A.multiply(x);
        expect(maxAbsDiff(ax, b) < 1e-8, "ftran residual too large for EigenFactor");

        auto y_idx = toIndexed(b);
        factor.btran(y_idx);
        const auto y = toDense(y_idx);
        const auto aty = A.transposeMultiply(y);
        expect(maxAbsDiff(aty, b) < 1e-8, "btran residual too large for EigenFactor");
    }
}

void testEtaFileSparseStorageAndApply() {
    constexpr int n = 200;
    lp_solver::linalg::EtaFile file;
    lp_solver::util::IndexedVector eta_vec(n);
    eta_vec.set(3, 2.0);
    eta_vec.set(99, -1.5);
    eta_vec.set(150, 0.5);
    file.append(150, eta_vec);
    expect(file.length() == 1, "eta file length");
    expect(file.updates().size() == 1, "one eta update");
    expect(file.updates().front().indices.size() == 3, "store sparse eta nnz only");

    lp_solver::util::IndexedVector v(n);
    v.set(10, 1.0);
    file.applyForward(v);

    lp_solver::util::IndexedVector v_ref(n);
    v_ref.set(10, 1.0);
    const double xp = v_ref[150] / 0.5;
    v_ref.set(150, xp);
    v_ref.add(3, -2.0 * xp);
    v_ref.add(99, 1.5 * xp);

    expect(std::abs(v[10] - v_ref[10]) < 1e-12, "untouched index unchanged");
    expect(std::abs(v[150] - v_ref[150]) < 1e-12, "sparse forward pivot");
    expect(std::abs(v[3] - v_ref[3]) < 1e-12, "sparse forward fill");
    expect(std::abs(v[99] - v_ref[99]) < 1e-12, "sparse forward fill");
}

void testEigenFactorEtaUpdateMath() {
    lp_solver::util::PackedMatrix::Builder b(3, 3);
    b.appendColumn(std::vector<int>{0}, std::vector<double>{1.0});
    b.appendColumn(std::vector<int>{1}, std::vector<double>{1.0});
    b.appendColumn(std::vector<int>{2}, std::vector<double>{1.0});
    const auto I = std::move(b).build();

    lp_solver::linalg::EigenFactor factor;
    expect(factor.factorize(I), "identity factorization should succeed");

    lp_solver::util::IndexedVector eta_col0(3);
    eta_col0.set(0, 2.0);
    eta_col0.set(1, 0.5);
    eta_col0.set(2, -1.0);
    factor.updateEta(0, eta_col0);

    lp_solver::util::IndexedVector eta_col2(3);
    eta_col2.set(0, 0.25);
    eta_col2.set(1, -0.5);
    eta_col2.set(2, 1.5);
    factor.updateEta(2, eta_col2);
    expect(factor.etaFileLength() == 2, "eta length should reflect two updates");

    // M^{-1} = E2^{-1} * E1^{-1} for ftran path.
    const std::vector<double> rhs{3.0, -1.0, 2.0};
    auto f_idx = toIndexed(rhs);
    factor.ftran(f_idx);
    const auto got_f = toDense(f_idx);
    const std::vector<double> expect_f{
        0.9166666666666666,
        -0.5833333333333333,
        2.3333333333333335
    };
    expect(maxAbsDiff(got_f, expect_f) < 1e-10, "ftran eta composition mismatch");

    // M^{-T} = E1^{-T} * E2^{-T} for btran path.
    auto b_idx = toIndexed(rhs);
    factor.btran(b_idx);
    const auto got_b = toDense(b_idx);
    const std::vector<double> expect_b{
        2.0,
        -1.0,
        0.5
    };
    expect(maxAbsDiff(got_b, expect_b) < 1e-10, "btran eta composition mismatch");
}

void testGoldfarbReidDseHypersparseMatchesDense() {
    constexpr int m = 5;
    lp_solver::util::PackedMatrix::Builder b(m, m);
    b.appendColumn(std::vector<int>{0}, std::vector<double>{1.0});
    b.appendColumn(std::vector<int>{0, 1}, std::vector<double>{1.0, 1.0});
    b.appendColumn(std::vector<int>{1, 2}, std::vector<double>{1.0, 1.0});
    b.appendColumn(std::vector<int>{2, 3}, std::vector<double>{1.0, 1.0});
    b.appendColumn(std::vector<int>{3, 4}, std::vector<double>{1.0, 1.0});
    const auto basis = std::move(b).build();

    lp_solver::linalg::EigenFactor factor;
    expect(factor.factorize(basis), "DSE weight test factorization failed");

    lp_solver::util::IndexedVector eta0(m);
    eta0.set(0, 2.0);
    eta0.set(2, -0.5);
    factor.updateEta(0, eta0);

    lp_solver::util::IndexedVector eta2(m);
    eta2.set(2, 1.25);
    eta2.set(4, 0.75);
    factor.updateEta(2, eta2);

    const int leaving_row = 2;
    lp_solver::util::IndexedVector rho(m);
    rho.set(leaving_row, 1.0);
    factor.btran(rho);

    lp_solver::util::IndexedVector aq(m);
    aq.set(0, 1.0);
    aq.set(3, -0.5);
    factor.ftran(aq);

    const auto apply_ftran = [&factor](lp_solver::util::IndexedVector& v) { factor.ftran(v); };

    std::vector<double> w_sparse(m, 1.0);
    std::vector<double> w_dense(m, 1.0);
    lp_solver::util::IndexedVector v_sparse(m);
    lp_solver::util::IndexedVector v_dense(m);

    lp_solver::simplex::detail::goldfarbReidDseWeightUpdate(
        leaving_row, rho, aq, v_sparse, apply_ftran, w_sparse);
    lp_solver::simplex::detail::goldfarbReidDseWeightUpdateDense(
        leaving_row, rho, aq, v_dense, apply_ftran, w_dense);

    expect(maxAbsDiff(w_sparse, w_dense) < 1e-12, "hypersparse DSE update should match dense reference");
}

lp_solver::util::PackedMatrix buildIdentityWithSparseExtraCols(int m, int n) {
    lp_solver::util::PackedMatrix::Builder builder(m, n);
    for (int j = 0; j < m; ++j) {
        builder.appendColumn(std::vector<int>{j}, std::vector<double>{1.0});
    }
    for (int j = m; j < n; ++j) {
        const int r1 = j % m;
        const int r2 = (j * 37 + 11) % m;
        if (r1 == r2) {
            builder.appendColumn(std::vector<int>{r1}, std::vector<double>{0.5});
        } else {
            builder.appendColumn(std::vector<int>{r1, r2}, std::vector<double>{0.5, -0.25});
        }
    }
    return std::move(builder).build();
}

void testDseSolveOnSparseModel() {
    constexpr int m = 80;
    constexpr int n = 160;
    lp_solver::model::ProblemData prob{
        buildIdentityWithSparseExtraCols(m, n),
        std::vector<double>(n, 1.0),
        std::vector<double>(m, 0.0),
        std::vector<double>(n, 0.0),
        std::vector<double>(n, 1.0e20)
    };

    lp_solver::model::SolverState state;
    state.basic_indices.resize(m);
    state.x_basic.assign(m, 1.0);
    for (int i = 0; i < m; ++i) {
        state.basic_indices[i] = i;
    }

    lp_solver::simplex::SolverConfig cfg;
    cfg.use_dual_steepest_edge = true;
    cfg.use_presolve = false;
    cfg.enable_big_m_phase_one = false;
    cfg.refactor_frequency = 10;

    auto factor = lp_solver::linalg::makeFactor(lp_solver::linalg::FactorBackend::Eigen);
    lp_solver::simplex::DualSimplex solver(std::move(factor), nullptr, nullptr);
    const auto status = solver.solve(prob, state, cfg);
    expect(status == lp_solver::simplex::DualSimplex::Status::Optimal, "sparse DSE solve should be optimal");
    expect(state.dse_weights.size() == static_cast<size_t>(m), "DSE weights sized to basis rows");
}

void testPresolveSparseStructure() {
    constexpr int m = 40;
    constexpr int n = 500;
    lp_solver::util::PackedMatrix::Builder builder(m, n);
    builder.appendColumn(std::vector<int>{0}, std::vector<double>{1.0});
    for (int j = 1; j < n; ++j) {
        const int row = 1 + ((j - 1) % (m - 1));
        builder.appendColumn(std::vector<int>{row}, std::vector<double>{1.0});
    }
    lp_solver::model::ProblemData prob{
        std::move(builder).build(),
        std::vector<double>(n, 0.0),
        std::vector<double>(m, 0.0),
        std::vector<double>(n, 0.0),
        std::vector<double>(n, 1.0e30)
    };
    prob.b[0] = 7.0;
    prob.c[0] = 2.0;

    lp_solver::presolve::Presolver presolver;
    const auto reduced = presolver.run(prob);
    expect(reduced.status == lp_solver::presolve::Presolver::Status::Reduced, "sparse presolve should reduce");
    expect(prob.A.numNonZeros() == n, "fixture stays sparse in CSC");
    expect(reduced.reduced_problem.A.numNonZeros() <= n, "reduced nnz does not exceed original");
    expect(reduced.reduced_problem.A.numNonZeros() < m * n, "avoid dense m-by-n fill");
    expect(std::abs(reduced.fixed_values[0] - 7.0) < 1e-10, "singleton primal fix");

    const std::vector<double> core(
        static_cast<size_t>(reduced.reduced_problem.numCols()),
        1.0
    );
    const auto full_x = presolver.postsolvePrimal(reduced, core);
    expect(static_cast<int>(full_x.size()) == prob.numCols(), "sparse postsolve primal size");
    expect(std::abs(full_x[0] - 7.0) < 1e-10, "sparse postsolve restores singleton");
}

}  // namespace

int main() {
    try {
        testEtaLengthAndRefactorReset();
        testBigMEntryPathRuns();
        testPresolveAndPostsolveDual();
        testPresolveSolvePostsolveDualEndToEnd();
        testPresolveAndPostsolve();
        testPresolveSparseStructure();
        testDefaultFactorBackendSelection();
        testDualSimplexWithDefaultFactor();
        testSparseFactorBackendsAgreeOnSolve();
        testEigenFactorRandomResiduals();
        testEtaFileSparseStorageAndApply();
        testEigenFactorEtaUpdateMath();
        testGoldfarbReidDseHypersparseMatchesDense();
        testDseSolveOnSparseModel();
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
    return 0;
}
