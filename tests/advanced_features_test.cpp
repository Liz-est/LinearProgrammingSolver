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

}  // namespace

int main() {
    try {
        testEtaLengthAndRefactorReset();
        testBigMEntryPathRuns();
        testPresolveAndPostsolve();
        testSparseFactorBackendsAgreeOnSolve();
        testEigenFactorRandomResiduals();
        testEigenFactorEtaUpdateMath();
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
    return 0;
}
