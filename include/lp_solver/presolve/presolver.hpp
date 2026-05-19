#pragma once

#include <vector>

#include "../model/problem_data.hpp"

namespace lp_solver::presolve {

class Presolver {
public:
    enum class Status { Reduced, Infeasible, Unbounded };
    enum class ReductionKind { FixedZeroColumn, SingletonRow };

    struct PostsolveRecord {
        ReductionKind kind{ReductionKind::FixedZeroColumn};
        int column{-1};
        int row{-1};
        double value{0.0};
        /// Singleton row: coefficient a_{row,column} at elimination time.
        double coefficient{0.0};
        /// Singleton row: objective c_column at elimination time.
        double objective_coef{0.0};
        /// Singleton row: other active rows m with a_{m,column} != 0 (for dual recovery).
        std::vector<int> dual_sum_rows;
        std::vector<double> dual_sum_coefs;
    };

    struct ReductionResult {
        Status status{Status::Reduced};
        int original_num_rows{0};
        model::ProblemData reduced_problem;
        std::vector<int> kept_rows;
        std::vector<int> kept_cols;
        std::vector<double> fixed_values;
        std::vector<PostsolveRecord> postsolve_stack;
        double objective_offset{0.0};
    };

    [[nodiscard]] ReductionResult run(const model::ProblemData& problem) const;
    [[nodiscard]] std::vector<double> postsolvePrimal(
        const ReductionResult& reduction,
        const std::vector<double>& reduced_primal
    ) const;
    [[nodiscard]] std::vector<double> postsolveDual(
        const ReductionResult& reduction,
        const std::vector<double>& reduced_dual
    ) const;
};

}  // namespace lp_solver::presolve
