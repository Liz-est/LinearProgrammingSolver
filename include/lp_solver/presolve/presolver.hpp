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
        double value{0.0};
    };

    struct ReductionResult {
        Status status{Status::Reduced};
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
};

}  // namespace lp_solver::presolve
