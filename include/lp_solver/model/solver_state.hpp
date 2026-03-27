#pragma once

#include <vector>

namespace lp_solver::model {

struct SolverState {
    std::vector<int> basic_indices;
    std::vector<int> nonbasic_indices;
    std::vector<double> x_basic;
    std::vector<double> reduced_costs;
    std::vector<double> dual_pi;

    int iteration{0};
    double objective{0.0};
};

}  // namespace lp_solver::model
