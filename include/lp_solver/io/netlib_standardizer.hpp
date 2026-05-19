#pragma once

#include <string>
#include <vector>

#include "../model/problem_data.hpp"
#include "mps_types.hpp"

namespace lp_solver::io {

struct StandardizationResult {
    bool ok{false};
    std::string error;

    model::ProblemData problem;
    std::vector<int> initial_basis_indices;
    double objective_offset{0.0};
};

[[nodiscard]] StandardizationResult standardizeNetlibModel(const RawLpModel& raw_model);

}  // namespace lp_solver::io
