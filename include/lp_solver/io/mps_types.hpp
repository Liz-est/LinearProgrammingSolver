#pragma once

#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lp_solver::io {

struct RawConstraint {
    enum class Type { LessEqual, GreaterEqual, Equal };

    std::string name;
    Type type{Type::Equal};
    double rhs{0.0};
    bool has_range{false};
    double range{0.0};
};

struct RawVariableBounds {
    double lower{0.0};
    double upper{std::numeric_limits<double>::infinity()};
};

struct RawLpModel {
    std::string name;
    std::string objective_row;
    bool maximize{false};

    std::vector<std::string> variable_names;
    std::unordered_map<std::string, int> variable_index;

    std::vector<RawConstraint> constraints;
    std::unordered_map<std::string, int> constraint_index;

    // Sparse column-wise entries: [column] -> [(row, value)].
    std::vector<std::vector<std::pair<int, double>>> columns;

    // Objective coefficient per variable index.
    std::vector<double> objective;

    // Bounds per variable index.
    std::vector<RawVariableBounds> bounds;
};

}  // namespace lp_solver::io
