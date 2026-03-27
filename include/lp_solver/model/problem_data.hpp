#pragma once

#include <vector>

#include "../util/packed_matrix.hpp"

namespace lp_solver::model {

struct ProblemData {
    util::PackedMatrix A;
    std::vector<double> c;
    std::vector<double> b;
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;

    [[nodiscard]] int numRows() const { return A.numRows(); }
    [[nodiscard]] int numCols() const { return A.numCols(); }
};

}  // namespace lp_solver::model
