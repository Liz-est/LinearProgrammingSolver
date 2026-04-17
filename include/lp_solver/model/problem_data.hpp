#pragma once

#include <utility>
#include <vector>

#include "../util/packed_matrix.hpp"

namespace lp_solver::model {

struct ProblemData {
    ProblemData()
        : A(util::PackedMatrix::Builder(0, 0).build()),
          c(),
          b(),
          lower_bounds(),
          upper_bounds() {}
    ProblemData(
        util::PackedMatrix matrix,
        std::vector<double> objective,
        std::vector<double> rhs,
        std::vector<double> lower,
        std::vector<double> upper
    )
        : A(std::move(matrix)),
          c(std::move(objective)),
          b(std::move(rhs)),
          lower_bounds(std::move(lower)),
          upper_bounds(std::move(upper)) {}

    util::PackedMatrix A;
    std::vector<double> c;
    std::vector<double> b;
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;

    [[nodiscard]] int numRows() const { return A.numRows(); }
    [[nodiscard]] int numCols() const { return A.numCols(); }
};

}  // namespace lp_solver::model
