#pragma once

#include "../model/solver_state.hpp"

namespace lp_solver::simplex {

class IRowPivot {
public:
    virtual ~IRowPivot() = default;

    [[nodiscard]] virtual int chooseRow(const model::SolverState& state) const = 0;
};

}  // namespace lp_solver::simplex
