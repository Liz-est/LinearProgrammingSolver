#pragma once

#include "../model/solver_state.hpp"

namespace lp_solver {
namespace simplex {

class ISolverObserver {
public:
    virtual ~ISolverObserver() = default;
    virtual void onIterationBegin(const model::SolverState&) {}
    virtual void onPivot(int leaving_row, int entering_col, double ratio) {
        (void)leaving_row;
        (void)entering_col;
        (void)ratio;
    }
    virtual void onIterationEnd(const model::SolverState&) {}
    virtual void onTermination(const model::SolverState&, const char* reason) { (void)reason; }
};

}  // namespace simplex
}  // namespace lp_solver
