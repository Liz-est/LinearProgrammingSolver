# Linear Programming Solver

**Language** | [中文版本](README_zh.md)

A C++20 library for **standard-form linear programs** (minimize \(c^\top x\) subject to \(Ax=b\), \(x \ge 0\)) using a **revised dual simplex** core with sparse CSC storage, optional presolve, and a pluggable basis factorization interface.

### Features

- **Revised dual simplex**: `CHUZR → BTRAN → CHUZC → FTRAN → pivot` loop with primal/dual updates
- **Phase I (Big-M, textbook-style)**: explicit bounding row on the current nonbasic set, artificial variable in the basis, one forcing pivot, then Phase II on the augmented problem (see `mat3007h_Project_Manual.tex` in the repo)
- **Basis maintenance**: dense LU factorization of \(B\) plus **ETA-file** (product-form) updates; periodic **refactor** when the ETA chain grows
- **Pricing**: optional **dual steepest-edge** row selection with **Goldfarb–Reid–style** weight updates; **Harris two-pass** dual ratio test for entering columns
- **Presolve / postsolve**: lightweight algebraic reductions with a LIFO stack and primal recovery (`Presolver`)
- **Sparse primitives**: `PackedMatrix` (CSC), `IndexedVector` (tracked nonzeros), `A x` and \(A^\top y\) helpers
- **Factor backends**: `EigenFactor` and `UmfpackFactor` implement the same **built-in dense LU** today (no external Eigen or SuiteSparse link required to build); names are kept for future true backend swap-ins
- **Cross-platform**: CMake, Windows / Linux / macOS

## Project Structure

```
LinearProgramingSolver/
├── include/lp_solver/
│   ├── linalg/ # IBasisFactor, LU + ETA sweep
│   ├── model/                  # ProblemData, SolverState
│   ├── presolve/               # Presolver + postsolve stack
│   ├── simplex/                # DualSimplex, hooks
│   └── util/                   # PackedMatrix, IndexedVector
├── src/
├── tests/
│   ├── smoke_test.cpp
│   ├── stress_test.cpp
│   └── advanced_features_test.cpp
├── mat3007h_Project_Manual.tex # Course implementation manual (reference)
└── CMakeLists.txt
```

## Requirements

- **C++20** (MSVC, GCC, or Clang)
- **CMake 3.20+**

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
```

Release:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Tests

```bash
ctest --test-dir build -C Debug --output-on-failure
```

Windows helper:

```powershell
.\run-test-plan.ps1 -Config Debug -BuildDir build
```

## Usage

```cpp
#include "lp_solver/lp_solver.hpp"

lp_solver::model::ProblemData problem{
    /* PackedMatrix A */,
    /* c */,
    /* b */,
    /* lower_bounds (per variable, often all0) */,
    /* upper_bounds (per variable) */
};

lp_solver::model::SolverState state;
state.basic_indices = /* size m, valid basis column indices */;

auto factor = lp_solver::linalg::makeFactor(lp_solver::linalg::FactorBackend::Eigen);
lp_solver::simplex::DualSimplex solver(std::move(factor), nullptr, nullptr);

lp_solver::simplex::SolverConfig cfg;
// cfg.use_presolve, cfg.enable_big_m_phase_one, cfg.use_dual_steepest_edge,
// cfg.use_harris_two_pass, cfg.refactor_frequency, etc.

const auto status = solver.solve(problem, state, cfg);
if (status == lp_solver::simplex::DualSimplex::Status::Optimal) {
    // state.primal_solution, state.dual_solution, state.objective
}
```

## API Overview

| Component | Role |
|-----------|------|
| `DualSimplex` | Main solver; `SolverConfig` toggles presolve, Big-M, DSE, Harris, refactor cadence |
| `IBasisFactor` | `factorize`, `ftran`, `btran`, `updateEta`, `etaFileLength` |
| `Presolver` | Reduce problem; `postsolvePrimal` maps a core solution back |
| `ProblemData` | \(A\), \(b\), \(c\), bounds |
| `SolverState` | Basis / nonbasis, \(x_B\), \(\pi\), reduced costs, DSE weights, solutions |

## Implementation Notes

- The manual’s **hypersparse triangular solves** and **third-party sparse LU** are described in `mat3007h_Project_Manual.tex`; the current code uses **dense** basis solves for correctness and a smaller dependency surface.
- After Big-M Phase I, the working problem has **one extra row and artificial column** until the end of the solve; reported `primal_solution` strips the artificial column when present.

## References

- Course manual: `mat3007h_Project_Manual.tex`
- Eigen (optional future backend): https://eigen.tuxfamily.org/
- SuiteSparse / UMFPACK (optional future backend): https://people.engr.tamu.edu/davis/suitesparse.html

---

**Last updated**: April 2026
