# Linear Programming Solver

**Language** | [中文版本](README_zh.md)

A C++20 library for **standard-form linear programs** (minimize \(c^\top x\) subject to \(Ax=b\), \(x \ge 0\)) using a **revised dual simplex** core with sparse CSC storage, optional presolve/postsolve, and a pluggable basis factorization interface.

## Features

- **Revised dual simplex**: `CHUZR → BTRAN → CHUZC → FTRAN → pivot` with primal/dual updates and periodic basis refactorization
- **Phase I (Big-M, textbook-style)**: bounding row on the current nonbasic set, artificial in basis, one forcing pivot, then Phase II (see `mat3007h_Project_Manual.tex`)
- **Basis maintenance**: sparse LU of \(B\) plus a **sparse ETA file** (product-form inverse); refactor when `etaFileLength() ≥ refactor_frequency`
- **Hypersparse triangular solves**: Gilbert–Peierls reachability + substitution on extracted CSC \(L/U\) factors (`O(\mathrm{nnz})\) work per solve when RHS is sparse)
- **Pricing**: optional **dual steepest-edge (DSE)** with **Goldfarb–Reid** weight recurrence updated only on FTRAN/BTRAN nonzeros; **Harris two-pass** dual ratio test for entering columns
- **Presolve / postsolve**: column-oriented sparse reductions (no full dense \(A\)); LIFO stack; **`postsolvePrimal`** and **`postsolveDual`** (complementary slackness on the reduction stack)
- **Sparse primitives**: `PackedMatrix` (CSC), `IndexedVector` (tracked nonzeros), `multiply` / `transposeMultiply`
- **Factor backends**: `EigenFactor` (Eigen SparseLU extract + GP solves); `UmfpackFactor` (SuiteSparse UMFPACK when linked, else same sparse engine as Eigen); **`makeDefaultFactor()`** picks UMFPACK when available
- **Cross-platform**: CMake, Windows / Linux / macOS

## Project Structure

```
LinearProgramingSolver/
├── include/lp_solver/
│   ├── linalg/           # IBasisFactor, EigenFactor, UmfpackFactor, EtaFile, sparse LU engine
│   ├── model/            # ProblemData, SolverState
│   ├── presolve/         # Presolver (primal + dual postsolve)
│   ├── simplex/          # DualSimplex, SolverConfig, hooks
│   │   └── detail/       # e.g. hypersparse DSE weight update
│   └── util/             # PackedMatrix, IndexedVector
├── src/
├── tests/
│   ├── smoke_test.cpp
│   ├── stress_test.cpp
│   ├── advanced_features_test.cpp   # presolve, DSE, ETA, default factor, backends
│   └── hypersparse_triangular_test.cpp
├── mat3007h_Project_Manual.tex      # Course implementation manual (reference)
└── CMakeLists.txt
```

## Requirements

- **C++20** (MSVC, GCC, or Clang)
- **CMake 3.20+**
- **Eigen 3.4+** (fetched automatically if not installed and `LP_SOLVER_FETCH_EIGEN=ON`)
- **SuiteSparse UMFPACK** (optional; see build options)

## Third-party Dependencies

- **Eigen** is required for sparse LU extraction and dense fallbacks.
- **SuiteSparse / UMFPACK** is optional (`-DLP_SOLVER_WITH_UMFPACK=ON` by default). If not found, `UmfpackFactor` and `FactorBackend::Default` use the same Eigen-based sparse path as `EigenFactor`.
- If you keep local source trees (e.g. `eigen-5.0.0/`) under the repo for experiments, add them to `.gitignore` to avoid accidental commits.

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

Disable UMFPACK detection:

```bash
cmake -S . -B build -DLP_SOLVER_WITH_UMFPACK=OFF
```

When UMFPACK is linked, the build defines `LP_SOLVER_HAVE_UMFPACK=1` on the `lp_solver` target.

## Tests

| CTest name | Focus |
|------------|--------|
| `smoke_test` | Large `IndexedVector` / `PackedMatrix`, dual simplex with default factor |
| `stress_test` | Medium-scale feasible-basis solve |
| `advanced_features_test` | Presolve/postsolve (primal + dual), Big-M, DSE, ETA file, factor backends, default factor |
| `hypersparse_triangular_test` | Gilbert–Peierls triangular solves vs dense reference |

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
    /* lower_bounds (per variable, often all 0) */,
    /* upper_bounds (per variable) */
};

lp_solver::model::SolverState state;
state.basic_indices = /* size m, valid basis column indices */;

// Recommended: UMFPACK when linked, otherwise Eigen SparseLU
auto factor = lp_solver::linalg::makeDefaultFactor();
// Or explicitly:
// auto factor = lp_solver::linalg::makeFactor(lp_solver::linalg::FactorBackend::Eigen);

lp_solver::simplex::DualSimplex solver(std::move(factor), nullptr, nullptr);

lp_solver::simplex::SolverConfig cfg;
cfg.use_presolve = true;
cfg.enable_big_m_phase_one = true;
cfg.use_dual_steepest_edge = true;
cfg.use_harris_two_pass = true;
cfg.refactor_frequency = 100;

const auto status = solver.solve(problem, state, cfg);
if (status == lp_solver::simplex::DualSimplex::Status::Optimal) {
    // state.primal_solution, state.dual_solution, state.objective
}
```

### `SolverConfig` (selected fields)

| Field | Default | Meaning |
|-------|---------|---------|
| `use_presolve` | `true` | Run sparse presolve before solve; postsolve primal/dual on success |
| `enable_big_m_phase_one` | `true` | Textbook Big-M when dual infeasible at start |
| `use_dual_steepest_edge` | `true` | DSE leaving-row rule (`chuzr`) |
| `use_harris_two_pass` | `true` | Harris ratio test (`chuzc`) |
| `refactor_frequency` | `100` | Refactor LU when ETA length reaches this (0 = never) |
| `big_m_scale` | `1000` | Big-M multiplier vs largest \|c\| |
| `max_iterations` | `10000` | Iteration cap |

### Factor factory

| API | Description |
|-----|-------------|
| `makeDefaultFactor()` | `makeFactor(FactorBackend::Default)` |
| `defaultFactorBackend()` | `Umfpack` if `LP_SOLVER_HAVE_UMFPACK`, else `Eigen` |
| `makeFactor(FactorBackend::Eigen \| Umfpack)` | Force a specific backend |

## API Overview

| Component | Role |
|-----------|------|
| `DualSimplex` | Main solver loop; owns refactor + postsolve on optimal exit |
| `IBasisFactor` | `factorize`, `ftran`, `btran`, `updateEta`, `etaFileLength` |
| `EtaFile` | Sparse product-form eta updates (`O(\sum \mathrm{nnz}(d_i))` apply) |
| `Presolver` | Sparse presolve; `postsolvePrimal` / `postsolveDual` |
| `ProblemData` | \(A\), \(b\), \(c\), bounds |
| `SolverState` | Basis / nonbasis, \(x_B\), \(\pi\), reduced costs, DSE weights, solutions |

## Implementation Notes

- **Hypersparse paths**: triangular solves (Gilbert–Peierls), ETA apply, and DSE Goldfarb–Reid updates iterate `IndexedVector::nonZeroIndices()` instead of full dimension loops where possible.
- **`EigenFactor`**: Eigen `SparseLU` factor extraction + GP-style `ftran`/`btran`; dense fallback inside the engine if extract validation fails.
- **`UmfpackFactor`**: UMFPACK numeric factor + extract when `LP_SOLVER_HAVE_UMFPACK`; otherwise identical sparse engine path as `EigenFactor`.
- **Big-M**: the working problem gains one row and one artificial column until solve end; `primal_solution` omits the artificial column when present.
- **DSE weights**: reset to `1.0` on refactor; between refactors updated via `simplex/detail/dse_weight_update.hpp`.

## References

- Course manual: `mat3007h_Project_Manual.tex`
- Eigen: https://eigen.tuxfamily.org/
- SuiteSparse / UMFPACK: https://people.engr.tamu.edu/davis/suitesparse.html

---

**Last updated**: May 2026
