# Linear Programming Solver

**Language** | [中文版本](README_zh.md)

A C++20 library for **standard-form linear programs** (minimize c^\top x subject to Ax=b, x \ge 0) using a **revised dual simplex** core with sparse CSC storage, optional presolve/postsolve, and a pluggable basis factorization interface.

## Features

- **Revised dual simplex**: `CHUZR → BTRAN → CHUZC → FTRAN → pivot` with primal/dual updates and periodic basis refactorization
- **Phase I (dual Big-M)**: when the initial basis is dual-infeasible, add a bounding row on the current nonbasic set, put an artificial column in the basis, perform one forcing pivot, then continue Phase II (see `mat3007h_Project_Manual.tex`)
- **Robustness extras**: primal recovery when primal-feasible but dual-infeasible; refactor + alternate-row retry when `chuzc` fails; presolve fallback when the presolved basis cannot factorize
- **Basis maintenance**: sparse LU of B plus a **sparse ETA file** (product-form inverse); refactor when `etaFileLength() ≥ refactor_frequency`
- **Hypersparse triangular solves**: Gilbert–Peierls reachability + substitution on extracted CSC L/U factors (`O(\mathrm{nnz}) work per solve when RHS is sparse)
- **Pricing**: optional **dual steepest-edge (DSE)** with **Goldfarb–Reid** weight recurrence updated only on FTRAN/BTRAN nonzeros; **Harris two-pass** dual ratio test for entering columns
- **Presolve / postsolve**: column-oriented sparse reductions (no full dense A); LIFO stack; `**postsolvePrimal`** and `**postsolveDual`** (complementary slackness on the reduction stack)
- **Sparse primitives**: `PackedMatrix` (CSC), `IndexedVector` (tracked nonzeros), `multiply` / `transposeMultiply`
- **Factor backends**: `EigenFactor` (Eigen SparseLU extract + GP solves); `UmfpackFactor` (SuiteSparse UMFPACK when linked, else same sparse engine as Eigen); `**makeDefaultFactor()`** picks UMFPACK when available
- **Cross-platform**: CMake, Windows / Linux / macOS

## Project Structure

```
LinearProgramingSolver/
├── include/lp_solver/
│   ├── io/               # MPS reader, Netlib standardizer (RawLpModel → ProblemData)
│   ├── linalg/           # IBasisFactor, EigenFactor, UmfpackFactor, EtaFile, sparse LU engine
│   ├── model/            # ProblemData, SolverState
│   ├── presolve/         # Presolver (primal + dual postsolve)
│   ├── simplex/          # DualSimplex, SolverConfig, hooks
│   │   └── detail/       # e.g. hypersparse DSE weight update
│   └── util/             # PackedMatrix, IndexedVector
├── src/
│   └── io/               # mps_reader.cpp, netlib_standardizer.cpp
├── tests/
│   ├── smoke_test.cpp
│   ├── stress_test.cpp
│   ├── advanced_features_test.cpp   # presolve, DSE, ETA, default factor, backends
│   ├── hypersparse_triangular_test.cpp
│   ├── netlib_parser_test.cpp       # MPS parse + standardization smoke checks
│   ├── netlib_runner.cpp            # CLI runner for Netlib `.mps` files
│   └── netlib_baseline.csv          # reference objectives / tolerances (starter set)
├── netlib/                          # local Netlib `.mps` data (gitignored; see below)
├── scripts/
│   └── run-netlib.ps1               # batch runner → CSV summary
├── docs/
│   └── netlib_format_notes.md       # supported MPS subset + standardization notes
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


| CTest name                    | Focus                                                                                     |
| ----------------------------- | ----------------------------------------------------------------------------------------- |
| `smoke_test`                  | Large `IndexedVector` / `PackedMatrix`, dual simplex with default factor                  |
| `stress_test`                 | Medium-scale feasible-basis solve                                                         |
| `advanced_features_test`      | Presolve/postsolve (primal + dual), Big-M, DSE, ETA file, factor backends, default factor |
| `hypersparse_triangular_test` | Gilbert–Peierls triangular solves vs dense reference                                      |
| `netlib_parser_test`          | MPS subset parse + Netlib standardization smoke checks                                    |


```bash
ctest --test-dir build -C Debug --output-on-failure
```

Windows helper:

```powershell
.\run-test-plan.ps1 -Config Debug -BuildDir build
```

## Netlib Benchmark Workflow

The repo ships a Netlib test path: read `.mps` files, standardize to `Ax=b, x>=0`, solve with `DualSimplex`, and compare objectives against reference values.

### Netlib-related files


| Path                                           | Role                                                                                        |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `include/lp_solver/io/mps_reader.hpp`          | MPS reader API (`readMpsFile`)                                                              |
| `include/lp_solver/io/netlib_standardizer.hpp` | Convert parsed model to `ProblemData` + initial slack basis                                 |
| `include/lp_solver/io/mps_types.hpp`           | Intermediate `RawLpModel` types                                                             |
| `src/io/mps_reader.cpp`                        | Fixed-field MPS parser with free-format fallback for loose layouts                          |
| `src/io/netlib_standardizer.cpp`               | Inequalities, bounds, ranges → standard form                                                |
| `tests/netlib_runner.cpp`                      | CLI executable source (`lp_solver_netlib_runner`)                                           |
| `tests/netlib_parser_test.cpp`                 | CTest: parse + standardize smoke checks                                                     |
| `tests/netlib_baseline.csv`                    | Local reference objectives / tolerances                                                     |
| `scripts/run-netlib.ps1`                       | Batch over a data directory; writes a results CSV                                           |
| `docs/netlib_format_notes.md`                  | Supported MPS sections and conversion rules                                                 |
| `netlib/`                                      | Place decompressed `.mps` files here (directory is gitignored)                              |
| `netlib/README.netlib`                         | Official Netlib problem table; batch script reads optimal objectives from here when present |
| `netlib-results-*.csv`                         | Batch output (gitignored); one row per problem with status, objective, match flag           |


### Prepare data

1. Obtain Netlib LP models (e.g. [Netlib LP test set](http://www.netlib.org/lp/data/)).
2. Decompress `.mps.gz` → plain `.mps` (the reader does not read gzip directly).
3. Put `.mps` files under `netlib/` or any directory you pass to `-DataDir`.

Ground truth for objectives: `**netlib/README.netlib**` (problem summary table). `run-netlib.ps1` merges it with `tests/netlib_baseline.csv` (CSV entries override on name collision).

### Build the runner

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug --target lp_solver_netlib_runner
```

On single-config generators (Ninja, Make), the binary is usually `build/lp_solver_netlib_runner.exe` (Windows) or `build/lp_solver_netlib_runner`.

### Run one problem

```powershell
.\build\lp_solver_netlib_runner.exe .\netlib\afiro.mps --ref -4.6475314286E+02 --tol 1e-6
```

Key-value lines are printed to stdout (`status`, `objective`, `iterations`, `objective_match`, …). Exit code `0` = optimal and reference match (when `--ref` is given); `10` = objective mismatch; `11` = non-optimal status.

#### `lp_solver_netlib_runner` options


| Flag                           | Default            | Meaning                                                      |
| ------------------------------ | ------------------ | ------------------------------------------------------------ |
| `--ref <obj>`                  | (none)             | Reference optimal objective; enables `objective_match` check |
| `--tol <t>`                    | `1e-6`             | Relative tolerance scaled by                                 |
| `--max-iters <N>`              | `50000`            | Iteration cap                                                |
| `--no-presolve` / `--presolve` | presolve on        | Toggle presolve                                              |
| `--no-big-m` / `--big-m`       | **no Big-M first** | Toggle dual Big-M Phase I for a single run                   |
| `--no-dse`                     | DSE on             | Disable dual steepest-edge leaving-row rule                  |
| `--no-harris`                  | Harris on          | Disable Harris two-pass entering-column rule                 |
| `--big-m-scale <X>`            | `1000`             | Big-M multiplier vs max                                      |
| `--refactor-freq <N>`          | `100`              | Refactor when ETA length ≥ N                                 |
| `--verbose`                    | off                | Print pivot / termination diagnostics                        |


**Strategy fallback (runner only):** the runner tries up to four `(Big-M on/off) × (DSE on/off)` combinations in order and stops at the first `Optimal` result. This is generic robustness logic.

### Batch run

```powershell
.\scripts\run-netlib.ps1 -DataDir .\netlib -BuildDir build -Config Debug `
  -BaselineCsv tests\netlib_baseline.csv -OutputCsv netlib-results.csv
```

The script:

- Finds `lp_solver_netlib_runner` under `BuildDir` (with or without `Config` subfolder).
- Runs every `*.mps` in `-DataDir`.
- Passes `--ref` / `--tol` when a baseline row exists (from CSV and/or `README.netlib`).
- Writes `-OutputCsv` (default: `netlib-results-<timestamp>.csv`).

With the bundled `netlib/` set and current solver, all **40** models pass objective checks against `README.netlib` (tolerance mostly `1e-6`; `GANGES` uses `1e-5` in baseline CSV).

### Standardization

The standardizer converts general MPS rows/bounds/ranges into `min c'x, Ax=b, x>=0`, records objective offsets, and builds an explicit initial basis from slack/unit columns. Details: `docs/netlib_format_notes.md`.

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


| Field                    | Default | Meaning                                                            |
| ------------------------ | ------- | ------------------------------------------------------------------ |
| `use_presolve`           | `true`  | Run sparse presolve before solve; postsolve primal/dual on success |
| `enable_big_m_phase_one` | `true`  | Big-M when dual infeasible at start                                |
| `use_dual_steepest_edge` | `true`  | DSE leaving-row rule (`chuzr`)                                     |
| `use_harris_two_pass`    | `true`  | Harris ratio test (`chuzc`)                                        |
| `refactor_frequency`     | `100`   | Refactor LU when ETA length reaches this (0 = never)               |
| `big_m_scale`            | `1000`  | Big-M multiplier vs largest                                        |
| `max_iterations`         | `10000` | Iteration cap                                                      |


Note: `lp_solver_netlib_runner` uses `max_iterations = 50000` and tries multiple `(enable_big_m_phase_one, use_dual_steepest_edge)` combinations before giving up; library callers set `SolverConfig` directly.

### Factor factory


| API                              | Description                                         |
| -------------------------------- | --------------------------------------------------- |
| `makeDefaultFactor()`            | `makeFactor(FactorBackend::Default)`                |
| `defaultFactorBackend()`         | `Umfpack` if `LP_SOLVER_HAVE_UMFPACK`, else `Eigen` |
| `makeFactor(FactorBackend::Eigen | Umfpack)`                                           |


## API Overview


| Component      | Role                                                                |
| -------------- | ------------------------------------------------------------------- |
| `DualSimplex`  | Main solver loop; owns refactor + postsolve on optimal exit         |
| `IBasisFactor` | `factorize`, `ftran`, `btran`, `updateEta`, `etaFileLength`         |
| `EtaFile`      | Sparse product-form eta updates (`O(\sum \mathrm{nnz}(d_i))` apply) |
| `Presolver`    | Sparse presolve; `postsolvePrimal` / `postsolveDual`                |
| `ProblemData`  | A, b, c, bounds                                                     |
| `SolverState`  | Basis / nonbasis, x_B, \pi, reduced costs, DSE weights, solutions   |


## Implementation Notes

- **Hypersparse paths**: triangular solves (Gilbert–Peierls), ETA apply, and DSE Goldfarb–Reid updates iterate `IndexedVector::nonZeroIndices()` instead of full dimension loops where possible.
- `**EigenFactor`**: Eigen `SparseLU` factor extraction + GP-style `ftran`/`btran`; dense fallback inside the engine if extract validation fails.
- `**UmfpackFactor`**: UMFPACK numeric factor + extract when `LP_SOLVER_HAVE_UMFPACK`; otherwise identical sparse engine path as `EigenFactor`.
- **Dual Big-M**: if reduced costs are dual-infeasible at start, the working problem gains one bounding row and one artificial column; one forcing pivot is applied, then Phase II continues. Returned `primal_solution` omits the artificial column when present.
- **Optimality check**: when no leaving row is found (primal feasible), the solver verifies dual feasibility (`minReducedCost`) before stopping; if dual-infeasible, it runs primal simplex pivots from the current basis until dual feasibility is restored or progress stops.
- `**chuzc` recovery**: on failed entering-column selection, refactor and retry; if still stuck, scan other primal-infeasible rows; only then report infeasible.
- **Presolve fallback**: if the presolved initial basis cannot be factorized, the solve is retried without presolve.
- **MPS reader**: tries fixed-column fields first; falls back to whitespace tokenization when fixed fields do not match known row/column names (needed for some Netlib layouts such as FORPLAN).
- **DSE weights**: reset to `1.0` on refactor; between refactors updated via `simplex/detail/dse_weight_update.hpp`.

## References

- Course manual: `mat3007h_Project_Manual.tex`
- Eigen: [https://eigen.tuxfamily.org/](https://eigen.tuxfamily.org/)
- SuiteSparse / UMFPACK: [https://people.engr.tamu.edu/davis/suitesparse.html](https://people.engr.tamu.edu/davis/suitesparse.html)

---

**Last updated**: May 2026