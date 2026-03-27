# Executable Test Plan

This plan is command-driven: every step is runnable from PowerShell and produces a pass/fail signal.

## Scope

- Validate current build and test flow for `lp_solver` and `lp_solver_tests`.
- Establish a repeatable baseline before adding Phase 1 contract tests.
- Provide CI-ready commands with non-zero exit codes on failure.

## One-Command Runner

Use the included script:

```powershell
.\run-test-plan.ps1 -Config Debug -BuildDir build
```

Release run:

```powershell
.\run-test-plan.ps1 -Config Release -BuildDir build-release
```

The script fails fast and returns a non-zero exit code if configure, build, or tests fail.

## Preconditions

- Windows 10/11 with PowerShell.
- CMake `>= 3.20`.
- A C++20 compiler toolchain (MSVC, Clang, or GCC via MinGW).

## Quick Start (single-command pipeline)

Run from repository root:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug; `
cmake --build build --config Debug; `
ctest --test-dir build -C Debug --output-on-failure
```

Pass criteria:
- All commands exit with code `0`.
- `ctest` reports `100% tests passed` (currently `smoke_test` + `stress_test`).

Fail criteria:
- Any configure/build/test command returns non-zero.

## Step-by-Step Plan

### 1) Configure

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
```

Checks:
- Build system generated successfully.
- No missing source/include path errors.

### 2) Build

```powershell
cmake --build build --config Debug
```

Checks:
- `lp_solver` library builds.
- `lp_solver_tests` executable links successfully.

### 3) Execute tests

```powershell
ctest --test-dir build -C Debug --output-on-failure
```

Checks:
- `smoke_test` executes and passes.
- No runtime assertion/crash in test binary.

### 4) Verbose rerun for debugging failures

```powershell
ctest --test-dir build -C Debug -V
```

Use when Step 3 fails to inspect full test command/output.

### 5) Run fast or stress tests selectively

Fast test only:

```powershell
ctest --test-dir build -C Debug -R smoke_test --output-on-failure
```

Stress test only:

```powershell
ctest --test-dir build -C Debug -R stress_test --output-on-failure
```

## Clean Rebuild (determinism check)

```powershell
if (Test-Path build) { Remove-Item -Recurse -Force build }
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

Pass criteria:
- Rebuild from empty `build/` reproduces a green run.

## Release Configuration Sanity

```powershell
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --config Release
ctest --test-dir build-release -C Release --output-on-failure
```

Pass criteria:
- Configure/build/test succeed in Release as well as Debug.

## Expansion Gates for Phase 1

As tests are added, this plan remains executable; only expected test count changes.

- Gate A: `IndexedVector` invariants test file added and passing.
- Gate B: `PackedMatrix` CSC contract tests added and passing.
- Gate C: `factorize/ftran/btran` backend contract tests added and passing.
- Gate D: Strategy injection and observer behavior tests added and passing.

Recommended naming:
- `tests/indexed_vector_test.cpp`
- `tests/packed_matrix_test.cpp`
- `tests/factor_contract_test.cpp`
- `tests/dual_simplex_hooks_test.cpp`

Each gate is complete when:
- Test source is compiled into `lp_solver_tests` (or dedicated test executables).
- `ctest --output-on-failure` remains fully green.

## CI Command Set (drop-in)

Use exactly these commands in CI jobs:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

This ensures local and CI behavior stay aligned.
