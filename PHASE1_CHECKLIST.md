# Phase 1 Implementation Checklist (Mapped to Current Files)

This checklist follows the project manual Phase 1 scope and maps each item to the scaffolded file layout.

## 1) Sparse Primitives (`util/`)

- [x] Implement `IndexedVector` non-zero invariant maintenance  
  - `include/lp_solver/util/indexed_vector.hpp`  
  - `src/util/indexed_vector.cpp`
- [x] Support accumulation and overwrite with zero-erasure semantics  
  - `src/util/indexed_vector.cpp`
- [x] Expose active index iteration for hypersparse loops  
  - `include/lp_solver/util/indexed_vector.hpp`
- [x] Build immutable CSC matrix via builder  
  - `include/lp_solver/util/packed_matrix.hpp`  
  - `src/util/packed_matrix.cpp`
- [x] Validate CSC construction constraints (sizes, row bounds, final `col_starts`)  
  - `src/util/packed_matrix.cpp`
- [ ] Add matrix-vector multiply helpers (`A*x`, `A^T*x`) for pricing and updates  
  - `src/util/packed_matrix.cpp` (next)

## 2) Factorization Abstraction (`linalg/`)

- [x] Define backend-agnostic basis interface (`factorize/ftran/btran/updateEta`)  
  - `include/lp_solver/linalg/i_basis_factor.hpp`
- [x] Provide backend factory  
  - `src/linalg/factor_factory.cpp`
- [ ] Integrate real SparseLU backend (Eigen)  
  - `include/lp_solver/linalg/eigen_factor.hpp`  
  - `src/linalg/eigen_factor.cpp`
- [ ] Integrate real UMFPACK backend  
  - `include/lp_solver/linalg/umfpack_factor.hpp`  
  - `src/linalg/umfpack_factor.cpp`
- [ ] Add singularity/condition diagnostics and error propagation  
  - `src/linalg/eigen_factor.cpp`  
  - `src/linalg/umfpack_factor.cpp`

## 3) Problem Model and Mutable State (`model/`)

- [x] Define immutable LP container (`A, b, c, bounds`)  
  - `include/lp_solver/model/problem_data.hpp`
- [x] Define mutable solver state (`basis/nonbasis/x_B/reduced costs/dual`)  
  - `include/lp_solver/model/solver_state.hpp`
- [ ] Add robust initialization helpers from a standard-form LP  
  - `src/model/*` (new files suggested)
- [ ] Add basis validity checks (unique indices, dimensions, bounds)  
  - `src/model/*` or `src/simplex/dual_simplex.cpp`

## 4) Dual Simplex Interfaces and Orchestration (`simplex/`)

- [x] Define row-pivot strategy seam  
  - `include/lp_solver/simplex/i_row_pivot.hpp`
- [x] Define observer seam  
  - `include/lp_solver/simplex/i_solver_observer.hpp`
- [x] Implement a minimal working `solve()` loop with:
  - input/state validation
  - factorization of current basis
  - CHUZR fallback (most-infeasible row)
  - CHUZC fallback (first nonbasic)
  - pivot state update
  - observer callbacks
  - iteration limit and termination statuses  
  - `src/simplex/dual_simplex.cpp`
- [ ] Implement true BTRAN computation (`B^T pi = e_p`) and pivot row values  
  - `src/simplex/dual_simplex.cpp`
- [ ] Implement true CHUZC dual ratio test (`min cbar_j / |alpha_pj|` for `alpha_pj < 0`)  
  - `src/simplex/dual_simplex.cpp`
- [ ] Implement true FTRAN update (`B d = A_q`) and primal update rule  
  - `src/simplex/dual_simplex.cpp`
- [ ] Maintain dual feasibility numerically with tolerances in updates  
  - `src/simplex/dual_simplex.cpp`

## 5) Public API Surface

- [x] Provide umbrella header exports  
  - `include/lp_solver/lp_solver.hpp`
- [ ] Add stable solver result/report object (status, iterations, objective, timings)  
  - `include/lp_solver/simplex/*` (new type suggested)

## 6) Testing and Build

- [x] Provide CMake target skeleton  
  - `CMakeLists.txt`
- [x] Add starter smoke test  
  - `tests/smoke_test.cpp`
- [ ] Add unit tests from manual contract:
  - `IndexedVector` invariants
  - `PackedMatrix` CSC correctness
  - backend contract tests (`factorize/ftran/btran`)
  - strategy-injection and observer tests  
  - `tests/*` (next)
- [ ] Enable CI build + test matrix (MSVC/Clang/GCC)  
  - `.github/workflows/*` (new)
