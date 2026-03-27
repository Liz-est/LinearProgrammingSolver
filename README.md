# Linear Programming Solver

**🌐 Language** | [中文版本](README_zh.md)

A high-performance C++ library for solving linear programming problems using the dual simplex algorithm with sparse matrix representations and pluggable linear algebra backends.

### Features

- **Dual Simplex Algorithm**: Efficient implementation of the dual simplex method for LP optimization
- **Sparse Matrix Representation**: CSC (Compressed Sparse Column) format for memory-efficient storage and computation
- **Multiple LA Backends**: Support for both Eigen and UMFPACK factorization methods
- **Indexed Vector Support**: Hypersparse vector operations for efficient pricing and tableau updates
- **C++20 Implementation**: Modern C++ with strict type safety and Move semantics optimization
- **Cross-Platform**: Built with CMake, supports Windows, Linux, and macOS

## Project Structure

```
LinearProgramingSolver/
├── include/lp_solver/          # Public headers
│   ├── linalg/                 # Linear algebra abstraction
│   │   ├── i_basis_factor.hpp  # Basis factorization interface
│   │   ├── eigen_factor.hpp    # Eigen backend
│   │   └── umfpack_factor.hpp  # UMFPACK backend
│   ├── model/                  # Problem data structures
│   ├── simplex/                # Dual simplex algorithm
│   └── util/                   # Sparse data structures
│       ├── indexed_vector.hpp  # Hypersparse vectors
│       └── packed_matrix.hpp   # CSC matrix representation
├── src/                        # Implementation files
├── tests/                      # Test suites
│   ├── smoke_test.cpp          # Basic functionality tests
│   └── stress_test.cpp         # Performance and robustness tests
└── CMakeLists.txt              # Build configuration
```

## Requirements

- **C++20** compiler (MSVC, GCC, or Clang)
- **CMake 3.20+**
- **Windows 10/11**, Linux, or macOS
- Optional: Eigen3 or UMFPACK libraries for advanced factorization

## Installation & Build

### Clone the Repository

```bash
git clone https://github.com/Liz-est/LinearProgramingSolver.git
cd LinearProgramingSolver
```

### Build with CMake

#### Debug Configuration
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
```

#### Release Configuration
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Run Tests

```bash
ctest --test-dir build -C Debug --output-on-failure
```

Or use the automated test runner (Windows):
```powershell
.\run-test-plan.ps1 -Config Debug -BuildDir build
```

## Usage

### Basic Example

```cpp
#include "lp_solver/lp_solver.hpp"

// Create problem data
ProblemData problem = /* ... */;

// Initialize solver
DualSimplex solver(problem);

// Solve
SolverState result = solver.solve();

// Access solution
if (result.status == OptimalityStatus::OPTIMAL) {
    auto solution = result.primal_solution;
    auto objective = result.objective_value;
}
```

## API Overview

### Core Components

- **`DualSimplex`**: Main solver class implementing the dual simplex algorithm
- **`IBasisFactor`**: Abstract interface for basis factorization
- **`PackedMatrix`**: Immutable CSC format sparse matrix
- **`IndexedVector`**: Sparse vector with active index tracking
- **`ProblemData`**: Standard form LP problem definition
- **`SolverState`**: Solution and solver statistics

### Architecture

The solver uses a modular architecture with pluggable backends:

```
DualSimplex (Algorithm)
    ↓
IBasisFactor (Abstract Interface)
    ├─ EigenFactor (Eigen3 backend)
    └─ UmfpackFactor (UMFPACK backend)
```

## Test Coverage

The project includes comprehensive test suites:

- **Smoke Tests** (`smoke_test.cpp`): Validates basic solver functionality and correctness
- **Stress Tests** (`stress_test.cpp`): Performance testing with large-scale problems

Run tests with:
```bash
ctest --test-dir build --output-on-failure -V
```

## Implementation Status

### Phase 1 (Active Development)

✅ Completed:
- Sparse primitives (IndexedVector, PackedMatrix)
- Factorization abstraction layer
- Dual simplex algorithm skeleton

🔄 In Progress:
- LA backend integration (Eigen, UMFPACK)
- Matrix-vector operations
- Pricing and update operations

## Development

### Build Configuration

- **C++ Standard**: C++20
- **Compiler Flags**: Strict error checking enabled
- **Build System**: CMake with native toolchain support

### Code Organization

- Headers in `include/lp_solver/`
- Implementations in `src/`
- Tests in `tests/`
- Build artifacts in `build/`

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Performance Considerations

- **Sparse Operations**: Leverages CSC format and indexed vectors for efficiency
- **Factorization**: Pluggable backends allow algorithm optimization per use case
- **Memory**: Zero-erasure for sparse vectors prevents memory leaks

## References

- Dual Simplex Method: [Academic reference]
- Sparse Matrix Formats: [Technical documentation]
- Eigen3: https://eigen.tuxfamily.org/
- UMFPACK: https://people.engr.tamu.edu/davis/suitesparse.html

## Support

For issues, questions, or suggestions, please:
- Open an issue on GitHub
- Submit a pull request with improvements
- Contact: [your contact information]

---

**Status**: Phase 1 Development | **Last Updated**: March 27 2026
