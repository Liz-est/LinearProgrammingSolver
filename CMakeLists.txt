cmake_minimum_required(VERSION 3.20)
project(lp_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(lp_solver
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util/indexed_vector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util/packed_matrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linalg/factor_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linalg/eigen_factor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linalg/umfpack_factor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/presolve/presolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simplex/dual_simplex.cpp
)

target_include_directories(lp_solver
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_testing()

add_executable(lp_solver_smoke_tests
    tests/smoke_test.cpp
)

target_link_libraries(lp_solver_smoke_tests PRIVATE lp_solver)

add_test(NAME smoke_test COMMAND lp_solver_smoke_tests)

add_executable(lp_solver_stress_tests
    tests/stress_test.cpp
)

target_link_libraries(lp_solver_stress_tests PRIVATE lp_solver)

add_test(NAME stress_test COMMAND lp_solver_stress_tests)

add_executable(lp_solver_advanced_features_tests
    tests/advanced_features_test.cpp
)

target_link_libraries(lp_solver_advanced_features_tests PRIVATE lp_solver)

add_test(NAME advanced_features_test COMMAND lp_solver_advanced_features_tests)
