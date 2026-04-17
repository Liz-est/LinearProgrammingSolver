cmake_minimum_required(VERSION 3.20)
project(lp_solver LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Eigen3 (SparseLU): Fetch if not installed ---
set(LP_SOLVER_FETCH_EIGEN ON CACHE BOOL "Download Eigen 3.4 via FetchContent if not found")
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND AND LP_SOLVER_FETCH_EIGEN)
    include(FetchContent)
    set(_LP_SOLVER_SAVED_BUILD_TESTING "${BUILD_TESTING}")
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(eigen)
    if(_LP_SOLVER_SAVED_BUILD_TESTING STREQUAL "")
        set(BUILD_TESTING ON CACHE BOOL "" FORCE)
    else()
        set(BUILD_TESTING ${_LP_SOLVER_SAVED_BUILD_TESTING} CACHE BOOL "" FORCE)
    endif()
endif()
if(NOT TARGET Eigen3::Eigen)
    message(FATAL_ERROR "Eigen3 is required. Install Eigen3 or enable LP_SOLVER_FETCH_EIGEN.")
endif()

# --- Optional UMFPACK (SuiteSparse) ---
option(LP_SOLVER_WITH_UMFPACK "Link SuiteSparse UMFPACK for UmfpackFactor" ON)
set(LP_SOLVER_HAVE_UMFPACK 0)
if(LP_SOLVER_WITH_UMFPACK)
    find_path(
        UMFPACK_INCLUDE_DIR
        NAMES umfpack.h
        PATH_SUFFIXES suitesparse
        DOC "Directory containing umfpack.h"
    )
    find_library(UMFPACK_LIBRARY NAMES umfpack libumfpack)
    find_library(AMD_LIBRARY NAMES amd libamd)
    find_library(SUITESPARSE_CONFIG_LIBRARY NAMES suitesparseconfig libsuitesparseconfig)
    if(UMFPACK_INCLUDE_DIR AND UMFPACK_LIBRARY AND AMD_LIBRARY)
        set(LP_SOLVER_HAVE_UMFPACK 1)
        message(STATUS "UMFPACK found: ${UMFPACK_LIBRARY}")
    else()
        message(STATUS "UMFPACK not found; UmfpackFactor will use Eigen SparseLU (same as EigenFactor).")
    endif()
endif()

add_library(lp_solver
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util/indexed_vector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util/packed_matrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linalg/sparse_triangular.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linalg/sparse_lu_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linalg/factor_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linalg/eigen_factor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linalg/umfpack_factor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/presolve/presolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simplex/dual_simplex.cpp
)

target_link_libraries(lp_solver PUBLIC Eigen3::Eigen)

target_include_directories(lp_solver
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(lp_solver PUBLIC LP_SOLVER_USE_EIGEN_SPARSE_LU=1)

if(LP_SOLVER_HAVE_UMFPACK)
    target_include_directories(lp_solver PRIVATE ${UMFPACK_INCLUDE_DIR})
    target_compile_definitions(lp_solver PRIVATE LP_SOLVER_HAVE_UMFPACK=1)
    target_link_libraries(lp_solver PRIVATE ${UMFPACK_LIBRARY} ${AMD_LIBRARY})
    if(SUITESPARSE_CONFIG_LIBRARY)
        target_link_libraries(lp_solver PRIVATE ${SUITESPARSE_CONFIG_LIBRARY})
    endif()
endif()

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
