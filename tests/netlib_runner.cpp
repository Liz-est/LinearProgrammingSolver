#include "../include/lp_solver/io/mps_reader.hpp"
#include "../include/lp_solver/io/netlib_standardizer.hpp"
#include "../include/lp_solver/linalg/i_basis_factor.hpp"
#include "../include/lp_solver/model/solver_state.hpp"
#include "../include/lp_solver/simplex/dual_simplex.hpp"
#include "../include/lp_solver/simplex/i_solver_observer.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    std::string mps_path;
    bool has_reference{false};
    double reference_objective{0.0};
    double tolerance{1e-6};
    int max_iterations{50'000};
    bool no_presolve{false};
    bool no_big_m{false};
    bool verbose{false};
    bool no_dse{false};
    bool no_harris{false};
    double big_m_scale{1000.0};
    int refactor_frequency{100};
};

void printUsage(const char* exe) {
    std::cerr << "Usage: " << exe
              << " <file.mps> [--ref <objective>] [--tol <abs_or_rel_tol>] [--max-iters <N>] [--no-presolve] [--no-big-m] [--big-m-scale <X>] [--verbose]" << '\n';
}

class DiagObserver : public lp_solver::simplex::ISolverObserver {
public:
    void onPivot(int leaving_row, int entering_col, double ratio) override {
        if (pivot_count_ < 5 || pivot_count_ % 100 == 0) {
            std::cout << "pivot iter=" << pivot_count_ << " leaving_row=" << leaving_row
                      << " entering_col=" << entering_col << " ratio=" << ratio << '\n';
        }
        ++pivot_count_;
    }
    void onTermination(const lp_solver::model::SolverState&, const char* reason) override {
        std::cout << "termination=" << (reason ? reason : "") << '\n';
    }

private:
    int pivot_count_{0};
};

bool parseDouble(const std::string& s, double& out) {
    char* end = nullptr;
    out = std::strtod(s.c_str(), &end);
    return end != nullptr && *end == '\0';
}

bool parseInt(const std::string& s, int& out) {
    char* end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (end == nullptr || *end != '\0' || v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

bool parseArgs(int argc, char** argv, CliOptions& options) {
    if (argc < 2) {
        return false;
    }
    options.mps_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--ref") {
            if (i + 1 >= argc || !parseDouble(argv[i + 1], options.reference_objective)) {
                return false;
            }
            options.has_reference = true;
            ++i;
        } else if (arg == "--tol") {
            if (i + 1 >= argc || !parseDouble(argv[i + 1], options.tolerance)) {
                return false;
            }
            ++i;
        } else if (arg == "--max-iters") {
            if (i + 1 >= argc || !parseInt(argv[i + 1], options.max_iterations)) {
                return false;
            }
            ++i;
        } else if (arg == "--no-presolve") {
            options.no_presolve = true;
        } else if (arg == "--presolve") {
            options.no_presolve = false;
        } else if (arg == "--no-big-m") {
            options.no_big_m = true;
        } else if (arg == "--big-m-scale") {
            if (i + 1 >= argc || !parseDouble(argv[i + 1], options.big_m_scale)) {
                return false;
            }
            ++i;
        } else if (arg == "--refactor-freq") {
            if (i + 1 >= argc || !parseInt(argv[i + 1], options.refactor_frequency)) {
                return false;
            }
            ++i;
        } else if (arg == "--no-dse") {
            options.no_dse = true;
        } else if (arg == "--no-harris") {
            options.no_harris = true;
        } else if (arg == "--verbose") {
            options.verbose = true;
        } else {
            return false;
        }
    }
    return true;
}

const char* statusToString(lp_solver::simplex::DualSimplex::Status st) {
    using Status = lp_solver::simplex::DualSimplex::Status;
    switch (st) {
    case Status::Optimal:
        return "optimal";
    case Status::Infeasible:
        return "infeasible";
    case Status::Unbounded:
        return "unbounded";
    case Status::IterationLimit:
        return "iteration_limit";
    default:
        return "unknown";
    }
}

}  // namespace

int main(int argc, char** argv) {
    CliOptions options;
    if (!parseArgs(argc, argv, options)) {
        printUsage(argv[0]);
        return 1;
    }

    const auto parsed = lp_solver::io::readMpsFile(options.mps_path);
    if (!parsed.ok) {
        std::cout << "classification=parse_error\n";
        std::cout << "error=" << parsed.error << '\n';
        return 2;
    }

    const auto standardized = lp_solver::io::standardizeNetlibModel(parsed.model);
    if (!standardized.ok) {
        std::cout << "classification=standardize_error\n";
        std::cout << "error=" << standardized.error << '\n';
        return 3;
    }

    lp_solver::model::SolverState state;
    state.basic_indices = standardized.initial_basis_indices;

    lp_solver::simplex::SolverConfig cfg;
    cfg.use_presolve = !options.no_presolve;
    cfg.enable_big_m_phase_one = !options.no_big_m;
    cfg.use_dual_steepest_edge = !options.no_dse;
    cfg.use_harris_two_pass = !options.no_harris;
    cfg.refactor_frequency = options.refactor_frequency;
    cfg.max_iterations = options.max_iterations;
    cfg.big_m_scale = options.big_m_scale;

    auto factor = lp_solver::linalg::makeDefaultFactor();
    DiagObserver diag_observer;
    lp_solver::simplex::ISolverObserver* observer_ptr = options.verbose ? &diag_observer : nullptr;
    lp_solver::simplex::DualSimplex solver(std::move(factor), nullptr, observer_ptr);

    const auto start = std::chrono::steady_clock::now();
    lp_solver::simplex::DualSimplex::Status status = lp_solver::simplex::DualSimplex::Status::Infeasible;
    try {
        status = solver.solve(standardized.problem, state, cfg);
    } catch (const std::exception& ex) {
        std::cout << "classification=solver_exception\n";
        std::cout << "error=" << ex.what() << '\n';
        return 4;
    }
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    const double reported_objective = state.objective + standardized.objective_offset;
    std::cout << std::setprecision(16);
    std::cout << "classification=ok\n";
    std::cout << "status=" << statusToString(status) << '\n';
    std::cout << "rows=" << standardized.problem.numRows() << '\n';
    std::cout << "cols=" << standardized.problem.numCols() << '\n';
    std::cout << "iterations=" << state.iteration << '\n';
    std::cout << "time_ms=" << elapsed_ms << '\n';
    std::cout << "objective=" << reported_objective << '\n';

    if (options.has_reference) {
        const double diff = std::abs(reported_objective - options.reference_objective);
        const double scaled_tol = options.tolerance * std::max(1.0, std::abs(options.reference_objective));
        const bool match = diff <= scaled_tol;
        std::cout << "reference=" << options.reference_objective << '\n';
        std::cout << "objective_diff=" << diff << '\n';
        std::cout << "objective_tol=" << scaled_tol << '\n';
        std::cout << "objective_match=" << (match ? "true" : "false") << '\n';
        if (!match || status != lp_solver::simplex::DualSimplex::Status::Optimal) {
            std::cout << "classification=objective_mismatch\n";
            return 10;
        }
    } else if (status != lp_solver::simplex::DualSimplex::Status::Optimal) {
        std::cout << "classification=solver_status_" << statusToString(status) << '\n';
        return 11;
    }

    return 0;
}
