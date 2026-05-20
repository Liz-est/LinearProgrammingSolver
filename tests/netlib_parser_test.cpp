#include "../include/lp_solver/io/mps_reader.hpp"
#include "../include/lp_solver/io/netlib_standardizer.hpp"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

void expect(bool cond, const char* msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

std::string writeTempMps() {
    const char* path = "netlib_parser_test_tmp.mps";
    std::ofstream out(path, std::ios::trunc);
    out << "NAME          TINY\n";
    out << "OBJSENSE\n";
    out << "  MIN\n";
    out << "ROWS\n";
    out << " N  COST\n";
    out << " L  R1\n";
    out << " G  R2\n";
    out << " E  R3\n";
    out << "COLUMNS\n";
    out << "    X1      COST      1      R1        1\n";
    out << "    X1      R2        1\n";
    out << "    X2      COST      2      R1        1\n";
    out << "    X2      R3        1\n";
    out << "    X3      COST     -1      R2        1\n";
    out << "RHS\n";
    out << "    RHS1    R1        5      R2        1\n";
    out << "    RHS1    R3        2\n";
    out << "BOUNDS\n";
    out << " LO BND     X1        1\n";
    out << " UP BND     X1        4\n";
    out << " FR BND     X2\n";
    out << " MI BND     X3\n";
    out << " UP BND     X3       10\n";
    out << "RANGES\n";
    out << " RNG1       R3        3\n";
    out << "ENDATA\n";
    out.close();
    return path;
}

void testReadAndStandardize() {
    const std::string path = writeTempMps();

    const auto parsed = lp_solver::io::readMpsFile(path);
    expect(parsed.ok, "parser should succeed");
    expect(parsed.model.variable_names.size() == 3, "expected 3 variables");
    expect(parsed.model.constraints.size() == 3, "expected 3 constraints");
    expect(parsed.model.constraints[2].has_range, "expected range on R3");

    const auto standardized = lp_solver::io::standardizeNetlibModel(parsed.model);
    expect(standardized.ok, "standardizer should succeed");
    expect(standardized.problem.numRows() > 0, "standardized rows > 0");
    expect(standardized.problem.numCols() > standardized.problem.numRows(), "expect slacks appended");
    expect(standardized.initial_basis_indices.size() == static_cast<size_t>(standardized.problem.numRows()),
           "basis size mismatch");

    for (int idx : standardized.initial_basis_indices) {
        expect(idx >= 0 && idx < standardized.problem.numCols(), "basis index out of bounds");
    }

    std::remove(path.c_str());
}

void testRejectGzPath() {
    const auto parsed = lp_solver::io::readMpsFile("dummy.mps.gz");
    expect(!parsed.ok, "gz path should be rejected");
}

}  // namespace

int main() {
    try {
        testReadAndStandardize();
        testRejectGzPath();
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "parser test failure: %s\n", ex.what());
        return 1;
    }
    return 0;
}
