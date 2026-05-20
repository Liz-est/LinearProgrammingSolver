#include "../../include/lp_solver/io/mps_reader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lp_solver::io {
namespace {

enum class Section { None, Name, ObjSense, Rows, Columns, Rhs, Bounds, Ranges };

std::string trim(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) {
        ++b;
    }
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
        --e;
    }
    return s.substr(b, e - b);
}

std::vector<std::string> splitWs(const std::string& line) {
    std::istringstream iss(line);
    std::vector<std::string> out;
    std::string tok;
    while (iss >> tok) {
        out.push_back(tok);
    }
    return out;
}

std::string toUpper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

std::string normalizeName(std::string s) {
    s = trim(s);
    std::string out;
    out.reserve(s.size());
    bool pending_space = false;
    for (char ch : s) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            pending_space = !out.empty();
        } else {
            if (pending_space) {
                out.push_back(' ');
                pending_space = false;
            }
            out.push_back(ch);
        }
    }
    return out;
}

bool parseDouble(const std::string& s, double& out) {
    if (s.empty()) {
        return false;
    }
    char* end = nullptr;
    out = std::strtod(s.c_str(), &end);
    while (end != nullptr && *end != '\0' && std::isspace(static_cast<unsigned char>(*end))) {
        ++end;
    }
    return end != s.c_str() && (end == nullptr || *end == '\0');
}

int ensureVariable(RawLpModel& model, const std::string& var_name) {
    const auto it = model.variable_index.find(var_name);
    if (it != model.variable_index.end()) {
        return it->second;
    }
    const int idx = static_cast<int>(model.variable_names.size());
    model.variable_index[var_name] = idx;
    model.variable_names.push_back(var_name);
    model.columns.emplace_back();
    model.objective.push_back(0.0);
    model.bounds.push_back(RawVariableBounds{});
    return idx;
}

bool isMarkerLine(const std::vector<std::string>& toks) {
    for (const std::string& tok : toks) {
        if (tok == "'MARKER'" || toUpper(tok) == "MARKER" || toUpper(tok) == "INTORG" || toUpper(tok) == "INTEND") {
            return true;
        }
    }
    return false;
}

std::string mpsField(const std::string& line, int start_col_1based, int width) {
    const int begin = start_col_1based - 1;
    if (begin >= static_cast<int>(line.size())) {
        return "";
    }
    const int end = std::min(begin + width, static_cast<int>(line.size()));
    return trim(line.substr(static_cast<size_t>(begin), static_cast<size_t>(end - begin)));
}

bool addColumnEntry(
    RawLpModel& model,
    int var_idx,
    const std::string& row_name,
    double val,
    int line_no,
    MpsReadResult& result
) {
    if (!model.objective_row.empty() && row_name == model.objective_row) {
        model.objective[static_cast<size_t>(var_idx)] += val;
        return true;
    }
    const auto row_it = model.constraint_index.find(normalizeName(row_name));
    if (row_it == model.constraint_index.end()) {
        result.error = "Unknown row '" + row_name + "' in COLUMNS at line " + std::to_string(line_no);
        return false;
    }
    model.columns[static_cast<size_t>(var_idx)].emplace_back(row_it->second, val);
    return true;
}

bool knownRow(const RawLpModel& model, const std::string& row_name) {
    if (!model.objective_row.empty() && row_name == model.objective_row) {
        return true;
    }
    return model.constraint_index.find(normalizeName(row_name)) != model.constraint_index.end();
}

bool parseColumnsFreeLine(const std::string& line, int line_no, RawLpModel& model, MpsReadResult& result) {
    const std::vector<std::string> toks = splitWs(line);
    if (toks.empty()) {
        return true;
    }
    if (isMarkerLine(toks)) {
        return true;
    }
    if (toks.size() < 3) {
        result.error = "Malformed COLUMNS line " + std::to_string(line_no);
        return false;
    }
    const int var_idx = ensureVariable(model, toks[0]);
    for (size_t i = 1; i + 1 < toks.size(); i += 2) {
        const std::string& row_name = toks[i];
        double val = 0.0;
        if (!parseDouble(toks[i + 1], val)) {
            result.error = "Invalid numeric value in COLUMNS at line " + std::to_string(line_no);
            return false;
        }
        if (!addColumnEntry(model, var_idx, row_name, val, line_no, result)) {
            return false;
        }
    }
    return true;
}

bool parseColumnsFixedLine(const std::string& line, int line_no, RawLpModel& model, MpsReadResult& result) {
    const std::string var_name = mpsField(line, 5, 8);
    if (var_name.empty()) {
        return true;
    }
    if (isMarkerLine(splitWs(trim(line)))) {
        return true;
    }

    // Decide between fixed and free format by validating the first row token
    // against the parsed constraints: if the fixed-column row name is unknown
    // but a whitespace token matches, use free-format parsing.
    const std::string fixed_row = mpsField(line, 15, 8);
    if (!fixed_row.empty() && !knownRow(model, fixed_row)) {
        const std::vector<std::string> toks = splitWs(line);
        if (toks.size() >= 3 && knownRow(model, toks[1])) {
            return parseColumnsFreeLine(line, line_no, model, result);
        }
    }

    const int var_idx = ensureVariable(model, var_name);
    const struct {
        int row_col;
        int val_col;
    } slots[] = {{15, 25}, {40, 50}};
    for (const auto& slot : slots) {
        const std::string row_name = mpsField(line, slot.row_col, 8);
        const std::string val_str = mpsField(line, slot.val_col, 12);
        if (row_name.empty() || val_str.empty()) {
            continue;
        }
        double val = 0.0;
        if (!parseDouble(val_str, val)) {
            result.error = "Invalid numeric value in COLUMNS at line " + std::to_string(line_no);
            return false;
        }
        if (!addColumnEntry(model, var_idx, row_name, val, line_no, result)) {
            return false;
        }
    }
    return true;
}

bool parseRhsFreeLine(
    const std::string& line,
    int line_no,
    RawLpModel& model,
    std::string& selected_rhs_name,
    MpsReadResult& result
) {
    const std::vector<std::string> toks = splitWs(line);
    if (toks.empty()) {
        return true;
    }
    // Token 0 may be an RHS name (no matching row) or a row name (matches a row).
    size_t pair_start = 0;
    if (!knownRow(model, toks[0])) {
        if (selected_rhs_name.empty()) {
            selected_rhs_name = toks[0];
        }
        if (toks[0] != selected_rhs_name) {
            return true;
        }
        pair_start = 1;
    }
    for (size_t i = pair_start; i + 1 < toks.size(); i += 2) {
        double val = 0.0;
        if (!parseDouble(toks[i + 1], val)) {
            result.error = "Invalid numeric value in RHS at line " + std::to_string(line_no);
            return false;
        }
        const auto row_it = model.constraint_index.find(normalizeName(toks[i]));
        if (row_it != model.constraint_index.end()) {
            model.constraints[static_cast<size_t>(row_it->second)].rhs = val;
        }
    }
    return true;
}

bool parseRhsFixedLine(
    const std::string& line,
    int line_no,
    RawLpModel& model,
    std::string& selected_rhs_name,
    bool& rhs_initialized,
    MpsReadResult& result
) {
    if (!rhs_initialized) {
        rhs_initialized = true;
        for (RawConstraint& row : model.constraints) {
            row.rhs = 0.0;
        }
    }

    const std::string fixed_first = mpsField(line, 5, 8);
    const std::string fixed_row = mpsField(line, 15, 8);
    if (!fixed_row.empty() && !knownRow(model, fixed_row)) {
        const std::vector<std::string> toks = splitWs(line);
        const bool free_ok =
            (toks.size() >= 2 && knownRow(model, toks[0])) ||
            (toks.size() >= 3 && knownRow(model, toks[1]));
        if (free_ok) {
            return parseRhsFreeLine(line, line_no, model, selected_rhs_name, result);
        }
    }

    std::string rhs_name = fixed_first;
    int pair_start_row = 15;
    if (!rhs_name.empty() && model.constraint_index.find(rhs_name) != model.constraint_index.end()) {
        pair_start_row = 5;
        rhs_name.clear();
    } else if (!rhs_name.empty()) {
        if (selected_rhs_name.empty()) {
            selected_rhs_name = rhs_name;
        }
        if (rhs_name != selected_rhs_name) {
            return true;
        }
    }

    const struct {
        int row_col;
        int val_col;
    } slots[] = {{pair_start_row, pair_start_row + 10}, {40, 50}};
    for (const auto& slot : slots) {
        const std::string row_name = mpsField(line, slot.row_col, 8);
        const std::string val_str = mpsField(line, slot.val_col, 12);
        if (row_name.empty() || val_str.empty()) {
            continue;
        }
        double val = 0.0;
        if (!parseDouble(val_str, val)) {
            result.error = "Invalid numeric value in RHS at line " + std::to_string(line_no);
            return false;
        }
        const auto row_it = model.constraint_index.find(normalizeName(row_name));
        if (row_it != model.constraint_index.end()) {
            model.constraints[static_cast<size_t>(row_it->second)].rhs = val;
        }
    }
    return true;
}

bool parseRangesFreeLine(
    const std::string& line,
    int line_no,
    RawLpModel& model,
    std::string& selected_ranges_name,
    MpsReadResult& result
) {
    const std::vector<std::string> toks = splitWs(line);
    if (toks.empty()) {
        return true;
    }
    size_t pair_start = 0;
    if (!knownRow(model, toks[0])) {
        if (selected_ranges_name.empty()) {
            selected_ranges_name = toks[0];
        }
        if (toks[0] != selected_ranges_name) {
            return true;
        }
        pair_start = 1;
    }
    for (size_t i = pair_start; i + 1 < toks.size(); i += 2) {
        double val = 0.0;
        if (!parseDouble(toks[i + 1], val)) {
            result.error = "Invalid numeric value in RANGES at line " + std::to_string(line_no);
            return false;
        }
        const auto row_it = model.constraint_index.find(normalizeName(toks[i]));
        if (row_it != model.constraint_index.end()) {
            RawConstraint& row = model.constraints[static_cast<size_t>(row_it->second)];
            row.has_range = true;
            row.range = val;
        }
    }
    return true;
}

bool parseRangesFixedLine(
    const std::string& line,
    int line_no,
    RawLpModel& model,
    std::string& selected_ranges_name,
    MpsReadResult& result
) {
    const std::string fixed_row = mpsField(line, 15, 8);
    if (!fixed_row.empty() && !knownRow(model, fixed_row)) {
        const std::vector<std::string> toks = splitWs(line);
        const bool free_ok =
            (toks.size() >= 2 && knownRow(model, toks[0])) ||
            (toks.size() >= 3 && knownRow(model, toks[1]));
        if (free_ok) {
            return parseRangesFreeLine(line, line_no, model, selected_ranges_name, result);
        }
    }

    std::string ranges_name = mpsField(line, 5, 8);
    int pair_start_row = 15;
    if (!ranges_name.empty() && model.constraint_index.find(ranges_name) != model.constraint_index.end()) {
        pair_start_row = 5;
        ranges_name.clear();
    } else if (!ranges_name.empty()) {
        if (selected_ranges_name.empty()) {
            selected_ranges_name = ranges_name;
        }
        if (ranges_name != selected_ranges_name) {
            return true;
        }
    }

    const struct {
        int row_col;
        int val_col;
    } slots[] = {{pair_start_row, pair_start_row + 10}, {40, 50}};
    for (const auto& slot : slots) {
        const std::string row_name = mpsField(line, slot.row_col, 8);
        const std::string val_str = mpsField(line, slot.val_col, 12);
        if (row_name.empty() || val_str.empty()) {
            continue;
        }
        double val = 0.0;
        if (!parseDouble(val_str, val)) {
            result.error = "Invalid numeric value in RANGES at line " + std::to_string(line_no);
            return false;
        }
        const auto row_it = model.constraint_index.find(normalizeName(row_name));
        if (row_it != model.constraint_index.end()) {
            RawConstraint& row = model.constraints[static_cast<size_t>(row_it->second)];
            row.has_range = true;
            row.range = val;
        }
    }
    return true;
}

bool applyBoundType(const std::string& bound_type, RawVariableBounds& b, bool has_value, double value, int line_no, MpsReadResult& result) {
    if (bound_type == "LO" || bound_type == "LI") {
        b.lower = has_value ? value : 0.0;
    } else if (bound_type == "UP" || bound_type == "UI") {
        b.upper = has_value ? value : 0.0;
    } else if (bound_type == "FX") {
        b.lower = has_value ? value : 0.0;
        b.upper = has_value ? value : 0.0;
    } else if (bound_type == "FR") {
        b.lower = -std::numeric_limits<double>::infinity();
        b.upper = std::numeric_limits<double>::infinity();
    } else if (bound_type == "MI") {
        b.lower = -std::numeric_limits<double>::infinity();
    } else if (bound_type == "PL") {
        b.upper = std::numeric_limits<double>::infinity();
    } else if (bound_type == "BV") {
        b.lower = 0.0;
        b.upper = 1.0;
    } else {
        result.error = "Unsupported BOUNDS type '" + bound_type + "' at line " + std::to_string(line_no);
        return false;
    }
    return true;
}

bool isBoundType(const std::string& s) {
    static const char* const kTypes[] = {"LO", "UP", "FX", "FR", "MI", "PL", "BV", "LI", "UI"};
    for (const char* t : kTypes) {
        if (s == t) {
            return true;
        }
    }
    return false;
}

bool parseBoundsFreeLine(
    const std::string& line,
    int line_no,
    RawLpModel& model,
    std::string& selected_bounds_name,
    MpsReadResult& result
) {
    const std::vector<std::string> toks = splitWs(line);
    if (toks.size() < 3) {
        return true;
    }
    const std::string bound_type = toUpper(toks[0]);
    const std::string& bounds_name = toks[1];
    const std::string& var_name = toks[2];
    if (selected_bounds_name.empty()) {
        selected_bounds_name = bounds_name;
    }
    if (bounds_name != selected_bounds_name) {
        return true;
    }
    const int var_idx = ensureVariable(model, var_name);
    RawVariableBounds& b = model.bounds[static_cast<size_t>(var_idx)];
    const bool has_value = toks.size() >= 4;
    double value = 0.0;
    if (has_value && !parseDouble(toks[3], value)) {
        result.error = "Invalid numeric value in BOUNDS at line " + std::to_string(line_no);
        return false;
    }
    return applyBoundType(bound_type, b, has_value, value, line_no, result);
}

bool parseBoundsFixedLine(
    const std::string& line,
    int line_no,
    RawLpModel& model,
    std::string& selected_bounds_name,
    MpsReadResult& result
) {
    const std::string fixed_type = toUpper(mpsField(line, 2, 2));
    const std::string fixed_name = mpsField(line, 5, 8);
    const std::string fixed_var = mpsField(line, 15, 8);

    // Fall back to free-format if the fixed columns do not produce a known
    // bound type or contain a variable name that does not exist.
    const bool fixed_valid_type = isBoundType(fixed_type);
    const bool fixed_valid_var = !fixed_var.empty() &&
        model.variable_index.find(fixed_var) != model.variable_index.end();
    if (!fixed_valid_type || (!fixed_valid_var && !fixed_var.empty())) {
        const std::vector<std::string> toks = splitWs(line);
        if (toks.size() >= 3 && isBoundType(toUpper(toks[0]))) {
            return parseBoundsFreeLine(line, line_no, model, selected_bounds_name, result);
        }
    }

    const std::string bound_type = fixed_type;
    const std::string bounds_name = fixed_name;
    const std::string var_name = fixed_var;
    if (bound_type.empty() || bounds_name.empty() || var_name.empty()) {
        return true;
    }
    if (selected_bounds_name.empty()) {
        selected_bounds_name = bounds_name;
    }
    if (bounds_name != selected_bounds_name) {
        return true;
    }
    const int var_idx = ensureVariable(model, var_name);
    RawVariableBounds& b = model.bounds[static_cast<size_t>(var_idx)];
    const std::string value_str = mpsField(line, 25, 12);
    const bool has_value = !value_str.empty();
    double value = 0.0;
    if (has_value && !parseDouble(value_str, value)) {
        result.error = "Invalid numeric value in BOUNDS at line " + std::to_string(line_no);
        return false;
    }
    return applyBoundType(bound_type, b, has_value, value, line_no, result);
}

}  // namespace

MpsReadResult readMpsFile(const std::string& file_path) {
    MpsReadResult result;

    if (file_path.size() >= 3 && toUpper(file_path.substr(file_path.size() - 3)) == ".GZ") {
        result.error = "Compressed .gz files are not read directly; please decompress to .mps first.";
        return result;
    }

    std::ifstream fin(file_path);
    if (!fin) {
        result.error = "Failed to open MPS file: " + file_path;
        return result;
    }

    RawLpModel model;
    Section section = Section::None;
    std::string selected_rhs_name;
    std::string selected_bounds_name;
    std::string selected_ranges_name;
    bool rhs_initialized = false;
    bool ranges_initialized = false;

    std::string line;
    int line_no = 0;
    while (std::getline(fin, line)) {
        ++line_no;
        if (line.empty() || line[0] == '*') {
            continue;
        }
        const bool is_section_header = !std::isspace(static_cast<unsigned char>(line[0]));
        const std::string stripped = trim(line);
        if (stripped.empty()) {
            continue;
        }

        const auto tokens = splitWs(stripped);
        if (tokens.empty()) {
            continue;
        }

        if (is_section_header) {
            const std::string first_upper = toUpper(tokens[0]);
            if (first_upper == "NAME") {
                section = Section::Name;
                if (tokens.size() >= 2) {
                    model.name = tokens[1];
                }
                continue;
            }
            if (first_upper == "OBJSENSE") {
                section = Section::ObjSense;
                continue;
            }
            if (first_upper == "ROWS") {
                section = Section::Rows;
                continue;
            }
            if (first_upper == "COLUMNS") {
                section = Section::Columns;
                continue;
            }
            if (first_upper == "RHS") {
                section = Section::Rhs;
                continue;
            }
            if (first_upper == "BOUNDS") {
                section = Section::Bounds;
                continue;
            }
            if (first_upper == "RANGES") {
                section = Section::Ranges;
                continue;
            }
            if (first_upper == "ENDATA") {
                break;
            }
        }

        switch (section) {
        case Section::ObjSense: {
            const std::string sense = toUpper(tokens[0]);
            if (sense == "MAX" || sense == "MAXIMIZE") {
                model.maximize = true;
            } else if (sense == "MIN" || sense == "MINIMIZE") {
                model.maximize = false;
            }
            break;
        }
        case Section::Rows: {
            if (tokens.size() < 2) {
                result.error = "Invalid ROWS line at " + std::to_string(line_no);
                return result;
            }
            const std::string type = toUpper(tokens[0]);
            std::string row_name = tokens[1];
            for (size_t i = 2; i < tokens.size(); ++i) {
                row_name += " ";
                row_name += tokens[i];
            }
            if (type == "N") {
                model.objective_row = normalizeName(row_name);
                break;
            }

            RawConstraint row;
            row.name = normalizeName(row_name);
            if (type == "L") {
                row.type = RawConstraint::Type::LessEqual;
            } else if (type == "G") {
                row.type = RawConstraint::Type::GreaterEqual;
            } else if (type == "E") {
                row.type = RawConstraint::Type::Equal;
            } else {
                result.error = "Unsupported ROW type '" + tokens[0] + "' at line " + std::to_string(line_no);
                return result;
            }
            const int idx = static_cast<int>(model.constraints.size());
            model.constraint_index[row.name] = idx;
            model.constraints.push_back(row);
            break;
        }
        case Section::Columns: {
            if (line.size() >= 25 && std::isspace(static_cast<unsigned char>(line[0]))) {
                if (!parseColumnsFixedLine(line, line_no, model, result)) {
                    return result;
                }
                break;
            }
            if (tokens.size() < 3) {
                result.error = "Invalid COLUMNS line at " + std::to_string(line_no);
                return result;
            }
            if (isMarkerLine(tokens)) {
                break;
            }
            const std::string var_name = tokens[0];
            const int var_idx = ensureVariable(model, var_name);
            for (size_t i = 1; i + 1 < tokens.size(); i += 2) {
                const std::string row_name = tokens[i];
                double val = 0.0;
                if (!parseDouble(tokens[i + 1], val)) {
                    result.error = "Invalid numeric value in COLUMNS at line " + std::to_string(line_no);
                    return result;
                }
                if (!model.objective_row.empty() && normalizeName(row_name) == model.objective_row) {
                    model.objective[static_cast<size_t>(var_idx)] += val;
                    continue;
                }
                if (!addColumnEntry(model, var_idx, row_name, val, line_no, result)) {
                    return result;
                }
            }
            break;
        }
        case Section::Rhs: {
            if (line.size() >= 25 && std::isspace(static_cast<unsigned char>(line[0]))) {
                if (!parseRhsFixedLine(line, line_no, model, selected_rhs_name, rhs_initialized, result)) {
                    return result;
                }
                break;
            }
            if (tokens.size() < 2) {
                result.error = "Invalid RHS line at " + std::to_string(line_no);
                return result;
            }
            if (!rhs_initialized) {
                rhs_initialized = true;
                for (RawConstraint& row : model.constraints) {
                    row.rhs = 0.0;
                }
            }
            size_t pair_start = 1;
            if (model.constraint_index.find(tokens[0]) != model.constraint_index.end()) {
                pair_start = 0;
            } else {
                const std::string rhs_name = tokens[0];
                if (selected_rhs_name.empty()) {
                    selected_rhs_name = rhs_name;
                }
                if (rhs_name != selected_rhs_name) {
                    break;
                }
            }
            for (size_t i = pair_start; i + 1 < tokens.size(); i += 2) {
                const std::string row_name = tokens[i];
                double val = 0.0;
                if (!parseDouble(tokens[i + 1], val)) {
                    result.error = "Invalid numeric value in RHS at line " + std::to_string(line_no);
                    return result;
                }
                const auto row_it = model.constraint_index.find(normalizeName(row_name));
                if (row_it != model.constraint_index.end()) {
                    model.constraints[static_cast<size_t>(row_it->second)].rhs = val;
                }
            }
            break;
        }
        case Section::Ranges: {
            if (line.size() >= 25 && std::isspace(static_cast<unsigned char>(line[0]))) {
                if (!parseRangesFixedLine(line, line_no, model, selected_ranges_name, result)) {
                    return result;
                }
                break;
            }
            if (tokens.size() < 2) {
                result.error = "Invalid RANGES line at " + std::to_string(line_no);
                return result;
            }
            if (!ranges_initialized) {
                ranges_initialized = true;
            }
            size_t pair_start = 1;
            if (model.constraint_index.find(tokens[0]) != model.constraint_index.end()) {
                pair_start = 0;
            } else {
                const std::string ranges_name = tokens[0];
                if (selected_ranges_name.empty()) {
                    selected_ranges_name = ranges_name;
                }
                if (ranges_name != selected_ranges_name) {
                    break;
                }
            }
            for (size_t i = pair_start; i + 1 < tokens.size(); i += 2) {
                const std::string row_name = tokens[i];
                double val = 0.0;
                if (!parseDouble(tokens[i + 1], val)) {
                    result.error = "Invalid numeric value in RANGES at line " + std::to_string(line_no);
                    return result;
                }
                const auto row_it = model.constraint_index.find(normalizeName(row_name));
                if (row_it != model.constraint_index.end()) {
                    RawConstraint& row = model.constraints[static_cast<size_t>(row_it->second)];
                    row.has_range = true;
                    row.range = val;
                }
            }
            break;
        }
        case Section::Bounds: {
            if (line.size() >= 22 && std::isspace(static_cast<unsigned char>(line[0]))) {
                if (!parseBoundsFixedLine(line, line_no, model, selected_bounds_name, result)) {
                    return result;
                }
                break;
            }
            if (tokens.size() < 3) {
                result.error = "Invalid BOUNDS line at " + std::to_string(line_no);
                return result;
            }
            const std::string bound_type = toUpper(tokens[0]);
            const std::string bounds_name = tokens[1];
            const std::string var_name = tokens[2];
            if (selected_bounds_name.empty()) {
                selected_bounds_name = bounds_name;
            }
            if (bounds_name != selected_bounds_name) {
                break;
            }
            const int var_idx = ensureVariable(model, var_name);
            RawVariableBounds& b = model.bounds[static_cast<size_t>(var_idx)];
            const bool has_value = tokens.size() >= 4;
            double value = 0.0;
            if (has_value && !parseDouble(tokens[3], value)) {
                result.error = "Invalid numeric value in BOUNDS at line " + std::to_string(line_no);
                return result;
            }
            if (!applyBoundType(bound_type, b, has_value, value, line_no, result)) {
                return result;
            }
            break;
        }
        case Section::None:
        case Section::Name:
            // Ignore loose lines before ROWS.
            break;
        }
    }

    if (model.constraints.empty()) {
        result.error = "No constraints parsed from MPS file.";
        return result;
    }
    if (model.variable_names.empty()) {
        result.error = "No variables parsed from MPS file.";
        return result;
    }

    for (std::vector<std::pair<int, double>>& col : model.columns) {
        std::sort(col.begin(), col.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        std::vector<std::pair<int, double>> merged;
        for (const auto& [row, val] : col) {
            if (!merged.empty() && merged.back().first == row) {
                merged.back().second += val;
            } else {
                merged.push_back({row, val});
            }
        }
        col.clear();
        col.reserve(merged.size());
        for (const auto& [row, val] : merged) {
            if (std::abs(val) > 1e-16) {
                col.push_back({row, val});
            }
        }
    }

    if (model.maximize) {
        for (double& c : model.objective) {
            c = -c;
        }
        model.maximize = false;
    }

    result.ok = true;
    result.model = std::move(model);
    return result;
}

}  // namespace lp_solver::io
