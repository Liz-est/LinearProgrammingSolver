#include "../../include/lp_solver/io/netlib_standardizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lp_solver::io {
namespace {

constexpr double kTol = 1e-12;

struct VarExpansion {
    double shift{0.0};
    std::vector<std::pair<int, double>> terms;
};

struct InequalityRow {
    enum class Type { LessEqual, GreaterEqual };
    Type type{Type::LessEqual};
    double rhs{0.0};
    std::unordered_map<int, double> coeffs;
};

bool isFinite(double x) { return std::isfinite(x); }

void pushIneqFromBounds(
    const RawConstraint& row,
    double rhs,
    const std::unordered_map<int, double>& coeffs,
    std::vector<InequalityRow>& out
) {
    double lower = -std::numeric_limits<double>::infinity();
    double upper = std::numeric_limits<double>::infinity();

    if (row.has_range) {
        const double abs_range = std::abs(row.range);
        if (row.type == RawConstraint::Type::Equal) {
            if (row.range >= 0.0) {
                lower = rhs;
                upper = rhs + row.range;
            } else {
                lower = rhs + row.range;
                upper = rhs;
            }
        } else if (row.type == RawConstraint::Type::LessEqual) {
            lower = rhs - abs_range;
            upper = rhs;
        } else {
            lower = rhs;
            upper = rhs + abs_range;
        }
    } else {
        if (row.type == RawConstraint::Type::Equal) {
            lower = rhs;
            upper = rhs;
        } else if (row.type == RawConstraint::Type::LessEqual) {
            upper = rhs;
        } else {
            lower = rhs;
        }
    }

    if (isFinite(upper)) {
        InequalityRow ineq;
        ineq.type = InequalityRow::Type::LessEqual;
        ineq.rhs = upper;
        ineq.coeffs = coeffs;
        out.push_back(std::move(ineq));
    }
    if (isFinite(lower)) {
        InequalityRow ineq;
        ineq.type = InequalityRow::Type::GreaterEqual;
        ineq.rhs = lower;
        ineq.coeffs = coeffs;
        out.push_back(std::move(ineq));
    }
}

}  // namespace

StandardizationResult standardizeNetlibModel(const RawLpModel& raw_model) {
    StandardizationResult out;

    const int m_raw = static_cast<int>(raw_model.constraints.size());
    const int n_raw = static_cast<int>(raw_model.variable_names.size());
    if (m_raw <= 0 || n_raw <= 0) {
        out.error = "Raw model dimensions are invalid.";
        return out;
    }
    if (raw_model.columns.size() != static_cast<size_t>(n_raw) || raw_model.objective.size() != static_cast<size_t>(n_raw) ||
        raw_model.bounds.size() != static_cast<size_t>(n_raw)) {
        out.error = "Raw model vectors are inconsistent.";
        return out;
    }

    std::vector<VarExpansion> expansions(static_cast<size_t>(n_raw));
    std::vector<double> objective_new;
    std::vector<std::pair<int, double>> extra_upper_bounds;
    objective_new.reserve(static_cast<size_t>(n_raw) * 2);

    auto add_new_var = [&objective_new](double c_coef) {
        const int idx = static_cast<int>(objective_new.size());
        objective_new.push_back(c_coef);
        return idx;
    };

    for (int j = 0; j < n_raw; ++j) {
        const auto& bounds = raw_model.bounds[static_cast<size_t>(j)];
        const double l = bounds.lower;
        const double u = bounds.upper;
        const bool has_l = isFinite(l);
        const bool has_u = isFinite(u);
        if (has_l && has_u && u + kTol < l) {
            out.error = "Invalid bounds: upper < lower at variable index " + std::to_string(j);
            return out;
        }

        VarExpansion exp;
        if (!has_l && !has_u) {
            const int pos = add_new_var(raw_model.objective[static_cast<size_t>(j)]);
            const int neg = add_new_var(-raw_model.objective[static_cast<size_t>(j)]);
            exp.terms.push_back({pos, 1.0});
            exp.terms.push_back({neg, -1.0});
        } else if (has_l && !has_u) {
            const int y = add_new_var(raw_model.objective[static_cast<size_t>(j)]);
            exp.shift = l;
            exp.terms.push_back({y, 1.0});
        } else if (!has_l && has_u) {
            const int y = add_new_var(-raw_model.objective[static_cast<size_t>(j)]);
            exp.shift = u;
            exp.terms.push_back({y, -1.0});
        } else {
            const int y = add_new_var(raw_model.objective[static_cast<size_t>(j)]);
            exp.shift = l;
            exp.terms.push_back({y, 1.0});
            extra_upper_bounds.push_back({y, u - l});
        }
        expansions[static_cast<size_t>(j)] = std::move(exp);
    }

    std::vector<double> rhs_adjusted(static_cast<size_t>(m_raw), 0.0);
    for (int i = 0; i < m_raw; ++i) {
        rhs_adjusted[static_cast<size_t>(i)] = raw_model.constraints[static_cast<size_t>(i)].rhs;
    }
    std::vector<std::unordered_map<int, double>> row_coeffs(static_cast<size_t>(m_raw));

    for (int j = 0; j < n_raw; ++j) {
        const VarExpansion& exp = expansions[static_cast<size_t>(j)];
        out.objective_offset += raw_model.objective[static_cast<size_t>(j)] * exp.shift;
        for (const auto& [row_idx, a] : raw_model.columns[static_cast<size_t>(j)]) {
            rhs_adjusted[static_cast<size_t>(row_idx)] -= a * exp.shift;
            for (const auto& [new_var, coeff] : exp.terms) {
                row_coeffs[static_cast<size_t>(row_idx)][new_var] += a * coeff;
            }
        }
    }

    std::vector<InequalityRow> inequalities;
    inequalities.reserve(static_cast<size_t>(m_raw) * 2 + extra_upper_bounds.size());
    for (int i = 0; i < m_raw; ++i) {
        pushIneqFromBounds(
            raw_model.constraints[static_cast<size_t>(i)],
            rhs_adjusted[static_cast<size_t>(i)],
            row_coeffs[static_cast<size_t>(i)],
            inequalities
        );
    }
    for (const auto& [new_var, ub] : extra_upper_bounds) {
        if (ub < -kTol) {
            out.error = "Invalid finite upper bound after shift.";
            return out;
        }
        InequalityRow ineq;
        ineq.type = InequalityRow::Type::LessEqual;
        ineq.rhs = ub;
        ineq.coeffs[new_var] = 1.0;
        inequalities.push_back(std::move(ineq));
    }

    if (inequalities.empty()) {
        out.error = "No inequalities produced after standardization.";
        return out;
    }

    const int n_struct = static_cast<int>(objective_new.size());
    const int m_eq = static_cast<int>(inequalities.size());
    const int n_total = n_struct + m_eq;

    std::vector<std::vector<std::pair<int, double>>> cols(static_cast<size_t>(n_total));
    std::vector<double> b(static_cast<size_t>(m_eq), 0.0);
    std::vector<int> basis(static_cast<size_t>(m_eq), -1);

    for (int i = 0; i < m_eq; ++i) {
        const auto& row = inequalities[static_cast<size_t>(i)];
        b[static_cast<size_t>(i)] = row.rhs;
        for (const auto& [j, a] : row.coeffs) {
            if (std::abs(a) > kTol) {
                cols[static_cast<size_t>(j)].push_back({i, a});
            }
        }

        const int slack_col = n_struct + i;
        const double s_coeff = (row.type == InequalityRow::Type::LessEqual) ? 1.0 : -1.0;
        cols[static_cast<size_t>(slack_col)].push_back({i, s_coeff});
        basis[static_cast<size_t>(i)] = slack_col;
    }

    util::PackedMatrix::Builder builder(m_eq, n_total);
    for (int j = 0; j < n_total; ++j) {
        auto& entries = cols[static_cast<size_t>(j)];
        std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

        std::vector<int> rows;
        std::vector<double> vals;
        rows.reserve(entries.size());
        vals.reserve(entries.size());
        for (const auto& [r, v] : entries) {
            if (!rows.empty() && rows.back() == r) {
                vals.back() += v;
            } else {
                rows.push_back(r);
                vals.push_back(v);
            }
        }

        std::vector<int> compact_rows;
        std::vector<double> compact_vals;
        compact_rows.reserve(rows.size());
        compact_vals.reserve(vals.size());
        for (size_t k = 0; k < rows.size(); ++k) {
            if (std::abs(vals[k]) > kTol) {
                compact_rows.push_back(rows[k]);
                compact_vals.push_back(vals[k]);
            }
        }
        builder.appendColumn(compact_rows, compact_vals);
    }

    std::vector<double> c(static_cast<size_t>(n_total), 0.0);
    for (int j = 0; j < n_struct; ++j) {
        c[static_cast<size_t>(j)] = objective_new[static_cast<size_t>(j)];
    }
    std::vector<double> lower(static_cast<size_t>(n_total), 0.0);
    std::vector<double> upper(static_cast<size_t>(n_total), 1.0e30);

    out.problem = model::ProblemData(std::move(builder).build(), std::move(c), std::move(b), std::move(lower), std::move(upper));
    out.initial_basis_indices = std::move(basis);
    out.ok = true;
    return out;
}

}  // namespace lp_solver::io
