#include "../../include/lp_solver/presolve/presolver.hpp"

#include <cmath>

namespace lp_solver::presolve {

namespace {
constexpr double kTol = 1e-12;

bool sameSign(const std::vector<double>& coeffs) {
    int sign = 0;
    for (double v : coeffs) {
        if (std::abs(v) <= kTol) {
            continue;
        }
        const int s = v > 0.0 ? 1 : -1;
        if (sign == 0) {
            sign = s;
        } else if (sign != s) {
            return false;
        }
    }
    return sign != 0;
}
}  // namespace

Presolver::ReductionResult Presolver::run(const model::ProblemData& problem) const {
    ReductionResult out;
    out.reduced_problem = problem;

    const int m = problem.numRows();
    const int n = problem.numCols();
    out.kept_rows.resize(m);
    out.kept_cols.resize(n);
    out.fixed_values.assign(n, 0.0);
    for (int i = 0; i < m; ++i) {
        out.kept_rows[i] = i;
    }
    for (int j = 0; j < n; ++j) {
        out.kept_cols[j] = j;
    }

    auto dense = problem.A.toDense();
    std::vector<bool> active_rows(m, true);
    std::vector<bool> active_cols(n, true);
    std::vector<double> rhs = problem.b;
    std::vector<double> c = problem.c;

    bool changed = true;
    while (changed) {
        changed = false;

        for (int i = 0; i < m; ++i) {
            if (!active_rows[i]) {
                continue;
            }
            std::vector<int> nz_cols;
            for (int j = 0; j < n; ++j) {
                if (active_cols[j] && std::abs(dense[i][j]) > kTol) {
                    nz_cols.push_back(j);
                }
            }
            if (nz_cols.empty()) {
                if (std::abs(rhs[i]) > kTol) {
                    out.status = Status::Infeasible;
                    return out;
                }
                active_rows[i] = false;
                changed = true;
                continue;
            }
            if (nz_cols.size() == 1) {
                const int k = nz_cols.front();
                const double aik = dense[i][k];
                const double xk = rhs[i] / aik;
                if (xk < -kTol) {
                    out.status = Status::Infeasible;
                    return out;
                }
                out.fixed_values[k] = xk;
                out.postsolve_stack.push_back(Presolver::PostsolveRecord{
                    Presolver::ReductionKind::SingletonRow, k, xk
                });
                out.objective_offset += c[k] * xk;
                for (int r = 0; r < m; ++r) {
                    if (!active_rows[r] || r == i) {
                        continue;
                    }
                    rhs[r] -= dense[r][k] * xk;
                    dense[r][k] = 0.0;
                }
                active_cols[k] = false;
                active_rows[i] = false;
                changed = true;
            }
        }

        for (int i = 0; i < m; ++i) {
            if (!active_rows[i] || std::abs(rhs[i]) > kTol) {
                continue;
            }
            std::vector<double> coeffs;
            std::vector<int> nz_cols;
            for (int j = 0; j < n; ++j) {
                if (active_cols[j] && std::abs(dense[i][j]) > kTol) {
                    coeffs.push_back(dense[i][j]);
                    nz_cols.push_back(j);
                }
            }
            if (coeffs.empty()) {
                continue;
            }
            if (sameSign(coeffs)) {
                for (int col : nz_cols) {
                    active_cols[col] = false;
                    out.fixed_values[col] = 0.0;
                    out.postsolve_stack.push_back(Presolver::PostsolveRecord{
                        Presolver::ReductionKind::FixedZeroColumn, col, 0.0
                    });
                    for (int r = 0; r < m; ++r) {
                        dense[r][col] = 0.0;
                    }
                }
                active_rows[i] = false;
                changed = true;
            }
        }

        for (int j = 0; j < n; ++j) {
            if (!active_cols[j]) {
                continue;
            }
            bool empty = true;
            for (int i = 0; i < m; ++i) {
                if (active_rows[i] && std::abs(dense[i][j]) > kTol) {
                    empty = false;
                    break;
                }
            }
            if (!empty) {
                continue;
            }
            if (c[j] < -kTol) {
                out.status = Status::Unbounded;
                return out;
            }
            active_cols[j] = false;
            out.fixed_values[j] = 0.0;
            out.postsolve_stack.push_back(Presolver::PostsolveRecord{
                Presolver::ReductionKind::FixedZeroColumn, j, 0.0
            });
            changed = true;
        }
    }

    std::vector<int> row_map;
    std::vector<int> col_map;
    for (int i = 0; i < m; ++i) {
        if (active_rows[i]) {
            row_map.push_back(i);
        }
    }
    for (int j = 0; j < n; ++j) {
        if (active_cols[j]) {
            col_map.push_back(j);
        }
    }

    out.kept_rows = row_map;
    out.kept_cols = col_map;

    util::PackedMatrix::Builder builder(static_cast<int>(row_map.size()), static_cast<int>(col_map.size()));
    for (int new_col = 0; new_col < static_cast<int>(col_map.size()); ++new_col) {
        const int old_col = col_map[new_col];
        std::vector<int> rows;
        std::vector<double> vals;
        for (int new_row = 0; new_row < static_cast<int>(row_map.size()); ++new_row) {
            const int old_row = row_map[new_row];
            const double aij = dense[old_row][old_col];
            if (std::abs(aij) > kTol) {
                rows.push_back(new_row);
                vals.push_back(aij);
            }
        }
        builder.appendColumn(rows, vals);
    }

    out.reduced_problem.A = std::move(builder).build();
    out.reduced_problem.b.assign(row_map.size(), 0.0);
    for (int i = 0; i < static_cast<int>(row_map.size()); ++i) {
        out.reduced_problem.b[i] = rhs[row_map[i]];
    }
    out.reduced_problem.c.assign(col_map.size(), 0.0);
    for (int j = 0; j < static_cast<int>(col_map.size()); ++j) {
        out.reduced_problem.c[j] = c[col_map[j]];
    }
    out.reduced_problem.lower_bounds.assign(col_map.size(), 0.0);
    out.reduced_problem.upper_bounds.assign(col_map.size(), 1.0e30);
    return out;
}

std::vector<double> Presolver::postsolvePrimal(
    const ReductionResult& reduction,
    const std::vector<double>& reduced_primal
) const {
    std::vector<double> full = reduction.fixed_values;
    for (size_t j = 0; j < reduction.kept_cols.size(); ++j) {
        const int original_col = reduction.kept_cols[j];
        if (j < reduced_primal.size() && original_col >= 0 && original_col < static_cast<int>(full.size())) {
            full[original_col] = reduced_primal[j];
        }
    }
    for (auto it = reduction.postsolve_stack.rbegin(); it != reduction.postsolve_stack.rend(); ++it) {
        if (it->column >= 0 && it->column < static_cast<int>(full.size())) {
            full[it->column] = it->value;
        }
    }
    return full;
}

}  // namespace lp_solver::presolve
