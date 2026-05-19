#include "../../include/lp_solver/presolve/presolver.hpp"

#include <cmath>
#include <vector>

namespace lp_solver::presolve {

namespace {
constexpr double kTol = 1e-12;

struct MutableCsc {
    int m{0};
    int n{0};
    std::vector<double> el;
    std::vector<int> ri;
    std::vector<int> cp;
};

MutableCsc copyFromPacked(const util::PackedMatrix& A) {
    MutableCsc out;
    out.m = A.numRows();
    out.n = A.numCols();
    out.el = A.elements();
    out.ri = A.rowIndices();
    out.cp = A.colStarts();
    return out;
}

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

void scanActiveRowStructure(
    const MutableCsc& mat,
    const std::vector<bool>& active_rows,
    const std::vector<bool>& active_cols,
    std::vector<int>& row_nz_count,
    std::vector<int>& row_single_col,
    std::vector<double>& row_single_val
) {
    row_nz_count.assign(static_cast<size_t>(mat.m), 0);
    row_single_col.assign(static_cast<size_t>(mat.m), -1);
    row_single_val.assign(static_cast<size_t>(mat.m), 0.0);

    for (int j = 0; j < mat.n; ++j) {
        if (!active_cols[static_cast<size_t>(j)]) {
            continue;
        }
        for (int p = mat.cp[static_cast<size_t>(j)]; p < mat.cp[static_cast<size_t>(j + 1)]; ++p) {
            const int i = mat.ri[static_cast<size_t>(p)];
            if (!active_rows[static_cast<size_t>(i)]) {
                continue;
            }
            ++row_nz_count[static_cast<size_t>(i)];
            row_single_col[static_cast<size_t>(i)] = j;
            row_single_val[static_cast<size_t>(i)] = mat.el[static_cast<size_t>(p)];
        }
    }
}

void collectActiveRowEntries(
    const MutableCsc& mat,
    int row,
    const std::vector<bool>& active_rows,
    const std::vector<bool>& active_cols,
    std::vector<int>& nz_cols,
    std::vector<double>& coeffs
) {
    nz_cols.clear();
    coeffs.clear();
    for (int j = 0; j < mat.n; ++j) {
        if (!active_cols[static_cast<size_t>(j)]) {
            continue;
        }
        for (int p = mat.cp[static_cast<size_t>(j)]; p < mat.cp[static_cast<size_t>(j + 1)]; ++p) {
            if (mat.ri[static_cast<size_t>(p)] != row) {
                continue;
            }
            nz_cols.push_back(j);
            coeffs.push_back(mat.el[static_cast<size_t>(p)]);
            break;
        }
    }
}

bool columnHasActiveEntry(
    const MutableCsc& mat,
    int col,
    const std::vector<bool>& active_rows
) {
    for (int p = mat.cp[static_cast<size_t>(col)]; p < mat.cp[static_cast<size_t>(col + 1)]; ++p) {
        if (active_rows[static_cast<size_t>(mat.ri[static_cast<size_t>(p)])]) {
            return true;
        }
    }
    return false;
}

void substituteColumn(
    const MutableCsc& mat,
    int col,
    double xk,
    int skip_row,
    const std::vector<bool>& active_rows,
    std::vector<double>& rhs
) {
    for (int p = mat.cp[static_cast<size_t>(col)]; p < mat.cp[static_cast<size_t>(col + 1)]; ++p) {
        const int r = mat.ri[static_cast<size_t>(p)];
        if (r == skip_row || !active_rows[static_cast<size_t>(r)]) {
            continue;
        }
        rhs[static_cast<size_t>(r)] -= mat.el[static_cast<size_t>(p)] * xk;
    }
}

void recordDualSumForColumn(
    const MutableCsc& mat,
    int col,
    int skip_row,
    const std::vector<bool>& active_rows,
    Presolver::PostsolveRecord& rec
) {
    for (int p = mat.cp[static_cast<size_t>(col)]; p < mat.cp[static_cast<size_t>(col + 1)]; ++p) {
        const int r = mat.ri[static_cast<size_t>(p)];
        if (r == skip_row || !active_rows[static_cast<size_t>(r)]) {
            continue;
        }
        rec.dual_sum_rows.push_back(r);
        rec.dual_sum_coefs.push_back(mat.el[static_cast<size_t>(p)]);
    }
}
}  // namespace

Presolver::ReductionResult Presolver::run(const model::ProblemData& problem) const {
    ReductionResult out;
    out.reduced_problem = problem;

    const int m = problem.numRows();
    const int n = problem.numCols();
    out.original_num_rows = m;
    out.kept_rows.resize(m);
    out.kept_cols.resize(n);
    out.fixed_values.assign(n, 0.0);
    for (int i = 0; i < m; ++i) {
        out.kept_rows[i] = i;
    }
    for (int j = 0; j < n; ++j) {
        out.kept_cols[j] = j;
    }

    const MutableCsc mat = copyFromPacked(problem.A);
    std::vector<bool> active_rows(static_cast<size_t>(m), true);
    std::vector<bool> active_cols(static_cast<size_t>(n), true);
    std::vector<double> rhs = problem.b;
    std::vector<double> c = problem.c;

    std::vector<int> row_nz_count;
    std::vector<int> row_single_col;
    std::vector<double> row_single_val;
    std::vector<int> nz_cols;
    std::vector<double> coeffs;

    bool changed = true;
    while (changed) {
        changed = false;

        scanActiveRowStructure(mat, active_rows, active_cols, row_nz_count, row_single_col, row_single_val);

        for (int i = 0; i < m; ++i) {
            if (!active_rows[static_cast<size_t>(i)]) {
                continue;
            }
            const int nnz = row_nz_count[static_cast<size_t>(i)];
            if (nnz == 0) {
                if (std::abs(rhs[static_cast<size_t>(i)]) > kTol) {
                    out.status = Status::Infeasible;
                    return out;
                }
                active_rows[static_cast<size_t>(i)] = false;
                changed = true;
                continue;
            }
            if (nnz != 1) {
                continue;
            }
            const int k = row_single_col[static_cast<size_t>(i)];
            const double aik = row_single_val[static_cast<size_t>(i)];
            const double xk = rhs[static_cast<size_t>(i)] / aik;
            if (xk < -kTol) {
                out.status = Status::Infeasible;
                return out;
            }
            out.fixed_values[static_cast<size_t>(k)] = xk;
            PostsolveRecord rec;
            rec.kind = ReductionKind::SingletonRow;
            rec.column = k;
            rec.row = i;
            rec.value = xk;
            rec.coefficient = aik;
            rec.objective_coef = c[static_cast<size_t>(k)];
            recordDualSumForColumn(mat, k, i, active_rows, rec);
            out.postsolve_stack.push_back(std::move(rec));
            out.objective_offset += c[static_cast<size_t>(k)] * xk;
            substituteColumn(mat, k, xk, i, active_rows, rhs);
            active_cols[static_cast<size_t>(k)] = false;
            active_rows[static_cast<size_t>(i)] = false;
            changed = true;
        }

        for (int i = 0; i < m; ++i) {
            if (!active_rows[static_cast<size_t>(i)] || std::abs(rhs[static_cast<size_t>(i)]) > kTol) {
                continue;
            }
            collectActiveRowEntries(mat, i, active_rows, active_cols, nz_cols, coeffs);
            if (coeffs.empty()) {
                continue;
            }
            if (sameSign(coeffs)) {
                for (int col : nz_cols) {
                    active_cols[static_cast<size_t>(col)] = false;
                    out.fixed_values[static_cast<size_t>(col)] = 0.0;
                    PostsolveRecord rec;
                    rec.kind = ReductionKind::FixedZeroColumn;
                    rec.column = col;
                    rec.value = 0.0;
                    out.postsolve_stack.push_back(std::move(rec));
                }
                active_rows[static_cast<size_t>(i)] = false;
                changed = true;
            }
        }

        for (int j = 0; j < n; ++j) {
            if (!active_cols[static_cast<size_t>(j)]) {
                continue;
            }
            if (columnHasActiveEntry(mat, j, active_rows)) {
                continue;
            }
            if (c[static_cast<size_t>(j)] < -kTol) {
                out.status = Status::Unbounded;
                return out;
            }
            active_cols[static_cast<size_t>(j)] = false;
            out.fixed_values[static_cast<size_t>(j)] = 0.0;
            PostsolveRecord rec;
            rec.kind = ReductionKind::FixedZeroColumn;
            rec.column = j;
            rec.value = 0.0;
            out.postsolve_stack.push_back(std::move(rec));
            changed = true;
        }
    }

    std::vector<int> row_map;
    std::vector<int> col_map;
    for (int i = 0; i < m; ++i) {
        if (active_rows[static_cast<size_t>(i)]) {
            row_map.push_back(i);
        }
    }
    for (int j = 0; j < n; ++j) {
        if (active_cols[static_cast<size_t>(j)]) {
            col_map.push_back(j);
        }
    }

    out.kept_rows = row_map;
    out.kept_cols = col_map;

    std::vector<int> old_row_to_new(static_cast<size_t>(m), -1);
    for (int new_row = 0; new_row < static_cast<int>(row_map.size()); ++new_row) {
        old_row_to_new[static_cast<size_t>(row_map[static_cast<size_t>(new_row)])] = new_row;
    }

    util::PackedMatrix::Builder builder(static_cast<int>(row_map.size()), static_cast<int>(col_map.size()));
    for (int new_col = 0; new_col < static_cast<int>(col_map.size()); ++new_col) {
        const int old_col = col_map[static_cast<size_t>(new_col)];
        std::vector<int> rows;
        std::vector<double> vals;
        for (int p = mat.cp[static_cast<size_t>(old_col)]; p < mat.cp[static_cast<size_t>(old_col + 1)]; ++p) {
            const int old_row = mat.ri[static_cast<size_t>(p)];
            const int mapped = old_row_to_new[static_cast<size_t>(old_row)];
            if (mapped < 0) {
                continue;
            }
            const double aij = mat.el[static_cast<size_t>(p)];
            if (std::abs(aij) > kTol) {
                rows.push_back(mapped);
                vals.push_back(aij);
            }
        }
        builder.appendColumn(rows, vals);
    }

    out.reduced_problem.A = std::move(builder).build();
    out.reduced_problem.b.assign(row_map.size(), 0.0);
    for (int i = 0; i < static_cast<int>(row_map.size()); ++i) {
        out.reduced_problem.b[i] = rhs[static_cast<size_t>(row_map[static_cast<size_t>(i)])];
    }
    out.reduced_problem.c.assign(col_map.size(), 0.0);
    for (int j = 0; j < static_cast<int>(col_map.size()); ++j) {
        out.reduced_problem.c[j] = c[static_cast<size_t>(col_map[static_cast<size_t>(j)])];
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
            full[static_cast<size_t>(original_col)] = reduced_primal[j];
        }
    }
    for (auto it = reduction.postsolve_stack.rbegin(); it != reduction.postsolve_stack.rend(); ++it) {
        if (it->column >= 0 && it->column < static_cast<int>(full.size())) {
            full[static_cast<size_t>(it->column)] = it->value;
        }
    }
    return full;
}

std::vector<double> Presolver::postsolveDual(
    const ReductionResult& reduction,
    const std::vector<double>& reduced_dual
) const {
    const int num_rows = reduction.original_num_rows > 0
                             ? reduction.original_num_rows
                             : static_cast<int>(reduced_dual.size());

    std::vector<double> full(static_cast<size_t>(num_rows), 0.0);
    for (size_t i = 0; i < reduction.kept_rows.size(); ++i) {
        const int original_row = reduction.kept_rows[i];
        if (original_row >= 0 && original_row < num_rows && i < reduced_dual.size()) {
            full[static_cast<size_t>(original_row)] = reduced_dual[i];
        }
    }

    for (auto it = reduction.postsolve_stack.rbegin(); it != reduction.postsolve_stack.rend(); ++it) {
        if (it->kind != ReductionKind::SingletonRow) {
            continue;
        }
        if (it->row < 0 || it->row >= num_rows || std::abs(it->coefficient) <= kTol) {
            continue;
        }
        double sum = 0.0;
        for (size_t t = 0; t < it->dual_sum_rows.size(); ++t) {
            const int mrow = it->dual_sum_rows[t];
            if (mrow >= 0 && mrow < num_rows) {
                sum += full[static_cast<size_t>(mrow)] * it->dual_sum_coefs[t];
            }
        }
        full[static_cast<size_t>(it->row)] = (it->objective_coef - sum) / it->coefficient;
    }
    return full;
}

}  // namespace lp_solver::presolve
