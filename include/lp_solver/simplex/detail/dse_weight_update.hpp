#pragma once

#include "../../util/indexed_vector.hpp"

#include <cmath>
#include <vector>

namespace lp_solver::simplex::detail {

/// Goldfarb–Reid DSE weight recurrence; only touches btran/ftran nonzeros (hypersparse).
template<typename FtranFn>
void goldfarbReidDseWeightUpdate(
    int leaving_row,
    const util::IndexedVector& btran_row,
    const util::IndexedVector& ftran_col,
    util::IndexedVector& v_rhs,
    FtranFn&& apply_ftran,
    std::vector<double>& dse_weights,
    double weight_floor = 1.0,
    double tiny = 1e-12
) {
    const int m = static_cast<int>(dse_weights.size());
    if (m != ftran_col.capacity() || m != v_rhs.capacity()) {
        return;
    }
    const double dp = ftran_col[leaving_row];
    if (std::abs(dp) <= tiny) {
        return;
    }

    const double w_p_old = dse_weights[leaving_row];
    dse_weights[leaving_row] = std::max(weight_floor, w_p_old / (dp * dp));

    v_rhs.clear();
    for (int i : btran_row.nonZeroIndices()) {
        v_rhs.set(i, btran_row[i]);
    }
    apply_ftran(v_rhs);

    for (int i : ftran_col.nonZeroIndices()) {
        if (i == leaving_row) {
            continue;
        }
        const double di = ftran_col[i];
        const double ratio = di / dp;
        const double vi = v_rhs[i];
        const double next = dse_weights[i] - 2.0 * ratio * vi + ratio * ratio * w_p_old;
        dse_weights[i] = std::max(weight_floor, next);
    }
}

/// Dense-index reference for tests (scans all rows for nonzero FTRAN entries).
template<typename FtranFn>
void goldfarbReidDseWeightUpdateDense(
    int leaving_row,
    const util::IndexedVector& btran_row,
    const util::IndexedVector& ftran_col,
    util::IndexedVector& v_rhs,
    FtranFn&& apply_ftran,
    std::vector<double>& dse_weights,
    double weight_floor = 1.0,
    double tiny = 1e-12
) {
    const int m = static_cast<int>(dse_weights.size());
    if (m != ftran_col.capacity() || m != v_rhs.capacity()) {
        return;
    }
    const double dp = ftran_col[leaving_row];
    if (std::abs(dp) <= tiny) {
        return;
    }

    const double w_p_old = dse_weights[leaving_row];
    dse_weights[leaving_row] = std::max(weight_floor, w_p_old / (dp * dp));

    v_rhs.clear();
    for (int i = 0; i < m; ++i) {
        const double val = btran_row[i];
        if (std::abs(val) > tiny) {
            v_rhs.set(i, val);
        }
    }
    apply_ftran(v_rhs);

    for (int i = 0; i < m; ++i) {
        if (i == leaving_row) {
            continue;
        }
        const double di = ftran_col[i];
        if (std::abs(di) <= tiny) {
            continue;
        }
        const double ratio = di / dp;
        const double vi = v_rhs[i];
        const double next = dse_weights[i] - 2.0 * ratio * vi + ratio * ratio * w_p_old;
        dse_weights[i] = std::max(weight_floor, next);
    }
}

}  // namespace lp_solver::simplex::detail
