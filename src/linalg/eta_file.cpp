#include "../../include/lp_solver/linalg/eta_file.hpp"

#include <cmath>
namespace lp_solver::linalg {

namespace {
constexpr double kSingularTol = 1e-12;
constexpr double kZeroTol = 1e-14;
}  // namespace

void EtaFile::clear() { updates_.clear(); }

bool EtaFile::append(int pivot_row, const util::IndexedVector& ftran_col) {
    if (pivot_row < 0 || pivot_row >= ftran_col.capacity()) {
        return false;
    }

    const double pivot_value = ftran_col[pivot_row];
    if (std::abs(pivot_value) <= kSingularTol) {
        return false;
    }

    Update eta;
    eta.pivot_row = pivot_row;
    eta.pivot_value = pivot_value;

    const auto& nz = ftran_col.nonZeroIndices();
    eta.indices.reserve(nz.size() + 1);
    eta.values.reserve(nz.size() + 1);

    bool have_pivot = false;
    for (int idx : nz) {
        const double val = ftran_col[idx];
        if (std::abs(val) <= kZeroTol) {
            continue;
        }
        eta.indices.push_back(idx);
        eta.values.push_back(val);
        if (idx == pivot_row) {
            have_pivot = true;
        }
    }
    if (!have_pivot) {
        eta.indices.push_back(pivot_row);
        eta.values.push_back(eta.pivot_value);
    }

    updates_.push_back(std::move(eta));
    return true;
}

void EtaFile::applyForward(util::IndexedVector& v) const {
    if (v.capacity() <= 0) {
        return;
    }
    for (const Update& eta : updates_) {
        const int p = eta.pivot_row;
        if (p < 0 || p >= v.capacity()) {
            continue;
        }
        const double dp = eta.pivot_value;
        if (std::abs(dp) <= kSingularTol) {
            continue;
        }
        const double xp = v[p] / dp;
        v.set(p, xp);
        for (size_t k = 0; k < eta.indices.size(); ++k) {
            const int i = eta.indices[k];
            if (i == p) {
                continue;
            }
            v.add(i, -eta.values[k] * xp);
        }
    }
}

void EtaFile::applyBackward(util::IndexedVector& v) const {
    if (v.capacity() <= 0) {
        return;
    }
    for (auto it = updates_.rbegin(); it != updates_.rend(); ++it) {
        const int p = it->pivot_row;
        if (p < 0 || p >= v.capacity()) {
            continue;
        }
        const double dp = it->pivot_value;
        if (std::abs(dp) <= kSingularTol) {
            continue;
        }
        double sum = 0.0;
        for (size_t k = 0; k < it->indices.size(); ++k) {
            const int i = it->indices[k];
            if (i == p) {
                continue;
            }
            sum += it->values[k] * v[i];
        }
        v.set(p, (v[p] - sum) / dp);
    }
}

int EtaFile::length() const { return static_cast<int>(updates_.size()); }

}  // namespace lp_solver::linalg
