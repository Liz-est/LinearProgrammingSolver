#include "../../include/lp_solver/linalg/eigen_factor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lp_solver::linalg {

namespace {
constexpr double kSingularTol = 1e-12;

std::vector<double> indexedToDense(const util::IndexedVector& rhs, int n) {
    std::vector<double> dense(static_cast<size_t>(n), 0.0);
    const auto& raw = rhs.rawValues();
    const int copy_n = std::min(static_cast<int>(raw.size()), n);
    for (int i = 0; i < copy_n; ++i) {
        dense[static_cast<size_t>(i)] = raw[static_cast<size_t>(i)];
    }
    return dense;
}

void denseToIndexed(const std::vector<double>& dense, util::IndexedVector& out) {
    out.clear();
    for (int i = 0; i < static_cast<int>(dense.size()); ++i) {
        if (std::abs(dense[static_cast<size_t>(i)]) > 1e-14) {
            out.set(i, dense[static_cast<size_t>(i)]);
        }
    }
}
}  // namespace

void EigenFactor::applyEtaForward(std::vector<double>& v, const std::vector<EtaUpdate>& etas) {
    for (const auto& eta : etas) {
        const int p = eta.pivot_row;
        const double dp = eta.d[static_cast<size_t>(p)];
        if (std::abs(dp) <= kSingularTol) {
            continue;
        }
        const double xp = v[static_cast<size_t>(p)] / dp;
        for (int i = 0; i < static_cast<int>(v.size()); ++i) {
            if (i == p) {
                continue;
            }
            v[static_cast<size_t>(i)] -= eta.d[static_cast<size_t>(i)] * xp;
        }
        v[static_cast<size_t>(p)] = xp;
    }
}

void EigenFactor::applyEtaBackward(std::vector<double>& v, const std::vector<EtaUpdate>& etas) {
    for (auto it = etas.rbegin(); it != etas.rend(); ++it) {
        const int p = it->pivot_row;
        const double dp = it->d[static_cast<size_t>(p)];
        if (std::abs(dp) <= kSingularTol) {
            continue;
        }
        double sum = 0.0;
        for (int i = 0; i < static_cast<int>(v.size()); ++i) {
            if (i == p) {
                continue;
            }
            sum += it->d[static_cast<size_t>(i)] * v[static_cast<size_t>(i)];
        }
        v[static_cast<size_t>(p)] = (v[static_cast<size_t>(p)] - sum) / dp;
    }
}

bool EigenFactor::factorize(const util::PackedMatrix& basis_matrix) {
    eta_updates_.clear();
    return engine_.factorize(basis_matrix);
}

void EigenFactor::ftran(util::IndexedVector& rhs) const {
    if (!engine_.ok()) {
        throw std::logic_error("EigenFactor::ftran called before successful factorization");
    }
    engine_.ftran(rhs);
    if (!eta_updates_.empty()) {
        auto v = indexedToDense(rhs, engine_.dimension());
        applyEtaForward(v, eta_updates_);
        denseToIndexed(v, rhs);
    }
}

void EigenFactor::btran(util::IndexedVector& rhs) const {
    if (!engine_.ok()) {
        throw std::logic_error("EigenFactor::btran called before successful factorization");
    }
    if (!eta_updates_.empty()) {
        auto v = indexedToDense(rhs, engine_.dimension());
        applyEtaBackward(v, eta_updates_);
        denseToIndexed(v, rhs);
    }
    engine_.btran(rhs);
}

void EigenFactor::updateEta(int pivot_row, const util::IndexedVector& ftran_col) {
    const int n = engine_.dimension();
    if (pivot_row < 0 || pivot_row >= n) {
        return;
    }
    EtaUpdate eta;
    eta.pivot_row = pivot_row;
    eta.d = indexedToDense(ftran_col, n);
    if (eta.d[static_cast<size_t>(pivot_row)] == 0.0) {
        eta.d[static_cast<size_t>(pivot_row)] = 1.0;
    }
    eta_updates_.push_back(std::move(eta));
}

int EigenFactor::etaFileLength() const { return static_cast<int>(eta_updates_.size()); }

}  // namespace lp_solver::linalg
