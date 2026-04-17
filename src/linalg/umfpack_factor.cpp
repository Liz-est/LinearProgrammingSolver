#include "../../include/lp_solver/linalg/umfpack_factor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace lp_solver::linalg {

namespace {
constexpr double kSingularTol = 1e-12;

std::vector<double> indexedToDense(const util::IndexedVector& rhs, int n) {
    std::vector<double> dense(n, 0.0);
    const auto& raw = rhs.rawValues();
    const int copy_n = std::min(static_cast<int>(raw.size()), n);
    for (int i = 0; i < copy_n; ++i) {
        dense[i] = raw[i];
    }
    return dense;
}

void denseToIndexed(const std::vector<double>& dense, util::IndexedVector& out) {
    out.clear();
    for (int i = 0; i < static_cast<int>(dense.size()); ++i) {
        if (std::abs(dense[i]) > 1e-14) {
            out.set(i, dense[i]);
        }
    }
}

bool luFactorize(std::vector<std::vector<double>>& lu, std::vector<int>& pivot) {
    const int n = static_cast<int>(lu.size());
    pivot.resize(n);
    for (int i = 0; i < n; ++i) {
        pivot[i] = i;
    }

    for (int k = 0; k < n; ++k) {
        int pivot_row = k;
        double best = std::abs(lu[k][k]);
        for (int i = k + 1; i < n; ++i) {
            const double cand = std::abs(lu[i][k]);
            if (cand > best) {
                best = cand;
                pivot_row = i;
            }
        }
        if (best <= kSingularTol) {
            return false;
        }
        if (pivot_row != k) {
            std::swap(lu[pivot_row], lu[k]);
            std::swap(pivot[pivot_row], pivot[k]);
        }
        for (int i = k + 1; i < n; ++i) {
            lu[i][k] /= lu[k][k];
            for (int j = k + 1; j < n; ++j) {
                lu[i][j] -= lu[i][k] * lu[k][j];
            }
        }
    }
    return true;
}

std::vector<double> luSolve(const std::vector<std::vector<double>>& lu, const std::vector<int>& pivot, const std::vector<double>& rhs) {
    const int n = static_cast<int>(lu.size());
    std::vector<double> x(n, 0.0);
    for (int i = 0; i < n; ++i) {
        x[i] = rhs[pivot[i]];
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            x[i] -= lu[i][j] * x[j];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) {
            x[i] -= lu[i][j] * x[j];
        }
        x[i] /= lu[i][i];
    }
    return x;
}

std::vector<double> luSolveTranspose(
    const std::vector<std::vector<double>>& lu,
    const std::vector<int>& pivot,
    const std::vector<double>& rhs
) {
    const int n = static_cast<int>(lu.size());
    std::vector<double> y(rhs);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            y[i] -= lu[j][i] * y[j];
        }
        y[i] /= lu[i][i];
    }
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) {
            y[i] -= lu[j][i] * y[j];
        }
    }

    std::vector<double> x(n, 0.0);
    for (int i = 0; i < n; ++i) {
        x[pivot[i]] = y[i];
    }
    return x;
}
}  // namespace

bool UmfpackFactor::factorize(const util::PackedMatrix& basis_matrix) {
    if (basis_matrix.numRows() != basis_matrix.numCols()) {
        is_factorized_ = false;
        return false;
    }
    dimension_ = basis_matrix.numRows();
    lu_ = basis_matrix.toDense();
    eta_updates_.clear();
    is_factorized_ = luFactorize(lu_, pivot_);
    return is_factorized_;
}

void UmfpackFactor::ftran(util::IndexedVector& rhs) const {
    if (!is_factorized_) {
        throw std::logic_error("ftran called before successful factorization");
    }
    std::vector<double> v = indexedToDense(rhs, dimension_);
    v = luSolve(lu_, pivot_, v);
    for (const auto& eta : eta_updates_) {
        const int p = eta.pivot_row;
        const double dp = eta.d[p];
        if (std::abs(dp) <= kSingularTol) {
            continue;
        }
        const double xp = v[p] / dp;
        for (int i = 0; i < dimension_; ++i) {
            if (i == p) {
                continue;
            }
            v[i] -= eta.d[i] * xp;
        }
        v[p] = xp;
    }
    denseToIndexed(v, rhs);
}

void UmfpackFactor::btran(util::IndexedVector& rhs) const {
    if (!is_factorized_) {
        throw std::logic_error("btran called before successful factorization");
    }
    std::vector<double> v = indexedToDense(rhs, dimension_);
    for (auto it = eta_updates_.rbegin(); it != eta_updates_.rend(); ++it) {
        const int p = it->pivot_row;
        const double dp = it->d[p];
        if (std::abs(dp) <= kSingularTol) {
            continue;
        }
        double sum = 0.0;
        for (int i = 0; i < dimension_; ++i) {
            if (i == p) {
                continue;
            }
            sum += it->d[i] * v[i];
        }
        v[p] = (v[p] - sum) / dp;
    }
    v = luSolveTranspose(lu_, pivot_, v);
    denseToIndexed(v, rhs);
}

void UmfpackFactor::updateEta(int pivot_row, const util::IndexedVector& ftran_col) {
    if (pivot_row < 0 || pivot_row >= dimension_) {
        return;
    }
    EtaUpdate eta;
    eta.pivot_row = pivot_row;
    eta.d = indexedToDense(ftran_col, dimension_);
    if (eta.d[pivot_row] == 0.0) {
        eta.d[pivot_row] = 1.0;
    }
    eta_updates_.push_back(std::move(eta));
}

int UmfpackFactor::etaFileLength() const { return static_cast<int>(eta_updates_.size()); }

}  // namespace lp_solver::linalg
