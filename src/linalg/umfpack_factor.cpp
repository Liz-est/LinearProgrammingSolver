#include "../../include/lp_solver/linalg/umfpack_factor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef LP_SOLVER_HAVE_UMFPACK
#include <umfpack.h>
#endif

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

/// Convert square CSR (row-major sparse) to CSC. Indices may be unsorted per column; GP solves
/// tolerate the lower-triangular structure here.
void csrToCsc(
    int n,
    const std::vector<int>& Rp,
    const std::vector<int>& Rj,
    const std::vector<double>& Rx,
    std::vector<int>& Cp,
    std::vector<int>& Ci,
    std::vector<double>& Cx
) {
    std::vector<int> count(static_cast<size_t>(n), 0);
    const int nnz = Rp[static_cast<size_t>(n)];
    for (int p = 0; p < nnz; ++p) {
        ++count[static_cast<size_t>(Rj[static_cast<size_t>(p)])];
    }
    Cp.assign(static_cast<size_t>(n + 1), 0);
    for (int j = 1; j <= n; ++j) {
        Cp[static_cast<size_t>(j)] = Cp[static_cast<size_t>(j - 1)] + count[static_cast<size_t>(j - 1)];
    }
    std::vector<int> next = Cp;
    Ci.resize(static_cast<size_t>(nnz));
    Cx.resize(static_cast<size_t>(nnz));
    for (int i = 0; i < n; ++i) {
        for (int p = Rp[static_cast<size_t>(i)]; p < Rp[static_cast<size_t>(i + 1)]; ++p) {
            const int j = Rj[static_cast<size_t>(p)];
            const int q = next[static_cast<size_t>(j)]++;
            Ci[static_cast<size_t>(q)] = i;
            Cx[static_cast<size_t>(q)] = Rx[static_cast<size_t>(p)];
        }
    }
    for (int j = 0; j < n; ++j) {
        const int p0 = Cp[static_cast<size_t>(j)];
        const int p1 = Cp[static_cast<size_t>(j + 1)];
        for (int a = p0 + 1; a < p1; ++a) {
            const int row = Ci[static_cast<size_t>(a)];
            const double val = Cx[static_cast<size_t>(a)];
            int b = a;
            while (b > p0 && Ci[static_cast<size_t>(b - 1)] > row) {
                Ci[static_cast<size_t>(b)] = Ci[static_cast<size_t>(b - 1)];
                Cx[static_cast<size_t>(b)] = Cx[static_cast<size_t>(b - 1)];
                --b;
            }
            Ci[static_cast<size_t>(b)] = row;
            Cx[static_cast<size_t>(b)] = val;
        }
    }
}

}  // namespace

UmfpackFactor::~UmfpackFactor() {
#ifdef LP_SOLVER_HAVE_UMFPACK
    clearUmfpackState();
#endif
}

#ifdef LP_SOLVER_HAVE_UMFPACK
void UmfpackFactor::clearUmfpackState() {
    if (umf_numeric_ != nullptr) {
        umfpack_di_free_numeric(&umf_numeric_);
    }
    use_umfpack_direct_ = false;
    Ap_.clear();
    Ai_.clear();
    Ax_.clear();
}

bool UmfpackFactor::factorizeWithUmfpackDirect(const util::PackedMatrix& basis_matrix) {
    if (basis_matrix.numRows() != basis_matrix.numCols()) {
        return false;
    }
    const int n = basis_matrix.numRows();
    if (n <= 0) {
        return false;
    }

    const auto& cs = basis_matrix.colStarts();
    const auto& ri = basis_matrix.rowIndices();
    const auto& el = basis_matrix.elements();
    if (static_cast<int>(cs.size()) != n + 1) {
        return false;
    }

    Ap_.assign(cs.begin(), cs.end());
    Ai_.assign(ri.begin(), ri.end());
    Ax_.assign(el.begin(), el.end());

    double Control[UMFPACK_CONTROL];
    double Info[UMFPACK_INFO];
    umfpack_di_defaults(Control);
    // Keep row scaling disabled so behavior matches EigenFactor and backend-consistency tests.
    Control[UMFPACK_SCALE] = UMFPACK_SCALE_NONE;

    void* Symbolic = nullptr;
    int status = umfpack_di_symbolic(n, n, Ap_.data(), Ai_.data(), Ax_.data(), &Symbolic, Control, Info);
    if (status != UMFPACK_OK) {
        umfpack_di_free_symbolic(&Symbolic);
        return false;
    }

    status = umfpack_di_numeric(Ap_.data(), Ai_.data(), Ax_.data(), Symbolic, &umf_numeric_, Control, Info);
    umfpack_di_free_symbolic(&Symbolic);
    if (status != UMFPACK_OK) {
        clearUmfpackState();
        return false;
    }

    use_umfpack_direct_ = true;
    return true;
}

void UmfpackFactor::umfpackSolve(int sys, util::IndexedVector& rhs) const {
    if (!use_umfpack_direct_ || umf_numeric_ == nullptr) {
        throw std::logic_error("UmfpackFactor::umfpackSolve called without UMFPACK numeric factorization");
    }
    if (rhs.capacity() < dimension_) {
        throw std::logic_error("UmfpackFactor::umfpackSolve rhs capacity mismatch");
    }
    std::vector<double> b = indexedToDense(rhs, dimension_);
    std::vector<double> x(static_cast<size_t>(dimension_), 0.0);
    const int status = umfpack_di_solve(
        sys,
        Ap_.data(),
        Ai_.data(),
        Ax_.data(),
        x.data(),
        b.data(),
        umf_numeric_,
        nullptr,
        nullptr
    );
    if (status != UMFPACK_OK) {
        throw std::runtime_error("umfpack_di_solve failed");
    }
    denseToIndexed(x, rhs);
}
#endif

void UmfpackFactor::applyEtaForward(std::vector<double>& v, const std::vector<EtaUpdate>& etas) {
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

void UmfpackFactor::applyEtaBackward(std::vector<double>& v, const std::vector<EtaUpdate>& etas) {
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

bool UmfpackFactor::factorize(const util::PackedMatrix& basis_matrix) {
    eta_updates_.clear();
    dimension_ = basis_matrix.numRows();
#ifdef LP_SOLVER_HAVE_UMFPACK
    clearUmfpackState();
    if (factorizeWithUmfpackDirect(basis_matrix)) {
        return true;
    }
    return engine_.factorize(basis_matrix);
#else
    return engine_.factorize(basis_matrix);
#endif
}

void UmfpackFactor::ftran(util::IndexedVector& rhs) const {
    bool solved = false;
#ifdef LP_SOLVER_HAVE_UMFPACK
    if (use_umfpack_direct_) {
        umfpackSolve(UMFPACK_A, rhs);
        solved = true;
    }
#endif
    if (!solved) {
        if (!engine_.ok()) {
            throw std::logic_error("UmfpackFactor::ftran called before successful factorization");
        }
        engine_.ftran(rhs);
    }
    if (!eta_updates_.empty()) {
        auto v = indexedToDense(rhs, dimension_);
        applyEtaForward(v, eta_updates_);
        denseToIndexed(v, rhs);
    }
}

void UmfpackFactor::btran(util::IndexedVector& rhs) const {
    if (!eta_updates_.empty()) {
        auto v = indexedToDense(rhs, dimension_);
        applyEtaBackward(v, eta_updates_);
        denseToIndexed(v, rhs);
    }
    bool solved = false;
#ifdef LP_SOLVER_HAVE_UMFPACK
    if (use_umfpack_direct_) {
        umfpackSolve(UMFPACK_At, rhs);
        solved = true;
    }
#endif
    if (!solved) {
        if (!engine_.ok()) {
            throw std::logic_error("UmfpackFactor::btran called before successful factorization");
        }
        engine_.btran(rhs);
    }
}

void UmfpackFactor::updateEta(int pivot_row, const util::IndexedVector& ftran_col) {
    const int n = dimension_;
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

int UmfpackFactor::etaFileLength() const { return static_cast<int>(eta_updates_.size()); }

}  // namespace lp_solver::linalg
