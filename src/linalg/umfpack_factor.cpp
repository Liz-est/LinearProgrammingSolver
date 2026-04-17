#include "../../include/lp_solver/linalg/umfpack_factor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef LP_SOLVER_HAVE_UMFPACK
extern "C" {
#include <umfpack.h>
}
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

#ifdef LP_SOLVER_HAVE_UMFPACK
[[nodiscard]] bool factorizeWithUmfpack(detail::SparseLuEngine& engine, const util::PackedMatrix& A) {
    engine = detail::SparseLuEngine{};
    if (A.numRows() != A.numCols()) {
        return false;
    }
    const int n = A.numRows();
    if (n <= 0) {
        return false;
    }

    const auto& cs = A.colStarts();
    const auto& ri = A.rowIndices();
    const auto& el = A.elements();
    if (static_cast<int>(cs.size()) != n + 1) {
        return false;
    }
    std::vector<int> Ap(cs.begin(), cs.end());
    std::vector<int> Ai(ri.begin(), ri.end());
    std::vector<double> Ax(el.begin(), el.end());

    double Control[UMFPACK_CONTROL];
    double Info[UMFPACK_INFO];
    umfpack_di_defaults(Control);
    Control[UMFPACK_SCALE] = UMFPACK_SCALE_NONE;

    void* Symbolic = nullptr;
    int status = umfpack_di_symbolic(n, n, Ap.data(), Ai.data(), Ax.data(), &Symbolic, Control, Info);
    if (status != UMFPACK_OK) {
        umfpack_di_free_symbolic(&Symbolic);
        return false;
    }

    void* Numeric = nullptr;
    status = umfpack_di_numeric(Ap.data(), Ai.data(), Ax.data(), Symbolic, &Numeric, Control, Info);
    umfpack_di_free_symbolic(&Symbolic);
    if (status != UMFPACK_OK) {
        umfpack_di_free_numeric(&Numeric);
        return false;
    }

    int lnz = 0;
    int unz = 0;
    int nnz_lu = 0;
    int n_row = 0;
    int n_col = 0;
    status = umfpack_di_get_lunz(&lnz, &unz, &nnz_lu, &n_row, &n_col, Numeric);
    if (status != UMFPACK_OK || n_row != n || n_col != n) {
        umfpack_di_free_numeric(&Numeric);
        return false;
    }

    std::vector<int> Lp_csr(static_cast<size_t>(n + 1));
    std::vector<int> Lj_csr(static_cast<size_t>(lnz));
    std::vector<double> Lx_csr(static_cast<size_t>(lnz));
    std::vector<int> Up_csc(static_cast<size_t>(n + 1));
    std::vector<int> Ui_csc(static_cast<size_t>(unz));
    std::vector<double> Ux_csc(static_cast<size_t>(unz));
    std::vector<int> Pperm(static_cast<size_t>(n));
    std::vector<int> Qperm(static_cast<size_t>(n));
    std::vector<double> Dx(static_cast<size_t>(n));
    std::vector<double> Rs(static_cast<size_t>(n));
    int do_recip = 0;

    status = umfpack_di_get_numeric(
        Lp_csr.data(),
        Lj_csr.data(),
        Lx_csr.data(),
        Up_csc.data(),
        Ui_csc.data(),
        Ux_csc.data(),
        Pperm.data(),
        Qperm.data(),
        Dx.data(),
        &do_recip,
        Rs.data(),
        Numeric
    );
    umfpack_di_free_numeric(&Numeric);
    if (status != UMFPACK_OK) {
        return false;
    }

    std::vector<int> Lp_csc;
    std::vector<int> Li_csc;
    std::vector<double> Lx_csc;
    csrToCsc(n, Lp_csr, Lj_csr, Lx_csr, Lp_csc, Li_csc, Lx_csc);

    engine.adoptFactorData(n, std::move(Lp_csc), std::move(Li_csc), std::move(Lx_csc), std::move(Up_csc), std::move(Ui_csc), std::move(Ux_csc), std::move(Pperm), std::move(Qperm));
    return engine.ok();
}
#endif
}  // namespace

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
#ifdef LP_SOLVER_HAVE_UMFPACK
    return factorizeWithUmfpack(engine_, basis_matrix);
#else
    return engine_.factorize(basis_matrix);
#endif
}

void UmfpackFactor::ftran(util::IndexedVector& rhs) const {
    if (!engine_.ok()) {
        throw std::logic_error("UmfpackFactor::ftran called before successful factorization");
    }
    engine_.ftran(rhs);
    if (!eta_updates_.empty()) {
        auto v = indexedToDense(rhs, engine_.dimension());
        applyEtaForward(v, eta_updates_);
        denseToIndexed(v, rhs);
    }
}

void UmfpackFactor::btran(util::IndexedVector& rhs) const {
    if (!engine_.ok()) {
        throw std::logic_error("UmfpackFactor::btran called before successful factorization");
    }
    if (!eta_updates_.empty()) {
        auto v = indexedToDense(rhs, engine_.dimension());
        applyEtaBackward(v, eta_updates_);
        denseToIndexed(v, rhs);
    }
    engine_.btran(rhs);
}

void UmfpackFactor::updateEta(int pivot_row, const util::IndexedVector& ftran_col) {
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

int UmfpackFactor::etaFileLength() const { return static_cast<int>(eta_updates_.size()); }

}  // namespace lp_solver::linalg
