#include "../../include/lp_solver/linalg/detail/sparse_lu_engine.hpp"
#include "../../include/lp_solver/linalg/detail/sparse_triangular.hpp"

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace lp_solver::linalg::detail {

namespace {
using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using MappedSupernodal = Eigen::internal::MappedSuperNodalMatrix<SpMat::Scalar, SpMat::StorageIndex>;

void packedToEigen(const util::PackedMatrix& A, SpMat& out) {
    const int m = A.numRows();
    const int n = A.numCols();
    const auto& cs = A.colStarts();
    const auto& ri = A.rowIndices();
    const auto& el = A.elements();
    std::vector<Eigen::Triplet<double>> t;
    t.reserve(static_cast<size_t>(A.numNonZeros()));
    for (int j = 0; j < n; ++j) {
        for (int p = cs[j]; p < cs[j + 1]; ++p) {
            t.emplace_back(ri[static_cast<size_t>(p)], j, el[static_cast<size_t>(p)]);
        }
    }
    out.resize(m, n);
    out.setFromTriplets(t.begin(), t.end());
}

void eigenPermToVector(const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>& perm, std::vector<int>& out) {
    const int n = static_cast<int>(perm.size());
    out.resize(n);
    for (int i = 0; i < n; ++i) {
        out[static_cast<size_t>(i)] = static_cast<int>(perm.indices()[i]);
    }
}

void buildInversePerm(const std::vector<int>& perm, std::vector<int>& inv) {
    const int n = static_cast<int>(perm.size());
    inv.assign(static_cast<size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
        inv[static_cast<size_t>(perm[static_cast<size_t>(i)])] = i;
    }
}

void extractLowerUnitCsc(const SpMat& L, std::vector<int>& Lp, std::vector<int>& Li, std::vector<double>& Lx) {
    const int n = static_cast<int>(L.cols());
    Lp.assign(static_cast<size_t>(n + 1), 0);
    int nnz = 0;
    for (int j = 0; j < n; ++j) {
        for (SpMat::InnerIterator it(L, j); it; ++it) {
            if (it.row() > j) {
                ++nnz;
            }
        }
    }
    Li.resize(static_cast<size_t>(nnz));
    Lx.resize(static_cast<size_t>(nnz));
    int t = 0;
    Lp[0] = 0;
    for (int j = 0; j < n; ++j) {
        for (SpMat::InnerIterator it(L, j); it; ++it) {
            if (it.row() > j) {
                Li[static_cast<size_t>(t)] = static_cast<int>(it.row());
                Lx[static_cast<size_t>(t)] = it.value();
                ++t;
            }
        }
        Lp[static_cast<size_t>(j + 1)] = t;
    }
}

void extractUpperCsc(const SpMat& U, std::vector<int>& Up, std::vector<int>& Ui, std::vector<double>& Ux) {
    const int n = static_cast<int>(U.cols());
    Up.assign(static_cast<size_t>(n + 1), 0);
    int nnz = 0;
    for (int j = 0; j < n; ++j) {
        for (SpMat::InnerIterator it(U, j); it; ++it) {
            if (it.row() <= j) {
                ++nnz;
            }
        }
    }
    Ui.resize(static_cast<size_t>(nnz));
    Ux.resize(static_cast<size_t>(nnz));
    int t = 0;
    Up[0] = 0;
    for (int j = 0; j < n; ++j) {
        for (SpMat::InnerIterator it(U, j); it; ++it) {
            if (it.row() <= j) {
                Ui[static_cast<size_t>(t)] = static_cast<int>(it.row());
                Ux[static_cast<size_t>(t)] = it.value();
                ++t;
            }
        }
        Up[static_cast<size_t>(j + 1)] = t;
    }
}

template <class TIter>
void sortByRowIndex(TIter begin, TIter end) {
    std::sort(begin, end, [](const auto& a, const auto& b) { return a.first < b.first; });
}

void transposeToCsc(
    const std::vector<int>& Ap,
    const std::vector<int>& Ai,
    const std::vector<double>& Ax,
    int n,
    std::vector<int>& Atp,
    std::vector<int>& Ati,
    std::vector<double>& Atx
) {
    std::vector<int> count(static_cast<size_t>(n), 0);
    const int nnz = Ap[static_cast<size_t>(n)];
    for (int p = 0; p < nnz; ++p) {
        const int i = Ai[static_cast<size_t>(p)];
        ++count[static_cast<size_t>(i)];
    }
    Atp.assign(static_cast<size_t>(n + 1), 0);
    for (int i = 1; i <= n; ++i) {
        Atp[static_cast<size_t>(i)] = Atp[static_cast<size_t>(i - 1)] + count[static_cast<size_t>(i - 1)];
    }
    std::vector<int> next = Atp;
    Ati.resize(static_cast<size_t>(nnz));
    Atx.resize(static_cast<size_t>(nnz));
    for (int j = 0; j < n; ++j) {
        for (int p = Ap[static_cast<size_t>(j)]; p < Ap[static_cast<size_t>(j + 1)]; ++p) {
            const int i = Ai[static_cast<size_t>(p)];
            const int q = next[static_cast<size_t>(i)]++;
            Ati[static_cast<size_t>(q)] = j;
            Atx[static_cast<size_t>(q)] = Ax[static_cast<size_t>(p)];
        }
    }
}
}  // namespace

bool SparseLuEngine::factorize(const util::PackedMatrix& basis_matrix) {
    factor_ok_ = false;
    if (basis_matrix.numRows() != basis_matrix.numCols()) {
        return false;
    }
    n_ = basis_matrix.numRows();
    if (n_ <= 0) {
        return false;
    }

    SpMat A;
    packedToEigen(basis_matrix, A);

    Eigen::SparseLU<SpMat> lu;
    lu.compute(A);
    if (lu.info() != Eigen::Success) {
        return false;
    }

    eigenPermToVector(lu.rowsPermutation(), perm_r_);
    eigenPermToVector(lu.colsPermutation(), perm_c_);

    const auto Lwrap = lu.matrixL();
    const MappedSupernodal& Lstore = Lwrap.m_mapL;
    std::vector<double> u_diag_from_l(static_cast<size_t>(n_), 0.0);

    Lp_.assign(static_cast<size_t>(n_ + 1), 0);
    Li_.clear();
    Lx_.clear();
    Li_.reserve(static_cast<size_t>(std::max<Eigen::Index>(0, lu.nnzL())));
    Lx_.reserve(static_cast<size_t>(std::max<Eigen::Index>(0, lu.nnzL())));
    for (int j = 0; j < n_; ++j) {
        std::vector<std::pair<int, double>> col;
        for (typename MappedSupernodal::InnerIterator it(Lstore, static_cast<Eigen::Index>(j)); it; ++it) {
            const int row = static_cast<int>(it.row());
            const double val = it.value();
            if (row == j) {
                u_diag_from_l[static_cast<size_t>(j)] = val;
            } else if (row > j) {
                col.emplace_back(row, val);
            }
        }
        sortByRowIndex(col.begin(), col.end());
        for (const auto& [row, val] : col) {
            Li_.push_back(row);
            Lx_.push_back(val);
        }
        Lp_[static_cast<size_t>(j + 1)] = static_cast<int>(Li_.size());
    }

    const auto Uwrap = lu.matrixU();
    using MappedU = std::decay_t<decltype(Uwrap.m_mapU)>;
    Up_.assign(static_cast<size_t>(n_ + 1), 0);
    Ui_.clear();
    Ux_.clear();
    Ui_.reserve(static_cast<size_t>(std::max<Eigen::Index>(0, lu.nnzU()) + n_));
    Ux_.reserve(static_cast<size_t>(std::max<Eigen::Index>(0, lu.nnzU()) + n_));
    for (int j = 0; j < n_; ++j) {
        std::vector<std::pair<int, double>> col;
        bool have_u_diag = false;
        double ujj = 0.0;
        for (typename MappedU::InnerIterator it(Uwrap.m_mapU, j); it; ++it) {
            const int row = static_cast<int>(it.row());
            const double val = it.value();
            if (row < j) {
                col.emplace_back(row, val);
            } else if (row == j) {
                have_u_diag = true;
                ujj = val;
            }
        }
        double diag = (have_u_diag && std::abs(ujj) > 1e-19) ? ujj : u_diag_from_l[static_cast<size_t>(j)];
        if (std::abs(diag) > 1e-19) {
            col.emplace_back(j, diag);
        }
        sortByRowIndex(col.begin(), col.end());
        for (const auto& [row, val] : col) {
            Ui_.push_back(row);
            Ux_.push_back(val);
        }
        Up_[static_cast<size_t>(j + 1)] = static_cast<int>(Ui_.size());
    }

    transposeToCsc(Up_, Ui_, Ux_, n_, Ut_p_, Ut_i_, Ut_x_);
    transposeToCsc(Lp_, Li_, Lx_, n_, Lt_p_, Lt_i_, Lt_x_);

    gp_mark_.assign(static_cast<size_t>(n_), 0);
    gp_stack_.assign(static_cast<size_t>(n_), 0);

    factor_ok_ = true;
    return true;
}

void SparseLuEngine::adoptFactorData(
    int dim,
    std::vector<int> Lp,
    std::vector<int> Li,
    std::vector<double> Lx,
    std::vector<int> Up,
    std::vector<int> Ui,
    std::vector<double> Ux,
    std::vector<int> perm_r,
    std::vector<int> perm_c
) {
    factor_ok_ = false;
    n_ = 0;
    if (dim <= 0) {
        return;
    }
    if (static_cast<int>(Lp.size()) != dim + 1 || static_cast<int>(Up.size()) != dim + 1) {
        return;
    }
    if (static_cast<int>(perm_r.size()) != dim || static_cast<int>(perm_c.size()) != dim) {
        return;
    }
    const int l_end = Lp[static_cast<size_t>(dim)];
    const int u_end = Up[static_cast<size_t>(dim)];
    if (l_end < 0 || u_end < 0) {
        return;
    }
    if (static_cast<int>(Li.size()) != l_end || static_cast<int>(Lx.size()) != l_end) {
        return;
    }
    if (static_cast<int>(Ui.size()) != u_end || static_cast<int>(Ux.size()) != u_end) {
        return;
    }

    n_ = dim;
    Lp_ = std::move(Lp);
    Li_ = std::move(Li);
    Lx_ = std::move(Lx);
    Up_ = std::move(Up);
    Ui_ = std::move(Ui);
    Ux_ = std::move(Ux);
    perm_r_ = std::move(perm_r);
    perm_c_ = std::move(perm_c);

    transposeToCsc(Up_, Ui_, Ux_, n_, Ut_p_, Ut_i_, Ut_x_);
    transposeToCsc(Lp_, Li_, Lx_, n_, Lt_p_, Lt_i_, Lt_x_);

    gp_mark_.assign(static_cast<size_t>(n_), 0);
    gp_stack_.assign(static_cast<size_t>(n_), 0);

    factor_ok_ = true;
}

void SparseLuEngine::applyRowPermForward(const std::vector<int>& perm, util::IndexedVector& x) const {
    const auto& raw = x.rawValues();
    util::IndexedVector tmp(n_);
    for (int i = 0; i < n_; ++i) {
        const int src = perm[static_cast<size_t>(i)];
        if (src >= 0 && src < n_ && std::abs(raw[static_cast<size_t>(src)]) > 1e-19) {
            tmp.set(i, raw[static_cast<size_t>(src)]);
        }
    }
    x.clear();
    for (int idx : tmp.nonZeroIndices()) {
        x.set(idx, tmp[idx]);
    }
}

void SparseLuEngine::applyRowPermInverseTranspose(const std::vector<int>& perm, util::IndexedVector& x) const {
    std::vector<int> inv;
    buildInversePerm(perm, inv);
    applyRowPermForward(inv, x);
}

void SparseLuEngine::applyColPermForward(const std::vector<int>& perm, util::IndexedVector& x) const {
    std::vector<int> inv;
    buildInversePerm(perm, inv);
    const auto& raw = x.rawValues();
    util::IndexedVector tmp(n_);
    for (int i = 0; i < n_; ++i) {
        const int src = inv[static_cast<size_t>(i)];
        if (src >= 0 && src < n_ && std::abs(raw[static_cast<size_t>(src)]) > 1e-19) {
            tmp.set(i, raw[static_cast<size_t>(src)]);
        }
    }
    x.clear();
    for (int idx : tmp.nonZeroIndices()) {
        x.set(idx, tmp[idx]);
    }
}

void SparseLuEngine::applyColPermInverseTranspose(const std::vector<int>& perm, util::IndexedVector& x) const {
    applyRowPermForward(perm, x);
}

void SparseLuEngine::ftran(util::IndexedVector& rhs) const {
    if (!factor_ok_) {
        throw std::logic_error("SparseLuEngine::ftran: not factorized");
    }
    applyRowPermForward(perm_r_, rhs);

    CscMatrixView L{ n_, Lp_.data(), Li_.data(), Lx_.data() };
    gpLowerUnitSolve(L, rhs, gp_mark_, gp_stack_);

    CscMatrixView U{ n_, Up_.data(), Ui_.data(), Ux_.data() };
    gpUpperSolve(U, rhs, gp_mark_, gp_stack_, false);

    applyColPermForward(perm_c_, rhs);
}

void SparseLuEngine::btran(util::IndexedVector& rhs) const {
    if (!factor_ok_) {
        throw std::logic_error("SparseLuEngine::btran: not factorized");
    }
    applyColPermInverseTranspose(perm_c_, rhs);

    CscMatrixView Ut{ n_, Ut_p_.data(), Ut_i_.data(), Ut_x_.data() };
    gpLowerDiagSolve(Ut, rhs, gp_mark_, gp_stack_);

    CscMatrixView Lt{ n_, Lt_p_.data(), Lt_i_.data(), Lt_x_.data() };
    gpUpperSolve(Lt, rhs, gp_mark_, gp_stack_, true);

    applyRowPermInverseTranspose(perm_r_, rhs);
}

}  // namespace lp_solver::linalg::detail
