#include "../../include/lp_solver/util/packed_matrix.hpp"

#include <stdexcept>

namespace lp_solver::util {

PackedMatrix::Builder::Builder(int num_rows, int num_cols)
    : num_rows_(num_rows), num_cols_(num_cols), col_starts_{0} {
    if (num_rows < 0 || num_cols < 0) {
        throw std::invalid_argument("matrix dimensions must be non-negative");
    }
}

PackedMatrix::Builder& PackedMatrix::Builder::appendColumn(
    const std::vector<int>& row_indices,
    const std::vector<double>& values
) {
    if (row_indices.size() != values.size()) {
        throw std::invalid_argument("row_indices and values must have same size");
    }
    if (current_col_ >= num_cols_) {
        throw std::logic_error("appendColumn called too many times");
    }

    for (size_t i = 0; i < values.size(); ++i) {
        if (row_indices[i] < 0 || row_indices[i] >= num_rows_) {
            throw std::out_of_range("row index out of bounds in appendColumn");
        }
        row_indices_.push_back(row_indices[i]);
        elements_.push_back(values[i]);
    }
    col_starts_.push_back(static_cast<int>(elements_.size()));
    ++current_col_;
    return *this;
}

PackedMatrix PackedMatrix::Builder::build() && {
    if (current_col_ != num_cols_) {
        throw std::logic_error("build() called before all columns are appended");
    }
    return PackedMatrix(
        num_rows_,
        num_cols_,
        std::move(elements_),
        std::move(row_indices_),
        std::move(col_starts_)
    );
}

IndexedVector PackedMatrix::column(int j) const {
    if (j < 0 || j >= num_cols_) {
        throw std::out_of_range("column index out of range");
    }

    IndexedVector out(num_rows_);
    const int begin = col_starts_[j];
    const int end = col_starts_[j + 1];
    for (int k = begin; k < end; ++k) {
        out.add(row_indices_[k], elements_[k]);
    }
    return out;
}

std::vector<double> PackedMatrix::multiply(const std::vector<double>& x) const {
    if (static_cast<int>(x.size()) != num_cols_) {
        throw std::invalid_argument("multiply input size must equal numCols");
    }
    std::vector<double> y(num_rows_, 0.0);
    for (int col = 0; col < num_cols_; ++col) {
        const double xj = x[col];
        if (xj == 0.0) {
            continue;
        }
        const int begin = col_starts_[col];
        const int end = col_starts_[col + 1];
        for (int k = begin; k < end; ++k) {
            y[row_indices_[k]] += elements_[k] * xj;
        }
    }
    return y;
}

std::vector<double> PackedMatrix::transposeMultiply(const std::vector<double>& y) const {
    if (static_cast<int>(y.size()) != num_rows_) {
        throw std::invalid_argument("transposeMultiply input size must equal numRows");
    }
    std::vector<double> x(num_cols_, 0.0);
    for (int col = 0; col < num_cols_; ++col) {
        double sum = 0.0;
        const int begin = col_starts_[col];
        const int end = col_starts_[col + 1];
        for (int k = begin; k < end; ++k) {
            sum += elements_[k] * y[row_indices_[k]];
        }
        x[col] = sum;
    }
    return x;
}

std::vector<std::vector<double>> PackedMatrix::toDense() const {
    std::vector<std::vector<double>> dense(num_rows_, std::vector<double>(num_cols_, 0.0));
    for (int col = 0; col < num_cols_; ++col) {
        const int begin = col_starts_[col];
        const int end = col_starts_[col + 1];
        for (int k = begin; k < end; ++k) {
            dense[row_indices_[k]][col] = elements_[k];
        }
    }
    return dense;
}

const std::vector<int>& PackedMatrix::rowIndices() const { return row_indices_; }
const std::vector<int>& PackedMatrix::colStarts() const { return col_starts_; }
const std::vector<double>& PackedMatrix::elements() const { return elements_; }

int PackedMatrix::numRows() const { return num_rows_; }
int PackedMatrix::numCols() const { return num_cols_; }
int PackedMatrix::numNonZeros() const { return static_cast<int>(elements_.size()); }

PackedMatrix::PackedMatrix(
    int nr,
    int nc,
    std::vector<double> el,
    std::vector<int> ri,
    std::vector<int> cs
)
    : num_rows_(nr),
      num_cols_(nc),
      elements_(std::move(el)),
      row_indices_(std::move(ri)),
      col_starts_(std::move(cs)) {
    if (num_rows_ < 0 || num_cols_ < 0) {
        throw std::invalid_argument("matrix dimensions must be non-negative");
    }
    if (col_starts_.size() != static_cast<size_t>(num_cols_ + 1)) {
        throw std::invalid_argument("col_starts must have num_cols + 1 entries");
    }
    if (elements_.size() != row_indices_.size()) {
        throw std::invalid_argument("elements and row_indices must have equal size");
    }
    if (col_starts_.back() != static_cast<int>(elements_.size())) {
        throw std::invalid_argument("last col_starts entry must equal nnz");
    }
}

}  // namespace lp_solver::util
