#include "../../include/lp_solver/util/indexed_vector.hpp"

#include <cmath>
#include <stdexcept>

namespace lp_solver::util {

namespace {
constexpr double kZeroTol = 1e-12;
[[nodiscard]] bool isZero(double x) { return std::abs(x) <= kZeroTol; }
}  // namespace

IndexedVector::IndexedVector(int capacity)
    : elements_(capacity, 0.0), pos_in_indices_(capacity, -1), non_zero_values_cache_() {}

void IndexedVector::add(int index, double value) {
    if (index < 0 || index >= static_cast<int>(elements_.size())) {
        throw std::out_of_range("IndexedVector::add index out of range");
    }
    const double old = elements_[index];
    const double next = old + value;
    elements_[index] = next;
    if (isZero(old) && !isZero(next)) {
        insertNonZeroIndex(index);
    } else if (!isZero(old) && isZero(next)) {
        elements_[index] = 0.0;
        eraseNonZeroIndex(index);
    }
}

void IndexedVector::set(int index, double value) {
    if (index < 0 || index >= static_cast<int>(elements_.size())) {
        throw std::out_of_range("IndexedVector::set index out of range");
    }
    const double old = elements_[index];
    elements_[index] = value;
    if (isZero(old) && !isZero(value)) {
        insertNonZeroIndex(index);
    } else if (!isZero(old) && isZero(value)) {
        elements_[index] = 0.0;
        eraseNonZeroIndex(index);
    }
}

void IndexedVector::clear() {
    for (int idx : indices_) {
        elements_[idx] = 0.0;
        pos_in_indices_[idx] = -1;
    }
    indices_.clear();
    nnz_ = 0;
}

double IndexedVector::operator[](int index) const { return elements_[index]; }

int IndexedVector::numNonZeros() const { return nnz_; }

int IndexedVector::capacity() const { return static_cast<int>(elements_.size()); }

const std::vector<int>& IndexedVector::nonZeroIndices() const { return indices_; }

const std::vector<double>& IndexedVector::nonZeroValues() const {
    non_zero_values_cache_.clear();
    non_zero_values_cache_.reserve(indices_.size());
    for (int idx : indices_) {
        non_zero_values_cache_.push_back(elements_[idx]);
    }
    return non_zero_values_cache_;
}

const std::vector<double>& IndexedVector::rawValues() const { return elements_; }

void IndexedVector::insertNonZeroIndex(int index) {
    if (pos_in_indices_[index] != -1) {
        return;
    }
    pos_in_indices_[index] = static_cast<int>(indices_.size());
    indices_.push_back(index);
    ++nnz_;
}

void IndexedVector::eraseNonZeroIndex(int index) {
    const int pos = pos_in_indices_[index];
    if (pos == -1) {
        return;
    }
    const int last_pos = static_cast<int>(indices_.size()) - 1;
    const int last_idx = indices_[last_pos];
    indices_[pos] = last_idx;
    pos_in_indices_[last_idx] = pos;
    indices_.pop_back();
    pos_in_indices_[index] = -1;
    --nnz_;
}

}  // namespace lp_solver::util
