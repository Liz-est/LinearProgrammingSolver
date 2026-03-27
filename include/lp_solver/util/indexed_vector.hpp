#pragma once

#include <vector>

namespace lp_solver::util {

class IndexedVector {
public:
    explicit IndexedVector(int capacity);

    void add(int index, double value);
    void set(int index, double value);
    void clear();

    [[nodiscard]] double operator[](int index) const;
    [[nodiscard]] int numNonZeros() const;
    [[nodiscard]] int capacity() const;

    [[nodiscard]] const std::vector<int>& nonZeroIndices() const;
    [[nodiscard]] const std::vector<double>& nonZeroValues() const;
    [[nodiscard]] const std::vector<double>& rawValues() const;

private:
    void insertNonZeroIndex(int index);
    void eraseNonZeroIndex(int index);

    std::vector<double> elements_;
    std::vector<int> indices_;
    std::vector<int> pos_in_indices_;
    mutable std::vector<double> non_zero_values_cache_;
    int nnz_{0};
};

}  // namespace lp_solver::util
