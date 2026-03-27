#pragma once

#include <vector>

#include "indexed_vector.hpp"

namespace lp_solver::util {

class PackedMatrix {
public:
    class Builder {
    public:
        Builder(int num_rows, int num_cols);
        Builder& appendColumn(const std::vector<int>& row_indices, const std::vector<double>& values);
        PackedMatrix build() &&;

    private:
        int num_rows_;
        int num_cols_;
        int current_col_{0};
        std::vector<double> elements_;
        std::vector<int> row_indices_;
        std::vector<int> col_starts_;
    };

    [[nodiscard]] IndexedVector column(int j) const;
    [[nodiscard]] const std::vector<int>& rowIndices() const;
    [[nodiscard]] const std::vector<int>& colStarts() const;
    [[nodiscard]] const std::vector<double>& elements() const;
    [[nodiscard]] int numRows() const;
    [[nodiscard]] int numCols() const;
    [[nodiscard]] int numNonZeros() const;

private:
    explicit PackedMatrix(
        int nr,
        int nc,
        std::vector<double> el,
        std::vector<int> ri,
        std::vector<int> cs
    );

    int num_rows_;
    int num_cols_;
    std::vector<double> elements_;
    std::vector<int> row_indices_;
    std::vector<int> col_starts_;

    friend class Builder;
};

}  // namespace lp_solver::util
