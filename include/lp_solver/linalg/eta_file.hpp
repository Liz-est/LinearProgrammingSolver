#pragma once

#include <vector>

#include "../util/indexed_vector.hpp"

namespace lp_solver::linalg {

/// Product-form inverse eta updates stored in sparse form (only non-zeros of each eta vector).
class EtaFile {
public:
    struct Update {
        int pivot_row{0};
        double pivot_value{1.0};
        std::vector<int> indices;
        std::vector<double> values;
    };

    void clear();

    /// Append $E = I + (d - e_p) e_p^T$ using non-zeros from `ftran_col`.
    /// Returns false when the pivot entry is too small to form a reliable eta update.
    [[nodiscard]] bool append(int pivot_row, const util::IndexedVector& ftran_col);

    /// Apply $E_k^{-1} \cdots E_1^{-1}$ to `v` (forward sweep, ftran path).
    void applyForward(util::IndexedVector& v) const;

    /// Apply $E_k^{-T} \cdots E_1^{-T}$ to `v` (backward sweep, btran path).
    void applyBackward(util::IndexedVector& v) const;

    [[nodiscard]] int length() const;
    [[nodiscard]] const std::vector<Update>& updates() const { return updates_; }

private:
    std::vector<Update> updates_;
};

}  // namespace lp_solver::linalg
