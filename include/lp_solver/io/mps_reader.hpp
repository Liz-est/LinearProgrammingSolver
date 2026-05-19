#pragma once

#include <string>

#include "mps_types.hpp"

namespace lp_solver::io {

struct MpsReadResult {
    bool ok{false};
    std::string error;
    RawLpModel model;
};

[[nodiscard]] MpsReadResult readMpsFile(const std::string& file_path);

}  // namespace lp_solver::io
