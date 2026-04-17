# 线性规划求解器

**语言** | [English](README.md)

用于 **标准型线性规划**（在 \(Ax=b,\ x\ge 0\) 下最小化 \(c^\top x\)）的 **C++20** 库：核心是 **修正对偶单纯形**，矩阵以 **CSC** 稀疏格式存放，可选 **预处理/后处理**，基矩阵通过可插拔的 **`IBasisFactor`** 维护。

## 特性

- **修正对偶单纯形**：`CHUZR → BTRAN → CHUZC → FTRAN → 转轴` 及对偶、原始更新
- **Phase I Big-M（教科书流程）**：对**当前非基变量**增加显式界约束行与人工变量入基，做一次**强制转轴**（最负 reduced cost 进基、人工离基），再进入 Phase II；与仓库内 `mat3007h_Project_Manual.tex` 一致
- **基维护**：对 \(B\) 做 **稀疏 LU** 分解，迭代间用 **ETA 链**（乘积形式逆）修正；达到 `refactor_frequency` 时 **重构** LU
- **定价**：可选 **对偶最陡边（DSE）** 选离基行及 **Goldfarb–Reid 型** 权重递推；进基采用 **Harris 两阶段**比率检验
- **预处理/后处理**：代数化简 + **LIFO 栈** 记录，`postsolvePrimal` 恢复原始维度上的原始解
- **稀疏基础结构**：`PackedMatrix`（CSC）、`IndexedVector`；`multiply` / `transposeMultiply`
- **超稀疏三角解**：基于 Gilbert-Peierls 风格的 CSC 稀疏前/回代
- **因子后端**：`EigenFactor` 使用 Eigen SparseLU；`UmfpackFactor` 在检测到 SuiteSparse UMFPACK 时使用 UMFPACK，否则回退到同一套稀疏引擎

## 项目结构

```
LinearProgramingSolver/
├── include/lp_solver/
│   ├── linalg/ # 基分解接口与 LU + ETA
│   ├── model/                  # ProblemData, SolverState
│   ├── presolve/               # Presolver
│   ├── simplex/                # DualSimplex
│   └── util/                   # PackedMatrix, IndexedVector
├── src/
├── tests/
│   ├── smoke_test.cpp
│   ├── stress_test.cpp
│   └── advanced_features_test.cpp
├── mat3007h_Project_Manual.tex # 课程实现手册（参考）
└── CMakeLists.txt
```

## 环境要求

- **C++20**（MSVC、GCC 或 Clang）
- **CMake 3.20+**

## 第三方依赖说明

- 项目可使用 **Eigen**，并可选启用 **SuiteSparse / UMFPACK**。
- 若你是本地手动下载源码压缩包并解压（例如 `eigen-5.0.0/`），建议尽量放在仓库目录之外。
- 若需要临时放在仓库根目录，请加入 `.gitignore`，避免误提交第三方源码目录。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
```

## 测试

```bash
ctest --test-dir build -C Debug --output-on-failure
```

Windows：

```powershell
.\run-test-plan.ps1 -Config Debug -BuildDir build
```

## 使用示例

```cpp
#include "lp_solver/lp_solver.hpp"

lp_solver::model::ProblemData problem{
    /* PackedMatrix A */,
    /* c */, /* b */,
    /* lower_bounds */, /* upper_bounds */
};

lp_solver::model::SolverState state;
state.basic_indices = /* m 个合法基列下标 */;

auto factor = lp_solver::linalg::makeFactor(lp_solver::linalg::FactorBackend::Eigen);
lp_solver::simplex::DualSimplex solver(std::move(factor), nullptr, nullptr);

lp_solver::simplex::SolverConfig cfg;
const auto status = solver.solve(problem, state, cfg);

if (status == lp_solver::simplex::DualSimplex::Status::Optimal) {
    // state.primal_solution, state.dual_solution, state.objective
}
```

`SolverConfig` 可开关：预处理、Big-M Phase I、DSE、Harris、重构频率、`big_m_scale` 等。

## 实现说明

- 手册中的 **超稀疏三角解** 与 **第三方稀疏 LU** 已实现：基求解采用稀疏 LU 因子 + 稀疏三角前/回代。
- `EigenFactor` 走 Eigen SparseLU 提取 + GP 风格解法；`UmfpackFactor` 在可用时走 UMFPACK 提取，不可用时回退同一稀疏路径。
- Big-M 扩维后工作问题多一行一列；返回的 **`primal_solution` 会去掉人工列**（若存在）。

## 参考

- 实现手册：`mat3007h_Project_Manual.tex`
- Eigen：https://eigen.tuxfamily.org/
- SuiteSparse / UMFPACK：https://people.engr.tamu.edu/davis/suitesparse.html

---

**最后更新**：2026 年 4 月
