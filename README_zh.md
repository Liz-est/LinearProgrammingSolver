# 线性规划求解器

**语言** | [English](README.md)

用于 **标准型线性规划**（在 \(Ax=b,\ x\ge 0\) 下最小化 \(c^\top x\)）的 **C++20** 库：核心是 **修正对偶单纯形**，矩阵以 **CSC** 稀疏格式存放，可选 **预处理/后处理**，基矩阵通过可插拔的 **`IBasisFactor`** 维护。

## 特性

- **修正对偶单纯形**：`CHUZR → BTRAN → CHUZC → FTRAN → 转轴`，含原始/对偶更新与周期性基重构
- **Phase I Big-M（教科书流程）**：对当前非基变量增加界约束行与人工变量入基，做一次强制转轴后进入 Phase II（见 `mat3007h_Project_Manual.tex`）
- **基维护**：对 \(B\) 做稀疏 LU，迭代间用 **稀疏 ETA 文件**（乘积形式逆）修正；`etaFileLength()` 达到 `refactor_frequency` 时重构 LU
- **超稀疏三角解**：Gilbert–Peierls 可达性 + 在提取的 CSC \(L/U\) 上前/回代；右端稀疏时工作量约为 \(O(\mathrm{nnz})\)
- **定价**：可选 **对偶最陡边（DSE）** 及仅在 FTRAN/BTRAN 非零元上更新的 **Goldfarb–Reid** 权重递推；进基采用 **Harris 两阶段**比率检验
- **预处理/后处理**：按列稀疏扫描化简（不构造稠密 \(A\)）；LIFO 栈记录；**`postsolvePrimal`** 与 **`postsolveDual`**（在化简栈上恢复对偶）
- **稀疏基础结构**：`PackedMatrix`（CSC）、`IndexedVector`（跟踪非零元）；`multiply` / `transposeMultiply`
- **因子后端**：`EigenFactor`（Eigen SparseLU 提取 + GP 解法）；`UmfpackFactor`（链接 UMFPACK 时使用，否则与 Eigen 相同稀疏路径）；**`makeDefaultFactor()`** 在可用时优先 UMFPACK

## 项目结构

```
LinearProgramingSolver/
├── include/lp_solver/
│   ├── linalg/           # IBasisFactor、EigenFactor、UmfpackFactor、EtaFile、稀疏 LU 引擎
│   ├── model/            # ProblemData, SolverState
│   ├── presolve/         # Presolver（原始/对偶后处理）
│   ├── simplex/          # DualSimplex, SolverConfig, 钩子
│   │   └── detail/       # 如超稀疏 DSE 权重更新
│   └── util/             # PackedMatrix, IndexedVector
├── src/
├── tests/
│   ├── smoke_test.cpp
│   ├── stress_test.cpp
│   ├── advanced_features_test.cpp   # 预处理、DSE、ETA、默认因子、后端一致性
│   └── hypersparse_triangular_test.cpp
├── mat3007h_Project_Manual.tex      # 课程实现手册（参考）
└── CMakeLists.txt
```

## 环境要求

- **C++20**（MSVC、GCC 或 Clang）
- **CMake 3.20+**
- **Eigen 3.4+**（未安装且 `LP_SOLVER_FETCH_EIGEN=ON` 时可自动拉取）
- **SuiteSparse UMFPACK**（可选，见构建选项）

## 第三方依赖说明

- **Eigen** 为必需依赖（稀疏 LU 提取及稠密回退）。
- **SuiteSparse / UMFPACK** 可选（默认 `-DLP_SOLVER_WITH_UMFPACK=ON`）。未找到时，`UmfpackFactor` 与 `FactorBackend::Default` 均走与 `EigenFactor` 相同的 Eigen 稀疏路径。
- 若将本地源码目录（如 `eigen-5.0.0/`）放在仓库内做实验，请加入 `.gitignore`，避免误提交。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
```

Release：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

关闭 UMFPACK 检测：

```bash
cmake -S . -B build -DLP_SOLVER_WITH_UMFPACK=OFF
```

成功链接 UMFPACK 时，构建会为 `lp_solver` 目标定义 `LP_SOLVER_HAVE_UMFPACK=1`。

## 测试

| CTest 名称 | 内容 |
|------------|------|
| `smoke_test` | 大规模 `IndexedVector` / `PackedMatrix`，默认因子下的单纯形 |
| `stress_test` | 中等规模可行基求解 |
| `advanced_features_test` | 预处理/后处理（原始+对偶）、Big-M、DSE、ETA、默认因子、后端一致性 |
| `hypersparse_triangular_test` | Gilbert–Peierls 三角解与稠密参考对比 |

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

// 推荐：有 UMFPACK 时用 UMFPACK，否则 Eigen SparseLU
auto factor = lp_solver::linalg::makeDefaultFactor();
// 或显式指定：
// auto factor = lp_solver::linalg::makeFactor(lp_solver::linalg::FactorBackend::Eigen);

lp_solver::simplex::DualSimplex solver(std::move(factor), nullptr, nullptr);

lp_solver::simplex::SolverConfig cfg;
cfg.use_presolve = true;
cfg.enable_big_m_phase_one = true;
cfg.use_dual_steepest_edge = true;
cfg.use_harris_two_pass = true;
cfg.refactor_frequency = 100;

const auto status = solver.solve(problem, state, cfg);

if (status == lp_solver::simplex::DualSimplex::Status::Optimal) {
    // state.primal_solution, state.dual_solution, state.objective
}
```

### `SolverConfig`（主要字段）

| 字段 | 默认值 | 含义 |
|------|--------|------|
| `use_presolve` | `true` | 求解前稀疏预处理；成功时对原始/对偶做后处理 |
| `enable_big_m_phase_one` | `true` | 起始对偶不可行时启用教科书 Big-M |
| `use_dual_steepest_edge` | `true` | DSE 离基行规则（`chuzr`） |
| `use_harris_two_pass` | `true` | Harris 比率检验（`chuzc`） |
| `refactor_frequency` | `100` | ETA 长度达到该值时重构 LU（0 表示不重构） |
| `big_m_scale` | `1000` | Big-M 相对 max\|c\| 的倍数 |
| `max_iterations` | `10000` | 最大迭代次数 |

### 因子工厂

| API | 说明 |
|-----|------|
| `makeDefaultFactor()` | 等价于 `makeFactor(FactorBackend::Default)` |
| `defaultFactorBackend()` | 有 `LP_SOLVER_HAVE_UMFPACK` 时为 `Umfpack`，否则 `Eigen` |
| `makeFactor(FactorBackend::Eigen \| Umfpack)` | 强制指定后端 |

## API 概览

| 组件 | 作用 |
|------|------|
| `DualSimplex` | 主求解循环；最优时执行后处理 |
| `IBasisFactor` | `factorize`、`ftran`、`btran`、`updateEta`、`etaFileLength` |
| `EtaFile` | 稀疏乘积形式 ETA 更新（应用代价 \(O(\sum \mathrm{nnz}(d_i))\)） |
| `Presolver` | 稀疏预处理；`postsolvePrimal` / `postsolveDual` |
| `ProblemData` | \(A\)、\(b\)、\(c\)、界 |
| `SolverState` | 基/非基、\(x_B\)、\(\pi\)、检验数、DSE 权重、解向量 |

## 实现说明

- **超稀疏路径**：三角解（Gilbert–Peierls）、ETA 应用、DSE 的 Goldfarb–Reid 更新在可能处仅遍历 `IndexedVector::nonZeroIndices()`，避免对全维度的 \(O(n)\) 扫描。
- **`EigenFactor`**：Eigen `SparseLU` 因子提取 + GP 风格 `ftran`/`btran`；提取校验失败时引擎内部可回退稠密路径。
- **`UmfpackFactor`**：在 `LP_SOLVER_HAVE_UMFPACK` 下使用 UMFPACK 分解与提取；否则与 `EigenFactor` 相同的稀疏引擎路径。
- **Big-M**：工作问题在求解结束前多一行一列人工结构；返回的 **`primal_solution` 会去掉人工列**（若存在）。
- **DSE 权重**：重构时重置为 `1.0`；迭代间由 `simplex/detail/dse_weight_update.hpp` 递推更新。

## 参考

- 实现手册：`mat3007h_Project_Manual.tex`
- Eigen：https://eigen.tuxfamily.org/
- SuiteSparse / UMFPACK：https://people.engr.tamu.edu/davis/suitesparse.html

---

**最后更新**：2026 年 5 月
