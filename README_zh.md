# 线性规划求解器

**语言** | [English](README.md)

用于 **标准型线性规划**（在 Ax=b,\ x\ge 0 下最小化 c^\top x）的 **C++20** 库：核心是 **修正对偶单纯形**，矩阵以 **CSC** 稀疏格式存放，可选 **预处理/后处理**，基矩阵通过可插拔的 `**IBasisFactor`** 维护。

## 特性

- **修正对偶单纯形**：`CHUZR → BTRAN → CHUZC → FTRAN → 转轴`，含原始/对偶更新与周期性基重构
- **Phase I Big-M（对偶起步）**：起始检验数对偶不可行时，对当前非基变量增加界约束行与人工变量入基，做一次强制转轴后进入 Phase II（见 `mat3007h_Project_Manual.tex`）
- **鲁棒性增强**：原始可行但对偶不可行时做原始单纯形恢复；`chuzc` 失败时重因子化并尝试其他离基行；预处理后初始基无法分解时自动关闭 presolve 重试
- **基维护**：对 B 做稀疏 LU，迭代间用 **稀疏 ETA 文件**（乘积形式逆）修正；`etaFileLength()` 达到 `refactor_frequency` 时重构 LU
- **超稀疏三角解**：Gilbert–Peierls 可达性 + 在提取的 CSC L/U 上前/回代；右端稀疏时工作量约为 O(\mathrm{nnz})
- **定价**：可选 **对偶最陡边（DSE）** 及仅在 FTRAN/BTRAN 非零元上更新的 **Goldfarb–Reid** 权重递推；进基采用 **Harris 两阶段**比率检验
- **预处理/后处理**：按列稀疏扫描化简（不构造稠密 A）；LIFO 栈记录；`**postsolvePrimal`** 与 `**postsolveDual`**（在化简栈上恢复对偶）
- **稀疏基础结构**：`PackedMatrix`（CSC）、`IndexedVector`（跟踪非零元）；`multiply` / `transposeMultiply`
- **因子后端**：`EigenFactor`（Eigen SparseLU 提取 + GP 解法）；`UmfpackFactor`（链接 UMFPACK 时使用，否则与 Eigen 相同稀疏路径）；`**makeDefaultFactor()`** 在可用时优先 UMFPACK

## 项目结构

```
LinearProgramingSolver/
├── include/lp_solver/
│   ├── io/               # MPS 读取、Netlib 标准化（RawLpModel → ProblemData）
│   ├── linalg/           # IBasisFactor、EigenFactor、UmfpackFactor、EtaFile、稀疏 LU 引擎
│   ├── model/            # ProblemData, SolverState
│   ├── presolve/         # Presolver（原始/对偶后处理）
│   ├── simplex/          # DualSimplex, SolverConfig, 钩子
│   │   └── detail/       # 如超稀疏 DSE 权重更新
│   └── util/             # PackedMatrix, IndexedVector
├── src/
│   └── io/               # mps_reader.cpp, netlib_standardizer.cpp
├── tests/
│   ├── smoke_test.cpp
│   ├── stress_test.cpp
│   ├── advanced_features_test.cpp   # 预处理、DSE、ETA、默认因子、后端一致性
│   ├── hypersparse_triangular_test.cpp
│   ├── netlib_parser_test.cpp       # MPS 解析 + 标准化 smoke 检查
│   ├── netlib_runner.cpp            # Netlib CLI 运行器源码
│   └── netlib_baseline.csv          # 参考目标值 / 容差（起步集）
├── netlib/                          # 本地 Netlib `.mps` 数据（已 gitignore）
├── scripts/
│   ├── run-test-plan.ps1            # 配置、构建并运行 CTest
│   └── run-netlib.ps1               # 批量运行并输出 CSV
├── docs/
│   └── netlib_format_notes.md       # 支持的 MPS 子集与转换说明
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


| CTest 名称                      | 内容                                             |
| ----------------------------- | ---------------------------------------------- |
| `smoke_test`                  | 大规模 `IndexedVector` / `PackedMatrix`，默认因子下的单纯形 |
| `stress_test`                 | 中等规模可行基求解                                      |
| `advanced_features_test`      | 预处理/后处理（原始+对偶）、Big-M、DSE、ETA、默认因子、后端一致性        |
| `hypersparse_triangular_test` | Gilbert–Peierls 三角解与稠密参考对比                     |
| `netlib_parser_test`          | MPS 子集解析 + Netlib 标准化 smoke 检查                 |


```bash
ctest --test-dir build -C Debug --output-on-failure
```

Windows：

```powershell
.\scripts\run-test-plan.ps1 -Config Debug -BuildDir build
```

## Netlib 测试流程

仓库提供 Netlib 测试路径：读取 `.mps`、标准化为 `Ax=b, x>=0`、调用 `DualSimplex` 求解，并与参考目标值对比。

### 相关文件说明


| 路径                                             | 作用                                                            |
| ---------------------------------------------- | ------------------------------------------------------------- |
| `include/lp_solver/io/mps_reader.hpp`          | MPS 读取 API（`readMpsFile`）                                     |
| `include/lp_solver/io/netlib_standardizer.hpp` | 将解析结果转为 `ProblemData` 与 slack 初始基                             |
| `include/lp_solver/io/mps_types.hpp`           | 中间结构 `RawLpModel`                                             |
| `src/io/mps_reader.cpp`                        | 固定列 MPS 解析；列对齐失败时回退到空白分词                                      |
| `src/io/netlib_standardizer.cpp`               | 不等式、边界、range → 标准型                                            |
| `tests/netlib_runner.cpp`                      | CLI 可执行文件源码（`lp_solver_netlib_runner`）                        |
| `tests/netlib_parser_test.cpp`                 | CTest：解析 + 标准化 smoke 检查                                       |
| `tests/netlib_baseline.csv`                    | 本地参考目标值与容差                                                    |
| `scripts/run-test-plan.ps1`                    | 配置、构建并运行 CTest 测试套件                                           |
| `scripts/run-netlib.ps1`                       | 批量遍历数据目录并写 CSV 汇总                                             |
| `docs/netlib_format_notes.md`                  | 支持的 MPS 区段与标准化规则                                              |
| `netlib/`                                      | 存放解压后的 `.mps`（目录已 gitignore）                                  |
| `netlib/README.netlib`                         | Netlib 官方问题表；批量脚本会从中读取最优目标值                                   |
| `netlib-results-*.csv`                         | 批量运行输出（已 gitignore）；每题一行，含 status、objective、objective_match 等 |


### 准备数据

1. 获取 Netlib LP 模型（如 [Netlib LP 数据集](http://www.netlib.org/lp/data/)）。
2. 将 `.mps.gz` 解压为 `.mps`（读取器不支持直接读 gzip）。
3. 放入 `netlib/` 或通过 `-DataDir` 指定任意目录。

**目标值基准**：以 `**netlib/README.netlib`** 中的问题汇总表为准。`run-netlib.ps1` 会将其与 `tests/netlib_baseline.csv` 合并（同名时 CSV 优先）。

### 构建 runner

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug --target lp_solver_netlib_runner
```

单配置生成器（Ninja、Make）下，可执行文件通常在 `build/lp_solver_netlib_runner.exe`（Windows）或 `build/lp_solver_netlib`。

### 单题运行

```powershell
.\build\lp_solver_netlib_runner.exe .\netlib\afiro.mps --ref -4.6475314286E+02 --tol 1e-6
```

stdout 输出键值对（`status`、`objective`、`iterations`、`objective_match` 等）。退出码：`0` = 最优且与参考匹配（需 `--ref`）；`10` = 目标值不匹配；`11` = 非最优状态。

#### `lp_solver_netlib_runner` 参数


| 参数                             | 默认             | 含义                              |
| ------------------------------ | -------------- | ------------------------------- |
| `--ref <obj>`                  | 无              | 参考最优目标值；启用 `objective_match` 检查 |
| `--tol <t>`                    | `1e-6`         | 相对容差（按                          |
| `--max-iters <N>`              | `50000`        | 最大迭代次数                          |
| `--no-presolve` / `--presolve` | 开启 presolve    | 开关预处理                           |
| `--no-big-m` / `--big-m`       | **优先不用 Big-M** | 单次运行是否启用对偶 Big-M Phase I        |
| `--no-dse`                     | 开启 DSE         | 关闭对偶最陡边离基规则                     |
| `--no-harris`                  | 开启 Harris      | 关闭 Harris 两阶段进基比率检验             |
| `--big-m-scale <X>`            | `1000`         | Big-M 相对 max                    |
| `--refactor-freq <N>`          | `100`          | ETA 长度 ≥ N 时重构 LU               |
| `--verbose`                    | 关              | 打印转轴 / 终止原因                     |


**策略回退（仅 runner）**：按顺序尝试最多四种 `{Big-M 开/关} × {DSE 开/关}` 组合，首个返回 `Optimal` 的结果即采用。这是通用鲁棒性逻辑。

### 批量运行

```powershell
.\scripts\run-netlib.ps1 -DataDir .\netlib -BuildDir build -Config Debug `
  -BaselineCsv tests\netlib_baseline.csv -OutputCsv netlib-results.csv
```

脚本会：

- 在 `BuildDir` 下查找 `lp_solver_netlib_runner`（含或不含 `Config` 子目录）。
- 对 `-DataDir` 中每个 `*.mps` 运行一次。
- 若存在基准行（来自 CSV 和/或 `README.netlib`），自动传入 `--ref` / `--tol`。
- 写入 `-OutputCsv`（默认 `netlib-results-<时间戳>.csv`）。

当前 solver 在完整 `netlib/` 集上相对 `README.netlib` **40/40** 通过目标值检查（容差多为 `1e-6`；`GANGES` 在 baseline CSV 中为 `1e-5`）。

### 标准化说明

标准化器将 MPS 的行/边界/range 转为 `min c'x, Ax=b, x>=0`，记录目标偏移，并为 slack/单位列构造显式初始基。详见 `docs/netlib_format_notes.md`。

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


| 字段                       | 默认值     | 含义                        |
| ------------------------ | ------- | ------------------------- |
| `use_presolve`           | `true`  | 求解前稀疏预处理；成功时对原始/对偶做后处理    |
| `enable_big_m_phase_one` | `true`  | 起始对偶不可行时启用Big-M           |
| `use_dual_steepest_edge` | `true`  | DSE 离基行规则（`chuzr`）        |
| `use_harris_two_pass`    | `true`  | Harris 比率检验（`chuzc`）      |
| `refactor_frequency`     | `100`   | ETA 长度达到该值时重构 LU（0 表示不重构） |
| `big_m_scale`            | `1000`  | Big-M 相对 max              |
| `max_iterations`         | `10000` | 最大迭代次数                    |


说明：`lp_solver_netlib_runner` 使用 `max_iterations = 50000`，并在放弃前自动尝试多种 `(enable_big_m_phase_one, use_dual_steepest_edge)` 组合；库内调用方直接设置 `SolverConfig` 即可。

### 因子工厂


| API                              | 说明                                                 |
| -------------------------------- | -------------------------------------------------- |
| `makeDefaultFactor()`            | 等价于 `makeFactor(FactorBackend::Default)`           |
| `defaultFactorBackend()`         | 有 `LP_SOLVER_HAVE_UMFPACK` 时为 `Umfpack`，否则 `Eigen` |
| `makeFactor(FactorBackend::Eigen | Umfpack)`                                          |


## API 概览


| 组件             | 作用                                                      |
| -------------- | ------------------------------------------------------- |
| `DualSimplex`  | 主求解循环；最优时执行后处理                                          |
| `IBasisFactor` | `factorize`、`ftran`、`btran`、`updateEta`、`etaFileLength` |
| `EtaFile`      | 稀疏乘积形式 ETA 更新（应用代价 O(\sum \mathrm{nnz}(d_i))）           |
| `Presolver`    | 稀疏预处理；`postsolvePrimal` / `postsolveDual`               |
| `ProblemData`  | A、b、c、界                                                 |
| `SolverState`  | 基/非基、x_B、\pi、检验数、DSE 权重、解向量                             |


## 实现说明

- **超稀疏路径**：三角解（Gilbert–Peierls）、ETA 应用、DSE 的 Goldfarb–Reid 更新在可能处仅遍历 `IndexedVector::nonZeroIndices()`，避免对全维度的 O(n) 扫描。
- `**EigenFactor`**：Eigen `SparseLU` 因子提取 + GP 风格 `ftran`/`btran`；提取校验失败时引擎内部可回退稠密路径。
- `**UmfpackFactor`**：在 `LP_SOLVER_HAVE_UMFPACK` 下使用 UMFPACK 分解与提取；否则与 `EigenFactor` 相同的稀疏引擎路径。
- **对偶 Big-M**：起始检验数对偶不可行时，工作问题临时增加一行界约束和一列人工变量；强制转轴一次后继续 Phase II。若存在人工列，返回的 `**primal_solution` 会去掉该列**。
- **最优性检查**：找不到离基行（原始可行）时，先检查对偶可行性（`minReducedCost`）；若对偶仍不可行，则在当前基上执行原始单纯形转轴，直到对偶可行或无法继续改进。
- `**chuzc` 恢复**：进基列选择失败时先重因子化并重试；仍失败则扫描其他原始不可行行；最后才报 infeasible。
- **Presolve 回退**：预处理后初始基无法分解时，自动关闭 presolve 重算。
- **MPS 读取**：优先固定列格式；若固定列解析出的行名无效，则回退到空白分词（部分 Netlib 文件如 FORPLAN 需要）。
- **DSE 权重**：重构时重置为 `1.0`；迭代间由 `simplex/detail/dse_weight_update.hpp` 递推更新。

## 参考

- 实现手册：`mat3007h_Project_Manual.tex`
- Eigen：[https://eigen.tuxfamily.org/](https://eigen.tuxfamily.org/)
- SuiteSparse / UMFPACK：[https://people.engr.tamu.edu/davis/suitesparse.html](https://people.engr.tamu.edu/davis/suitesparse.html)

---

**最后更新**：2026 年 5 月