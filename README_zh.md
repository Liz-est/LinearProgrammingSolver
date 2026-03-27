# 线性规划求解器

**🌐 语言** | [English](README.md)

一个高性能的 C++ 库，使用对偶单纯形算法、稀疏矩阵表示和可插拔的线性代数后端，用于求解线性规划问题。

## 特性

- **对偶单纯形算法**：高效的对偶单纯形法实现
- **稀疏矩阵表示**：采用 CSC（压缩稀疏列）格式，内存高效
- **多后端支持**：支持 Eigen 和 UMFPACK 因式分解方法
- **索引向量**：超稀疏向量操作，用于高效的定价和表计算更新
- **C++20 实现**：现代 C++ 代码，具有严格的类型安全和 Move 语义优化
- **跨平台支持**：使用 CMake 构建，支持 Windows、Linux 和 macOS

## 项目结构

```
LinearProgramingSolver/
├── include/lp_solver/          # 公共头文件
│   ├── linalg/                 # 线性代数抽象层
│   │   ├── i_basis_factor.hpp  # 基因式分解接口
│   │   ├── eigen_factor.hpp    # Eigen 后端
│   │   └── umfpack_factor.hpp  # UMFPACK 后端
│   ├── model/                  # 问题数据结构
│   ├── simplex/                # 对偶单纯形算法
│   └── util/                   # 稀疏数据结构
│       ├── indexed_vector.hpp  # 超稀疏向量
│       └── packed_matrix.hpp   # CSC 矩阵表示
├── src/                        # 实现文件
├── tests/                      # 测试套件
│   ├── smoke_test.cpp          # 基础功能测试
│   └── stress_test.cpp         # 性能和鲁棒性测试
└── CMakeLists.txt              # 构建配置
```

## 环境要求

- **C++20** 编译器（MSVC、GCC 或 Clang）
- **CMake 3.20+**
- **Windows 10/11**、Linux 或 macOS
- 可选：Eigen3 或 UMFPACK 库用于高级因式分解

## 安装与构建

### 克隆仓库

```bash
git clone https://github.com/Liz-est/LinearProgramingSolver.git
cd LinearProgramingSolver
```

### 使用 CMake 构建

#### 调试构建
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
```

#### 发布构建
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## 运行测试

```bash
ctest --test-dir build -C Debug --output-on-failure
```

或使用自动化测试运行器（Windows）：
```powershell
.\run-test-plan.ps1 -Config Debug -BuildDir build
```

## 使用示例

```cpp
#include "lp_solver/lp_solver.hpp"

// 创建问题数据
ProblemData problem = /* ... */;

// 初始化求解器
DualSimplex solver(problem);

// 求解
SolverState result = solver.solve();

// 获取解
if (result.status == OptimalityStatus::OPTIMAL) {
    auto solution = result.primal_solution;
    auto objective = result.objective_value;
}
```

## API 概览

### 核心组件

- **`DualSimplex`**：实现对偶单纯形算法的主求解器类
- **`IBasisFactor`**：基因式分解的抽象接口
- **`PackedMatrix`**：不可变的 CSC 格式稀疏矩阵
- **`IndexedVector`**：带活跃指标跟踪的稀疏向量
- **`ProblemData`**：标准型 LP 问题定义
- **`SolverState`**：解和求解器统计

### 架构

求解器采用模块化架构，支持可插拔的后端：

```
DualSimplex（算法层）
    ↓
IBasisFactor（抽象接口）
    ├─ EigenFactor（Eigen3 后端）
    └─ UmfpackFactor（UMFPACK 后端）
```

## 测试覆盖

项目包含全面的测试套件：

- **冒烟测试**（`smoke_test.cpp`）：验证求解器基本功能和正确性
- **压力测试**（`stress_test.cpp`）：大规模问题的性能测试

运行测试：
```bash
ctest --test-dir build --output-on-failure -V
```

## 实现状态

### 第一阶段（开发中）

✅ 已完成：
- 稀疏基元（IndexedVector、PackedMatrix）
- 因式分解抽象层
- 对偶单纯形算法框架

🔄 进行中：
- LA 后端集成（Eigen、UMFPACK）
- 矩阵-向量操作
- 定价和更新操作

## 开发

### 构建配置

- **C++ 标准**：C++20
- **编译器标志**：启用严格错误检查
- **构建系统**：支持本地工具链的 CMake

### 代码组织

- 头文件在 `include/lp_solver/`
- 实现文件在 `src/`
- 测试在 `tests/`
- 构建产物在 `build/`

## 贡献指南

欢迎贡献！请：

1. Fork 本仓库
2. 创建功能分支
3. 为新功能添加测试
4. 确保所有测试通过
5. 提交 Pull Request

## 性能考虑

- **稀疏操作**：利用 CSC 格式和索引向量的高效性
- **因式分解**：可插拔后端允许针对特定用例的算法优化
- **内存**：稀疏向量的零擦除防止内存泄漏


## 参考资源

- 对偶单纯形法：[学术参考]
- 稀疏矩阵格式：[技术文档]
- Eigen3：https://eigen.tuxfamily.org/
- UMFPACK：https://people.engr.tamu.edu/davis/suitesparse.html

## 支持

如有问题、疑问或建议，请：
- 在 GitHub 上提交 Issue
- 提交改进的 Pull Request
- 联系：[你的联系方式]

---

**状态**：第一阶段开发中 | **最后更新**：2026年3月
