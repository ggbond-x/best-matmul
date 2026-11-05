# H100 PCIe FP16 GEMM 对比测试框架

本目录是 **plan1：H100 PCIe FP16 GEMM 公平基线对比** 的统一测试工程。它在同一块 GPU、同一数学语义、同一计时口径下比较仓库自研 Hopper TMA+WGMMA Kernel 与 cuBLAS、cuBLASLt、CUTLASS SM90，输出可追溯的原始 CSV、NCU 报告和中文分析。

> 当前仓库只在 **RTX 3050 Laptop（SM86）** 上完成过构建、正确性/计时链路验证；自研 Kernel 与 CUTLASS SM90 因 `min_cc=9.0` 在该设备上标记 `UNSUPPORTED`。**H100 实测数据尚未产生，任何文件都不会伪造 H100 结果。**

## 1. 目录结构

```text
cuBLAS_test/
├── README.md                  # 本文件：目标、结构、构建与数据约定
├── CMakeLists.txt             # 统一构建（SM90a gencode、CUTLASS、cuBLAS/cuBLASLt）
├── .gitignore                 # build/、缓存、NCU 大报告等
├── include/gemm/              # 公共接口：Problem/Options、计时、遥测、CSV
├── src/
│   ├── main.cpp               # CLI、Case 编排、轮转顺序、manifest/正确性日志
│   ├── host_data.cpp          # 输入生成、TN/NT 物理打包、FP64 参考
│   ├── timing.cpp             # CUDA Event / wall / CUDA Graph 计时与 L2 冲刷
│   ├── csv.cpp                # raw_runs.csv schema
│   ├── telemetry.cpp          # NVML（dlopen）时钟/功耗/温度采样
│   ├── backends/              # cuBLAS、cuBLASLt、CUTLASS 适配与注册表
│   └── custom/                # V1 / V2(PIPE=2,3,4) 原始 Kernel 适配（不改数学逻辑）
├── scripts/
│   ├── README.md              # 各计划索引与脚本总览
│   ├── common/                # 跨计划复用：配置解析、环境快照、GPU 监控、运行编排
│   └── plan1/                 # 本次计划：构建/quick/full/NCU/汇总
├── build/plan1/               # 编译产物（不提交）
└── output/plan1/<timestamped_run>/   # 每次运行的唯一结果目录
```

## 2. 对比实现

| 名称 | 来源 | 累加精度 | 最低 CC |
|---|---|---|---|
| `CUSTOM_V1_F16ACC` | `kernels/wgmma_tma_fp16_v1.cuh` | FP16 | 9.0 |
| `CUSTOM_V2_PIPE2_F16ACC` | `kernels/wgmma_tma_fp16_pipeline_v2.cuh`，`PIPE=2` | FP16 | 9.0 |
| `CUSTOM_V2_PIPE3_F16ACC` | 同上，`PIPE=3` | FP16 | 9.0 |
| `CUSTOM_V2_PIPE4_F16ACC` | 同上，`PIPE=4` | FP16 | 9.0 |
| `CUBLAS_EX_F16ACC` / `CUBLAS_EX_F32ACC` | `cublasGemmEx`，`CUBLAS_COMPUTE_16F/32F` | FP16 / FP32 | 0.0 |
| `CUBLAS_LT_F16ACC` / `CUBLAS_LT_F32ACC` | `cublasLtMatmul` + heuristic/tuning | FP16 / FP32 | 0.0 |
| `CUTLASS_SM90_128x128x64_C1x1x1_F16ACC/F32ACC` | CUTLASS 4.4.1 CollectiveBuilder | FP16 / FP32 | 9.0 |
| `CUTLASS_SM90_128x256x64_C2x1x1_F16ACC/F32ACC` | 同上，2-SM cluster | FP16 / FP32 | 9.0 |

**排名只在相同累加精度组内进行**：FP16 累加（自研、`*_F16ACC`）与 FP32 累加（`*_F32ACC`）不直接等同。性能比 `custom_tflops / best_matched_library_tflops` 仅在相同 Shape、布局、累加精度、计时模式、缓存模式下计算。

## 3. 布局定义（与仓库 `host/common.cuh` 一致）

逻辑问题固定为：

```text
A: M×K, B: N×K, C: M×N（列主序）
C(m,n) = Σ_p A(m,p) * B(n,p)
alpha = 1, beta = 0
```

| 布局 | A 存储 | B 存储 | cuBLAS op(A),op(B) | CUTLASS |
|---|---|---|---|---|
| **NT** | `a[row + p*M]`（M 连续，`lda=M`） | `b[col + p*N]`（N 连续，`ldb=N`） | `N, T` | A/B ColumnMajor |
| **TN** | `a[row*K + p]`（K 连续，`lda=K`） | `b[col*K + p]`（K 连续，`ldb=K`） | `T, N` | A/B RowMajor |

C 始终是列主序 `c[row + col*M]`。测试程序先按规范逻辑生成一份 A/B，再为每个布局做物理打包；打包与初始化不计时，但输入数据完全一致。

## 4. 依赖

- NVIDIA GPU；自研 Kernel/CUTLASS 需要 **SM90a（H100/H800 等 Hopper）**。
- CUDA Toolkit **≥ 12.0**（本机用 12.6 验证）；`build.sh` 会自动选择 `/usr/local/cuda*` 中版本最高的 nvcc，也可用 `CUDACXX`/`CUDA_HOME` 指定。
- CMake ≥ 3.16。
- CUTLASS 4.x（本机 `../cutlass`，版本 4.4.1）；可用 `CUTLASS_ROOT` 覆盖。
- Python 3.8+；`PyYAML` 可选（缺失时使用内置受限解析器，已验证两者 config hash 一致）。
- Nsight Compute（仅 `run_ncu.sh` 需要）。

## 5. 构建与运行

```bash
cd matmul/best-matmul/cuBLAS_test

# 构建到 build/plan1/，并做 --list-impls/--list-cases 冒烟检查
bash scripts/plan1/build.sh

# 快速：S01+S02，TN/NT，先正确性后性能，新建运行目录
bash scripts/plan1/run_quick.sh

# 完整：S01–S12 + S01/S02 cold-cache，新建另一个运行目录
bash scripts/plan1/run_full.sh

# NCU 诊断（追加到已有运行，或省略 --run-dir 新建）
RUN_DIR="output/plan1/20261105_113000_a1b2c3"
bash scripts/plan1/run_ncu.sh --run-dir "$RUN_DIR"

# 只基于该运行目录的真实原始数据生成 summary/ncu_summary/analysis
python3 scripts/plan1/summarize.py --run-dir "$RUN_DIR"
```

所有入口都可以从任意工作目录调用；脚本会自行解析项目根目录。详细参数、输出字段和判读方法见 [scripts/plan1/README.md](scripts/plan1/README.md)。

## 6. 数据约定

- 每次 quick/full/NCU 都创建新的 `output/plan1/<timestamped_run>/`，**不覆盖历史**。
- `csvs/raw_runs.csv` 是唯一原始测量来源；`summary.csv` 由它生成；`ncu_summary.csv` 只用于诊断，不进入 benchmark 排名。
- `run_manifest.json` 区分 `OK / FAIL / UNSUPPORTED / NOT_RUN`；未测数据不插值、不推测。
- `environment.json` 记录 GPU/驱动/CUDA/cuBLAS/CUTLASS/Git/锁频状态；`resolved_plan.yaml` 是本轮实际生效配置。
- 默认不修改 GPU 时钟、不终止其他进程；未锁频时 `clock_locked=false`，报告只描述实际观测值。

## 7. 已知限制

- 没有 H100 时只能验证构建与 cuBLAS/cuBLASLt 路径；自研/CUTLASS 在非 Hopper 设备上会是 `UNSUPPORTED`。
- H100 PCIe 的 756 TFLOPS 是官方稠密 FP16 标称峰值，只在检测到 H100 时作为分母，且不代表当前时钟下的可持续上限。
- NCU 的 replay/cache-control 会改变运行条件，其 Duration 与普通 benchmark 不可混排。
