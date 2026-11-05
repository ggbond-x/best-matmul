# plan1：H100 PCIe FP16 GEMM 公平基线对比

本计划回答一个问题：在同一张 H100 PCIe、相同数学语义和测量条件下，仓库自研 Hopper TMA+WGMMA GEMM（V1/V2，`PIPE=2/3/4`）与 cuBLAS、cuBLASLt、CUTLASS SM90 的差距有多大，差距集中在哪些 Shape/布局/K 长度，以及下一步应优先优化什么。

## 1. 测试范围

### Shape（全部同时测 TN、NT）

| ID | M | N | K | 优先级 | 目的 |
|---|---:|---:|---:|---|---|
| S01 | 4096 | 4096 | 4096 | P0 | 核心锚点，对应历史 V1/V2 数据 |
| S02 | 4096 | 4096 | 256 | P0 | 短 K：启动、同步、Epilogue 占比 |
| S03 | 4096 | 4096 | 1024 | P0 | 中 K 吞吐增长 |
| S04 | 4096 | 4096 | 2048 | P0 | 长 K 收敛 |
| S05 | 2048 | 2048 | 2048 | P0 | 中等方阵 |
| S06 | 8192 | 8192 | 8192 | P0 | 大规模吞吐/功耗/频率持续性 |
| S07 | 1024 | 1024 | 1024 | P1 | 小/中方阵 |
| S08 | 4096 | 4096 | 512 | P1 | K 扫描补点 |
| S09 | 256 | 4096 | 4096 | P1 | 小 M、大 N |
| S10 | 4096 | 256 | 4096 | P1 | 大 M、小 N |
| S11 | 1024 | 8192 | 4096 | P1 | 非方阵，偏 N |
| S12 | 8192 | 1024 | 4096 | P1 | 非方阵，偏 M |

不支持某 Shape 的实现必须输出 `UNSUPPORTED` 及原因，不能静默跳过。

### 实现与累加精度

- 自研：`CUSTOM_V1_F16ACC`、`CUSTOM_V2_PIPE2/3/4_F16ACC`（源码不改，仅适配；FP16 累加）。
- cuBLAS：`CUBLAS_EX_F16ACC`、`CUBLAS_EX_F32ACC`。
- cuBLASLt：`CUBLAS_LT_F16ACC`、`CUBLAS_LT_F32ACC`（heuristic 枚举有限候选 + 逐候选校验，记录算法 ID 与 workspace）。
- CUTLASS SM90：`128×128×64 C1x1x1`、`128×256×64 C2x1x1`，各有 FP16/FP32 累加，共 4 个。

**FP16 累加与 FP32 累加分组排名**，不直接比较绝对误差。自定义 Kernel/CUTLASS 需要 CC ≥ 9.0。

### 自研算子测试配置

下表是当前 plan1 构建的配置。CTA 的 M/N/K 是一个线程块负责的 tile 尺寸；WGMMA 的 M/N/K 是单条指令的尺寸，两者不是同一个概念。

| 测试实现 | CTA_M | CTA_N | CTA_K | WGMMA 指令 M×N×K | GMMA_N | PIPE 深度 | WGMMA 在途批次上限 | 累加精度 | 线程/CTA |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| `CUSTOM_V1_F16ACC` | 128 | 128 | 64 | 64×128×16 | 128 | 3 | 1 | FP16 | 128 |
| `CUSTOM_V2_PIPE2_F16ACC` | 128 | 128 | 64 | 64×128×16 | 128 | 2 | 2 | FP16 | 128 |
| `CUSTOM_V2_PIPE3_F16ACC` | 128 | 128 | 64 | 64×128×16 | 128 | 3 | 2 | FP16 | 128 |
| `CUSTOM_V2_PIPE4_F16ACC` | 128 | 128 | 64 | 64×128×16 | 128 | 4 | 2 | FP16 | 128 |

- 四个实现的 A/B 输入、累加器和 C 输出均为 FP16，`alpha=1`、`beta=0`；A/B 通过共享内存描述符提供给 WGMMA（SS），CTA 内使用一个 128 线程 warpgroup，没有独立的 producer warpgroup。
- V1 每批使用 `warpgroup_wait<0>()` 等待完成；V2 允许两批在途，通过 `warpgroup_wait<1>()` 等较早批次完成后释放槽位，并在末尾使用 `wait<0>()` 收尾。PIPE 表示共享内存缓冲槽数，与 WGMMA 在途批次上限是两个不同参数。
- `CUSTOM_V1_F16ACC` 和 `CUSTOM_V2_PIPE3_F16ACC` 对应历史报告的指令和 tile 配置；V2 PIPE=2/4 用于流水深度对照。四种变体都采用原始直接写回路径，未启用合并写回。
- 配置来自顶层 `CMakeLists.txt` 中的 `GMMA_INST_N=128` 和 V2 编译宏，以及 `kernels/detail/wgmma_config.cuh` 的 CTA/PIPE 默认值。它们在编译时确定；修改后需要重新编译。运行时实际资源信息写入 `csvs/raw_runs.csv` 的 `notes`，包括寄存器、动态/静态 SMEM 和理论 CTA/SM；实际 CTA/SM 需要 profiling，未采集时标记 N/A。

### 数学与布局

```text
A: M×K, B: N×K, C: M×N（列主序）
C(m,n) = Σ_p A(m,p) * B(n,p),  alpha=1, beta=0
```

| 布局 | A | B | lda/ldb | cuBLAS op | CUTLASS A / B |
|---|---|---|---|---|---|
| NT | `a[row + p*M]` | `b[col + p*N]` | `M / N` | `N, T` | ColumnMajor / RowMajor |
| TN | `a[row*K + p]` | `b[col*K + p]` | `K / K` | `T, N` | RowMajor / ColumnMajor |

CUTLASS 的 B 标签按 GEMM 的 K×N 矩阵定义；上表 B 地址描述的是物理 N×K 矩阵，因此两者的行/列主序标签相反。

输入为固定 seed 的非平凡 FP16；同一运行内所有实现共享同一份逻辑 A/B，按布局物理打包后上传，打包与初始化不计时。

### 正确性

先正确性后性能，参考结果为 GPU FP64（失败时退化为采样 CPU FP64）。阈值在运行前固定：

- FP16 累加：`atol = 0.5 + 1e-4*K`，`rtol = 0.01`，`rel_l2 ≤ 0.01`。
- FP32 累加：`atol = 0.125`，`rtol = 0.002`，`rel_l2 ≤ 0.001`。

未通过或出现 NaN/Inf 的实现不参与最佳基线选择。`correctness.log` 记录每次检查的 max_abs/max_rel/rel_l2/mismatch/NaN-Inf/pass。
Shape 和阈值在解析 `plan.yaml` 时传给二进制，`resolved_plan.yaml` 与实际执行一致；`atol` 等数值支持数字或仅含 `K`、数字和 `+ - * /` 的表达式。改变输入类型、alpha/beta 或参考方式为当前程序不支持的值时，运行会直接拒绝。

### 计时

- 主榜：warm-cache + CUDA Event；`warmup=20`，每轮 `inner=100`，`samples=5`，`rounds=5`，每轮轮换实现顺序。
- 可选 `wall`（含 Host 发射）与 `graph`（CUDA Graph replay）默认关闭；启用时单独成榜。
- cold-cache：仅 S01/S02，`flush_mb=256`、`inner=1`，冲刷不计入 kernel 时间。
- NCU 只做诊断：S01/S02，sections 见 `plan.yaml`，`--cache-control all`、`--replay-mode kernel`、`--clock-control none`；其 Duration 不进入主榜。
  - 二进制在采集前先执行 1 次 warmup launch，随后执行 5 次被测 launch；实际 `launch-skip` 和 `launch-count` 来自快照。
  - 附加到 benchmark 时，cuBLASLt 使用已通过正确性验证的同 GPU、同布局、同 Shape、warm/event 结果所保存的完整算法，并检查库版本及算法合法性；没有唯一匹配时跳过并记录原因。单独运行 NCU 没有对应 benchmark，使用首个 heuristic 并标记无法与已调优的 benchmark 对应。

`workspace_mb` 是共同上限：cuBLAS Ex 显式配置该大小的用户 workspace（报告提供的预算，实际使用量无法查询），cuBLASLt 按此上限搜索，CUTLASS 超预算时记为 `UNSUPPORTED`。空间分配不计时。普通 launch 的 warmup 数是 kernel 启动数；graph 预热按完整 replay 向上取整，可能多于指定次数。

## 2. 构建

```bash
cd matmul/best-matmul/cuBLAS_test
bash scripts/plan1/build.sh                 # -> build/plan1/gemm_bench
bash scripts/plan1/build.sh --clean         # 重新配置
CUTLASS_ROOT=/path/to/cutlass bash scripts/plan1/build.sh
CUDACXX=/usr/local/cuda-12.6/bin/nvcc bash scripts/plan1/build.sh
```

构建脚本会：

1. 从任意 CWD 定位项目根与 CUTLASS；
2. 自动选择版本最高的 `/usr/local/cuda*/bin/nvcc`（避免误用旧 Toolkit）；
3. 以 `compute_90a/sm_90a` 生成代码，链接与编译器同源的 cuBLAS/cuBLASLt；
4. 写入 `build/plan1/build_info.txt`；
5. 执行 `--list-impls` / `--list-cases` 冒烟检查（12 个实现、12 个 Case）。

## 3. 运行

### quick（S01+S02）

```bash
bash scripts/plan1/run_quick.sh
# 缩短时间用于链路验证（不要用于正式排名）：
bash scripts/plan1/run_quick.sh --set warmup=2 --set inner=5 --set samples=2 --set rounds=2
# 只跑 cuBLAS/cuBLASLt：
bash scripts/plan1/run_quick.sh --impls CUBLAS_EX_F16ACC,CUBLAS_LT_F16ACC
```

### full（S01–S12 + S01/S02 cold）

```bash
bash scripts/plan1/run_full.sh
bash scripts/plan1/run_full.sh --timing wall --cases S01,S02
```

可用参数：`--set KEY=VALUE`，以及 `--warmup/--inner/--samples/--rounds/--seed/--device/--flush-mb/--lt-candidates/--lt-workspace-mb/--timing/--cache/--lt-selection/--cases/--cold-cases/--layouts/--impls`。所有覆盖都会写入 `resolved_plan.yaml` 并计入 `config_hash`。

### NCU

```bash
# 新建运行目录
bash scripts/plan1/run_ncu.sh

# 追加到已存在的 benchmark 运行（只写 reports/ 与 csvs/ncu_summary.csv）
RUN_DIR="output/plan1/20261105_113000_a1b2c3"
bash scripts/plan1/run_ncu.sh --run-dir "$RUN_DIR"
bash scripts/plan1/run_ncu.sh --run-dir "$RUN_DIR" --cases S01 --layouts TN \
  --impls CUSTOM_V1_F16ACC,CUBLAS_EX_F32ACC
```

缺少 `ncu` 时脚本记录 `SKIPPED` 和原因，不生成假指标。
每次 NCU 采集写入独立的 `reports/ncu_<时间戳>_<随机后缀>/`，重复附加不会覆盖已有报告；附加过程不改变原 benchmark 的运行状态。旧结果没有记录完整 cuBLASLt 算法时，该实现会跳过采集，需要用当前版本重跑 benchmark。

### 汇总

```bash
python3 scripts/plan1/summarize.py --run-dir "output/plan1/<timestamped_run>"
```

只读取指定目录的 `csvs/raw_runs.csv`、NCU CSV 与环境文件，生成/更新：

- `csvs/summary.csv`：按 Case×布局×累加×计时/缓存模式分组的中位数、p10/p90、最佳库基线、自研占比；
- `csvs/ncu_summary.csv`：NCU 指标长表；
- `analysis.md`：回答 6 个规定问题；
- `figures/*.svg`（可选）：仅由真实 CSV 生成。

`raw_runs.csv` 的遥测字段是每轮开始和结束的 NVML 端点采样，`telemetry_scope` 标记为 `round_endpoint_pair`；`summary.csv` 的时钟/功耗字段使用 `clock_power.csv` 中落入该组合测量时间窗口的周期采样，`telemetry_scope` 标注采样数。窗口内无采样或历史数据无时间戳时记为 N/A；采样间隔较长时不能据此归因单次 kernel 功耗。

## 4. 输出目录

```text
output/plan1/<YYYYMMDD_HHMMSS_xxxxxx>/
├── resolved_plan.yaml      # 实际生效配置（含 CLI 覆盖）
├── run_manifest.json       # 运行级/Case 级状态与原因
├── environment.json        # GPU/驱动/CUDA/cuBLAS/CUTLASS/Git/锁频状态
├── commands.log            # 所有命令与返回码
├── run.log                 # 执行过程与关键事件
├── correctness.log         # 每次正确性检查的误差
├── clock_power.csv         # 带时间戳的 GPU 频率/功耗/温度采样
├── interfering_processes_before.txt  # 仅当检测到其他进程
├── csvs/
│   ├── raw_runs.csv        # 唯一原始记录（每轮一行）
│   ├── summary.csv
│   └── ncu_summary.csv     # 仅 NCU 运行后生成
├── reports/                # .ncu-rep / ncu_attempts.csv
├── figures/                # 可选 SVG
└── analysis.md
```

运行目录由 `plan_utils.py` 创建，已存在时拒绝复用；quick/full 永远新建目录，历史结果不会被覆盖。

## 5. 如何判读

1. 先看 `run_manifest.json`：哪些是 `OK/FAIL/UNSUPPORTED/NOT_RUN`；`UNSUPPORTED` 不等于性能差。
2. 再看 `csvs/summary.csv`：只在同 Case、布局、累加精度、timing、cache、timer 口径下比较；`custom_pct_of_best_library` 是该组最佳正确库基线的百分比。
3. 756 TFLOPS 理论峰值只在检测到 H100 时计算；它不代表当前时钟下的可持续上限。
4. `analysis.md` 的 6 个问题必须引用同目录真实数据；没有数据时明确写“未测量”。
5. NCU 的 Duration/SOL 只用于解释 stall、占用率、TMA/WGMMA 行为，不能与 benchmark 混排。

## 6. 常见失败

| 现象 | 原因/处理 |
|---|---|
| 自研/CUTLASS 全部 `UNSUPPORTED` | 当前 GPU CC < 9.0；需 Hopper（SM90a）。 |
| `找不到 CUTLASS` | 设置 `CUTLASS_ROOT=/path/to/cutlass`。 |
| 链接到旧 cuBLAS / `CUBLAS_COMPUTE_16F` 未定义 | 默认 PATH 的 nvcc 过旧；用 `CUDACXX` 指定 CUDA ≥12 的 nvcc，或检查 `ldd build/plan1/gemm_bench`。 |
| `gemm_bench 不存在` | 先执行 `bash scripts/plan1/build.sh`。 |
| 没有 `clock_power.csv` 数据 | 无 NVML/`nvidia-smi` 权限；遥测字段为 N/A，不影响计时。 |
| NCU 全部 `SKIPPED` | 未安装 `ncu` 或无权限；报告保留原因，不伪造指标。 |
| 正确性 `FAIL` | 检查 `correctness.log` 的 `reason`、布局与累加精度；失败实现不参与基线。 |
| cold-cache 时间异常偏高 | `flush_mb` 过大/过小或 L2 实际大小不同；按实测调整并在报告中注明。 |

## 7. 当前验证状态

在 RTX 3050 Laptop（SM86，CUDA 12.6，CUTLASS 4.4.1）上已验证：构建、12 个实现/12 个 Case 枚举、cuBLAS/cuBLASLt 在 TN/NT 的 FP16/FP32 正确性、warm+cold 计时、NVML 遥测、运行目录/manifest/汇总/分析生成。自研与 CUTLASS SM90 在该设备上按设计为 `UNSUPPORTED`。**尚未在 H100 上实测，未产生 H100 性能数据。**
