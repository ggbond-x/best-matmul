# Hopper v1–v5 完整对照实验

在 Linux 上使用 Hopper GPU（compute capability 9.0，编译目标 `sm_90a`）运行。默认从项目目录逐级向上查找 `third_party/cutlass`，并优先使用 `/usr/local/cuda/bin/nvcc`；也可用 `CUTLASS_PATH`、`NVCC_PATH` 显式指定。脚本不修改时钟、功耗上限、MIG 或驱动计数器权限。

本测试是完整的版本与参数对照实验：编译 15 个配置及对应的全输出校验程序，先做数值校验和 SASS/资源导出，再执行核心五版本计时、PIPE 扫描、计算线程扫描和 Compute Sanitizer 检查。`test_ncu.sh` 额外采集 60 份 cold focused NCU 报告和 1 份 v5 warm full 报告，`--with-nsys` 时再加小矩阵时间线。`test_no_ncu.sh` 需要 `nvcc`、`nvidia-smi`、`compute-sanitizer`、`cuobjdump`、`python3` 和常见 shell 工具，但不需要安装 Nsight Compute 或 Nsight Systems；`test_ncu.sh` 还需要 `ncu`，`--with-nsys` 还需要 `nsys`。plan1 这个目录自身包含完整的构建和执行脚本；plan2 是独立的 V2 专项扫描，自带构建与执行脚本，不调用本目录。两者只共享不含实验默认值的执行器 `scripts/common/runner_common.sh`：它读取 manifest 并完成验证、计时、NCU 等通用流程，默认配置和入口仍由各自的 `run.sh` 维护。

## 一键编译与运行

在 `matmul/best-matmul` 目录下：

```bash
# 只显示计划：不检查工具/GPU，不编译，不创建目录。
bash scripts/plan1/test_no_ncu.sh --dry-run
bash scripts/plan1/test_ncu.sh --dry-run --with-nsys

# 一键编译并执行完整实验（不采集 NCU）。
CUTLASS_PATH=/path/to/cutlass NVCC_PATH=/usr/local/cuda/bin/nvcc GPU=0 \
  bash scripts/plan1/test_no_ncu.sh

# 完整实验，包含 60 份 cold focused + 1 份 v5 warm full NCU 报告。
CUTLASS_PATH=/path/to/cutlass NVCC_PATH=/usr/local/cuda/bin/nvcc GPU=0 \
  bash scripts/plan1/test_ncu.sh

# 再追加五个核心版本在小矩阵上的 Nsight Systems 时间线。
CUTLASS_PATH=/path/to/cutlass NVCC_PATH=/usr/local/cuda/bin/nvcc GPU=0 \
  bash scripts/plan1/test_ncu.sh --with-nsys
```

`GPU` 是 `nvidia-smi` 的索引或完整 GPU UUID，默认 `0`。`BUILD_DIR` 默认 `build/plan1/h100_pcie`，`EXP_DIR` 默认 `output/h100_pcie/manual_<本地时间戳>_<进程号>/`；`EXP_DIR` 必须不存在，且不能位于 `BUILD_DIR` 内部。相对路径以调用命令时的当前目录为基准。可用环境变量 `GPU`、`CUTLASS_PATH`、`NVCC_PATH`、`NCU`、`BUILD_DIR`、`EXP_DIR` 覆盖默认值；预热、批次数和重复轮数由脚本固定，分步运行见下文。构建结果在 `build/plan1/h100_pcie/`，实验材料在 `EXP_DIR`；正式成绩看各 `bench_*/summary.csv`。

以前构建在 `build/h100_pcie/` 的程序不会自动迁移；如需继续使用，可运行 `scripts/plan1/run.sh --build-dir "$PWD/build/h100_pcie" ...`。manifest 记录二进制绝对路径，若要使用新路径，请重新编译，不要只移动目录。

只编译、不启动 GPU：

```bash
# 15 个 benchmark + 15 个 .validate，输出到 build/plan1/h100_pcie。
bash scripts/plan1/build.sh --suite all --with-validation

# 只编译核心 5 个或额外 10 个配置；--dry-run 只打印命令和 manifest。
bash scripts/plan1/build.sh --suite core --with-validation --dry-run
bash scripts/plan1/build.sh --suite sweep --with-validation --dry-run
```

`build.sh` 使用 `-O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_90a -lineinfo -Xptxas=-v` 编译，不使用 `-G`、寄存器上限或 fast-math。每个配置的编译命令、诊断和 CPU 映射检查保存在构建目录。以核心 `v3` 为例，编译命令等价于：

```bash
bash build_benchmark.sh --kernel v3 --cutlass "$CUTLASS_PATH" \
  --nvcc "$NVCC_PATH" --output build/plan1/h100_pcie/v3 -- \
  -lineinfo -Xptxas=-v -DPEAK_CTA_M=128 -DPEAK_CTA_N=128 \
  -DPEAK_CTA_K=64 -DPIPE=3 -DGMMA_INST_N=128 \
  -DWGMMA_COMPUTE_THREADS=128 -DWGMMA_MAX_INFLIGHT=2 \
  -DEPILOGUE_COALESCED=1
```

`run.sh` 只执行已编译的程序，并从 `manifest.tsv` 读取实际配置，不根据文件名猜测宏值。单独运行某一阶段时，使用同一套 `BUILD_DIR` 和新的 `EXP_DIR`：

```bash
export BUILD_DIR=$PWD/build/plan1/h100_pcie
export EXP_DIR=$PWD/output/h100_pcie/step_$(date +%Y%m%d_%H%M%S)_$$

bash scripts/plan1/run.sh --mode validate --build-dir "$BUILD_DIR" \
  --gpu "${GPU:-0}" --variants all --output-dir "$EXP_DIR/validate"

bash scripts/plan1/run.sh --mode bench --build-dir "$BUILD_DIR" \
  --gpu "${GPU:-0}" --variants v1,v2,v3,v4,v5 --baseline v1 \
  --shapes 4096x4096x4096 --layouts NT,TN --repeats 5 \
  --per-batch 20 --min-warmup 300 --batches 100 --telemetry-ms 1000 \
  --output-dir "$EXP_DIR/bench_core"
```

`bash scripts/plan1/run.sh --help` 可查看全部参数。`--mode all` 依次执行 validate、bench、ncu、disasm，不包含 sanitize 和 nsys。

## 实验覆盖范围和目的

### 编译配置

核心 5 个版本统一为 CTA=`128×128×64`、`GMMA_INST_N=128`、`PIPE=3`：

| ID | 在途批数 | 合并写回 | 计算线程 | 搬运线程 | 主要变化 |
|---|---:|---|---:|---:|---|
| `v1` | 1 | 关 | 128 | 0 | 每批 WGMMA 完成后释放槽位 |
| `v2` | 2 | 关 | 128 | 0 | 两批在途的计算流水 |
| `v3` | 2 | 开 | 128 | 0 | 共享内存重排后合并写回 |
| `v4` | 2 | 开 | 128 | 128 | 独立 producer warpgroup |
| `v5` | 2 | 开 | 256 | 128 | 两个计算 warpgroup，沿 CTA 的 M 维分工 |

额外 10 个配置用于两组单变量扫描：

| 分组 | ID | 变化 | 固定条件 | 目的 |
|---|---|---|---|---|
| PIPE | `v3_p2/v3_p4`、`v4_p2/v4_p4`、`v5_p2/v5_p4` | `PIPE=2/4` | 各版本其余设置不变 | 与同版本默认 `PIPE=3` 比较流水深度和共享内存占用 |
| 线程 | `v4_m256_t128`、`v5_m256_t128`、`v5_m256_t256`、`v5_m256_t512` | 计算线程 `128/256/512` | CTA=`256×128×64`，`PIPE=3` | 固定 CTA 比较计算 warpgroup 数量；128 线程组还用于 v4/v5 实现等价对照 |

各配置只有在 CTA、`PIPE` 和线程组织满足模板、共享内存及寄存器约束时才能启动，实际资源以编译日志和运行结果为准。`v5_m256_t512` 是四组线程对照中资源需求最高的一组。

### 一键流程执行的阶段

| 顺序 | 阶段 / 输出目录 | 内容 |
|---|---|---|
| 0 | `build/` | 编译 15 个 benchmark + 15 个 `.validate`，并在 CPU 检查 CuTe 线程到输出坐标的映射 |
| 1 | 归档 | 复制构建目录到 `build_artifacts/`，打包源码到 `source.tar.gz` |
| 2 | `env/` | 记录 GPU 型号、UUID、计算能力、拓扑、进程、NCU/工具版本和二进制 SHA256 |
| 3 | `validate/` | 每个配置运行独立的全输出校验程序；失败即停止，不进入计时 |
| 4 | `disasm/` | `cuobjdump` 导出全部配置的资源、SASS 和 PTX（有嵌入 PTX 时） |
| 5 | `bench_core/` | 5 版本 × 6 形状 × NT/TN × 5 轮独立进程 |
| 6 | `bench_v3_pipe/`、`bench_v4_pipe/`、`bench_v5_pipe/` | 三个版本各自比较 `PIPE=2/3/4`，2 形状 × NT/TN × 3 轮 |
| 7 | `bench_m256_threads/` | 4 个 M256 配置 × 2 形状 × NT/TN × 4 轮 |
| 8 | `ncu_all_cold_4096x4096x4096/`、`ncu_all_cold_4096x4096x256/` | 仅 `test_ncu.sh`：15 配置 × NT/TN × 2 形状 = 60 份 focused 报告 |
| 9 | `ncu_v5_warm_full/` | 仅 `test_ncu.sh`：v5、NT、4096³ 的一份 full set warm 报告 |
| 10 | `nsys_small_gemm/` | 仅 `test_ncu.sh --with-nsys`：5 版本 × NT/TN 的 1024³ 时间线 |
| 11 | `sanitize/` | v4、v5、v5_m256_t512 × NT/TN × memcheck/racecheck/synccheck |

默认计时形状为 `1024³`、`4096³`、`8192³`、`4096x4096x256`、`16384x1024x4096`、`1024x16384x4096`，覆盖方阵、短 K、长 M、长 N 和 CTA 波次不足的场景。所有形状须分别被各配置的 CTA tile 整除，脚本会在运行前检查。NT/TN 保持逻辑运算不变，只改变 A/B 的物理布局，两种布局分别统计。

### 校验与计时口径

全输出校验由独立程序执行：遍历 NT/TN、四种输入、短 K、`K` 恰为 `PIPE` 个 tile、以及多次共享内存槽位回绕，将每个输出元素与 CPU 参考比较。输入使用可精确累加的小整数，避免把 FP16 累加舍入误判为同步或映射错误。这属于小矩阵回归，不等同于对 4096 形状全部元素的 CPU 校验。

正式计时使用普通 CUDA Event：默认每个独立进程至少预热 300 次，测量 100 批，每批 20 次启动；跨进程取 p50 的中位数。每个问题、布局和配置默认运行 5 个核心进程（PIPE 扫描 3 个、线程扫描 4 个），每轮轮换配置顺序以降低温度/boost 漂移与固定顺序的相关性。`summary.csv` 的 `speedup_vs_baseline` 只比较同形状、同布局的耗时：核心组基线为 `v1`，PIPE 组基线为各自版本的默认 `PIPE=3`，线程组基线为 `v5_m256_t128`。

NCU、Nsight Systems 和 Compute Sanitizer 的日志耗时不参与成绩。NCU 使用 `--launch-skip 50` 抓取第 51 个匹配 `gemm_device` 的启动，focused 模式采集 9 个 section，cold 为 kernel replay + 清缓存；warm full 使用 application replay，不能与其他版本的 cold 指标混作缓存条件对照。1%–3% 的差异需要重复确认，并检查 `runs.csv` 的 `warmup_stable`、`batch_drift_pct` 和 `telemetry.csv`。

## 输出文件

构建产物默认在 `build/plan1/h100_pcie/`：

```text
build/plan1/h100_pcie/
├── manifest.tsv                 # 本次实际配置的权威记录
├── build_metadata.txt           # 时间、工具链、CUTLASS git 状态、源码哈希
├── v1 ... v5_m256_t512          # 15 个 benchmark
├── <配置名>.validate            # 15 个全输出校验程序（--with-validation）
├── <配置名>.build.log, .command.sh, .help.txt, .sha256
├── <配置名>.validate.build.log, .validate.command.sh, .validate.mapping.txt
└── <配置名>.validate.sha256
```

实验总目录默认在 `output/h100_pcie/manual_<时间戳>_<进程号>/`：

```text
output/h100_pcie/manual_<timestamp>_<pid>/
├── status.txt                   # RUNNING、COMPLETE 或 FAILED stage=... exit=...
├── test.log                     # 全流程日志
├── source.tar.gz                # 明确列出的源码快照
├── build_artifacts/             # 构建目录副本
├── env/                         # 硬件、驱动、工具版本与进程记录
├── validate/                    # validate_<配置名>.log
├── disasm/                      # <配置名>.resources.txt/.sass.txt/.ptx.txt
├── bench_core/                  # 核心五版本计时
├── bench_v3_pipe/               # v3 PIPE 对照
├── bench_v4_pipe/               # v4 PIPE 对照
├── bench_v5_pipe/               # v5 PIPE 对照
├── bench_m256_threads/          # M256 计算线程对照
├── ncu_all_cold_4096x4096x4096/ # 仅 test_ncu.sh
├── ncu_all_cold_4096x4096x256/  # 仅 test_ncu.sh
├── ncu_v5_warm_full/            # 仅 test_ncu.sh
├── nsys_small_gemm/             # 仅 test_ncu.sh --with-nsys
└── sanitize/                    # Compute Sanitizer 日志
```

每个 `bench_*` 子目录包含：

- `summary.csv`：按配置、形状、布局聚合的 `median_p50_ms`、`min_p50_ms`、`max_p50_ms`、`tflops_from_median_p50`、`speedup_vs_baseline`、`flagged_runs`。
- `runs.csv`：每个独立进程的 `p50_ms/p99_ms/TFLOPS_p50`、`warmup_stable`、`batch_drift_pct` 和 warning；`warmup_not_stable`、`drift_over_5pct` 应触发复核。
- `bench_index.tsv`：配置、形状、布局、轮次到日志文件的索引。
- 每次运行的 `bench_<ID>_<shape>_<layout>_r<轮次>.log`；启用采样时的 `telemetry.csv`。
- `run_metadata.txt`、`commands.sh`、`timeline.tsv`、`manifest.tsv`、`build_metadata.txt`。

NCU 目录每份报告保存 `ncu_<ID>_<shape>_<layout>_<cold|warm>.ncu-rep`、`.raw.csv`、`.details.txt` 和 `.log`；Nsight Systems 保存 `nsys_<ID>_<shape>_<layout>.nsys-rep` 和 `.log`；`sanitize/` 保存 `sanitize_<ID>_<layout>_<工具>.log`。

`test.log` 用于排查流程；正式结果看各 `bench_*/summary.csv` 和 `runs.csv`。`status.txt` 只有 `COMPLETE` 才表示全部阶段完成，失败时会标记阶段和退出码，并保留已产生材料。`source.tar.gz` 与 `build_artifacts/` 在耗时的 GPU 阶段之前生成，后续重编译不会覆盖本次实验材料。

更完整的参数约束、NCU section 解释和成绩判读方法见 [docs/H100_BENCHMARK.md](../../docs/H100_BENCHMARK.md)。该文档写于脚本位于项目根目录时，命令中的 `build.sh`、`run.sh`、`test.sh` 现在分别对应 `scripts/plan1/` 下的同名脚本、`test_no_ncu.sh` 和 `test_ncu.sh`。
