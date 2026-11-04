# V2 PIPE、WGMMA 与 CTA 对照实验

在 Linux 上使用 Hopper GPU（compute capability 9.0，编译目标 `sm_90a`）运行。默认使用项目 `third_party/cutlass` 和 PATH 中的 `nvcc`；请使用支持 SM90a 的 CUDA 工具链，并确保 `nvidia-smi` 可访问目标 GPU。脚本使用已有的 `scripts/test1/run.sh` 执行验证和计时，不依赖 test1 的构建脚本。**不需要安装 Nsight Compute（NCU）**：所有性能结果来自普通 CUDA Event 计时。

## 一键编译与运行

在 `matmul/best-matmul` 目录下：

```bash
# 仅查看编译、验证和运行命令；不检查 GPU、编译或创建目录。
bash scripts/test2/run.sh --dry-run

# 一键编译、检查输出映射、全输出校验，再依次进行同机基准和四组扫描。
CUTLASS_ROOT=/path/to/cutlass CUDACXX=/usr/local/cuda/bin/nvcc GPU=0 \
  bash scripts/test2/run.sh

# 只编译和执行 CPU 输出映射检查，不启动 GPU。
CUTLASS_ROOT=/path/to/cutlass CUDACXX=/usr/local/cuda/bin/nvcc \
  OUTPUT_DIR=/tmp/v2-build bash scripts/test2/run.sh --build-only
```

`OUTPUT_DIR` 必须不存在；省略时生成 `reports/test2/<本地时间戳>_<进程号>/`。可用 `bash scripts/test2/run.sh --help` 查看参数。建议运行前检查目标 GPU 的其他进程并尽量独占；脚本记录前后的进程快照和计时期间的设备状态，但不会自动判定实验是否受到干扰。

每个配置都编译 benchmark 和独立的全输出验证程序。以报告基线为例，其 benchmark 编译命令等价于：

```bash
bash build_benchmark.sh --kernel v2 --cutlass "$CUTLASS_ROOT" \
  --nvcc "$CUDACXX" --output "$OUTPUT_DIR/build/v2_p3_n128_m128n128" -- \
  -lineinfo -Xptxas=-v -DPEAK_CTA_M=128 -DPEAK_CTA_N=128 \
  -DPEAK_CTA_K=64 -DPIPE=3 -DGMMA_INST_N=128 -DWGMMA_MAX_INFLIGHT=2
```

`build_benchmark.sh` 使用 `-O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_90a` 编译。验证程序使用相同的编译选项、内核和宏，入口为 `tests/full_output_validation.cu`。实际调用和 `nvcc` 参数写入本次结果的 `commands.sh` 和各配置的 `*.build.log`；`manifest.tsv` 是实际配置的权威记录。

若已通过 `--build-only` 得到一套构建，可以在 Hopper 上手动运行其中一组，下面的 `OUTPUT_DIR` 指向该套构建的根目录；各输出子目录必须不存在：

```bash
# 全部配置先做全输出验证。GPU 是 nvidia-smi 的索引或 GPU UUID。
export OUTPUT_DIR=/tmp/v2-build
bash scripts/test1/run.sh --mode validate --build-dir "$OUTPUT_DIR/build" \
  --gpu "${GPU:-0}" --variants all --output-dir "$OUTPUT_DIR/validation"

# 示例：PIPE 扫描；完整的四组命令可先用 --dry-run 查看。
bash scripts/test1/run.sh --mode bench --build-dir "$OUTPUT_DIR/build" \
  --gpu "${GPU:-0}" --layouts NT,TN \
  --shapes 4096x4096x256,4096x4096x4096 \
  --variants v2_p2_n128_m128n128,v2_p3_n128_m128n128,v2_p4_n128_m128n128 \
  --baseline v2_p3_n128_m128n128 --repeats 3 --per-batch 20 \
  --min-warmup 300 --batches 100 --telemetry-ms 1000 \
  --output-dir "$OUTPUT_DIR/01_pipe"
```

## 实验覆盖范围和目的

先在同一台服务器上比较 V1 和报告配置的 V2，建立同机基准。两者都使用 FP16 输入与累加、128 个计算线程、直接写回、CTA=128×128×64、PIPE=3、GMMA_N=128；V1 为单批在途，V2 为两批在途。后续扫描均使用 V2、128 计算线程、两批在途、直接写回和 CTA_K=64。每组分别测 NT/TN、`4096×4096×256` 和 `4096×4096×4096`。各扫描组只改变下表指定变量：

| 顺序 | 输出目录 | 变化的变量 | 固定条件 | 目的 |
|---|---|---|---|---|
| 0 | `00_baseline` | V1 / V2 | CTA=128×128×64，PIPE=3，GMMA_N=128 | 在当前服务器分别获得 V1、V2 的 NT/TN 时间，以同布局 V1 为基准比较 V2 |
| 1 | `01_pipe` | PIPE=2/3/4 | CTA=128×128×64，GMMA_N=128 | 比较缓冲深度和资源占用的折中；与 V2 报告的编译配置对应 |
| 2 | `02_gmma` | GMMA_N=64/128 | CTA=128×128×64，PIPE=3 | 观察覆盖相同 CTA 时指令 N 尺寸的影响 |
| 3 | `03_cta` | CTA M×N=64×128、128×64、128×128 | PIPE=3，GMMA_N=64 | 比较较小 CTA 的资源、并行度与数据复用 |
| 4 | `04_large_cta` | CTA M×N=128×128、256×128、128×256 | PIPE=3，GMMA_N=64 | 单独探索更大 CTA，避免同时改变指令尺寸 |

共编译 9 个去重配置（V1 基准及 8 个 V2 配置）；共享基线会在各自实验组重新计时。所有配置先执行输出映射检查和 GPU 全输出验证（包含 NT/TN、不同输入、短流水及多次槽位回绕）；验证失败则停止，不继续 benchmark。全输出验证是小矩阵回归，不等同于对 4096 形状全部元素进行 CPU 验证。当前内核要求 M/N/K 分别整除 CTA 对应维度，表中形状满足要求。

默认每组每个配置、形状和布局运行 3 个独立进程，轮换配置顺序；每次至少预热 300 次，测量 100 批，每批 20 次。可用 `REPEATS`、`MIN_WARMUP`、`BATCHES`、`PER_BATCH`、`TELEMETRY_MS` 环境变量调整；`TELEMETRY_MS=0` 关闭设备采样。每组 `summary.csv` 的 `speedup_vs_baseline` 由相同形状、布局的跨进程 p50 延迟中位数计算，大于 1 表示快于该组基线。`runs.csv` 包含逐进程 p50/p99、预热稳定性及漂移；若预热不稳定、漂移明显或 GPU 被其他进程占用，应重测。1%–3% 差异需要重复确认；NCU 不是此脚本的一部分，普通 benchmark 的时长才用于性能对比。

例如查看 `00_baseline/summary.csv` 中 `shape=4096x4096x4096`、`layout=TN` 的两行：`median_p50_ms` 是在这台服务器测得的时间，V2 的 `speedup_vs_baseline` 是同机同布局的 V1 时间除以 V2 时间。若要将 V2-NT 和 V1-TN 对照，应自行比较这两行的 `median_p50_ms`，不要把同布局 speedup 当作跨布局比较。租用服务器的型号、功耗和时钟可能不同，原 H100 PCIe 报告的绝对数字只作背景；判断本轮变体收益时以本机基准和相同形状、布局的时间为准。

PIPE=4 和大 CTA 消耗较多共享内存；实际资源限制和寄存器数以编译日志、设备和运行结果为准。这里固定 PIPE 和其他参数，并不自动把每组胜者用于下一组，因此可以清楚地解释单变量对照。

## 输出文件

```text
reports/test2/<时间戳>_<进程号>/
├── status.txt                  # RUNNING、COMPLETE 或失败阶段
├── run.log, commands.sh        # 总日志和实际命令
├── gpu_list.txt, gpu_before.txt, gpu_after.txt
├── processes_before.csv, processes_after.csv
├── build/
│   ├── manifest.tsv, build_metadata.txt
│   ├── <配置名>, <配置名>.validate
│   ├── <配置名>.build.log, <配置名>.validate.build.log
│   ├── <配置名>.mapping.log, <配置名>.sha256
├── validation/                 # 各配置 GPU 全输出验证日志
├── 00_baseline/                # 本机 V1/V2 基准对照
├── 01_pipe/                    # PIPE 对照
├── 02_gmma/                    # WGMMA N 对照
├── 03_cta/                     # 小/基准 CTA 对照
└── 04_large_cta/               # 大 CTA 对照
```

每个 benchmark 子目录包含 `summary.csv`、`runs.csv`、`bench_index.tsv`、每次运行的日志、命令和环境记录，以及启用采样时的 `telemetry.csv`。失败时已产生的文件会保留，`status.txt` 标记失败阶段；不要把未完成的目录当作完整结果。`--build-only` 只生成顶层日志、状态和 `build/`。
