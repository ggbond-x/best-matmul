# V2 PIPE、WGMMA 与 CTA 对照实验

在 Linux 上使用 Hopper GPU（compute capability 9.0，编译目标 `sm_90a`）运行。默认使用项目 `third_party/cutlass` 和 PATH 中的 `nvcc`；请使用支持 SM90a 的 CUDA 工具链，并确保 `nvidia-smi` 可访问目标 GPU。脚本自带 `scripts/plan2/run_variants.sh` 执行验证和计时，与 plan1 完全解耦，不调用 plan1 的任何脚本；两者仅共享通用执行器 `scripts/common/runner_common.sh`，plan2 的默认形状、基线和构建提示均由自己的入口提供。**不需要安装 Nsight Compute（NCU）**：所有性能结果来自普通 CUDA Event 计时。

## 环境准备

以下命令都在 `matmul/best-matmul` 目录下的 Bash 中执行。先设置服务器上的工具路径和目标 GPU：

```bash
export CUTLASS_ROOT=/path/to/cutlass
export CUDACXX=/usr/local/cuda/bin/nvcc
export GPU=0
```

`CUTLASS_ROOT` 要指向含 `include/cute/tensor.hpp` 的 CUTLASS 根目录；根据服务器实际位置修改。可用 `bash scripts/plan2/run.sh --help` 查看可选环境变量。建议运行前检查目标 GPU 的其他进程并尽量独占；脚本记录前后的进程快照和计时期间的设备状态，但不会自动判定实验是否受到干扰。

## 编译已有测试配置

```bash
# 仅预览计划，不检查 GPU、编译或创建目录。
bash scripts/plan2/run.sh --dry-run

# 只编译已有的九个配置及验证程序，并检查 CPU 输出映射。
export OUTPUT_DIR="$PWD/output/plan2/manual_$(date +%Y%m%d_%H%M%S)_$$"
export BUILD_DIR="$PWD/build/plan2/$(basename "$OUTPUT_DIR")"
bash scripts/plan2/run.sh --build-only
```

`--build-only` 不启动 GPU。`OUTPUT_DIR` 和 `BUILD_DIR` 必须不存在且不能相互嵌套。默认分别为 `output/plan2/<本地时间戳>_<进程号>/` 和 `build/plan2/<相同标识>/`；如果只指定 `OUTPUT_DIR`，构建目录默认使用其最后一级名称。脚本在末尾打印两个路径，结果目录的 `build_dir.txt` 和构建目录的 `output_dir.txt` 记录关联路径。**后续单独运行 `run_variants.sh` 时必须显式传入 `--build-dir`**，脚本不会猜测或复用任何历史构建；`run.sh` 结束时会打印可直接复制的命令示例。

`run.sh` 对每个配置先拼出参数，然后执行 `bash build_benchmark.sh ...`。以 v2 为例，在 `run.sh` 脚本中会组装成如下的 benchmark 编译命令：

```bash
bash build_benchmark.sh \
  --kernel v2 \
  --cutlass "$CUTLASS_ROOT" \
  --nvcc "$CUDACXX" \
  --output "$BUILD_DIR/v2_p3_n128_m128n128" \
  -- -lineinfo -Xptxas=-v \
  -DPEAK_CTA_M=128 -DPEAK_CTA_N=128 -DPEAK_CTA_K=64 \
  -DPIPE=3 -DGMMA_INST_N=128 -DWGMMA_MAX_INFLIGHT=2
```

编译完成后会生成如下的文件：

~~~text
build/plan2/<时间戳>_<pid>/
├── v2_p3_n128_m128n128            ← benchmark 程序
├── v2_p3_n128_m128n128.validate   ← 全输出验证程序
├── v2_p3_n128_m128n128.build.log
├── v2_p3_n128_m128n128.validate.build.log
└── v2_p3_n128_m128n128.mapping.log
~~~

为什么每个配置编译两个程序？一个 CUDA 程序只能有一个 `main`，而这两件事的目标不同：

|              | benchmark 程序              | `.validate` 程序                   |
| :----------- | :-------------------------- | :--------------------------------- |
| 入口         | `benchmark_host.cu`         | `tests/full_output_validation.cu`  |
| 用途         | 计时、算 TFLOPS             | 检查结果是否正确                   |
| 运行规模     | 4096³、4096×4096×256 等形状 | 小矩阵、多种输入、短流水、槽位回绕 |
| 输出         | 延迟、带宽、遥测            | PASS/FAIL 及错误位置               |
| 是否参与排名 | 是                          | 否，只决定能不能进入 benchmark     |

两者使用**完全相同的内核头文件和完全相同的 `-D` 宏**，例如都包含 `wgmma_tma_fp16_pipeline_v2.cuh`，都带 `PEAK_CTA_*=128,128,64`、`PIPE=3`、`GMMA_INST_N=128`、`WGMMA_MAX_INFLIGHT=2`。这样验证程序检查的就是 benchmark 实际使用的那个编译产物逻辑，而不是另一套配置。

## 运行已有测试配置

在 Hopper 服务器上，使用上一节 `--build-only` 生成的 `OUTPUT_DIR` 和 `BUILD_DIR`。执行器是 plan2 自带的 `scripts/plan2/run_variants.sh`，不依赖 plan1。**`--build-dir` 是必填项**：它必须指向某一次 `run.sh --build-only` 的构建目录，脚本不会默认使用 `build/plan2/` 下的某个固定或“最新”目录，也不会覆盖已有构建。如果在新 shell 中操作，可从结果目录的 `build_dir.txt` 取回路径：

```bash
export OUTPUT_DIR=/实际的结果目录
export BUILD_DIR="$(cat "$OUTPUT_DIR/build_dir.txt")"
```

也可以直接使用 `run.sh` 末尾打印的 `--build-dir` 路径。若漏传 `--build-dir`，脚本会报错并给出构建命令；不会静默选择其他实验的构建。下面各结果子目录必须不存在；若已有结果，使用新的结果目录名，避免覆盖。

```bash
# 保证验证或 benchmark 失败时停止后续命令。
set -e -o pipefail

# 全部配置先做全输出验证。GPU 是 nvidia-smi 的索引或 GPU UUID。
bash scripts/plan2/run_variants.sh --mode validate --build-dir "$BUILD_DIR" \
  --gpu "${GPU:-0}" --variants all --output-dir "$OUTPUT_DIR/validation"

# 以下每组只运行已编译的程序，不重新编译。按顺序运行基准及四组扫描。
common=(--mode bench --build-dir "$BUILD_DIR" --gpu "${GPU:-0}" --layouts NT,TN \
  --shapes 4096x4096x256,4096x4096x4096 \
  --repeats 3 --per-batch 20 --min-warmup 300 --batches 100 --telemetry-ms 1000)

bash scripts/plan2/run_variants.sh "${common[@]}" \
  --variants v1_p3_n128_m128n128,v2_p3_n128_m128n128 \
  --baseline v1_p3_n128_m128n128 --output-dir "$OUTPUT_DIR/00_baseline"
bash scripts/plan2/run_variants.sh "${common[@]}" \
  --variants v2_p2_n128_m128n128,v2_p3_n128_m128n128,v2_p4_n128_m128n128 \
  --baseline v2_p3_n128_m128n128 \
  --output-dir "$OUTPUT_DIR/01_pipe"
bash scripts/plan2/run_variants.sh "${common[@]}" \
  --variants v2_p3_n64_m128n128,v2_p3_n128_m128n128 \
  --baseline v2_p3_n128_m128n128 --output-dir "$OUTPUT_DIR/02_gmma"
bash scripts/plan2/run_variants.sh "${common[@]}" \
  --variants v2_p3_n64_m64n128,v2_p3_n64_m128n64,v2_p3_n64_m128n128 \
  --baseline v2_p3_n64_m128n128 --output-dir "$OUTPUT_DIR/03_cta"
bash scripts/plan2/run_variants.sh "${common[@]}" \
  --variants v2_p3_n64_m128n128,v2_p3_n64_m256n128,v2_p3_n64_m128n256 \
  --baseline v2_p3_n64_m128n128 --output-dir "$OUTPUT_DIR/04_large_cta"
```

如果希望一次完成编译和全部测试，也可以执行 `bash scripts/plan2/run.sh`。它会编译新的一套程序，要求指定的 `OUTPUT_DIR` 和 `BUILD_DIR` 都尚不存在；不要沿用上面已经创建的目录。

## 单独编译和运行未列出的 case

CTA、PIPE、GMMA 指令尺寸是**编译期**配置；想测试 `configs` 中没有的组合，需要重新编译。例如，下例使用 V2 的 **CTA=64×64×64、PIPE=2、GMMA_N=64**。先编译 benchmark 和采用相同参数的全输出验证程序：

```bash
# 独立目录，不修改 plan2 生成的 manifest 和已有二进制。
CUSTOM_ID="custom_$(date +%Y%m%d_%H%M%S)_$$"
CUSTOM_BUILD_DIR="$PWD/build/plan2/$CUSTOM_ID"
CUSTOM_OUTPUT_DIR="$PWD/output/plan2/$CUSTOM_ID"
mkdir -p "$CUSTOM_BUILD_DIR" "$CUSTOM_OUTPUT_DIR"
flags=(-lineinfo -Xptxas=-v -DPEAK_CTA_M=64 -DPEAK_CTA_N=64 \
  -DPEAK_CTA_K=64 -DPIPE=2 -DGMMA_INST_N=64 -DWGMMA_MAX_INFLIGHT=2)

bash build_benchmark.sh \
  --kernel v2 \
  --cutlass "$CUTLASS_ROOT" \
  --nvcc "$CUDACXX" \
  --output "$CUSTOM_BUILD_DIR/v2_custom" \
  -- "${flags[@]}"

"$CUDACXX" -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_90a \
  -I "$CUTLASS_ROOT/include" -I "$CUTLASS_ROOT/tools/util/include" \
  "-DGEMM_KERNEL_HEADER=\"$PWD/kernels/wgmma_tma_fp16_pipeline_v2.cuh\"" \
  "${flags[@]}" tests/full_output_validation.cu -o "$CUSTOM_BUILD_DIR/v2_custom.validate"
```

然后在目标 Hopper GPU 上先验证，再计时。`CUDA_VISIBLE_DEVICES` 可设为 GPU 索引或 UUID；这里使用上面设置的 `GPU`。

```bash
set -e -o pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

"$CUSTOM_BUILD_DIR/v2_custom.validate" --mapping-only
"$CUSTOM_BUILD_DIR/v2_custom.validate" 2>&1 | tee "$CUSTOM_OUTPUT_DIR/validation.log"

# 与 plan2 一样的 4096³、NT/TN、三次独立进程及 CUDA Event 计时参数。
for layout in NT TN; do
  for rep in 1 2 3; do
    "$CUSTOM_BUILD_DIR/v2_custom" 4096 4096 4096 "${layout:0:1}" "${layout:1:1}" \
      2 event 20 300 100 2>&1 | tee "$CUSTOM_OUTPUT_DIR/${layout}_r${rep}.log"
  done
done
```

日志中 `p50_ms` 是该进程的批均延迟中位数；比较时可对三个进程的 `p50_ms` 再取中位数，并在同一台服务器、相同形状和布局下与基准对照。直接运行不会自动生成 `summary.csv`，也不会自动采集 GPU telemetry。

命令位置参数依次是 `M N K transA transB input_mode event|wall per_batch min_warmup batches`：`N T` 表示 NT，`T N` 表示 TN；`2` 为随机 ±1 输入，`event 20 300 100` 分别表示 CUDA Event、每批 20 次、至少预热 300 次、测量 100 批。可用 `"$CUSTOM_BUILD_DIR/v2_custom" --help` 查看帮助。若只改变矩阵 M/N/K 或 NT/TN，直接修改运行参数即可，无需重新编译；本内核要求 M、N、K 分别能被编译时的 CTA_M、CTA_N、CTA_K 整除。

改变 `flags` 时，当前代码要求 CTA_M 是 64 的倍数、CTA_N 能被 GMMA_N 整除、CTA_K 是 16 的倍数；`GMMA_INST_N` 支持 8、16、32、64、128、256，V2 的两批在途要求 PIPE 至少为 2。还须满足 CuTe/TMA 布局、共享内存和设备资源约束；满足这些基本整除条件不保证所有组合都可编译或运行。自定义程序不在 plan2 的 `manifest.tsv` 中，不能直接以配置 ID 交给 `run_variants.sh`；上面的直接运行方式独立保存验证和计时日志。

## 实验覆盖范围和目的

先在同一台服务器上比较 V1 和报告配置的 V2，建立同机基准。两者都使用 FP16 输入与累加、128 个计算线程、直接写回、CTA=128×128×64、PIPE=3、GMMA_N=128；V1 为单批在途，V2 为两批在途。后续扫描均使用 V2、128 计算线程、两批在途、直接写回和 CTA_K=64。每组分别测 NT/TN、`4096×4096×256` 和 `4096×4096×4096`。各扫描组只改变下表指定变量：

| 顺序 | 输出目录       | 变化的变量                        | 固定条件                           | 目的                                                         |
| ---- | -------------- | --------------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| 0    | `00_baseline`  | V1 / V2                           | CTA=128×128×64，PIPE=3，GMMA_N=128 | 在当前服务器分别获得 V1、V2 的 NT/TN 时间，以同布局 V1 为基准比较 V2 |
| 1    | `01_pipe`      | PIPE=2/3/4                        | CTA=128×128×64，GMMA_N=128         | 比较缓冲深度和资源占用的折中；与 V2 报告的编译配置对应       |
| 2    | `02_gmma`      | GMMA_N=64/128                     | CTA=128×128×64，PIPE=3             | 观察覆盖相同 CTA 时指令 N 尺寸的影响                         |
| 3    | `03_cta`       | CTA M×N=64×128、128×64、128×128   | PIPE=3，GMMA_N=64                  | 比较较小 CTA 的资源、并行度与数据复用                        |
| 4    | `04_large_cta` | CTA M×N=128×128、256×128、128×256 | PIPE=3，GMMA_N=64                  | 单独探索更大 CTA，避免同时改变指令尺寸                       |

共编译 9 个去重配置（V1 基准及 8 个 V2 配置）；共享基线会在各自实验组重新计时。所有配置先执行输出映射检查和 GPU 全输出验证（包含 NT/TN、不同输入、短流水及多次槽位回绕）；验证失败则停止，不继续 benchmark。全输出验证是小矩阵回归，不等同于对 4096 形状全部元素进行 CPU 验证。当前内核要求 M/N/K 分别整除 CTA 对应维度，表中形状满足要求。

默认每组每个配置、形状和布局运行 3 个独立进程，轮换配置顺序；每次至少预热 300 次，测量 100 批，每批 20 次。可用 `REPEATS`、`MIN_WARMUP`、`BATCHES`、`PER_BATCH`、`TELEMETRY_MS` 环境变量调整；`TELEMETRY_MS=0` 关闭设备采样。每组 `summary.csv` 的 `speedup_vs_baseline` 由相同形状、布局的跨进程 p50 延迟中位数计算，大于 1 表示快于该组基线。`runs.csv` 包含逐进程 p50/p99、预热稳定性及漂移；若预热不稳定、漂移明显或 GPU 被其他进程占用，应重测。1%–3% 差异需要重复确认；NCU 不是此脚本的一部分，普通 benchmark 的时长才用于性能对比。

例如查看 `00_baseline/summary.csv` 中 `shape=4096x4096x4096`、`layout=TN` 的两行：`median_p50_ms` 是在这台服务器测得的时间，V2 的 `speedup_vs_baseline` 是同机同布局的 V1 时间除以 V2 时间。若要将 V2-NT 和 V1-TN 对照，应自行比较这两行的 `median_p50_ms`，不要把同布局 speedup 当作跨布局比较。租用服务器的型号、功耗和时钟可能不同，原 H100 PCIe 报告的绝对数字只作背景；判断本轮变体收益时以本机基准和相同形状、布局的时间为准。

PIPE=4 和大 CTA 消耗较多共享内存；实际资源限制和寄存器数以编译日志、设备和运行结果为准。这里固定 PIPE 和其他参数，并不自动把每组胜者用于下一组，因此可以清楚地解释单变量对照。

## 输出文件

```text
build/plan2/<时间戳>_<进程号>/
├── manifest.tsv, build_metadata.txt, output_dir.txt
├── <配置名>, <配置名>.validate
├── <配置名>.build.log, <配置名>.validate.build.log
└── <配置名>.mapping.log, <配置名>.sha256

output/plan2/<时间戳>_<进程号>/
├── status.txt                  # RUNNING、COMPLETE 或失败阶段
├── run.log, commands.sh, build_dir.txt  # 总日志、命令和构建目录
├── gpu_list.txt, gpu_before.txt, gpu_after.txt
├── processes_before.csv, processes_after.csv
├── validation/                 # 各配置 GPU 全输出验证日志
│   ├── validate_<id>.log
│   └── validate_<id>.status.tsv  # 退出码、PASS/FAIL/TIMEOUT/INTERRUPTED/RUNNING、构建哈希
├── 00_baseline/                # 本机 V1/V2 基准对照
├── 01_pipe/                    # PIPE 对照
├── 02_gmma/                    # WGMMA N 对照
├── 03_cta/                     # 小/基准 CTA 对照
├── 04_large_cta/               # 大 CTA 对照
└── csv/                        # collect_results.py 生成的汇总表（可选）
```

每个 benchmark 子目录包含 `summary.csv`、`runs.csv`、`bench_index.tsv`、每次运行的日志、命令和环境记录，以及启用采样时的 `telemetry.csv`。失败时已产生的文件会保留，`status.txt` 标记失败阶段；不要把未完成的目录当作完整结果。`--build-only` 只生成独立构建目录及输出目录中的顶层日志和状态。

## 汇总 CSV

`scripts/plan2/collect_results.py` 把各实验组的分散结果整理成扁平 CSV，不启动 GPU，也不重新计算耗时或 TFLOPS。原始 `summary.csv`、`runs.csv`、`manifest.tsv` 和验证日志只读，脚本不会修改或删除它们。

```bash
# 自动选择已完成的运行，输出到该结果目录的 csv/ 子目录。
python3 scripts/plan2/collect_results.py

# 指定结果目录和输出目录。
python3 scripts/plan2/collect_results.py "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR/csv"

# 从其他输出根目录自动选择（默认是 best-matmul/output/plan2）。
python3 scripts/plan2/collect_results.py --output-root /path/to/output/plan2
```

### 默认结果选择规则

不传结果目录时，脚本只扫描 `--output-root` 的直接子目录，并按以下规则选择：

1. 读取 `status.txt`，只考虑第一行为 `COMPLETE` 的运行；
2. 该运行必须同时包含 `00_baseline`、`01_pipe`、`02_gmma`、`03_cta`、`04_large_cta` 五个实验组；
3. 在满足条件的运行中，按 `status.txt` 的修改时间选择；**不依据各 `summary.csv` 的修改时间**，因此复制或重写旧 summary 不会让旧运行被误选；
4. `--build-only` 留下的 `COMPLETE` 没有实验组，归类为 `BUILD_ONLY`，不会被当成 benchmark 完成。

没有可确认完成的运行时，脚本以退出码 2 结束，并列出候选目录的运行状态（`FAILED`、`RUNNING`、`PARTIAL`、`BUILD_ONLY`、`UNKNOWN`）和缺失组，不会猜测。显式传入一个部分完成、失败或运行中的目录仍可汇总，但会打印警告，并在 `dataset.csv` 中设置 `partial=yes`、`run_state` 和 `missing_groups`；这明确表示结果不完整，不能当作完整实验结论。显式目录只包含 `--build-only` 的构建、没有任何 `summary.csv` 时直接报错。不同运行目录的数据不会被混合。

### 配置来源与冲突处理

`version`、`gmma_n`、`inflight` 等配置字段优先来自**每个实验组在运行时保存的 `manifest.tsv` 快照**（`run_variants.sh` 会把当时的构建 manifest 复制到结果子目录）。因此历史结果不会因为后来重新编译而改变标签：

- 结果目录内的快照足以完成汇总；原服务器的 `build_dir.txt` 路径不可访问时仍可运行，脚本只把快照当作唯一配置来源。
- 同一配置 ID 在不同实验组的快照中配置字段不一致时，脚本报错退出，不会静默挑一个值。
- 提供 `--manifest` 时，它只作为核对和缺失快照的补充；与组内快照冲突时脚本报错，不会覆盖组内快照。
- `build_dir.txt` 指向的当前构建 manifest 仅用于比较：若与历史组快照不同，保留组内快照，并在 `manifest_note` 中标记 `build_manifest_differs`，提示该历史结果并非由当前二进制产生。
- 某组缺少 manifest 快照时，该组配置字段留空，`manifest_source=missing`，并打印警告；脚本不猜测配置。

验证状态也按组绑定：`validation/validate_<id>.status.tsv` 记录验证时的 manifest SHA-256。只有哈希与该组快照一致时，验证结果才可能用于该组；同一配置 ID 来自不同构建的验证不会被误标为通过。

### 验证状态判定

`validation_status` 只取 `PASS`、`FAIL`、`UNKNOWN`：

- `PASS`：验证进程退出码为 0、侧车文件记录的构建哈希与组内快照一致，且日志包含最终成功标记 `All full-output regressions PASS`。仅有 `Mapping PASS` 或某条子检查的 `PASS` 不算通过。
- `FAIL`：日志中出现大小写不敏感的失败词（`failure`/`failed`/`error`/`mismatch` 等），或侧车状态为 `FAIL`、`TIMEOUT`、`INTERRUPTED`，或退出码非零。
- `UNKNOWN`：缺少验证日志或侧车文件、侧车仍为 `RUNNING`、退出状态缺失、构建哈希不匹配，或旧日志没有退出状态证据。历史日志不会被推断为通过。

`summary_all.csv`、`runs_all.csv` 与 `validation.csv` 使用同一套判定，同一配置的状态一致。`validation.csv` 还会列出结果中出现的验证记录，即使没有对应的实验组行（`group` 留空）。

### 生成的文件

```text
csv/
├── summary_all.csv    # 各组 summary.csv 合并；含 group、run_state、version、gmma_n、inflight、
│                      # manifest_source、manifest_note、manifest_sha256、validation_status/detail
├── runs_all.csv       # 各组 runs.csv 合并，附加字段与 summary_all.csv 相同
├── validation.csv     # group、id、status、process_status、exit_code、detail、log、status_file、
│                      # validate_binary、manifest_sha256
├── manifests_all.csv  # 组内快照、显式 manifest 和当前构建 manifest 的配置记录及其 SHA-256
└── dataset.csv        # result_dir、run_state、status_text、expected/present/missing_groups、
                       # partial、collected_utc
```

退出码：正常汇总为 0（部分结果也会打印警告但返回 0，方便检查）；结果目录不存在、无可选运行、manifest 冲突或没有实验组时为 2。