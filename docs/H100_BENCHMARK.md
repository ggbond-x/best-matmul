# H100 PCIe：v1–v5 对照实验

本文说明如何使用 `build.sh` 和 `run.sh` 比较调度、合并写回、warp specialization 与多个计算 warpgroup。当前开发环境没有 H100 和 Nsight Compute；本文提供实验方法及可复制命令，不包含 H100 性能实测结论。

本实验测量已驻留 GPU 显存的 dense FP16 GEMM。输入、累加和输出均为 FP16，`alpha=1`、`beta=0`，运算量按 `2*M*N*K` 计算。主机分配、TMA 描述符准备、Host↔Device 拷贝、数值校验及状态查询在正式计时之外。Event 区间包含连续提交的一批 GEMM，可能包含提交不及时造成的 GPU 空隙；它也不是端到端业务耗时。GPU 显存常驻不等于每次都从 HBM 读取：重复运行可能命中 L2。

比较理论峰值时，确认实际设备为 **H100 PCIe 80GB**，并使用对应 dense FP16 Tensor Core 口径、实际工作频率和功耗条件。SXM、FP8、结构化稀疏的峰值不能作为本实验的 PCIe dense FP16 基准。脚本检查 GPU 名称包含 H100，但不会替你证明其为 PCIe 80GB；以环境记录中的实际型号、形态及配置为准。

## 一键执行

[test.sh](test.sh) 已将下述步骤串联，带工具检查、失败中止、材料归档和阶段状态记录。在本目录执行：

```bash
# 只显示完整计划，不编译、不启动 GPU，也不创建实验目录。
bash test.sh --dry-run

# 按服务器的实际安装路径修改；默认 GPU=0。
CUTLASS_PATH=/workspace/peak_flops/third_party/cutlass \
NVCC_PATH=/usr/local/cuda/bin/nvcc bash test.sh

# 需要小矩阵 CPU 提交/GPU 执行时间线时，追加此开关。
CUTLASS_PATH=/workspace/peak_flops/third_party/cutlass \
NVCC_PATH=/usr/local/cuda/bin/nvcc bash test.sh --with-nsys
```

默认执行：15 配置编译及完整输出校验、全部 SASS/资源导出、核心五轮/PIPE 三轮/线程四轮计时、15 配置 × NT/TN × 长短 K 的 **60 份 cold focused NCU 报告**、一份 v5 NT 的 warm full 报告，以及三个代表配置的 Sanitizer 检查。`--with-nsys` 再增加核心五版本 NT/TN 的 10 份小矩阵时间线。完整采集耗时可能较长；warm full 用于深入查看 v5，不能直接与其他版本 cold 指标做缓存条件对照。

可用环境变量 `GPU`、`CUTLASS_PATH`、`NVCC_PATH`、`NCU`、`BUILD_DIR`、`EXP_DIR` 覆盖脚本默认值；相对路径以脚本目录为基准。`NCU` 可指向独立安装的 ncu 可执行文件。正式执行会提前检查所需工具，只有 `--with-nsys` 才要求安装 nsys。

本次实验目录额外保存 `build_artifacts/`、`source.tar.gz`、全流程 `test.log` 和 `status.txt`。状态为 `COMPLETE` 才表示全部阶段完成；失败时记录阶段和退出码，并保留已完成材料。归档与 SASS 导出安排在耗时的 profiler 阶段之前。`EXP_DIR` 必须不存在，且不能等于或位于 `BUILD_DIR` 内部，以避免归档递归复制自身。

## 1. 配置和对照关系

### 核心五组：逐步增加功能

`--suite core` 编译下列五组。共同设置为 **CTA=128×128×64、GMMA_INST_N=128、PIPE=3**，单条 WGMMA 形状为 `64×128×16`。每次都比较相同问题、布局和输入。

| ID | 在途批数 | 合并写回 | 计算线程 | 专用搬运线程 | CTA 总线程 | 相邻版本的主要变化 |
|---|---:|---|---:|---:|---:|---|
| `v1` | 1 | 关 | 128 | 0 | 128 | 每批 WGMMA 完成后再释放槽位 |
| `v2` | 2 | 关 | 128 | 0 | 128 | 增加计算流水 |
| `v3` | 2 | 开 | 128 | 0 | 128 | 增加共享内存重排与合并写回 |
| `v4` | 2 | 开 | 128 | 128 | 256 | 增加独立 producer warpgroup |
| `v5` | 2 | 开 | 256 | 128 | 384 | 增加到两个计算 warpgroup，沿 CTA 的 M 维分工 |

“专用搬运线程为 0”表示没有预留独立 warpgroup，原计算线程也承担搬运控制。v4/v5 虽为 producer 保留 128 个线程，主循环中实际由一个 leader 等待可写槽位、设置事务字节数并发起 TMA；其余 producer 线程不执行 WGMMA。合并写回阶段可使用整个 CTA 的线程。

固定 CTA 后，v5 将同一块输出分给两个 warpgroup，没有将每个 CTA 的运算量翻倍。增加计算线程会同时改变每线程累加器负担、寄存器分配和 CTA 资源，不能预先断言吞吐一定提高。

### 额外十组：流水深度和计算线程数

`--suite sweep` 只编译以下十组；`--suite all` 合并核心五组与这十组。所有额外配置均为 `GMMA_INST_N=128`、两批在途、合并写回开启。

| ID | CTA | PIPE | 计算线程 | 专用搬运线程 | 对照用途 |
|---|---|---:|---:|---:|---|
| `v3_p2` | 128×128×64 | 2 | 128 | 0 | 与 `v3`、`v3_p4` 比流水深度 |
| `v3_p4` | 128×128×64 | 4 | 128 | 0 | 同上 |
| `v4_p2` | 128×128×64 | 2 | 128 | 128 | 与 `v4`、`v4_p4` 比流水深度 |
| `v4_p4` | 128×128×64 | 4 | 128 | 128 | 同上 |
| `v5_p2` | 128×128×64 | 2 | 256 | 128 | 与 `v5`、`v5_p4` 比流水深度 |
| `v5_p4` | 128×128×64 | 4 | 256 | 128 | 同上 |
| `v4_m256_t128` | 256×128×64 | 3 | 128 | 128 | 与下一组做实现等价对照 |
| `v5_m256_t128` | 256×128×64 | 3 | 128 | 128 | 线程数扫描的参考配置 |
| `v5_m256_t256` | 256×128×64 | 3 | 256 | 128 | 固定 tile，增加至两个计算 warpgroup |
| `v5_m256_t512` | 256×128×64 | 3 | 512 | 128 | 固定 tile，增加至四个计算 warpgroup |

流水对照按版本分别比较 `p2 / 默认 p3 / p4`，保持 tile 和线程数一致。线程对照固定 `256×128×64` tile 和 `PIPE=3`，比较 v5 的 `128/256/512` 个计算线程。不要将核心 `v5` 与 `v5_m256_t512` 的差值全部归因于线程数，因为它们的 M tile 也不同。

`v4_m256_t128` 与 `v5_m256_t128` 使用相同设备实现及线程组织，适合作为实现等价对照。两者是否在测量噪声内一致，需要实际数据判断。

## 2. 六种默认形状与 NT/TN

| M×N×K | A+B+C 的 FP16 占用 | 主要观察点 |
|---|---:|---|
| 1024×1024×1024 | 6 MiB | 核心 tile 下仅 64 个 CTA；观察波次不足、启动和固定开销 |
| 4096×4096×4096 | 96 MiB | 中等方阵，作为统一 NCU 分析形状 |
| 8192×8192×8192 | 384 MiB | 更多 CTA 和较长主循环，观察稳定吞吐与持续功耗 |
| 4096×4096×256 | 36 MiB | 只有四个 K tile；观察流水启动、排空和写回占比 |
| 16384×1024×4096 | 168 MiB | 长 M、窄 N；观察 CTA 网格、复用及布局敏感性 |
| 1024×16384×4096 | 168 MiB | 窄 M、长 N；与上一组形成形状对照 |

占用仅计三份 GPU 矩阵，不含主机副本和运行时开销。NT/TN 保持逻辑运算不变，改变 A/B 的物理布局及对应 TMA/GMMA major 组织；两种布局分别出结果，不能混合取中位数。所有问题须能被所选配置的 CTA tile 整除，脚本会在运行前检查。

## 3. 按步骤执行

以下命令在 Linux/Bash 的 `matmul/best-matmul/` 目录执行，同一实验使用同一个终端中的变量。先设置实际 GPU 索引或完整 GPU UUID；每次新实验重新生成 `EXP_DIR`。`run.sh` 要求结果目录尚不存在，防止覆盖已有结果。

```bash
GPU=0
# 让 cuobjdump、compute-sanitizer 等也使用目标 CUDA 工具链；按实际安装位置调整。
export PATH="/usr/local/cuda/bin:$PATH"
BUILD_DIR="$PWD/build/h100_pcie"
EXP_DIR="$PWD/reports/h100_pcie/manual_$(date +%Y%m%d_%H%M%S)_$$"
CORE=v1,v2,v3,v4,v5

# 只显示计划，无需安装 CUDA/CUTLASS，也不会创建输出文件。
bash build.sh --suite all --with-validation --output-dir "$BUILD_DIR" --dry-run

# 自动定位：CUTLASS_ROOT 或项目 third_party；CUDACXX 或标准 CUDA 路径/PATH。
bash build.sh --suite all --with-validation --output-dir "$BUILD_DIR"
```

需要指定工具链时，将上一条真正编译命令换为下面这条，并替换 CUTLASS 安装路径。不要让 PATH 意外选择不支持 SM90a 的旧 nvcc。

```bash
bash build.sh --suite all --with-validation \
  --cutlass /workspace/peak_flops/third_party/cutlass \
  --nvcc /usr/local/cuda/bin/nvcc --output-dir "$BUILD_DIR"
```

构建固定使用优化的 `sm_90a` 代码，增加 `-lineinfo -Xptxas=-v`，不使用 `-G`、寄存器上限或 fast-math。`--with-validation` 为每个配置另编译完全相同宏设置的 `.validate` 程序，并在 CPU 上检查真实 CuTe 输出归属；编译步骤不启动 GPU 内核。每次成功构建后，manifest 只列出本次选择的 suite，因而后续完整实验应使用 `--suite all`。

### 环境、数值校验与核心计时

```bash
# 环境模式无需 manifest，不启动内核；先确认实际型号、GPU UUID、MIG 和其他进程。
bash run.sh --mode env --gpu "$GPU" --output-dir "$EXP_DIR/env"

# 每个配置的独立校验程序检查 NT/TN、四种输入、短 K/边界/多次槽位回绕的每个 C 元素。
bash run.sh --mode validate --build-dir "$BUILD_DIR" --gpu "$GPU" \
  --variants all --output-dir "$EXP_DIR/validate"

# 核心五版本：默认六种形状、两种布局、每组 3 个独立进程。
bash run.sh --mode bench --build-dir "$BUILD_DIR" --gpu "$GPU" \
  --variants "$CORE" --baseline v1 --layouts NT,TN \
  --repeats 3 --per-batch 20 --min-warmup 300 --batches 100 \
  --telemetry-ms 1000 --output-dir "$EXP_DIR/bench_core"
```

`validate` 要求本次构建元数据明确包含 `with_validation=1`，避免误用旧的 `.validate` 文件。校验输入为小整数，限制参考部分和处于可精确表示的 FP16 范围，适合发现映射、同步及漏写问题；它不证明任意浮点输入和任意 K 的数值精度。`--mode check` 仅使用主程序的 16 点抽样校验；已有全输出校验时，按需要另做快速排查。

核心默认计时共运行 `5×6×2×3=180` 个独立进程，每个进程另有预检查、预热和后检查。可先用 `--shapes 4096x4096x4096` 确认流程，再执行六种形状。

### 单独执行流水与线程扫描

下面先用方阵和短 K 定位变化，再按需要扩展到其他默认形状；同一对照组保持形状、布局、预热和采样设置一致。

```bash
SWEEP_SHAPES=4096x4096x4096,4096x4096x256

# 三个版本各自扫描 PIPE=2/3/4；每组以自己的 PIPE=3 为参考。
for version in v3 v4 v5; do
  bash run.sh --mode bench --build-dir "$BUILD_DIR" --gpu "$GPU" \
    --variants "${version}_p2,${version},${version}_p4" --baseline "$version" \
    --shapes "$SWEEP_SHAPES" --layouts NT,TN --repeats 3 \
    --output-dir "$EXP_DIR/bench_${version}_pipe"
done

# M tile 固定 256；v4 与 v5 的 128 线程组同时保留，检查实现等价性。
bash run.sh --mode bench --build-dir "$BUILD_DIR" --gpu "$GPU" \
  --variants v4_m256_t128,v5_m256_t128,v5_m256_t256,v5_m256_t512 \
  --baseline v5_m256_t128 --shapes "$SWEEP_SHAPES" --layouts NT,TN \
  --repeats 3 --output-dir "$EXP_DIR/bench_m256_threads"
```

### NCU：先聚焦，再扩展

先收集一个布局下 v3/v4/v5 的 cold focused 报告。这里 cold 表示 **kernel replay + `--cache-control all`**；该缓存条件用于单内核分析，与普通重复计时的缓存历史不同。

```bash
bash run.sh --mode ncu --build-dir "$BUILD_DIR" --gpu "$GPU" \
  --variants v3,v4,v5 --layouts NT --profile-shape 4096x4096x4096 \
  --ncu-set focused --ncu-cache cold --ncu-clock none --launch-skip 50 \
  --output-dir "$EXP_DIR/ncu_cold_focused"

# 只对一个配置/布局收 full；warm 重启应用重建目标前的执行历史，代价更高。
bash run.sh --mode ncu --build-dir "$BUILD_DIR" --gpu "$GPU" \
  --variants v5 --layouts NT --profile-shape 4096x4096x4096 \
  --ncu-set full --ncu-cache warm --ncu-clock none --launch-skip 50 \
  --timeout 7200 --output-dir "$EXP_DIR/ncu_v5_warm_full"
```

warm 使用 **application replay + `--cache-control none`**。它保留应用执行历史来构造缓存条件，但不保证数据全部命中缓存，也不保证与未分析程序完全相同。不要用 kernel replay 加关闭清缓存来冒充每个 pass 都相同的热缓存。replay、缓存控制及动态时钟会影响分析；参阅 [NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。

脚本按函数名 `regex:gemm_device` 过滤，仅抓一个匹配启动。`--launch-skip 50` 跳过前 50 个匹配 GEMM，目标为第 51 个：第一启动用于预检查，所以默认抓到第 50 次预热。初始化用的其他 kernel 不在该计数中。NCU 的应用参数保证目标出现在预热停止之前。

NCU 还固定启用 `--check-exit-code yes`，让被分析程序的非零退出成为分析失败，避免将数值校验失败的程序当作成功采集。

更换布局、形状或指标时使用新的结果目录。附加 `--metrics` 前，先查看环境模式保存的 `env_ncu_metrics_gh100.txt`，按本机支持的完整名称填写；不要照搬不同架构或版本的指标名。CLI 的过滤、section、指标查询及导出方式见 [NVIDIA Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)。

**NCU 日志中的程序耗时和 TFLOPS 不作为成绩。** 指标收集可能多次 replay，带来插桩、缓存和时钟条件变化。正式成绩仅由 `--mode bench` 的普通 Event 计时生成；分析 cold/warm 的差异时，也不要混用它们的时长来排序版本。

### Nsight Systems：检查小 GEMM 的提交空隙

如果小矩阵的批均时间未随 kernel 优化明显下降，可单独检查 CPU 提交和 GPU 执行时间线。Event 覆盖整批 GEMM，GPU 等待 CPU 继续提交的空隙也可能落在两个 Event 之间；仅看单个 kernel 时长不足以解释批均结果。

```bash
bash run.sh --mode nsys --build-dir "$BUILD_DIR" --gpu "$GPU" \
  --variants v1,v5 --profile-shape 1024x1024x1024 --layouts NT \
  --output-dir "$EXP_DIR/nsys_small_gemm"

nsys-ui "$EXP_DIR/nsys_small_gemm/nsys_v5_1024x1024x1024_NT.nsys-rep"
```

此模式要求 PATH 中可用的 `nsys`，没有新增的专用脚本选项，沿用 `--profile-shape`、`--variants`、`--layouts` 和 `--timeout`。脚本固定使用 `nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none --stats=true`，收集 CUDA API/GPU 活动及 OS runtime 时间线，关闭 CPU 栈采样和上下文切换采集。被分析程序使用输入模式 2、Event 计时、每批 10 次、最低预热 100 次及 20 个正式批次。

报告包含初始化、校验、预热和正式运行，应在 GUI 中定位稳定的连续 GEMM 段，结合 CPU launch API 与同一 stream 的 GPU kernel 间隙判断提交是否及时，不使用全程序平均值替代稳定段分析。该 trace 也有采集开销，日志或统计表中的时间与 TFLOPS 不参与正式成绩汇总。CLI 与时间线说明见 [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。

### 同步诊断与最终指令

```bash
# 小矩阵、多次槽位回绕；分别运行三种工具，可能很慢。
bash run.sh --mode sanitize --build-dir "$BUILD_DIR" --gpu "$GPU" \
  --variants v4,v5,v5_m256_t512 --layouts NT,TN \
  --sanitizers memcheck,racecheck,synccheck --timeout 7200 \
  --output-dir "$EXP_DIR/sanitize"

# 不启动 GPU：保留最终二进制的资源、SASS 和可能嵌入的 PTX。
bash run.sh --mode disasm --build-dir "$BUILD_DIR" \
  --variants v1,v2,v3,v4,v5,v5_m256_t512 --output-dir "$EXP_DIR/disasm"
```

Compute Sanitizer 应足够新以识别 Hopper 的 TMA/WGMMA 同步语义。诊断失败时先看具体错误、工具版本和源码，再决定是程序问题还是工具覆盖问题；通过也不等于证明全部并发执行正确。诊断耗时不参与性能排名。仅包含 sm_90a cubin 的二进制可能没有嵌入 PTX，PTX 导出为空不表示没有生成 WGMMA；以 SASS 和资源报告为据。

`--mode all` 顺序执行环境记录、validate、bench、ncu、disasm，不包含 sanitize 或 nsys；这两种诊断分别调用。分步命令更便于发现问题后只重跑相应阶段。

## 4. 新脚本参数表

底层 `-D` 宏、单个 benchmark 的位置参数和六组专用回归脚本参数，见 [README 参数总表](README.md#参数总表)。这里只列本实验新增的两个入口。

### build.sh

| 参数 | 默认值 | 范围 / 约束 | 作用 |
|---|---|---|---|
| `--suite` | `core` | `core` / `sweep` / `all` | 选择 5 / 10 / 15 个配置；成功后替换所选 suite 的 manifest |
| `--cutlass DIR` | `CUTLASS_ROOT`，否则项目 `third_party` 自动定位 | 包含所需 CUTLASS 头文件的根目录 | 显式路径优先；自动尝试 `third_party/cutlass`、`third_party/CUTLASS`、`third_party` |
| `--nvcc PATH` | `CUDACXX`，否则 `/usr/local/cuda/bin/nvcc`，再查 PATH | 支持当前源码和 SM90a 的编译器 | 固定本次所有配置的工具链 |
| `--output-dir DIR` | `modular/build/h100_pcie` | 可写目录；相对当前目录解析 | 保存二进制、manifest、编译日志和构建元数据 |
| `--with-validation` | 关 | 无值开关 | 为每个配置编译 `.validate` 并执行 CPU 映射检查 |
| `--dry-run` | 关 | 无值开关 | 显示命令与 manifest，不要求 CUDA/CUTLASS 存在，不写文件 |
| `--help` / `-h` | 关 | 无值开关 | 显示帮助后退出 |

### run.sh

数值参数使用不超过 9 位、无前导零的非负十进制整数；各参数还有下表的下限。`min_warmup+21*per_batch<=2147483647`。CSV 列表用逗号分隔，不含空项或重复项；形状写作 `MxNxK`，使用小写 `x`。

| 参数 | 默认值 | 范围 / 约束 | 作用 |
|---|---|---|---|
| `--mode` | `bench` | `env/check/validate/bench/ncu/nsys/sanitize/disasm/all` | 选择记录、校验、计时或分析阶段；nsys 单独运行 |
| `--build-dir DIR` | `modular/build/h100_pcie` | 含本次构建 manifest 的目录；env 无需 manifest | 定位配置及二进制；validate/all 还要求校验 sidecar 和启用标记 |
| `--output-dir DIR` | `modular/reports/h100_pcie/<时间戳>_<mode>_<PID>` | 必须不存在；相对当前目录解析 | 新建本次结果目录，保留已有实验 |
| `--gpu ID` | `0` | 单张完整 GPU 的 nvidia-smi 索引或 GPU UUID | 先解析 UUID，再设置 CUDA_VISIBLE_DEVICES；进程内编号为 0 |
| `--variants CSV` | `all` | manifest 中互不重复的配置 ID | 选择本次比较组及初始顺序 |
| `--layouts CSV` | `NT,TN` | `NT`、`TN` 或二者 | 选择 bench/check/ncu/nsys/sanitize 布局；全输出 validate 自行覆盖两者 |
| `--shapes CSV` | 第 2 节六种形状 | 正整数形状列表，满足所选 tile 和矩阵元素数量限制 | 选择普通计时问题 |
| `--profile-shape` | `4096x4096x4096` | 单一合法形状 | 选择 NCU 或 Nsight Systems 问题，不影响 bench 的形状列表 |
| `--repeats N` | `3` | `2..999999999` | 每个问题/布局/配置的独立进程次数 |
| `--per-batch N` | `20` | `1..999999999`，满足预热计数联合限制 | 每个批均样本的 GEMM 次数 |
| `--min-warmup N` | `300` | `1..999999999`，满足联合限制 | 最少预热启动数；仍须满足自适应稳定判据 |
| `--batches N` | `100` | `2..999999999`，还受内存和实验时间限制 | 正式批均样本数 |
| `--baseline ID` | `v1` | 同一结果集中的参考 ID；不在结果中则相对列为空 | 计算相同形状/布局下的延迟比 |
| `--ncu PATH` | `NCU` 环境变量或 `ncu` | 可执行的 NCU CLI | 选择分析器版本 |
| `--ncu-set` | `focused` | `focused/full` | focused 收九个 section；full 使用本机 full set |
| `--ncu-cache` | `cold` | `cold/warm` | 选择 kernel replay+清缓存，或 application replay+不清缓存 |
| `--ncu-clock` | `none` | `none/base` | 传递 NCU 时钟控制；base 只影响分析阶段，普通 bench 不使用 |
| `--launch-skip N` | `50` | `1..999999999` | 跳过匹配 GEMM 的启动数，仅收下一次 |
| `--metrics CSV` | 空 | 本机查询确认支持的指标列表 | 在 section/set 之外请求指标 |
| `--sanitizers CSV` | `memcheck,racecheck,synccheck` | 这三种工具的非空列表 | 选择单独运行的诊断工具 |
| `--timeout N` | `1800` 秒 | `0..999999999`；0 不限时 | 单条命令超时，超时后再等 30 秒强制结束 |
| `--telemetry-ms N` | `1000` 毫秒 | `0` 或 `100..999999999` | bench 期间后台采样 GPU 状态；0 关闭 |
| `--dry-run` | 关 | 无值开关；除 env 外仍须合法 manifest | 仅显示命令，不启动 GPU、不新建结果目录 |
| `--help` / `-h` | 关 | 无值开关 | 显示帮助后退出 |

`--mode validate` 不使用 bench 的形状/输入/布局列表，而由校验程序生成边界用例。`ncu`、`nsys` 和 `sanitize` 也使用自己的小批次参数，普通 bench 的采样参数不改变它们的分析流程。

## 5. 如何读正式成绩

一个计时样本为“本批 Event 总耗时 / 本批启动次数”。默认每进程取 100 个这样的批均样本，输出 `p50_ms/p99_ms`；p99 是批均延迟的高分位，不能解释为每一次 kernel 的尾延迟。跨进程聚合使用各进程 p50 的中位数，不将多份进程 p99 拼成一个新的 p99。

每个形状/布局默认重复三个独立进程。每一轮循环移位版本执行顺序，例如核心第一轮从 v1 开始、第二轮从 v2 开始、第三轮从 v3 开始，降低固定顺序与温度、boost 漂移的相关性。轮换不能消除其他进程、降频或功耗约束。

默认三轮适合初筛；核心五版本正式对比可用 `--repeats 5` 或 5 的倍数，使每个版本各占一次顺序位置。PIPE 三配置的三轮已覆盖所有位置；M256 四配置对比可用 `--repeats 4` 或 4 的倍数。位置均衡仍不能代替检查实际温漂与进程间分散。

`--min-warmup 300` 是启动次数下限，不是 GPU 已达到热稳态的保证。不同形状的 300 次运行持续时间差异很大；短窗口批均延迟稳定也不代表温度和功耗已经稳定。结合 `telemetry.csv` 检查持续的频率、温度及功耗变化，必要时对所有对照配置一致地延长预热，然后重新采集。

| 文件 / 字段 | 解释 |
|---|---|
| `bench_index.tsv` | 配置、形状、布局、进程轮次到普通计时日志的索引 |
| `runs.csv` | 每进程的 `p50_ms/p99_ms/TFLOPS_p50`、预热和漂移记录 |
| `runs.csv: warning` | `warmup_not_stable` 表示预热未确认稳定；`drift_over_5pct` 表示批次漂移绝对值超过 5% |
| `batch_drift_pct` | 正式样本最后四分之一的中位数相对最前四分之一的变化；正值表示变慢 |
| `summary.csv: median_p50_ms` | 相同配置/形状/布局下，各独立进程 p50 的中位数 |
| `min_p50_ms / max_p50_ms` | 各进程 p50 的范围，用于检查进程间分散 |
| `tflops_from_median_p50` | `2*M*N*K/(median_p50_ms*1e9)`；dense GEMM 运算口径 |
| `speedup_vs_baseline` | 同形状/布局下参考配置中位延迟除以当前延迟；大于 1 表示当前更快 |
| `flagged_runs` | 该组含任一 warning 的进程个数，不是失败数或自动剔除数 |
| `cta / pipe / compute_threads / producer_threads` | 聚合时保留的配置字段，防止把不同资源组合混看 |

先检查数值校验、warning、进程间范围和环境，再解释收益。脚本保留有 warning 的结果，不自动删除慢样本；即使 `flagged_runs=0`，进程间仍可能有明显分散。核心和 sweep 都应按相同问题比较，跨 tile 的相对值只表示耗时比，不能直接解释为单一优化收益。

## 6. 从 NCU 观察解释变化

以下是需要用数据验证的分析假设，不是已经测得的结论。section 的具体指标及可用性以安装版本为准，先查看保存的 section/metric 目录。

| Focused section | 优先观察 | 对照假设与交叉检查 |
|---|---|---|
| `LaunchStats` | CTA 线程数、寄存器、动态共享内存、grid | 是否与 manifest 相符；线程增加、PIPE 增加有没有改变驻留 CTA 上限 |
| `Occupancy` | 理论/实测 occupancy、限制资源 | 低 occupancy 是否由共享内存、寄存器或波次数限制；不要把更高 occupancy 当作必然更快 |
| `SpeedOfLight` | 计算与内存子系统利用率 | 用于选择下一步调查方向；利用率百分比不是本项目按 dense `2MNK` 计算的 TFLOPS |
| `ComputeWorkloadAnalysis` | Tensor 管线负载和指令执行 | v2 流水、v5 多计算组是否减少 Tensor 管线空闲；同时排查供数和资源约束 |
| `MemoryWorkloadAnalysis` | DRAM/L2/shared 流量与利用率、缓存和存储事务 | v3 合并写回是否改善输出事务；新增 shared 重排是否有额外代价；cold/warm 条件是否改变供数 |
| `SchedulerStats` | eligible/issued warp 与调度空隙 | 多计算组是否提供更多可发射工作，还是因驻留 CTA 下降抵消收益 |
| `WarpStateStats` | 等待分布和采样范围 | 将等待对应到消费者计算和 producer 搬运角色；不要只按全 CTA 平均 stall 排名 |
| `InstructionStats` | Tensor、加载/存储和控制指令构成 | 与 SASS 交叉检查版本实际生成了哪些路径；区分静态指令和动态执行量 |
| `SourceCounters` | 源码/指令位置上的采样 | 用 lineinfo 将等待定位到 WGMMA 完成、TMA 可读、empty barrier 或写回位置 |

v4/v5 的 producer 在等待槽位或数据状态时出现 stall 可能正是预期行为；它们也不应执行 WGMMA。把 producer 与 consumer 混在一起的平均等待、active warp 或 occupancy，不能直接当作“计算线程更闲”。应结合消费者指令位置、Tensor 管线、可发射 warp、实际延迟和资源限制判断。本实现没有使用 `setmaxnreg` 在 producer/consumer 间重新分配寄存器，不能把多数时间空闲的 producer 当作零资源开销。

比较 PIPE 时，A/B 数据缓冲大致随 `2*(CTA_M+CTA_N)*CTA_K*PIPE` 字节线性增长，另有屏障及对齐开销。主循环结束后写回复用同一共享内存，动态分配取主循环空间与 C 暂存 `2*CTA_M*CTA_N` 的 **较大值**，不是两者相加。更深流水可能改善供数，也可能减少每 SM 驻留 CTA；用实际 Launch config/NCU 数据确认。

H100 每 SM 有 228 KiB 共享内存、64K 个 32 位寄存器，单 CTA 可使用的共享内存上限为 227 KiB。以下只按共享内存计算驻留上界，不能解释为已经达到的 occupancy；寄存器、warp 数及其他限制可能进一步降低驻留。硬件资源说明见 [NVIDIA Hopper Tuning Guide：Occupancy](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy)。

| CTA / PIPE | 每槽 A+B | 全部 A+B 槽位 | 仅共享内存决定的每 SM 驻留上界 |
|---|---:|---:|---|
| 128×128×64 / 2 | 32 KiB | 64 KiB，加屏障/对齐 | 最多 3 个 CTA |
| 128×128×64 / 3 | 32 KiB | 96 KiB，加屏障/对齐 | 最多 2 个 CTA |
| 128×128×64 / 4 | 32 KiB | 128 KiB，加屏障/对齐 | 最多 1 个 CTA |
| 256×128×64 / 3 | 48 KiB | 144 KiB，加屏障/对齐 | 最多 1 个 CTA |

所以 PIPE=4 即便改善重叠，也可能因只能驻留一个 CTA 而损失跨 CTA 的延迟隐藏。PIPE=3 的两个驻留 CTA 只是可能性，还要检查实际寄存器预算。M256 线程扫描本来就受共享内存约束至最多一个 CTA，因此应分析一个 CTA 内独立计算组、累加器和等待的变化，不能将更多 warp 直接等同于更多驻留 CTA。

`*.build.log` 的 ptxas 报告用于检查寄存器、stack 和 spill。发生 spill 时，再看 local memory 流量及延迟，不能只看寄存器数减少。动态共享内存由启动时配置，不能因为 ptxas 的静态 shared 数字很小就认为没有共享内存。`*.resources.txt`、程序 Launch config、NCU LaunchStats 应一起核对。

## 7. 保留实验材料并打开报告

构建目录保留 `manifest.tsv`、`build_metadata.txt`、每组 `.command.sh/.build.log/.help.txt/.sha256`。启用完整校验时另有 `.validate`、其命令/编译日志、`.validate.mapping.txt` 及校验和。元数据含 UTC 构建时间、nvcc 版本、CUTLASS git HEAD/分支/已跟踪文件状态和源码哈希；构建失败不会将 manifest 更新为半套配置。

每次运行的新结果目录保留 manifest 和构建元数据副本、`run_metadata.txt`、`commands.sh`、环境文件及相应阶段日志。`commands.sh` 是精确参数及 CUDA_VISIBLE_DEVICES UUID 的审计清单，包含持续采样命令；日志重定向和监控进程回收由 `run.sh` 管理，重跑实验请使用 `run.sh`，不要整体执行审计清单。`env_gpu_before.txt/env_gpu_after.txt` 包含时钟、功耗、温度、MIG/ECC、PCIe 链路及降频相关状态；`env_processes.txt` 用于检查其他进程。

bench 默认另保存 `telemetry.csv`，每秒采样 `timestamp,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used`；采样错误写入 `telemetry.stderr.txt`。这是粗粒度状态，不是每个 kernel 的瞬时功耗或精细能量测量。`--telemetry-ms 0` 可关闭；对照实验采用相同采样设置。监控进程在 bench 结束或脚本退出时回收。

`timeline.tsv` 没有表头，三列为 **UTC 纳秒时间、start/end、日志文件名**，记录每条主要命令的边界。telemetry 的 timestamp 为 nvidia-smi 本地时间，通过 `env_date.txt` 保存的时区偏移转换后，可与各进程区间对齐，检查时钟、温度及功率变化。

每份 NCU 分析保存 `.ncu-rep`、`.raw.csv`、`.details.txt` 及 `.log`。CSV 从报告导出，避免混入程序输出和 `==PROF==` 行。可在安装了匹配版本 Nsight Compute UI 的机器上打开报告，示例文件名与上面的命令一致：

```bash
ncu-ui "$EXP_DIR/ncu_cold_focused/ncu_v5_4096x4096x4096_NT_cold.ncu-rep"
```

Nsight Systems 分析保存 `nsys_<ID>_<MxNxK>_<布局>.nsys-rep` 和同名 `.log`；日志包含 `--stats=true` 的统计输出，按工具版本还可能产生辅助数据库。`env_nsys_version.txt` 保存采集器版本，env 模式也会尝试记录该版本。用 `nsys-ui` 打开 `.nsys-rep` 查看时间线，保留原始报告，避免只留下整体统计。

遇到 `ERR_NVGPUCTRPERM` 时，按 [NVIDIA 性能计数器权限说明](https://developer.nvidia.com/ERR_NVGPUCTRPERM) 与管理员确认访问策略。脚本不会自动修改驱动模块选项、容器权限或系统计数器权限；普通计时与性能计数器采集分别检查。

## 8. 时钟、功耗与可比条件

脚本不会调用 nvidia-smi 修改时钟、功耗上限、MIG 或持久化模式。普通计时采用当前机器条件；NCU 默认 `--ncu-clock none`，显式选择 `base` 才请求分析器的时钟控制，且只用于分析阶段。它不能代替普通计时的同条件实验。

先用环境记录检查实际频率、温度、功耗上限及限制原因。如果独占设备并得到相应管理权限，可手工查询支持频率，再选择适合该设备的锁定值；不要照抄另一张卡的固定 MHz 或 W。以下修改命令默认注释，确定 `LOCK_MHZ` 后再自行执行，并在实验记录中注明修改及恢复方式：

```bash
nvidia-smi -i "$GPU" -q -d SUPPORTED_CLOCKS
# LOCK_MHZ=从当前设备支持列表中选择的值
# sudo nvidia-smi -i "$GPU" -lgc "$LOCK_MHZ,$LOCK_MHZ"
# 实验结束、确认应恢复默认时：sudo nvidia-smi -i "$GPU" -rgc
```

具体支持项与权限见 [NVIDIA System Management Interface 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。比较版本时保持 GPU、驱动、CUDA/CUTLASS、功耗上限、时钟策略、MIG/ECC、其他进程、采样设置及输入问题一致，并保存实际观测值。先确认结果可重复，再用 NCU 的资源和指令证据解释差异。
