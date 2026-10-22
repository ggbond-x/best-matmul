# Hopper FP16 GEMM

独立的 Hopper FP16 GEMM 教学与基准仓库（CUDA 12.6、CUTLASS/CuTe、`sm_90a`）。主路径只有两个内核：`teach` 和 `fast`。它们共用同一套 tile、同一条 WGMMA 指令和同一份合并写回。

输入、累加和输出都是 FP16，`alpha=1`，`beta=0`。逻辑形状是 A(M,K)、B(N,K)、C(M,N)，布局只支持 NT 和 TN。

## 默认配置

不传 `-D` 时的默认值在 `gemm/config.cuh`。`bash build.sh teach|fast -- -D宏=值` 会改掉这一份二进制里的对应常量；不写 `-D` 时仍是下表。

| | teach | fast |
|---|---|---|
| CTA tile | 128×128×64 | 128×128×64 |
| PIPE（TMA 槽数） | 3 | 3 |
| WGMMA | `64×128×16` SS，FP16 | 相同 |
| 每个计算 warpgroup 的在途 WGMMA 批数 | 2 | 2 |
| producer 线程 | 0（计算 warpgroup 里选出的一个 lane 发 TMA） | 128（只有 thread 0 发 TMA） |
| 计算 warpgroup | 1（128 线程） | 2（256 线程，沿 M 维分工） |
| CTA 线程数 | 128 | 384 |
| 写回 | 16 字节合并写回 | 相同 |

默认问题规模的 `M`、`N`、`K` 必须分别是 128、128、64 的倍数。换了 CTA 之后，运行时的三维都要能被新 tile 整除。empty mbarrier 的到达计数等于计算线程数。`fast` 在 `WGMMA_MAX_INFLIGHT=2`（默认）时，producer 会排空自己用过的每个槽，最后一批 WGMMA 在 `wait<0>` 之后再对 empty 做一次 arrive。`WGMMA_MAX_INFLIGHT=1` 则每个 K tile 算完就释放，不再多一次 arrive。

## 编译期对照旋钮

每个二进制只烘进一组宏。A/B 时编两个输出，运行参数保持一致，用 `--help` 确认打印出来的配置不同。`--` 后面的参数原样传给 nvcc，所以 `-D` 必须写在 `--` 之后。

| 宏 | 默认 | 例子 | 对照时隔离的是什么 |
|---|---|---|---|
| `PEAK_CTA_M` | 128 | `-DPEAK_CTA_M=256` | CTA 的 M。`fast` 还要求它能被 `64 × 计算 warpgroup 数` 整除 |
| `PEAK_CTA_N` | 128 | `-DPEAK_CTA_N=256` | CTA 的 N。必须是 `GMMA_INST_N` 的倍数，并且是 64 的倍数（同一份二进制同时实例化 NT 与 TN 的 SW128） |
| `PEAK_CTA_K` | 64 | `-DPEAK_CTA_K=128` | 每个 TMA/WGMMA 的 K 步。必须是 64 的倍数 |
| `PIPE` | 3 | `-DPIPE=4` | 环形缓冲槽数，也就是可以同时驻留的 A/B tile 数。不改指令形状 |
| `GMMA_INST_N` | 128 | `-DGMMA_INST_N=64` | 一条 WGMMA 的 N，可选 8、16、32、64、128、256。默认 `PEAK_CTA_N=128` 时 64 和 128 都能过；256 要同时把 `PEAK_CTA_N` 设成 256 |
| `WGMMA_MAX_INFLIGHT` | 2 | `-DWGMMA_MAX_INFLIGHT=1` | 每个计算 warpgroup 允许多少批 WGMMA 未完成。1：算完再释放槽。2：`wait<1>` 让当前批和上一批重叠。等于 2 时 `PIPE` 至少为 2 |
| `EPILOGUE_COALESCED` | 1 | `-DEPILOGUE_COALESCED=0` | 1：累加器进共享内存，再以 16 字节写 C。0：计算线程直接 `axpby`。只改写回 |
| `WGMMA_COMPUTE_THREADS` | 256 | `-DWGMMA_COMPUTE_THREADS=128` | 只作用于 `fast` 的计算线程，必须是 128 的倍数，加上固定的 128 个 producer 不超过 1024。`teach` 忽略它，计算线程始终是 128 |

`fast` 的 producer 固定 128 个线程，其中只有 thread 0 发 TMA。旧套件也没有 producer 宏。`teach` 不单独留 producer warpgroup。

只改流水槽，其余保持 `fast` 默认（CTA 128×128×64，指令 N=128，两批在途，合并写回，256 个计算线程）：

```bash
bash build.sh fast --output build/gemm_fast_pipe3
bash build.sh fast --output build/gemm_fast_pipe4 -- -DPIPE=4
./build/gemm_fast_pipe3 --help
./build/gemm_fast_pipe4 --help
./build/gemm_fast_pipe3 4096 4096 4096 N T 2 event 20 20 200
./build/gemm_fast_pipe4 4096 4096 4096 N T 2 event 20 20 200
```

`gemm_fast_pipe4 --help` 里是 `PIPE=4`，不传 `-D` 的那份是 `PIPE=3`。

只改指令 N。默认 CTA_N=128，64 和 128 都能整除。每个 K tile 的 WGMMA 条数是 `(128/64)×(128/INST_N)×(64/16)`：N=128 时 8 条，N=64 时 16 条。

```bash
bash build.sh teach --output build/gemm_teach_n128
bash build.sh teach --output build/gemm_teach_n64 -- -DGMMA_INST_N=64
./build/gemm_teach_n128 --help
./build/gemm_teach_n64 --help
```

`fast` 上只改计算线程。CTA_M=128 仍能被一个 warpgroup 的 64 整除，所以 128 和默认 256 都能编过。这是默认 `fast`（v5 线程数）和 v4 线程数的差别，其余宏不动：

```bash
bash build.sh fast --output build/gemm_fast_t256
bash build.sh fast --output build/gemm_fast_t128 -- -DWGMMA_COMPUTE_THREADS=128
./build/gemm_fast_t256 --help
./build/gemm_fast_t128 --help
```

## 目录

```text
benchmark_host.cu          # 解析参数 → 检查 → prepare → 采样 → 打印
build.sh                   # 编译 teach、fast，以及全输出校验程序
gemm/
  config.cuh               # 默认宏；-D 可以覆盖
  smem_tma.cuh             # SharedStorage 和 TMA/SW128 布局
  epilogue.cuh             # 合并写回（EPILOGUE_COALESCED=1）
  gemm_teach.cuh           # 单 warpgroup 流水线
  gemm_fast.cuh            # producer + 可配数量的计算 warpgroup
  launch.cuh               # prepare() 建 TMA；operator() 只负责启动
host/
  common.cuh               # 问题规模、NT/TN 下标、矩阵缓冲
  driver.cuh               # 命令行、检查、结果打印
  measurement.cuh          # 预热和计时，不包含描述符构造
  validation.cuh           # 抽样 FP64 对照
tests/
  full_output_validation.cu
  run_validation.sh
archive/v1-v5/             # 原来的 v1–v5 源码，不是默认路径
experiments/               # 可选的 v1–v5 对照、NCU、nsys、sanitizer
```

## 编译和运行

这套代码验证过的工具链是 **CUDA 12.6（nvcc 12.6）**、支持 C++17 的主机 **g++**、**CUTLASS 4.x** 头文件。编译器必须是接受 `sm_90a` 的 CUDA 12 `nvcc`。有些机器上的 `/usr/bin/nvcc` 版本很旧，把 `CUDACXX` 指到 CUDA 12 的可执行文件。

CUTLASS 的查找顺序：`--cutlass DIR`，然后 `$CUTLASS_PATH`，然后 `<仓库根>/third_party/cutlass`（含有 `include/cute/tensor.hpp` 的那一层）。`$CUTLASS_ROOT` 只是 `$CUTLASS_PATH` 的旧名字，脚本仍接受。

nvcc 的查找顺序：`--nvcc PATH`，然后 `$CUDACXX`，然后 `/usr/local/cuda/bin/nvcc`，最后是 `PATH` 里的 `nvcc`。

下面的命令都在仓库根目录执行。

```bash
git clone --depth 1 --branch v4.3.3 https://github.com/NVIDIA/cutlass.git third_party/cutlass
# 或者：export CUTLASS_PATH=/path/to/cutlass
export CUDACXX=/usr/local/cuda/bin/nvcc

bash build.sh all
./build/gemm_teach --help
./build/gemm_fast  --help

# 基准程序自带抽样校验。下面是短跑，用来看对错，不是性能结果。
./build/gemm_teach 256 256 64 N T 3 event 1 1 2
./build/gemm_teach 256 256 832 T N 3 event 1 1 2
./build/gemm_fast  256 256 64 N T 3 event 1 1 2
./build/gemm_fast  256 256 832 T N 3 event 1 1 2

# 性能采样。分配和 TMA 描述符不在计时区间里。
./build/gemm_fast 4096 4096 4096 N T 2 event 20 20 200

# 先做主机侧 MMA 归属检查；不加 --compile-only 时，Hopper 上会再核对全部输出。
bash tests/run_validation.sh --compile-only
bash tests/run_validation.sh
```

`bash build.sh teach` 或 `bash build.sh fast` 只编译一个基准程序。加上 `--validate` 会再编译 `build/validate_teach` 或 `build/validate_fast`。`all` 总会编译两个基准程序和两个校验程序。

运行参数可以从尾部省略，默认是：

```text
M    N    K    transA transB input_mode timer per_batch min_warmup batches
4096 4096 4096 N      T      2          event 20        20         200
```

`input_mode`：`0` 全零，`1` 全一，`2` 随机 ±1，`3` 周期 −2..2。转置组合只能是 `N T` 或 `T N`。

## 从 v1–v5 迁到 teach / fast

旧文档在仓库根目录使用 `build_benchmark.sh --kernel v1` … `v5`，以及 `bash build.sh --suite ...` 和 `bash run.sh`。这些入口已经不在仓库根目录。根目录的 `build.sh` 只接受 `teach`、`fast`、`all`。

对应关系：

| 旧内核 | 旧行为 | 现在用什么 |
|---|---|---|
| v3 | 一个 warpgroup，两批在途，合并写回，套件里的 `GMMA_INST_N=128` | `teach` |
| v5 默认 | 128 个 producer + 256 个计算线程，同样的 tile / PIPE / 写回 / 指令 | `fast` |
| v1 | 一个 warpgroup，每批算完再释放，直接写回 | `bash build.sh teach -- -DWGMMA_MAX_INFLIGHT=1 -DEPILOGUE_COALESCED=0`，或下面的归档命令 |
| v2 | 一个 warpgroup，两批在途，直接写回 | `bash build.sh teach -- -DEPILOGUE_COALESCED=0`，或归档 |
| v4 | 128 个 producer + 128 个计算线程 | `bash build.sh fast -- -DWGMMA_COMPUTE_THREADS=128`，或归档。不传这个宏时 `fast` 是 256 个计算线程 |
| v5 且计算线程不是 256，或 tile 不是 128×128×64 | 宏可配 | `bash build.sh fast -- -DWGMMA_COMPUTE_THREADS=... -DPEAK_CTA_M=...`，或归档 |

当年单内核构建不传 `-DGMMA_INST_N` 时，头文件默认是 64。套件 `build.sh --suite core` 则强制 128。这个分裂已经去掉：不传 `-D` 时，`teach` / `fast` 和归档头文件的默认都是 `64×128×16`。要回到 N=64 的指令，在主路径或归档构建里写 `-DGMMA_INST_N=64`。

先在仓库根目录准备依赖，后面的新旧命令都用它：

```bash
export CUTLASS_PATH=/path/to/cutlass
export CUDACXX=/usr/local/cuda/bin/nvcc
```

旧文档里的 `cd .../matmul/modular` 就是现在的仓库根目录。`$CUTLASS_ROOT` 仍能用，新命令统一写 `$CUTLASS_PATH`。

### v3 → teach

旧写法：

```bash
bash build_benchmark.sh --kernel v3 --output build/gemm_v3 \
  -- -DPIPE=3 -DWGMMA_MAX_INFLIGHT=2 -DEPILOGUE_COALESCED=1 -DGMMA_INST_N=128
./build/gemm_v3 4096 4096 4096 N T 2 event 20 20 200
```

新写法：

```bash
bash build.sh teach
./build/gemm_teach 4096 4096 4096 N T 2 event 20 20 200
```

上面旧命令里的宏值就是现在不传 `-D` 时的默认值。要扫别的 `PIPE`、在途批数、写回或指令 N，把同样的 `-D` 放在 `bash build.sh teach --` 后面。TN 把运行参数里的 `N T` 改成 `T N`。

### v5 默认 → fast

旧写法（256 个计算线程，`PEAK_CTA_M=128`）：

```bash
bash build_benchmark.sh --kernel v5 --output build/gemm_v5 \
  -- -DPIPE=3 -DWGMMA_MAX_INFLIGHT=2 -DEPILOGUE_COALESCED=1 \
     -DWGMMA_COMPUTE_THREADS=256 -DPEAK_CTA_M=128 -DGMMA_INST_N=128
./build/gemm_v5 4096 4096 4096 N T 2 event 20 20 200
```

新写法：

```bash
bash build.sh fast
./build/gemm_fast 4096 4096 4096 N T 2 event 20 20 200
```

### v1、v2、v4 的归档源码

主路径上的对应宏在上面的对照表里。要编译原来的 v1/v2/v4 源文件时，用 `experiments/build_benchmark.sh`。`--kernel` 仍是 `v1` 到 `v5`，运行参数和以前相同。

旧写法：

```bash
bash build_benchmark.sh --kernel v1 --output build/gemm_v1
./build/gemm_v1 4096 4096 4096 N T 2 event 20 20 200

bash build_benchmark.sh --kernel v2 --output build/gemm_v2 \
  -- -DPIPE=3 -DWGMMA_MAX_INFLIGHT=2
./build/gemm_v2 4096 4096 4096 N T 2 event 20 20 200

bash build_benchmark.sh --kernel v4 --output build/gemm_v4 \
  -- -DPIPE=3 -DWGMMA_MAX_INFLIGHT=2 -DEPILOGUE_COALESCED=1
./build/gemm_v4 4096 4096 4096 N T 2 event 20 20 200
```

归档写法（在仓库根目录执行）：

```bash
bash experiments/build_benchmark.sh --kernel v1 --output build/gemm_v1
./build/gemm_v1 4096 4096 4096 N T 2 event 20 20 200

bash experiments/build_benchmark.sh --kernel v2 --output build/gemm_v2 \
  -- -DPIPE=3 -DWGMMA_MAX_INFLIGHT=2
./build/gemm_v2 4096 4096 4096 N T 2 event 20 20 200

bash experiments/build_benchmark.sh --kernel v4 --output build/gemm_v4 \
  -- -DPIPE=3 -DWGMMA_MAX_INFLIGHT=2 -DEPILOGUE_COALESCED=1
./build/gemm_v4 4096 4096 4096 N T 2 event 20 20 200
```

归档的 v1 固定是一批在途和直接写回，多传 `-DWGMMA_MAX_INFLIGHT` 不会改变它。归档的 v4 把计算线程固定成 128。主路径上对应的是上面的 `teach` / `fast` 宏，不是改归档头文件。

### v5 的非默认线程数

三个计算 warpgroup、`PEAK_CTA_M=192` 时，`M` 必须是 192 的倍数，下面用 3072。主路径和归档都能编出这一组宏。

旧写法：

```bash
bash build_benchmark.sh --kernel v5 --output build/gemm_v5_384 \
  -- -DPIPE=3 -DWGMMA_MAX_INFLIGHT=2 -DEPILOGUE_COALESCED=1 \
     -DWGMMA_COMPUTE_THREADS=384 -DPEAK_CTA_M=192
./build/gemm_v5_384 3072 4096 4096 N T 2 event 20 20 200
```

新写法，主路径：

```bash
bash build.sh fast --output build/gemm_fast_384 \
  -- -DWGMMA_COMPUTE_THREADS=384 -DPEAK_CTA_M=192
./build/gemm_fast_384 3072 4096 4096 N T 2 event 20 20 200
```

同一组宏的归档内核：

```bash
bash experiments/build_benchmark.sh --kernel v5 --output build/gemm_v5_384 \
  -- -DPIPE=3 -DWGMMA_MAX_INFLIGHT=2 -DEPILOGUE_COALESCED=1 \
     -DWGMMA_COMPUTE_THREADS=384 -DPEAK_CTA_M=192
./build/gemm_v5_384 3072 4096 4096 N T 2 event 20 20 200
```

### 套件：core / all / bench / ncu

旧写法在仓库根目录：

```bash
bash build.sh --suite all --with-validation
bash run.sh --mode validate --gpu 0
bash run.sh --mode bench --gpu 0 --variants v1,v2,v3,v4,v5
bash run.sh --mode ncu --gpu 0 --variants v1,v2,v3,v4,v5
```

新写法有两条。只比较现在的两条主路径时：

```bash
bash build.sh all
./build/gemm_teach 4096 4096 4096 N T 2 event 20 20 200
./build/gemm_fast  4096 4096 4096 N T 2 event 20 20 200
bash tests/run_validation.sh
```

还要编 v1–v5、扫 PIPE 或做 NCU 时，脚本在 `experiments/`。可以从仓库根目录这样调用，不必先 `cd`。默认产物目录是 `experiments/build/h100_pcie`，`run.sh` 默认也读这个目录。

```bash
bash experiments/build.sh --suite core
bash experiments/run.sh --mode bench --gpu 0 --variants v1,v2,v3,v4,v5

bash experiments/build.sh --suite all --with-validation
bash experiments/run.sh --mode validate --gpu 0
bash experiments/run.sh --mode ncu --gpu 0 --variants v1,v2,v3,v4,v5
```

`--suite core` 是 v1–v5，CTA `128×128×64`，`PIPE=3`，`GMMA_INST_N=128`。`--suite all` 再加上 10 个 PIPE / 线程对照。`--with-validation` 之后，`--mode validate` 才会找到每个配置的 `.validate` 程序。

也可以进入目录再跑，效果相同：

```bash
cd experiments
bash build.sh --suite core --with-validation
bash run.sh --mode validate --gpu 0
```

### 全输出校验脚本

旧写法：

```bash
bash tests/run_warp_specialized_tests.sh --cutlass "$CUTLASS_ROOT" --nvcc "$CUDACXX"
```

这个脚本现在转去编译并检查 `teach` 和 `fast`，不再编以前那六组 v4/v5 宏组合。新写法：

```bash
bash tests/run_validation.sh --cutlass "$CUTLASS_PATH" --nvcc "$CUDACXX"
bash build.sh teach --validate
./build/validate_teach --mapping-only
./build/validate_fast
```

`--mapping-only` 只在主机上检查线程是否盖住 CTA tile，不启动内核。

## 建议阅读顺序

1. `benchmark_host.cu`：`prepare()` 返回一个对象，调用它就提交一次 GEMM。
2. `gemm/config.cuh`：默认宏，以及按 `GMMA_INST_N` 选择的 SS 指令。
3. `gemm/gemm_teach.cuh`：一个 warpgroup 的预取。默认两批在途、合并写回；`if constexpr` 保留一批在途和直接写回。
4. `gemm/gemm_fast.cuh`：producer leader、可配数量的计算 warpgroup、尾部排空。默认两批在途时，最后一批在 `wait<0>` 之后再 arrive 一次 empty。
5. `gemm/epilogue.cuh` 和 `gemm/smem_tma.cuh`：两边共用。直接写回不经过 `epilogue.cuh`。

CuTe 仍负责 TMA 描述符、SW128 布局和 TiledMMA。`Fp16SsAtom` 按编译期的 `GMMA_INST_N` 选具体的 `SM90_64xNx16` 类型。

## 校验在查什么

基准程序会把 C 拷回主机，拒绝 NaN/Inf，再用 CPU FP64 点积比较 16 个分散位置。容差是 `1 + 2% * abs(reference)`。这是抽样，不是全矩阵证明。

`tests/full_output_validation.cu` 对 `teach` 和 `fast` 都检查每一个输出：NT 和 TN、四种输入，以及 PIPE 深度附近的 K tile 数（含多次 phase 翻转）。

## 归档实验

`archive/v1-v5/` 保留原来的五个入口。设备调度没有改写，只修正了搬目录之后的一个 include，并把 `GMMA_INST_N` 的默认值改成 128。NCU、nsys 和 sanitizer 的说明在 [experiments/H100_BENCHMARK.md](experiments/H100_BENCHMARK.md)。
