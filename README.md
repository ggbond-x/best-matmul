# Hopper FP16 GEMM

独立的 Hopper FP16 GEMM 教学与基准仓库（CUDA 12.6、CUTLASS/CuTe、`sm_90a`）。主路径只有两个内核：`teach` 和 `fast`。它们共用同一套 tile、同一条 WGMMA 指令和同一份合并写回。

输入、累加和输出都是 FP16，`alpha=1`，`beta=0`。逻辑形状是 A(M,K)、B(N,K)、C(M,N)，布局只支持 NT 和 TN。

## 默认配置

唯一的默认值在 `gemm/config.cuh`。根目录 `build.sh` 不会用 `-D` 改它们。

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

`M`、`N`、`K` 必须分别是 128、128、64 的倍数。empty mbarrier 的到达计数等于计算线程数。`fast` 里 producer 会排空自己用过的每个槽，最后一批 WGMMA 在 `wait<0>` 之后再对 empty 做一次 arrive。

## 目录

```text
benchmark_host.cu          # 解析参数 → 检查 → prepare → 采样 → 打印
build.sh                   # 编译 teach、fast，以及全输出校验程序
gemm/
  config.cuh               # 唯一的默认值
  smem_tma.cuh             # SharedStorage 和 TMA/SW128 布局
  epilogue.cuh             # 一份合并写回
  gemm_teach.cuh           # 单 warpgroup 流水线
  gemm_fast.cuh            # producer + 两个计算 warpgroup
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
| v1 | 一个 warpgroup，每批算完再释放，直接写回 | 没有对应的 teach/fast，用下面的归档命令 |
| v2 | 一个 warpgroup，两批在途，直接写回 | 同上，归档 |
| v4 | 128 个 producer + 128 个计算线程 | 同上，归档。`fast` 是 256 个计算线程，不能代替 v4 |
| v5 且计算线程不是 256，或 `PEAK_CTA_M` 不是 128 | 宏可配 | 归档。`teach` / `fast` 的 tile 和线程数是常量 |

当年单内核构建不传 `-DGMMA_INST_N` 时，头文件默认是 64。套件 `build.sh --suite core` 则强制 128。这个分裂已经去掉：`teach` / `fast` 和归档头文件的默认都是 `64×128×16`。要回到 N=64 的指令，只能在归档构建里显式写 `-DGMMA_INST_N=64`。

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

`-DPIPE`、`-DWGMMA_MAX_INFLIGHT`、`-DEPILOGUE_COALESCED`、`-DGMMA_INST_N` 对 `teach` 不起作用，这些数写死在 `gemm/config.cuh`。上面旧命令里的宏值就是现在的固定值。TN 把运行参数里的 `N T` 改成 `T N`。

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

### v1、v2、v4 仍要旧调度

这三条和 `teach` / `fast` 的线程或写回不一样。归档脚本是 `experiments/build_benchmark.sh`，`--kernel` 仍是 `v1` 到 `v5`，运行参数和以前相同。

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

新写法（归档，在仓库根目录执行）：

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

v1 固定是一批在途和直接写回，多传 `-DWGMMA_MAX_INFLIGHT` 不会改变它。v4 的计算线程固定是 128；`fast` 不能拿来跑这条。

### v5 的非默认线程数

三个计算 warpgroup、`PEAK_CTA_M=192` 的例子不在 `teach` / `fast` 里。`M` 必须是 192 的倍数，下面用 3072。

旧写法：

```bash
bash build_benchmark.sh --kernel v5 --output build/gemm_v5_384 \
  -- -DPIPE=3 -DWGMMA_MAX_INFLIGHT=2 -DEPILOGUE_COALESCED=1 \
     -DWGMMA_COMPUTE_THREADS=384 -DPEAK_CTA_M=192
./build/gemm_v5_384 3072 4096 4096 N T 2 event 20 20 200
```

新写法：

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
2. `gemm/config.cuh`：tile、PIPE、指令。
3. `gemm/gemm_teach.cuh`：一个 warpgroup 的预取、两批在途 WGMMA、合并写回。
4. `gemm/gemm_fast.cuh`：producer leader、两个计算 warpgroup、尾部排空、最后一次 empty arrive。
5. `gemm/epilogue.cuh` 和 `gemm/smem_tma.cuh`：两边共用。

CuTe 仍负责 TMA 描述符、SW128 布局和 TiledMMA。指令 N 已经固定，没有按 N 选择类型的模板。

## 校验在查什么

基准程序会把 C 拷回主机，拒绝 NaN/Inf，再用 CPU FP64 点积比较 16 个分散位置。容差是 `1 + 2% * abs(reference)`。这是抽样，不是全矩阵证明。

`tests/full_output_validation.cu` 对 `teach` 和 `fast` 都检查每一个输出：NT 和 TN、四种输入，以及 PIPE 深度附近的 K tile 数（含多次 phase 翻转）。

## 归档实验

`archive/v1-v5/` 保留原来的五个入口。设备调度没有改写，只修正了搬目录之后的一个 include，并把 `GMMA_INST_N` 的默认值改成 128。NCU、nsys 和 sanitizer 的说明在 [experiments/H100_BENCHMARK.md](experiments/H100_BENCHMARK.md)。
