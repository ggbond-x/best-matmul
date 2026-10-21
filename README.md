# Hopper FP16 GEMM

这里共用命令行、数据生成、校验与测量流程，在编译时选择 v1/v2/v3/v4/v5 内核。计算约定为 FP16 输入和输出、`alpha=1`、`beta=0`，支持 NT/TN。

## H100 PCIe 五版本对照实验

新增 [build.sh](build.sh) 和 [run.sh](run.sh)，分别负责统一编译与测试。完整配置表、脚本参数、实验顺序、NCU 指标解释和诊断命令见 [H100_BENCHMARK.md](H100_BENCHMARK.md)。在本目录的 Linux/Bash 中执行：

```bash
# 编译全部 15 个配置（5 个基础版本 + 10 个参数对照）及全输出校验程序。
# 默认自动定位项目 CUTLASS 和 /usr/local/cuda/bin/nvcc，也可显式传 --cutlass/--nvcc。
bash build.sh --suite all --with-validation

# 先在 H100 上校验，再独立计时；报告目录自动按时间新建。
bash run.sh --mode validate --gpu 0
bash run.sh --mode bench --gpu 0 --variants v1,v2,v3,v4,v5

# 默认抓取 4096^3、NT/TN 的一次预热后 GEMM，生成 .ncu-rep、CSV 和分析文本。
bash run.sh --mode ncu --gpu 0 --variants v1,v2,v3,v4,v5
```

`build.sh` 默认 `--suite core`，使用统一的 CTA=`128×128×64`、`GMMA_INST_N=128`、`PIPE=3`；这与下方单版本示例中 `GMMA_INST_N=64` 的配置不同。`run.sh` 从编译生成的 `manifest.tsv` 读取实际配置。正式性能只取未被 profiler 插桩的 `bench` 结果，NCU 和 Sanitizer 日志中的耗时不参与汇总。

## 阅读顺序

先看 `benchmark_host.cu` 的 `main` 和 `run_benchmark`，再根据关心的部分进入对应文件：

```text
modular/
├── benchmark_host.cu                   # 解析 → 检查 → 准备 → 测量 → 输出
├── build_benchmark.sh                  # 选择内核、检查依赖、调用 nvcc
├── build.sh                            # H100 PCIe 统一编译、参数对照与编译诊断
├── run.sh                              # 校验、独立计时、NCU、Sanitizer、SASS
├── H100_BENCHMARK.md                    # 对比设计、命令、参数范围与报告解读
├── host/
│   ├── common.cuh                      # 参数、NT/TN 索引、矩阵缓冲
│   ├── driver.cuh                      # 命令行、参数检查、结果输出
│   ├── measurement.cuh                 # 单批计时、有限预热、采样、统计
│   └── validation.cuh                  # 回读、checksum、CPU 抽样参考计算
├── kernels/
│   ├── wgmma_tma_fp16_v1.cuh            # 保守调度的版本入口
│   ├── wgmma_tma_fp16_pipeline_v2.cuh   # 流水线的版本入口
│   ├── wgmma_tma_fp16_epilogue_v3.cuh   # 流水线和合并写回的版本入口
│   ├── wgmma_tma_fp16_warp_specialized_v4.cuh # 独立 TMA producer/计算 consumer
│   ├── wgmma_tma_fp16_multi_warpgroup_v5.cuh # 可配置多个计算 warpgroup
│   └── detail/
│       ├── wgmma_config.cuh            # CTA/PIPE 参数、WGMMA 指令类型
│       ├── warp_specialized_config.cuh # v4/v5 的线程分工及配置约束
│       ├── tma_wgmma_device.cuh        # v1–v3 按阶段展开的设备流程
│       ├── tma_wgmma_warp_specialized_device.cuh # v4/v5 producer/consumer 流程
│       └── tma_wgmma_adapter.cuh       # CPU 侧构造 TMA、配置和启动内核
└── tests/
    ├── full_output_validation.cu      # 全输出数值校验、线程到输出的映射检查
    └── run_warp_specialized_tests.sh  # 多配置编译和校验入口
```

设备核心按“CTA 张量视图 → TMA 预取 → 寄存器累加器 → K 维主循环 → C 写回”组织。CUTE 所需模板保留在布局和指令选择处；版本差异由编译期分支选择。

| 版本 | 主循环与线程分工 | 写回 |
|---|---|---|
| v1 | 每个 K tile 提交后 `wait<0>`，完成后释放槽位 | 直接 `axpby` |
| v2 | 默认两批在途，`wait<1>` 后释放最旧槽，末尾 `wait<0>` | 直接 `axpby` |
| v3 | 与 v2 相同 | 默认经共享内存重排，以 16 字节向量连续写回 |
| v4 | 基于 v3；128 个 producer 线程负责 TMA，128 个 consumer 线程执行 WGMMA | 沿用 v3 的合并写回 |
| v5 | 基于 v4；consumer 默认 256 个线程，多个 warpgroup 沿 CTA 的 M 维分工 | 沿用合并写回，覆盖所有计算 warpgroup 的结果 |

`PIPE` 是共享内存槽位数，默认 `3`；`WGMMA_MAX_INFLIGHT` 是每个计算 warpgroup 的 WGMMA 在途批数。v2–v5 默认 `WGMMA_MAX_INFLIGHT=2`，也可设为 `1`；设为 `2` 时要求 `PIPE>=2`。v3–v5 默认 `EPILOGUE_COALESCED=1`，可设为 `0` 比较直接写回。v1–v3 的线程组织与行为保持不变。

v4/v5 为 producer 保留一个完整 warpgroup，其中只有一个 leader 线程等待可写槽位、设置事务字节数并发起 TMA；其余 producer 线程不执行 WGMMA。主循环结束后，合并写回可使用整个 CTA 的线程搬运输出。

v4 每个 CTA 共 `256=128+128` 个线程，计算线程固定为 `128`。v5 通过编译期参数 `WGMMA_COMPUTE_THREADS` 控制计算线程数，默认 `256`，所以每个 CTA 默认共 `384=128+256` 个线程。v5 的该参数须为正的 `128` 倍数；单条 WGMMA 始终由一个 `128` 线程 warpgroup 协同执行。v5 的多个计算 warpgroup 分摊同一个 CTA tile，因此须满足：

- `PEAK_CTA_M` 可被 `64*(WGMMA_COMPUTE_THREADS/128)` 整除。
- `128+WGMMA_COMPUTE_THREADS<=1024`。
- 所选 CTA、`PIPE` 和写回方式的共享内存及寄存器需求适合目标 GPU；满足线程数和 tile 约束并不保证一定可启动。

默认 CTA tile 为 `128×128×64`，适合 v5 的默认两个计算 warpgroup。若改为 `384` 个计算线程，可将 `PEAK_CTA_M` 改为 `192`；若改为 `512` 个，可将其改为 `256`。`M/N/K` 仍须分别为对应 CTA tile 的整数倍。

## 编译和运行

在本目录执行 Linux / WSL Bash 命令。编译器需要支持 C++17、`sm_90a` 和当前 CUTLASS 头文件；运行内核需要 Hopper GPU。仅编译、查看 `--help` 或运行主机侧映射检查不要求有 Hopper。

本项目已验证的编译环境是 **CUDA Toolkit 12.6、nvcc V12.6.85、g++ 9.4.0**，本地 CUTLASS 头文件版本为 **4.3.3**。这是本地已通过编译的组合，其他工具链版本需另行验证。`/usr/local/cuda` 当前指向 `/usr/local/cuda-12.6`；系统另有 `/usr/bin/nvcc`，版本为 CUDA 10.1，不能用于本项目，因此应显式选择编译器。

单版本入口 `build_benchmark.sh` 的编译器选择顺序为 `--nvcc PATH` → `CUDACXX` → `PATH` 中的 `nvcc`；CUTLASS 路径选择顺序为 `--cutlass DIR` → `CUTLASS_ROOT`。主机 C++ 编译器由 nvcc 选择，如需指定，可在该入口的 `--` 后传 `-ccbin /path/to/g++`；脚本不会读取 `CXX` 来替代该选项。对照实验入口 `build.sh` 还会优先于 PATH 尝试 `/usr/local/cuda/bin/nvcc`，并在未指定 CUTLASS 时自动查找项目的第三方目录，详见实验文档。

```bash
# 本地已验证的依赖路径；在其他机器上替换成实际安装位置。
cd /home/myCode/cutlass_study/peak_flops/src/sm90/nsight_compute/matmul/modular
export CUTLASS_ROOT=/home/myCode/cutlass_study/peak_flops/third_party/cutlass
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc

"$CUDACXX" --version
g++ --version
```

构建脚本固定使用 `-O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_90a`，自动添加 CUTLASS 头文件搜索路径并定义 `GEMM_KERNEL_HEADER`。`--` 后的内容作为额外 nvcc 参数传入，例如 `-DPIPE=3`、`-lineinfo` 或 `-Xptxas=-v`；这些参数仍须符合所选 nvcc 的规则。

### 参数总表

下表汇总本项目识别或生成的构建参数、编译期参数、运行参数及测试选项。编译期参数用 `-- -D名称=值` 传入，更换后需要重新编译；运行参数按表中位置顺序传入，只能从尾部省略。`INT_MAX=2147483647`。

| 类别 / 适用入口 | 参数 | 默认值 | 作用 | 可配置范围与约束 |
|---|---|---|---|---|
| 构建脚本 | `--kernel` | `v1` | 选择编译的内核 | `v1`、`v2`、`v3`、`v4`、`v5` 或符合后端接口的头文件路径；相对路径以 `modular/` 为基准，文件须存在，路径不能含双引号或换行 |
| 构建 / 测试脚本 | `--cutlass DIR` / `CUTLASS_ROOT` | 无，必须提供其中一个 | 指定 CUTLASS 根目录 | 包含 `include/cute/tensor.hpp` 等所需头文件的目录；命令行选项优先于环境变量 |
| 构建 / 测试脚本 | `--nvcc PATH` / `CUDACXX` | `PATH` 中的 `nvcc` | 指定 CUDA 编译器 | 支持本项目 C++17、SM90a 代码和当前 CUTLASS 的可执行文件；优先级为命令行、环境变量、默认名称；本地验证为 CUDA 12.6.85 |
| 构建脚本 | `--output PATH` | `modular/build/<内核头文件名去掉扩展名>` | 指定输出二进制 | 可写文件路径；显式相对路径以调用时的工作目录为基准，父目录自动创建 |
| 构建脚本 | `--` 后的额外参数 | 空 | 将后续参数原样交给 nvcc | 可传 `-D`、`-ccbin`、`-lineinfo` 等所选 nvcc 接受的选项；脚本选项须放在 `--` 之前 |
| nvcc 固定选项 | `-O3` | 固定开启 | 开启优化 | 脚本固定使用；未提供项目级优化等级开关 |
| nvcc 固定选项 | `-std=c++17` | 固定使用 | 选择 C++17 语言标准 | 当前代码使用 C++17 功能，保持此设置 |
| nvcc 固定选项 | `--expt-relaxed-constexpr` | 固定开启 | 允许主机与设备代码间的 constexpr 调用，供 CuTe 模板使用 | 保持开启 |
| nvcc 固定选项 | `-arch=sm_90a` | 固定使用 | 为 Hopper 的架构专属指令生成代码 | 本项目固定目标；改为其他架构不构成可用的移植 |
| nvcc 自动选项 | `-I` | CUTLASS 的 `include/`，存在时加 `tools/util/include/` | 指定头文件搜索路径 | 由 CUTLASS 根目录自动生成；额外搜索路径可通过 `-- -I...` 添加 |
| nvcc 自动宏 | `GEMM_KERNEL_HEADER` | 由 `--kernel` 解析得到 | 决定公共主机入口包含哪个内核头文件 | 由脚本自动定义为头文件路径字符串，使用 `--kernel` 配置，无需重复传 `-D` |
| nvcc 可选参数 | `-ccbin PATH` | nvcc 默认的主机 C++ 编译器 | 指定主机编译器或其目录 | 在 `--` 后传入；必须与所选 CUDA Toolkit 兼容，本地验证为 g++ 9.4.0 |
| 编译期，v1–v5 | `GMMA_INST_N` | `64` | 选择单条 WGMMA 的 N 维度，指令形状为 `64×N×16` | `{8,16,32,64,128,256}`；`PEAK_CTA_N` 必须是它的倍数；此参数不控制计算线程数 |
| 编译期，v1–v5 | `PEAK_CTA_M` | `128` | 每个 CTA 负责的输出行数 | 正的 `64` 倍数；v5 还须为 `64*(WGMMA_COMPUTE_THREADS/128)` 的倍数；受共享内存、寄存器及 CuTe/TMA 布局限制 |
| 编译期，v1–v5 | `PEAK_CTA_N` | `128` | 每个 CTA 负责的输出列数 | 当前 SW128 布局要求为正的 `64` 倍数，同时须为 `GMMA_INST_N` 的倍数，即 `lcm(64,GMMA_INST_N)` 的正倍数；受资源和布局限制 |
| 编译期，v1–v5 | `PEAK_CTA_K` | `64` | 每个 K tile 的归约长度 | 当前同时编译 NT/TN 的 SW128 布局要求为正的 `64` 倍数；配置文件中的基础断言仅检查 `16` 的倍数，不能单独据此判断布局可用；受资源和布局限制 |
| 编译期，v1–v5 | `PIPE` | `3` | A/B 的共享内存环形缓冲槽数 | 整数 `>=1`；`WGMMA_MAX_INFLIGHT=2` 时须 `>=2`；未设独立固定上限，实际受共享内存等资源限制 |
| 编译期，v2–v5 | `WGMMA_MAX_INFLIGHT` | `2` | 每个计算 warpgroup 的最大在途 WGMMA 批数 | `{1,2}`；`2` 要求 `PIPE>=2`；v1 固定为 `1`，传宏不会改变 v1 的调度 |
| 编译期，v3–v5 | `EPILOGUE_COALESCED` | `1` | 是否经共享内存重排并以 16 字节向量合并写回 | `{0,1}`：`0` 为直接写回，`1` 为合并写回；v1/v2 固定直接写回，不使用此宏 |
| 编译期，v4/v5 | `WGMMA_COMPUTE_THREADS` | v4：`128`；v5：`256` | 控制执行 WGMMA 的计算线程数，多个 warpgroup 沿 M 分工 | v4 仅允许 `128`；v5 允许 `{128,256,384,512,640,768,896}`，并须满足 M tile 约束；另有固定 `128` 个搬运线程，总数不超过 `1024`；v1–v3 不使用此宏 |
| 运行位置 1 | `M` | `4096` | 逻辑矩阵 A、C 的行数 | `1..INT_MAX`，且必须是 `PEAK_CTA_M` 的正整数倍；矩阵元素数量还须满足下述联合约束 |
| 运行位置 2 | `N` | `4096` | 逻辑矩阵 B 的行数、C 的列数 | `1..INT_MAX`，且必须是 `PEAK_CTA_N` 的正整数倍；受矩阵元素数量联合约束 |
| 运行位置 3 | `K` | `4096` | A/B 的归约维长度 | `1..INT_MAX`，且必须是 `PEAK_CTA_K` 的正整数倍；受矩阵元素数量联合约束 |
| 运行位置 4 | `transA` | `N` | 选择 A 的物理布局 | 大写 `N` 或 `T`；与 `transB` 组合仅支持 `N T`、`T N` |
| 运行位置 5 | `transB` | `T` | 选择 B 的物理布局 | 大写 `T` 或 `N`；与 `transA` 组合仅支持 `N T`、`T N` |
| 运行位置 6 | `input_mode` | `2` | 选择输入生成方式 | `{0,1,2,3}`：全零、全一、随机 `±1`、周期 `-2..2` |
| 运行位置 7 | `timer` | `event` | 选择计时范围 | `event` 为 GPU Event 时间间隔；`wall` 为主机提交到流完成的墙钟时间，区分大小写 |
| 运行位置 8 | `per_batch` | `20` | 每个测量批次的内核启动次数 | `1..INT_MAX`，并受下述预热计数联合约束；单个样本为本批总耗时除以启动次数 |
| 运行位置 9 | `min_warmup` | `20` | 最低预热启动次数 | `1..INT_MAX`，并受预热计数联合约束；实际预热次数由稳定性判断决定 |
| 运行位置 10 | `batches` | `200` | 正式测量的样本批数 | `2..INT_MAX`，且需有足够主机内存保存样本；不包含预热批次 |
| 基准程序 | `--help` / `-h` | 不启用 | 显示编译配置及运行参数 | 单独传入，无需 GPU 查询或启动内核 |
| 构建 / 测试脚本 | `--help` / `-h` | 不启用 | 显示对应脚本的参数说明 | 无值开关，输出帮助后退出 |
| 测试脚本 | `--output-dir DIR` | `modular/build/warp_specialized_tests` | 指定六组测试二进制的输出目录 | 可写目录；父目录自动创建，相对路径以调用时工作目录为基准 |
| 测试脚本 | `--compile-only` | 不启用 | 编译所有测试配置并运行 CPU 线程映射检查 | 无值开关；启用后跳过 GPU 数值测试，否则继续执行 Hopper 全输出校验 |
| 单个测试二进制 | `--mapping-only` | 不启用 | 仅检查 CuTe 线程到输出坐标的归属 | 作为唯一选项传入；无需 Hopper，未传时还会执行 GPU 数值校验 |

参数范围中的“正倍数”是必要条件，满足后仍须通过 CuTe 模板实例化、TMA 描述符构造和目标 GPU 的资源检查。当前 SW128 原子布局对 FP16 的形状分别为 MN-major 的 `64×8` 和 K-major 的 `8×64`；同一个二进制会实例化 NT/TN 两条路径，因此最终 N、K 的布局约束比基础静态断言更严格。

运行时还检查 `M*K<=INT_MAX`、`N*K<=INT_MAX`、`M*N<=INT_MAX`，显存须能容纳三份矩阵。预热参数须满足 `min_warmup + 21*per_batch <= INT_MAX`，防止启动计数溢出。`alpha=1`、`beta=0`、FP16 输入/累加/输出，以及 v4/v5 的 `128` 个搬运线程都是当前实现固定值，没有对应的运行参数或 `-D` 开关。

### 各版本编译和运行示例

先在同一个终端执行上面的目录切换和环境变量设置。下面每组先编译再运行对应二进制；v1–v5 使用相同的矩阵尺寸、NT 布局和测量参数，便于比较。运行命令需要在 Hopper 上执行。

#### v1：基础调度

```bash
bash build_benchmark.sh --kernel v1 --output build/gemm_v1
./build/gemm_v1 4096 4096 4096 N T 2 event 20 20 200
```

#### v2：WGMMA 流水线

```bash
bash build_benchmark.sh --kernel v2 --output build/gemm_v2 \
  -- -DPIPE=3 -DWGMMA_MAX_INFLIGHT=2
./build/gemm_v2 4096 4096 4096 N T 2 event 20 20 200
```

#### v3：流水线与合并写回

```bash
bash build_benchmark.sh --kernel v3 --output build/gemm_v3 \
  -- -DPIPE=3 -DWGMMA_MAX_INFLIGHT=2 -DEPILOGUE_COALESCED=1
./build/gemm_v3 4096 4096 4096 N T 2 event 20 20 200
```

#### v4：Warp specialization

使用 `128` 个搬运线程和 `128` 个 WGMMA 计算线程，每个 CTA 共 `256` 个线程。

```bash
bash build_benchmark.sh --kernel v4 --output build/gemm_v4 \
  -- -DPIPE=3 -DWGMMA_MAX_INFLIGHT=2 -DEPILOGUE_COALESCED=1
./build/gemm_v4 4096 4096 4096 N T 2 event 20 20 200
```

#### v5：可配置计算 warpgroup 数量

此例使用 `256` 个计算线程，加上 `128` 个搬运线程，每个 CTA 共 `384` 个线程。

```bash
bash build_benchmark.sh --kernel v5 --output build/gemm_v5 \
  -- -DPIPE=3 -DWGMMA_MAX_INFLIGHT=2 -DEPILOGUE_COALESCED=1 \
     -DWGMMA_COMPUTE_THREADS=256 -DPEAK_CTA_M=128
./build/gemm_v5 4096 4096 4096 N T 2 event 20 20 200
```

v5 扩展示例：使用 `384` 个计算线程（三个计算 warpgroup），M tile 同时改为 `192`；运行时 M 必须是 `192` 的倍数，此处取 `3072`。

```bash
bash build_benchmark.sh --kernel v5 --output build/gemm_v5_384 \
  -- -DPIPE=3 -DWGMMA_MAX_INFLIGHT=2 -DEPILOGUE_COALESCED=1 \
     -DWGMMA_COMPUTE_THREADS=384 -DPEAK_CTA_M=192
./build/gemm_v5_384 3072 4096 4096 N T 2 event 20 20 200
```

所有版本也支持 TN 布局，将运行参数中的 `N T` 改为 `T N` 即可。下面分别给出帮助命令和 TN 运行示例：

```bash
./build/gemm_v1 --help
./build/gemm_v5 4096 4096 4096 T N 2 event 20 20 200
```

也可用 `--cutlass DIR`、`--nvcc PATH` 显式指定依赖。`--kernel` 接受别名或头文件路径：相对内核路径以 `modular/` 为基准；显式 `--output` 相对当前工作目录。默认输出为 `modular/build/<内核文件名去掉扩展名>`。`--` 后的参数原样交给 nvcc，例如 `-DPEAK_CTA_N=256 -DGMMA_INST_N=64`。

运行参数允许从尾部省略，默认值为：

```text
M    N    K    transA transB input_mode timer per_batch min_warmup batches
4096 4096 4096 N      T      2          event 20        20         200
```

`input_mode`：`0=全零`、`1=全一`、`2=随机 ±1`、`3=周期 -2..2`。布局只支持 `N T` 或 `T N`；尺寸必须分别为所选 CTA 的 M/N/K tile 的整数倍。逻辑矩阵统一为 A(M,K)、B(N,K)、C(M,N)，物理布局由 `common.cuh` 的索引和适配器的 stride 对应。

## 测量与校验

流程为“预检查 → 预热 → 正式采样 → 后检查 → 汇总”。每个样本是“一批启动的总耗时 / 启动次数”。`per_batch` 控制每批启动次数，`batches` 控制样本数；p50/p99 是这些批均耗时的百分位，TFLOPS 使用 `2*M*N*K` 计算。

`min_warmup` 是最低启动次数。每 5 批比较相邻窗口的中位数，连续两次变化不超过 2% 才确认稳定。默认最早在第 15 批确认，即 300 次启动；最多运行 `ceil(min_warmup/per_batch)+20` 批，仍不稳定时输出警告并标记结果。

Event 模式测 GPU 时间线间隔；Wall 模式测 CPU 提交到流完成的时间。分配、描述符构建、GPU 状态查询和校验都在计时之外。比较内核时应保持问题、输入、计时模式和采样参数一致。

校验会回读全部 C、拒绝 NaN/Inf，再用 CPU FP64 点积检查分散的 4×4 个位置，允许误差为 `1 + 2%*abs(reference)`。这是抽样检查和 FP16 经验容差，不代表全矩阵数值保证。

v4/v5 应先检查短 K 和环形缓冲 phase 翻转。以下命令使用默认 CTA、`PIPE=3` 和两批在途，分别覆盖 1/2/3/4 个 K tile，以及 13 个 K tile 的多次槽位复用；NT/TN 都运行内置校验。`per_batch/min_warmup/batches` 减小以便检查正确性，不用于比较稳定性能：

```bash
for binary in build/gemm_v4 build/gemm_v5; do
  for k in 64 128 192 256 832; do
    for layout in 'N T' 'T N'; do
      read -r trans_a trans_b <<< "$layout"
      "$binary" 256 256 "$k" "$trans_a" "$trans_b" 3 event 1 1 2
    done
  done
done

# 单槽、单批在途也应通过同样的短 K / phase 翻转检查。
bash build_benchmark.sh --kernel v5 --output build/gemm_v5_pipe1 \
  -- -DPIPE=1 -DWGMMA_MAX_INFLIGHT=1 -DWGMMA_COMPUTE_THREADS=256
./build/gemm_v5_pipe1 256 256 64 N T 3 event 1 1 2
./build/gemm_v5_pipe1 256 256 832 T N 3 event 1 1 2

# 三个计算 warpgroup：M 取 192 的倍数，并检查两个布局。
./build/gemm_v5_384 384 256 832 N T 3 event 1 1 2
./build/gemm_v5_384 384 256 832 T N 3 event 1 1 2
```

`tests/` 另提供专用全输出校验：覆盖 v4 的单槽/三槽配置及 v5 的 `128/256/384/512` 个计算线程，组合两种写回方式和在途批数。GPU 检查遍历全部 C，覆盖 NT/TN、四种输入、短 K 和多次环形缓冲复用；线程到输出的映射也会在主机侧检查。

```bash
# 只编译，并运行主机侧映射检查；不启动 GPU 内核。
bash tests/run_warp_specialized_tests.sh \
  --cutlass "$CUTLASS_ROOT" --nvcc "$CUDACXX" --compile-only

# 在 Hopper 上编译并运行完整数值校验。
bash tests/run_warp_specialized_tests.sh \
  --cutlass "$CUTLASS_ROOT" --nvcc "$CUDACXX"
```

## 修改或新增内核

只调整已有开关时，直接传 `-D` 参数。修改算法时：

1. 复制一个版本入口为 `kernels/my_pipeline.cuh`，修改 `kName`。
2. 复制 `kernels/detail/tma_wgmma_device.cuh` 为 `kernels/detail/my_device.cuh`，让新入口包含它，再修改设备算法。
3. 保留 `SharedStorage`、`gemm_device` 的模板及参数接口，可继续复用适配器。改变线程组织、共享内存、TMA 布局或参数时，同步修改适配器。
4. 用 `--kernel kernels/my_pipeline.cuh` 编译，无需修改主机入口或登记别名。

复用 v1–v3 设备文件时，先定义 `kMaxInflight` 和 `kCoalescedEpilogue` 再包含它；v4/v5 由 `warp_specialized_config.cuh` 提供这两个值及线程分工常量。复制对应版本入口即可保留正确顺序。更换整个实现时，公共主机只要求 `gemm_backend` 提供：

| 接口 | 职责 |
|---|---|
| `kInterfaceVersion=1`、`kName` | 接口版本、日志名称 |
| `validate_problem(problem)` | 内核专属约束 |
| `print_configuration()` | 内核编译配置 |
| `with_prepared(options, buffers, runner)` | 准备后调用 `runner(launch, launch_config)` 并返回结果 |

`launch()` 向 `options.measurement.stream` 提交一次完整 GEMM，不做参数准备、分配或同步。描述符和参数必须存活到 `runner` 返回；不要将捕获它们的回调保存到外部。模板设备代码在同一翻译单元实例化，无需可分离编译。

各配置的数值正确性和性能仍需在 Hopper 上验证，特别是短 K、不同 PIPE 和 CTA 组合。
