# cuBLAS / cuBLASLt GEMM Benchmark

用于对比自定义 Tensor Core、CUTLASS、CuTe、WGMMA/TMA MatMul 的实测库基线。支持连续 **Row-Major / Column-Major** 矩阵，执行：

C = alpha × op(A) × op(B) + beta × C

本目录名称是 **cuBLAS _test**，cuBLAS 与下划线之间有一个空格。Shell 路径请加引号。

## 1. 目录与实现

~~~text
cuBLAS _test/
├── CMakeLists.txt
├── include/gemm/
│   ├── config.hpp          参数、逻辑尺寸、行跨度
│   ├── cuda_utils.hpp      CUDA/cuBLAS 错误检查与资源释放
│   └── backend.hpp         共用 GemmPlan 接口与精度定义
├── src/
│   ├── config.cpp          CLI 解析、内置测试集合
│   ├── cublas_gemm.cpp     cuBLAS GemmEx 与行列主序映射
│   ├── cublaslt_gemm.cpp   Row-Major 描述符、候选选择、调优
│   └── data.cu             FP16/BF16 转换、上传与下载
├── benchmark/main.cpp      设备检测、内存、计时、CPU 参考、CSV
├── scripts/
│   ├── cases.json          可修改的默认矩阵集合
│   ├── run_suite.py        串行批量运行、独立进程、实验清单
│   ├── run_csv_cases.sh    按 cuBLAS测试结果.csv 一键构建并运行
│   ├── run_csv_cases.py    CSV 校验、双后端调度和结果回填
│   └── compare.py          严格匹配语义与计时条件、计算性能比值
├── tests/
│   ├── gpu_smoke.py        92 个真实 GPU 回归用例
│   ├── test_tools.py       对比工具与批量配置测试
│   ├── cli_checks.py       非法参数与无 GPU 跳过路径
│   ├── gpu_edge_cases.py   显存、尺寸溢出与 Workspace 保护
│   └── cases_smoke.json    批量脚本的小型验证集合
├── results/
│   ├── example_schema.csv  完整 CSV 表头，可用作自定义结果格式
│   └── ...                 实际运行生成的 CSV / manifest / log
└── VALIDATION.md           本机实际验证记录与未验证范围
~~~

| CLI dtype | A/B 存储 | 乘法/累加计算 | 默认 C 存储 | 后端 |
|---|---|---|---|---|
| fp16 | FP16 | FP16 输入、FP32 累加 | FP16 | cuBLAS / cuBLASLt |
| bf16 | BF16 | BF16 输入、FP32 累加 | BF16 | cuBLAS / cuBLASLt |
| fp32 | FP32 | 严格 FP32，PEDANTIC | FP32 | cuBLAS / cuBLASLt |
| tf32 | FP32 | 允许 TF32 乘法、FP32 累加 | FP32 | cuBLAS / cuBLASLt |

FP16/BF16 可加 --output-dtype fp32，避免最终输出转换。CSV 单独记录 InputDtype、OutputDtype、Compute 和 Reduction。项目不启用 fast-math，不把 FP32 自动替换成 TF32，不含 FP8 扩展。

## 2. 编译

要求 CUDA Toolkit >= 11.0、支持该 Toolkit 的 C++17 编译器、CMake >= 3.16；Python >= 3.8 用于脚本和测试，无第三方 Python 包依赖。实际已在 CUDA 12.6 编译验证。GPU 架构必须是所选 nvcc 支持的架构，例如 Hopper SM90 需要足够新的 Toolkit。

本机默认 /usr/bin/nvcc 是 CUDA 10.1，必须明确选择 CUDA 12.6：

~~~bash
cd "/home/myCode/cutlass_study/peak_flops/src/sm90/nsight_compute/cuBLAS _test"

cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DGEMM_CUDA_ARCHITECTURES="80;86;90"
cmake --build build -j 4
~~~

H100 专用构建可设 -DGEMM_CUDA_ARCHITECTURES=90；RTX 3050 Laptop 可用 86。本工程调用的 cuBLAS 二进制会自行选择设备对应内核；这个 CMake 选项控制本项目 CUDA 源码的编译目标，**不会把 cuBLAS 强制固定成某个内核**。

改变 CUDA Toolkit 时使用新的 build 目录，避免 CMake 缓存混用旧头文件和库。CMake 从所选 nvcc 的 Toolkit 中查找 cudart、cublas、cublasLt，不混用系统 CUDA 10.1 的库。Ubuntu 20.04 的 CMake 3.16 所需 CUDA17 标志已补齐。

Windows 原生 CUDA + Visual Studio 也可用 CMake 构建，指定 nvcc.exe 的路径；可执行文件通常位于 build/Release/gemm_bench.exe。**Windows 原生构建尚未实测**，本次使用 WSL/Linux。

## 3. 单项运行

~~~bash
./build/gemm_bench --help
./build/gemm_bench --device-info

# 默认 FP16，C 也为 FP16，FP32 累加
./build/gemm_bench --m 4096 --n 4096 --k 4096 \
  --backend cublas --dtype fp16 \
  --warmup 20 --repeats 50 --inner 10 \
  --csv results/cublas_fp16.csv

# BF16 输入、FP32 输出
./build/gemm_bench --m 1024 --n 4096 --k 4096 \
  --dtype bf16 --output-dtype fp32 --csv results/bf16_fp32out.csv

# FP32 严格计算和 TF32 分开输出
./build/gemm_bench --m 2048 --n 2048 --k 2048 \
  --dtype fp32 --csv results/fp32.csv
./build/gemm_bench --m 2048 --n 2048 --k 2048 \
  --dtype tf32 --csv results/tf32.csv

# 检查非方阵、转置、alpha/beta；全量 CPU FP64 参考
./build/gemm_bench --m 37 --n 53 --k 29 \
  --transa T --transb N --alpha -0.75 --beta 0.5 \
  --check-all --csv results/transpose.csv

# cuBLASLt：最多 32 个 heuristic 候选中实测选择，64 MiB workspace
./build/gemm_bench --backend cublaslt --dtype fp16 \
  --m 4096 --n 4096 --k 4096 \
  --workspace-mb 64 --lt-candidates 32 --lt-selection tune \
  --csv results/lt_tuned.csv

# 选择本次 heuristic 返回列表中第 0 个候选
./build/gemm_bench --backend cublaslt --dtype bf16 \
  --lt-selection index --lt-index 0 --workspace-mb 32 \
  --csv results/lt_index.csv

# Graph 是单独的实验模式；捕获 inner 个串行 GEMM
./build/gemm_bench --backend cublaslt --dtype fp16 \
  --m 16 --n 4096 --k 4096 --mode graph --inner 50 \
  --csv results/lt_graph.csv
~~~

CSV 默认追加；已有文件表头不匹配时拒绝写入。不要让多个进程同时写同一个 CSV。批量脚本默认拒绝已有非空输出，需显式 --append 才追加。

## 4. 物理布局与转置

逻辑 op(A) 是 M×K，op(B) 是 K×N，输出 C 是 M×N。输入缓冲区存储的是 **执行 op 前** 的矩阵；T 标志不能仅当作改变库参数而不改变实际存储形状。

连续 Row-Major 矩阵的 leading dimension 是每行的元素数：

| TransA/TransB | A 存储形状 | B 存储形状 | 用户 lda | 用户 ldb | 用户 ldc |
|---|---|---|---|---|---|
| NN | M×K | K×N | K | N | N |
| NT | M×K | N×K | K | K | N |
| TN | K×M | K×N | M | N | N |
| TT | K×M | N×K | M | K | N |

同一块 Row-Major X 内存可以解释成 Column-Major Xᵀ。对整个等式转置：

Cᵀ = alpha × op(B)ᵀ × op(A)ᵀ + beta × Cᵀ

因此传统 cuBLAS 调用使用：

~~~cpp
cublasGemmEx(handle,
    transB, transA, N, M, K, &alpha,
    B, inputType, ldb,
    A, inputType, lda, &beta,
    C, outputType, N,
    computeType, CUBLAS_GEMM_DEFAULT);
~~~

交换 A/B、M/N，保留每个操作数自己的转置标志。对 cuBLAS 而言第一个参数叫 lda，但传的是用户 B 的 ldb；第二个参数传用户 A 的 lda。上述四种组合都适用，没有显式矩阵转置和额外复制。

cuBLASLt 直接设置 CUBLASLT_ORDER_ROW，描述符写入表中的实际存储形状与行跨度，再设置 TRANSA/TRANSB。C 和 D 使用同一指针，保证与 cuBLAS 一样的原地 C 更新语义。

使用 `--layout column_major` 时，传统 cuBLAS 直接按 `M,N,K` 和 `A,B,C` 顺序调用；cuBLASLt 使用 `CUBLASLT_ORDER_COL`。连续 Column-Major 的 leading dimensions 为：

| TransA/TransB | A 存储形状 | B 存储形状 | lda | ldb | ldc |
|---|---|---|---:|---:|---:|
| NN | M×K | K×N | M | K | M |
| NT | M×K | N×K | M | N | M |
| TN | K×M | K×N | K | K | M |
| TT | K×M | N×K | K | N | M |

当前接口只支持连续矩阵；CSV 记录 LDA/LDB/LDC。添加带 padding 的矩阵时，应同时扩展分配、参考索引和描述符，不能只修改 API 的 leading dimension。

## 5. 精度约定

FP16 与 BF16 是 **输入/输出格式**，FP32 累加是 **中间计算精度**，二者不可混为一谈。FP16 有 10 位显式尾数、BF16 有 7 位；BF16 指数范围更接近 FP32。FP32 累加不能恢复输入量化已经丢失的信息。默认最终 C 还会转换回 FP16/BF16。

严格 FP32 使用 CUBLAS_COMPUTE_32F_PEDANTIC；传统 cuBLAS 同时设置 PEDANTIC math mode。TF32 显式使用 CUBLAS_COMPUTE_32F_FAST_TF32：输入/输出内存仍是 FP32，但允许把乘法输入舍入到 TF32 的较短尾数，在支持的设备上可能更快。它不能作为严格 FP32 的等价精度结果。

低精度路径使用 CUBLAS_COMPUTE_32F。cuBLAS 禁止 reduced-precision reduction；Lt 的 reduction mask 只允许 compute-type reduction，且检查 split-K 候选不使用低精度输出类型归约。因此不会为提高分数而偷偷把 FP32 累加/归约改为 FP16/BF16。

Tensor Core 使用由库按设备、尺寸、布局与算法决定，尤其小尺寸/不对齐矩阵不保证一定选择 Tensor Core。TF32 也是允许使用的计算模式，不保证每个尺寸都以 TF32 内核执行；严格确认实际指令需要另外用 Nsight Compute。NVIDIA_TF32_OVERRIDE=0 时 TF32 用例明确 SKIP，而不是把禁用 TF32 的结果标为正常 TF32 测试。

参考：NVIDIA [cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/index.html)，尤其 Compute Type、GEMM Algorithms Numerical Behavior、cuBLASLt Matrix Layout 和 Heuristic API 章节。最终 API 可用性以本机 Toolkit 头文件为准。

## 6. 计时协议

1. 检测 GPU、驱动/runtime/cuBLAS 版本，预估显存并保留库加载余量。
2. 创建独立 non-blocking stream；分配内存；生成可复现数据；转换精度；上传。
3. 创建 Handle、描述符、Workspace，执行算法查询/调优。
4. 至少一次不计时的 GEMM，处理库的延迟初始化。Graph 捕获、实例化和首次 replay 也在计时外。
5. 执行 warmup 个 batch。每个 batch 包含 effective_inner 个串行 GEMM。
6. 同一个 stream 上记录 start event → GEMM batch → stop event。
7. 每个测量样本只等待 stop event，再取 EventElapsedTime / effective_inner。循环内部没有每 GEMM 的 CPU 同步。
8. repeats 个样本计算均值、中位数、最小/最大、P95 和总体标准差。
9. 在计时外下载、检查结果并写 CSV。

公式：

~~~text
FLOPs = 2 × M × N × K
TFLOPS = FLOPs / (mean_ms × 10^9)
~~~

没有把分配、初始化、H2D/D2H、句柄创建、算法选择、CPU 参考计算算入时间。算法本身包含的多个 GPU kernel、必要的 split-K reduction 等属于一次 GEMM 工作，应包含在 GEMM 的耗时中。

**普通 launch 的 CUDA Event 区间可能包含 host enqueue 不及时造成的 GPU 空闲间隔。**它测的是 stream 上这一段 GEMM 的完成时间，不能宣称是 profiler 逐 kernel duration 之和。短算子建议同时查看 min/median、增大 inner，并单独测 graph。Graph 结果不得和 launch 直接匹配比较。

默认复用 A/B/C，属于热缓存/重复操作数测试；没有做冷缓存刷新。温度、功耗限制、动态频率、共享 GPU 负载、WSL 调度均可能影响结果。本工具不会修改 GPU 时钟和功耗设置。建议在目标 GPU 空闲、热状态稳定时保存多次独立运行，并记录机器环境。

beta=0 默认每次覆盖输出，允许 inner>1。**beta 非零强制 effective_inner=1**，每次测量前在同一 stream 上 D2D 恢复 C0，复制位于 start event 之前。Graph 同样只捕获一个 GEMM，恢复操作不在图内。这样每个样本都执行相同的 alpha*AB+beta*C0，不发生多次递推累加。恢复仍会影响缓存，因此记录 BetaPolicy，只有相同策略才可比较。

alpha=0、beta 特殊值可能触发库的快捷路径；此时惯用的 2MNK 计数不代表实际执行了这些乘加，不宜用来评估 GEMM 算力。

## 7. 正确性检查

输入由固定种子的计数器 hash 生成 [-0.5, 0.5) 的有限数；A/B/C 使用不同 salt。CPU 保留 **实际量化后的值**，参考计算用 FP64 累加，独立实现逻辑转置，不调用另一个 GEMM 库充当参考。

默认 M×N <= 4096 全量检查；更大的矩阵抽查最多 512 个位置，包含角落/中心和固定种子的随机位置。--check-all 强制全量，代价为 O(MNK)，大矩阵可能非常慢。--check-samples N 控制抽样数。整个输出仍会扫描 NaN/Inf。PASS 对大矩阵只代表指定抽样验证通过，不能当作全量证明。

每个检查位置使用：

~~~text
abs(actual - reference)
  <= atol + rtol*abs(reference)
     + (8*epsilon_fp32 + tf32_input_round_bound)*abs(alpha)*sum(abs(a_k*b_k))
~~~

TF32 输入误差项为约 2^-10，其它模式为 0；相消严重的点因此不会仅因 reference 接近零被误判。默认 (atol, rtol) 按输出类型选择：

| C 输出 | atol | rtol |
|---|---|---|
| FP32 | 1e-5 | 2e-5 |
| FP16 | 5e-4 | 1e-3 |
| BF16 | 4e-3 | 8e-3 |

--atol/--rtol 可覆盖前两项，FP32 累加与 TF32 输入的条件相关项仍保留。这个界限偏保守，尤其 TF32 的大 K 相消情况；CSV 同时记录 MaxAbsError、RelativeL2、Checked、Bad，便于进一步审查。验证不是逐 bit 比较，也不是任意输入分布/溢出行为的形式化证明。

## 8. cuBLASLt 算法选择

- first：依 heuristic 顺序选择第一个通过状态、Workspace、精度与实际地址试运行检查的候选。
- index：选择返回列表中的指定零基索引，不是 cuBLASLt 全局算法 ID。
- tune：候选先预热，再各测 5 个样本，以中位数选择；使用与正式 benchmark 相同的 launch/graph 和 beta 策略。此过程全部在正式计时之外。

候选数量由 --lt-candidates 限定。调优不是穷举全部算法，不保证全局最优。最终选中算法由独立 CPU 参考检查，错误会使结果 FAIL，不会把错误结果当成优化成功。

CSV 保存 heuristic 索引、实际候选数量、algorithm ID/tile/stages/split-K/reduction/swizzle/custom/numerical flags、Workspace 预算与实际需求、库版本和 GPU UUID。复现实验请保留这些字段与批量 manifest；相同索引在不同库版本/硬件上不代表同一算法。当前不提供跨版本 opaque 算法对象回放。

Workspace 只直接配置 cuBLASLt；传统 cuBLAS 使用库管理的 workspace，CSV 明确标为 library_managed。环境变量 CUBLAS_WORKSPACE_CONFIG 也会记录。

## 9. 批量集合与命令

### 按 `cuBLAS测试结果.csv` 一键执行

下面的入口严格读取 CSV 中的 48 行 ColMajor 用例，并分别运行 cuBLASEx 与 cuBLASLt，共 96 个独立进程。默认针对 H100 编译 SM90，使用每行的 FP16/BF16、转置、Alpha/Beta、32 MiB Workspace、Warmup=50 和 Repeat=200；`inner=1`，对应 CSV 中一次采样一个 GEMM。cuBLASLt 默认在 32 个候选内进行计时选择。

~~~bash
cd /home/myCode/cutlass_study/peak_flops/src/sm90/nsight_compute/cuBLAS_test

# 只校验 CSV 并打印 96 条命令，不编译、不创建输出目录。
bash scripts/run_csv_cases.sh --dry-run

# 一键构建并执行；结果写入 outptu/cublas_csv_<时间戳>_<PID>/。
CUDA_HOME=/usr/local/cuda GPU_ARCHS=90 \
  bash scripts/run_csv_cases.sh
~~~

每次运行保存 `cuBLAS测试结果.csv`（回填后的同结构表）、`raw_results.csv`、`index.csv`、`run.log`、`build.log`、`manifest.json`、`run_info.txt` 和 `status.txt`。可使用 `LT_SELECTION=first` 跳过 cuBLASLt 候选实测选择，或用 `RUN_DIR` 指定准确输出目录。输出根目录按任务要求拼写为 `outptu`。

### 通用 JSON 集合

默认 20 个不同形状，三元组顺序均为 **(M, N, K)**：

- 方阵：256、512、1024、2048、4096、8192。
- 非方阵：(1024,4096,4096)、(4096,1024,4096)、(4096,4096,1024)、(8192,4096,4096)。
- LLM：M=1、4、16、64、128、512、2048，N=K=4096；小 M 对应 Decode，大 M 对应 Prefill。
- 非对齐：(127,255,513)、(1000,1031,997)、(33,4097,4093)。

这是典型形状集合，不绑定某一特定模型。可直接修改 scripts/cases.json，或 --cases 指向同样结构的 JSON。

~~~bash
# 一个后端/精度的内置集合
./build/gemm_bench --suite all --backend cublas --dtype fp16 \
  --csv results/fp16_all.csv

# 默认两个后端、四种精度、20 个形状，共 160 个进程；NN、launch
python3 scripts/run_suite.py --csv results/baseline.csv

# LLM + 特殊尺寸，增加四种转置、单独记录 launch/graph
python3 scripts/run_suite.py --groups llm irregular \
  --backends cublas cublaslt --dtypes fp16 bf16 \
  --transposes NN NT TN TT --modes launch graph \
  --csv results/llm.csv \
  --extra --warmup 20 --repeats 50 --inner 20 --lt-selection tune

# 先列出命令；不运行 GPU
python3 scripts/run_suite.py --dry-run

# 小规模验证批量脚本
python3 scripts/run_suite.py --cases tests/cases_smoke.json \
  --dtypes fp16 bf16 --csv results/batch_smoke.csv \
  --extra --warmup 2 --repeats 4 --inner 3
~~~

Python runner 串行执行，避免不同 case 竞争 GPU；每个 case 独立进程，单个执行错误不会污染后续 CUDA context。保存命令、二进制和源码 SHA256、环境、nvidia-smi、返回码到带时间戳的 manifest.json，同时保存日志。

显存不足、没有 GPU、不支持的精度或没有可用 Lt 算法输出 SKIP+原因，时间/TFLOPS 留空。FAIL 表示参考不匹配或真正执行错误。native --suite 遇到 CUDA 执行错误会停止该进程以避免继续使用受损 context；支持不足和 OOM 则继续。主程序退出码：0 至少一个 PASS 且无 FAIL；1 存在 FAIL；2 参数/全局设置错误；3 全部跳过。

## 10. CSV 与自定义算子对比

每个 case 一行。完整字段见 results/example_schema.csv，包含设备、输入输出精度、计算模式、leading dimensions、alpha/beta、事件计时协议、warmup/repeats/inner、统计、校验、算法、Workspace、逐样本 Samples_ms。

以下为本机 RTX 3050 Laptop **实际回归测试**中的节选，不是 H100 性能，也不是正式稳定性能报告：

~~~csv
GPU,Backend,Dtype,M,N,K,TransA,TransB,Mode,Time_ms,Median_ms,TFLOPS,Status
NVIDIA GeForce RTX 3050 Laptop GPU,cublaslt,FP16,1024,1024,1024,N,N,launch,0.187221328417,0.186709324519,11.470293829,PASS
~~~

对比工具使用 Time_ms（样本平均耗时），不是 Median_ms：

~~~text
Performance Ratio = Custom TFLOPS / Baseline TFLOPS
Speedup = Baseline Time_ms / Custom Time_ms
~~~

使用相同 FLOP 定义时这两个值应一致；>1 表示自定义算子更快。工具检查 CSV 的 TFLOPS 与时间是否一致，允许 0.5% 的打印舍入差异，随后由时间重算吞吐，避免混用不同统计量。

~~~bash
python3 scripts/compare.py \
  --baseline results/baseline.csv \
  --custom results/my_matmul.csv \
  --output results/comparison.csv --strict

# 只选择传统 cuBLAS 作为参照
python3 scripts/compare.py --baseline results/baseline.csv \
  --custom results/my_matmul.csv --output results/vs_cublas.csv \
  --baseline-backend cublas
~~~

--baseline-backend all 默认把 cuBLAS 和 cuBLASLt 分开输出两条对比，不静默混成一个参考。只比较 PASS 行。缺少必要语义字段会报错，不猜测自定义算子的精度。

自定义 CSV 必须填写 scripts/compare.py 中 KEY_FIELDS，尤其 GPU/GPU_UUID/SM、Dtype/InputDtype/OutputDtype、Compute、Reduction、Layout、M/N/K、TransA/B、LDA/B/C、Alpha/Beta、Mode、TimingProtocol、CachePolicy、BetaPolicy、Warmup/Repeats/Inner/Seed。字段约定与本程序保持一致。例如：

- FP16 输入/输出、FP32 累加：Dtype=FP16，InputDtype=FP16，OutputDtype=FP16，Compute=FP32_ACC，Reduction=FP32。
- 严格 FP32：Compute=FP32_PEDANTIC；仅当自定义内核确实保证严格 FP32 时才使用该标签。
- TF32：Dtype=TF32，InputDtype=FP32，OutputDtype=FP32，Compute=TF32_FP32_ACC。
- 计时遵循本文协议时才写 TimingProtocol=cuda_event_batch_v1。
- Backend 写你的算子名称；Time_ms/TFLOPS 必须填你的实际测量结果，Status 只能在你完成校验后标为 PASS。

只有实际语义相符时才能填写同样标签；不能靠修改 CSV 标签把不公平的实验变成可比较结果。量化、padding、不同归约精度、不同缓存策略均需明确区分。

重复匹配记录默认报错，避免自动挑最快结果。--duplicates median 可对相同算法/Workspace/版本的重复实验取各次平均时间的中位数；算法或 Workspace 配置不同仍拒绝合并。未匹配行会输出 UNMATCHED；--strict 使部分未匹配时返回 4。完全没有匹配或缺少字段时返回错误。

若希望把自定义 kernel 接入同一计时器，可派生 GemmPlan 并实现 run()：构造时准备资源，run() 只在传入 stream 上异步执行同样布局/精度的 GEMM，不能分配内存或自行同步。在 config.cpp 增加 backend 选项，在 main.cpp 的 factory 分支构造该 plan，即可复用数据、Events、Graph、校验及 CSV 流程。当前未附带一个冒充高性能自定义 kernel 的示例实现。

**cuBLAS 的测量值是库实现的基线，不是 GPU 理论峰值。**自定义内核超过某个 cuBLAS case 并不意味着超过硬件峰值。

## 11. 测试与限制

~~~bash
# 不需要可见 GPU：工具、参数、无 GPU 路径
(cd build && ctest --output-on-failure)

# 需要 GPU：92 个真实回归，涵盖四种转置、非零 beta、Graph、输出类型和 Lt 调优
python3 tests/gpu_smoke.py --exe build/gemm_bench \
  --csv results/gpu_regression_new.csv

# 可选把 GPU 回归注册进 CTest
cmake -S . -B build -DGEMM_ENABLE_GPU_TESTS=ON
(cd build && ctest --output-on-failure)
~~~

精度支持：FP16 Tensor Core 基线要求 SM70+；BF16/TF32 要求 SM80+。不支持时跳过；不做悄悄退回其它输入格式。自定义 kernel 需要自行满足相同要求。

本次真实验证范围见 VALIDATION.md。H100/WGMMA/TMA 性能、8192 大矩阵完整集合、长期稳态功耗、Windows 原生构建及不同 Toolkit 版本未在本机验证。采样检查不能覆盖每个大矩阵元素。没有自动锁频、NVML 温度采样或多进程 CSV 写锁；需要这些功能时可在当前接口上扩展。


