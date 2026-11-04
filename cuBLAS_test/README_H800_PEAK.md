# H800 PCIe FP16 Dense GEMM 峰值、功耗与时钟测试

此入口与原有 gemm_bench、run_csv_cases.sh、run_cublas_all.sh 并存。不会覆盖历史结果，不修改 Power Limit、锁频、持久化模式或其他 GPU 配置。

## 快速运行

在仓库根目录：

~~~bash
bash matmul/best-matmul/cuBLAS_test/scripts/run_h800_peak.sh --gpu 0 --mode all
~~~

脚本定位自身目录；也可以从任意目录用脚本的绝对路径调用。参数：

| 参数 | 默认 | 含义 |
|---|---|---|
| --gpu | 0 | nvidia-smi 的物理 GPU 编号 |
| --mode | all | scan / sustained / all |
| --quick | 关闭 | 5 个小用例 × 2 API，覆盖非方阵 NN/NT/TN/TT；再做 2 个不同尺寸 × 2 API 的 3 秒持续验证 |
| --dry-run | 关闭 | 只校验 CSV 并打印计划；不需要 GPU，不编译，不创建结果目录 |
| --shapes | csvs/h800_pcie_fp16_shapes.csv | 自定义 Shape 表 |
| --config | csvs/benchmark_config.csv | 自定义参数表 |
| --scan-results | 空 | sustained 模式可从此前 benchmark_summary.csv 选择有效高性能 Shape |
| --skip-build | 关闭 | 使用已编译程序 |
| --exe | build_h800_peak/h800_peak_bench | 自定义可执行程序 |
| --monitor-backend | auto | auto 优先 NVML，失败降级 nvidia-smi；可指定 nvml / nvidia-smi |
| --help | — | 使用说明 |

~~~bash
bash matmul/best-matmul/cuBLAS_test/scripts/run_h800_peak.sh --dry-run
bash matmul/best-matmul/cuBLAS_test/scripts/run_h800_peak.sh --quick
bash matmul/best-matmul/cuBLAS_test/scripts/run_h800_peak.sh --mode scan --gpu 0
bash matmul/best-matmul/cuBLAS_test/scripts/run_h800_peak.sh --mode sustained --scan-results /absolute/prior_run/benchmark_summary.csv
~~~

--quick 明确覆盖部分配置并使用专门的小尺寸表，适合验证流程，不用于报告 H800 峰值。sustained 未提供扫描结果时，使用配置表中的大尺寸 NN 候选，并明确标记它们不是实测最优。

--gpu 的物理编号不受原有 CUDA_VISIBLE_DEVICES 逻辑编号影响：先查询物理 UUID，再通过该 UUID 将子进程绑定为 CUDA device 0；程序返回 UUID 后必须一致才开始监控。原 CUDA_VISIBLE_DEVICES 保存于 environment.json。

## 依赖与构建

- Linux，Python 3.8+，CMake 3.16+，CUDA Toolkit 11+，cuBLAS/cuBLASLt，匹配的 NVIDIA 驱动。
- 推荐 CUDA 12.x 用于 H800。默认编译 SM80、SM90；SM80 cubin 可用于本机 SM86 功能验证。
- 必需 nvidia-smi 用于物理 GPU 身份确认及环境检查。
- nvidia-ml-py 为首选监控依赖；matplotlib 用于 PNG 图表。不自动修改系统 Python。

~~~bash
python3 -m venv /path/to/h800-venv
/path/to/h800-venv/bin/pip install -r /absolute/cuBLAS_test/requirements-h800-peak.txt
PYTHON_BIN=/path/to/h800-venv/bin/python bash /absolute/cuBLAS_test/scripts/run_h800_peak.sh --quick
~~~

如果未安装 NVML 包，会降级到 nvidia-smi；未安装 matplotlib 时 CSV 和报告照常生成，并明确提示图表缺失。安装后可单独重跑分析。脚本不把环境不支持的传感器值填成 0。

可配置环境变量：CUDA_HOME（默认 /usr/local/cuda）、NVCC、GPU_ARCHS（默认 80;90）、JOBS（默认 4）、H800_BUILD_DIR、PYTHON_BIN。例如：

~~~bash
CUDA_HOME=/usr/local/cuda-12.6 GPU_ARCHS=90 bash /absolute/cuBLAS_test/scripts/run_h800_peak.sh
~~~

CMake 新增 h800_peak_bench；复用现有配置结构、CUDA RAII、cuBLAS 后端和 cuBLASLt 后端。后端仅增加可选候选校验回调、候选记录及可用时的 cluster_id 查询；旧调用方不传回调时维持原选择流程。

## 测试集合和数值规范

默认 10 个 M/N/K 组合，每个覆盖四种转置，共 40 个 CSV 用例 × 2 API：

- 1024³、2048³、4096³、8192³、16384³。
- 4096×4096×8192、8192×4096×4096、4096×8192×4096。
- 2048×16384×4096；另加 4096×4096×256 观察短 K。

A、B、C 均 FP16，Compute Type 为 CUBLAS_COMPUTE_32F；alpha=1、beta=0；Dense，不启用结构化稀疏。禁止低精度 split-K reduction，与现有基准的 FP32 reduction 约束一致。FP16 输入不自动等于 Tensor Core 使用证据。

输入采用固定种子的哈希随机数，先量化为 FP16；GPU 初始化与 CPU 参考使用同一确定性函数。每个候选执行后、调优计时前，检查 CPU FP64 点积参考；选中算法在正式计时前与结束后再次校验。默认抽样 1024 个输出位置（含边界），并扫描整个输出的 NaN/Inf。

误差预算：

~~~text
abs_error <= 0.0005 + 0.001 * abs(reference)
             + 8 * float_epsilon * sum(abs(A[k] * B[k]))
~~~

参考使用实际量化后的输入，允许 FP16 输出舍入和 FP32 累加误差。大型矩阵是抽样参考，不宣称全元素参考验证。错误候选不参与调优择优；尾部校验失败使该 run_id 的全部窗口退出排名。candidate_validation.csv 即使在所有候选校验失败时也保留诊断。

Column-major 的物理尺寸和 leading dimension：

| 矩阵 | N | T |
|---|---|---|
| A | M×K，lda=M | K×M，lda=K |
| B | K×N，ldb=K | N×K，ldb=N |
| C | M×N，ldc=M | 不转置 |

分配前检查 A/B/C、workspace、参考缓冲以及显存预留；不足返回 SKIP。配置错误在启动 GPU 前拒绝；运行时 CUDA/cuBLAS 错误记录 FAIL/SKIP，不加入排名。

## 算法选择与实际 Tensor Core 证据

cuBLASGemmEx 使用 CUBLAS_GEMM_DEFAULT。cuBLASLt 请求最多 32 个 heuristic 候选，逐个执行 AlgoCheck，确认合法性、workspace 和 FP32 reduction，执行数值校验，再以 5 个调优批次的中位数择优。API 返回的候选可能少于请求值，实际数量保存在原始结果。

不手动拼接 Tile/Stages/Split-K/Cluster 配置。记录库返回的算法属性；cluster_id 只在头文件和运行库支持时读取。Tile/Stages/Cluster 的数字是枚举 ID，不直接等价于 CTA 的 M/N/K 或流水线层数。workspace_bytes 记录 Lt 实际选择值；GemmEx 未暴露的内部 workspace 留空。

tensor_core_evidence：

- lt_numerical_flags_tensor_core：所选 Lt 算法返回 HMMA 数值实现标志，可确认 Tensor Core 实现家族。
- unverified_requires_profiler：没有可用直接证据。尤其不能仅因 FP16 类型断言 GemmEx 使用 Tensor Core。
- instruction_path 默认 unverified_requires_sass：HMMA 数值标志不能区分传统 MMA 与 Hopper WGMMA。

提供可选独立 Nsight Compute 证据工具：

~~~bash
python3 /absolute/cuBLAS_test/scripts/profile_h800_peak.py /absolute/run_directory --api cublas --dry-run
python3 /absolute/cuBLAS_test/scripts/profile_h800_peak.py /absolute/run_directory --api cublas --ncu /path/to/ncu
~~~

该工具选择指定 API 的最佳有效记录，在独立 logs/profile_<timestamp>/ 下运行短 launch 模式探针。程序通过 cudaProfilerStart/Stop 限定正式调用，排除初始化/校验。NCU 使用 --clock-control none，不锁频；保存真实 tensor 指令计数和 SASS。需要适配设备的 ncu 及硬件计数器权限；不可用时保留 unverified 和错误信息，不修改系统权限。

**Profiler 运行的耗时绝不并入峰值统计。** Lt 会重新搜索，可能选择与原测试不同的算法，应查看 probe_raw.csv；SASS 的指令存在是静态证据，动态 tensor counter >0 才说明探针实际执行过 Tensor Core 指令。

## 计时和阶段

每组默认至少 2 秒和 20 批 warmup，随后校准每个 CUDA Event 批次约 20 ms（inner 限制为 1..4096），避免小尺寸主要测 CPU launch。默认 CUDA Graph，两个 API 使用同一协议；也可在配置中改为 launch。

正式扫描至少 3 秒及 10 个 Event 样本。持续负载默认 45 秒，按 1 秒窗口输出。持续阶段从扫描结果选 3 个不同 M/N/K 的高吞吐用例，并在选中的同一转置下测试两个 API；持续运行也重新校验及调优，记录实际算法。

每个 Event 只包围批量 GEMM，Event 同步后取平均每次 GEMM 的毫秒数。初始化、分配、H2D/D2H、参考计算和调优均排除。summary 从所有批次样本重算 median/mean/min，而不是对窗口中位数再取中位数。

~~~text
FLOPs = 2*M*N*K
TFLOPS(median) = FLOPs / (median_ms * 1e9)
TFLOPS(mean)   = FLOPs / (mean_ms * 1e9)
~~~

phase_events.csv 分别记录 initialization、reference、tuning_validation、warmup、timing_setup、scan/sustained、post_validation。监控与 Benchmark 都使用 Linux CLOCK_MONOTONIC 的纳秒时间，另存 UTC 时间。GPU Event 与主机单调时钟不是同一个时钟域，因此用主机时间包围事件区间，并保存 gpu_event_duty_ratio 说明正式窗口中的 GPU 计时占比。

逐窗口正式时间不含窗口间 CSV 写入间隙。只有整个传感器查询区间都落入窗口且 UUID 匹配的采样才统计。跨窗口、预热、空闲数据不混入正式统计。不能将主机窗口关联当作亚毫秒 GPU/传感器硬件对时。

## 功耗、时钟与限频

默认 150 ms 轮询。记录真实查询开始/结束/耗时，不做追赶式突发采样；NVML 失败优先降级。nvidia-smi 启动进程开销更高，应比较实际采样间隔、占用及 sensor errors。

- GPU utilization、power draw/limit、SM/memory clocks、温度、显存占用和 UUID。
- clock reasons 原始位掩码、名称及单独的 idle/application clock/HW slowdown/power brake 字段。
- Power throttling = SW Power Cap 或 HW Power Brake。
- Thermal throttling = SW Thermal 或 HW Thermal；通用 HW Slowdown 不能直接判断为热限制。
- NVML 可用时记录计算 PID；不可见不代表不存在其他任务。运行前另做 nvidia-smi 进程检查。
- Power Draw ≥ 95% Power Limit 只记为接近上限，不直接视为限频。

统计为有效采样均值/极值和事件采样比例，不宣称精确时间积分。传感器硬件更新周期可能比轮询慢，未知时标注 unknown。

质量规则：窗口/整组少于 3 个完整查询点，或采样时间跨度覆盖不到正式窗口总时长的 50%，为 INSUFFICIENT；无点为 MISSING。每个指标另存有效样本数。缺 reason 为 unknown，少量点未发现事件也为 unknown；只要确实采到事件可标 true，但仍保留采样不足提示。

SIGINT/SIGTERM、子程序失败或超时都会终止子程序、通知监控退出、保留数据并尝试生成报告。SIGKILL/断电不能运行清理逻辑，不能保证完整尾部记录。重分析时缺少完成进程记录的 run_id 排除排名。

## 输出文件

每次运行新建 output/h800_peak_<Asia-Shanghai timestamp>_<pid>/：

| 文件 | 内容 |
|---|---|
| environment.json | 型号/UUID/SM/版本、源文件与程序哈希、配置、环境限制、启动前进程与遥测 |
| benchmark_raw.csv | 每个正式窗口、完整 Event 样本、算法、正确性、双时间戳 |
| gpu_telemetry.csv | 原始传感器查询，不改写缺失值 |
| benchmark_summary.csv | 用户要求的全部字段；按 run_id 汇总 |
| window_summary.csv | 每个持续/扫描窗口对应的 TFLOPS、功耗、频率 |
| phase_events.csv | 正式负载与预热等阶段的准确主机时间范围 |
| telemetry_labeled.csv | 关联后的阶段标签；原始遥测保留 |
| algorithm_trials.csv | 实际测试候选的校验状态、调优中位数、workspace |
| candidate_validation.csv | 每个执行候选的抽样误差和正确性，全部拒绝时仍可诊断 |
| run_index.csv | 命令、PID、退出码、状态和进程时间范围 |
| performance_report.md | 最高成绩、API 对照、限频证据、持续趋势、归因边界 |
| plots/ | 三类散点图及每组持续测试的三联时间趋势 PNG |
| logs/ | 编译、监控、各组 stdout/stderr；可选 profiler 独立输出 |
| CSV 副本 | 本次实际使用的 Shape 和参数快照 |

benchmark_config.csv 是原始参数表副本；--quick 等实际覆盖结果以 environment.json 的 config 为准。图表英文轴标签避免服务器缺中文字体；报告用中文。任何失败记录的 TFLOPS 汇总留空，原始时间数据仍保留。

重新分析不启动 GPU：

~~~bash
python3 /absolute/cuBLAS_test/scripts/analyze_results.py /absolute/run_directory
~~~

## 理论峰值与归因方法

只有型号精确匹配 H800 PCIe、SM=114、compute capability=9.0、可见显存约 80GB 且为物理 GPU UUID，才启用 756.5 TFLOPS 的 Dense 参考。其他型号、MIG/不匹配配置默认留空，不能套 H100 SXM 的峰值。

- [H800 PCIe OEM 规格](https://lenovopress.lenovo.com/lp1814-thinksystem-nvidia-h800-pcie-gen5-gpu)：1513 FP16 含结构化稀疏，对应 Dense 756.5，350W。
- [Hopper 微基准研究](https://arxiv.org/html/2501.12084v1)：1620 MHz 参考频率及持续负载/功耗现象。平均频率线性折算仅是解释性模型，不是测得的硬件上限。
- [cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/)、[NVML 文档](https://docs.nvidia.com/deploy/nvml-api/)、[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)。

先检查同一 Shape/API/算法的持续窗口。正式负载早/晚三分之一的中位数用于描述变化；后段 TFLOPS 和 SM Clock 的 CV 均 ≤3% 仅作为稳定性筛选，不证明热稳态。只有通过筛选的后段才计算相关性；后段少于 8 个有效窗口、所需传感器每窗少于 3 个采样或变化过小时，明确标记数据不足。

明确 reason 位与时钟、TFLOPS 同步变化是线索，不是单独的因果证明。跨 Shape 散点图仅作描述。区分规模不足、算法差异、功耗限制、温度限制、参考频率差和其他进程。必要时另做固定条件重复试验；本工具不自动进行功耗上限或锁频对照。

## 开发验证

~~~bash
python3 /absolute/cuBLAS_test/tests/test_h800_peak.py
cmake --build /absolute/cuBLAS_test/build_h800_peak --target gemm_bench h800_peak_bench
ctest --test-dir /absolute/cuBLAS_test/build_h800_peak --output-on-failure
~~~

测试夹具全部在临时目录，用来检查时间边界、UUID、缺失值、限频位、失败过滤、原始样本汇总、CSV 和任意目录调用；不会将人工夹具写进 output 伪装成 GPU 结果。

本次开发环境是 WSL Ubuntu、CUDA 12.6、RTX 3050 Laptop。它能验证功能，不能替代 H800 PCIe 的实测验收。H800 长负载峰值、计数器权限、传感器可用性和 HMMA/HGMMA 路径仍须在目标服务器验证。


## 本次交付与实际验收记录（2026-10-24）

新增文件：

| 文件 | 作用 |
|---|---|
| csvs/h800_pcie_fp16_shapes.csv、csvs/benchmark_config.csv | 40 个用例及可调参数 |
| src/h800_peak.cu | 单组基准、CPU FP64 抽样校验、完整 NaN/Inf 检查、事件计时、阶段记录 |
| scripts/run_h800_peak.sh、scripts/run_h800_peak.py | 一键编译、扫描、选择不同尺寸、持续负载、清理与汇总 |
| scripts/monitor_gpu.py | NVML 优先、nvidia-smi 降级、UUID 绑定与原始遥测 |
| scripts/analyze_results.py | 时间窗口关联、缺失值处理、失败排除、CSV、报告及图表 |
| scripts/profile_h800_peak.py | 独立 Nsight Compute 动态 Tensor 指令计数与 SASS 证据 |
| tests/test_h800_peak.py | 18 项边界条件和解析回归测试 |
| requirements-h800-peak.txt、README_H800_PEAK.md | 可选 Python 依赖与完整使用说明 |

修改既有 CMakeLists.txt、include/gemm/backend.hpp、src/cublaslt_gemm.cpp，分别增加构建目标/测试、可选校验接口、逐候选校验及算法记录。原有命令行入口和历史测试数据保留。

本机 CUDA 12.6 已成功编译 h800_peak_bench 与旧 gemm_bench。CTest 4/4 通过；新工具单元测试 18/18 通过。真实 GPU 测试结果如下，目录相对于本项目：

| 验收项 | 实测结果 | 输出目录 |
|---|---|---|
| nvidia-smi 自动降级 Quick Test | 14/14 PASS，四种转置和两种 API 正确性通过 | output/h800_peak_20261024_074740_117008_4049146 |
| NVML Quick Test | 14/14 PASS，215 条遥测、40 个窗口、7 张 PNG | output/h800_peak_20261024_075041_792436_4098965 |
| 30 秒持续负载 / API | 2/2 PASS，60 个窗口、417 条遥测、5 张 PNG | output/h800_peak_20261024_075335_960526_4141623 |
| 超大矩阵显存预检查 | 两种 API 均 SKIP，未分配超量显存、TFLOPS 留空 | output/h800_peak_20261024_075544_914445_4178440 |
| SIGTERM 中断 | 基准和监控均退出，报告保留，中断 run 排除排名 | output/h800_peak_20261024_130843_512179_1284910 |

持续负载的独立 cuBLAS profiler 结果位于上述持续测试目录的 logs/profile_20261024_130858_810963/：
8 次 kernel 均采得 sm__inst_executed_pipe_tensor.sum=589824，并导出 HMMA SASS；其性能时间不进入正常排名。
Nsight Compute 同时报告驱动兼容性和模块加载诊断，已原样保存；这份证据仅描述本机独立探针，H800 上仍需重新验证。
较早的 profile_20261024_130737_676108 保留了 CSV 格式兼容修复前的诊断，不作为最终解析结果。

环境限制与后续验收：

- 本机为 RTX 3050 Laptop，尚未执行 H800 的完整 40 用例扫描与最优 3 组 × 两种 API 的 45 秒持续测试；不能据此宣布 H800 峰值或解释此前 470 TFLOPS 的唯一原因。
- 本机 WSL 驱动未提供当前 Power Limit，CSV 留空；实际功耗、时钟、温度及 reason 位已采集。
- 目标 H800 上先运行 --quick，再运行 --mode all；检查 GPU 身份、实际 Power Limit、计数器与算法记录。
- 优先比较固定 Shape/API/算法的窗口。reason 位、频率、TFLOPS 同步变化可支持限频解释；单独高功耗或一个相关系数不足以证明因果。
- 再次运行 analyze_results.py 会更新派生汇总和图表，原始 benchmark_raw.csv、gpu_telemetry.csv 与运行时 environment.json 保留。
