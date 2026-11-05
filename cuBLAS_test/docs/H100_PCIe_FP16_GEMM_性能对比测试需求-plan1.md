# H100 PCIe FP16 GEMM 性能对比测试需求（按测试计划归档版）

> **交付对象：Qoder**｜文档性质：测试需求与验收标准（非实现代码）  
> **核心目标：**在相同硬件、数学语义和测量条件下，评估自研 Hopper TMA+WGMMA GEMM 与 cuBLAS、cuBLASLt、CUTLASS 的性能差距，定位后续优化方向。  
> **执行要求：**请一次性完成所需的测试代码、构建配置、运行脚本、CSV 汇总和说明文档；不要只实现某一个阶段。若暂时没有 H100，仍需先完成可构建的代码与脚本，并清楚标记“未在目标 GPU 实测”，禁止编造测试数据。
>
> **项目组织原则（本版新增）：**`plan1` 表示**第一次独立的测试计划——H100 PCIe FP16 GEMM 公平基线对比**。测试程序及库适配放在公共 `src/`，每个计划有自己的 `scripts/planN/` 配置与执行入口，所有原始结果放在 `output/planN/<timestamped_run>/`。**本次仅创建 plan1，不预先创建空的 plan2/plan3。**

## 1. 背景与研究问题

当前自研 GEMM 使用 Hopper（SM90）上的 **TMA + WGMMA**，已有 V1/V2 版本。此前在 **H100 PCIe、FP16、4096×4096×4096** 上观察到约：

| 现有版本 | 布局 | 历史 Benchmark 吞吐（TFLOPS） | 备注                         |
| -------- | ---- | ----------------------------: | ---------------------------- |
| V1       | NT   |                        约 420 | 历史记录，仅作背景；必须重测 |
| V1       | TN   |                        约 470 | 历史记录，仅作背景；必须重测 |
| V2       | NT   |                         480.0 | 历史记录，仅作背景；必须重测 |
| V2       | TN   |                         470.7 | 历史记录，仅作背景；必须重测 |

H100 PCIe 官方公布的 FP16 Tensor Core 峰值含稀疏加速值，**稠密 FP16 参考峰值约为 756 TFLOPS**。该值是额定条件下的理论参考，不代表实际运行时钟下可以直接达到的持续吞吐。

本测试必须回答：

1. 在**同一张 H100 PCIe** 上，相同 GEMM 问题的 cuBLAS/cuBLASLt/CUTLASS 实测性能是多少？
2. 自研 V1/V2 达到最佳公平基线的多少百分比？差距集中在哪些 Shape、布局与 K 长度？
3. 测试期间 GPU 的**实际 SM 时钟、显存时钟、功耗、温度**分别是多少？是否发生降频、限功耗或外部负载干扰？
4. `PIPE=2/3`、CTA 驻留数量与流水线结构的变化，能否解释 TN、NT 的差异？（此项可作为可选扩展实验，不应阻塞核心对比。）
5. 最终应当优先优化 TMA 供数、WGMMA 发射/等待、CTA 并行度，还是 Epilogue 写回？

## 2. 公平性约束——不满足则不能进入主排行榜

### 2.1 数学与布局必须一致

所有实现统一计算：

$$
D = \alpha\,op(A)\,op(B) + \beta C,\quad op(A):M\times K,\quad op(B):K\times N.
$$

- 主测试设 `alpha=1`、`beta=0`，输出 D 的类型必须相同；不计数据初始化、Host↔Device 拷贝、内存分配和初始化时间。
- 输入 A/B 均为 FP16；**先检查当前自研 Kernel 的累加器真实精度**（FP16 还是 FP32），并记录。对比实现只有在相同精度/计算语义下才进入严格公平排名；如无法完全一致，拆分为独立精度组，禁止把 FP16 累加与 FP32 累加直接等同。
- TN、NT 必须严格按仓库现有定义解释：在报告中明确写出 `op(A)`/`op(B)` 的 N/T、物理行/列主序、leading dimension/stride、A/B 实际 Shape。不要把 CUTLASS 的布局缩写、cuBLAS 的转置标志、物理连续方向混为一谈。
- 不允许为某一实现偷偷额外做预转置却把预转置开销排除；若使用预转置，必须同时提供“纯 GEMM 性能”和“含预处理总时间”两个结果，前者单列而不进入不带预处理的严格比较。
- 使用相同的输入数据、相同的初值、相同的输出误差标准；严禁全零输入造成不真实的捷径。
- 若各库使用不同 workspace 或算法搜索策略，记录 workspace 大小与选中的算法 ID；所有算法挑选均使用相同的可用内存预算与独立复测。

### 2.2 基准实现

| 代号         | 要实现的对象                         | 关键要求                                                     |
| ------------ | ------------------------------------ | ------------------------------------------------------------ |
| CUSTOM_V1    | 仓库中现有 V1 GEMM                   | 保留原始源码和行为，不能为测试改动计算逻辑                   |
| CUSTOM_V2    | 仓库中现有 V2 GEMM                   | 保留原始源码和行为，不能为测试改动计算逻辑                   |
| CUBLAS_EX    | `cublasGemmEx` 等适配 API            | 明确 compute type、输入输出类型和算法策略                    |
| CUBLAS_LT    | `cublasLtMatmul`                     | 使用可配置 heuristic/候选算法，记录最佳合法算法、workspace 和搜索范围 |
| CUTLASS_SM90 | CUTLASS Hopper FP16 Tensor Core GEMM | 从可编译的 SM90/SM90a 候选内搜索；记录具体 Kernel 名称、CTA Shape、Stage、Cluster、调度器等 |

注意：先确认当前仓库所带 CUTLASS 版本与 CUDA Toolkit 的 API/架构支持。**禁止编译 CUTLASS 全部 Kernel**；先筛选 FP16、SM90、所需布局的有限候选，之后可以扩展搜索范围。CUTLASS 没有适用候选时如实标记，不得以不匹配的数据类型替代。

## 3. 测试 Shape 与实验矩阵

本章 S01～S12 的所有 Shape 都属于 **plan1**。下面的 P0/P1 是 **plan1 内部的测试优先级**，不是文件夹 `plan1`/`plan2` 的命名；后续的 `plan2`、`plan3` 应表示新的实验目标，例如流水深度调优或 Warp Specialization 对比。

### 3.1 P0：必做，直接回答当前问题

每个 Shape 都测试 TN、NT 两种布局以及所有可用对比实现：

| ID   |    M |    N |    K | 目的                                |
| ---- | ---: | ---: | ---: | ----------------------------------- |
| S01  | 4096 | 4096 | 4096 | **核心锚点**，直接对比历史 V1/V2    |
| S02  | 4096 | 4096 |  256 | 短 K，观察启动、同步、Epilogue 占比 |
| S03  | 4096 | 4096 | 1024 | 中 K，观察吞吐增长过程              |
| S04  | 4096 | 4096 | 2048 | 长 K 与吞吐收敛                     |
| S05  | 2048 | 2048 | 2048 | 中等方阵                            |
| S06  | 8192 | 8192 | 8192 | 大规模吞吐与功耗、频率持续性        |

### 3.2 P1：建议必做，有利于判断适用范围

| ID   |    M |    N |    K | 场景         |
| ---- | ---: | ---: | ---: | ------------ |
| S07  | 1024 | 1024 | 1024 | 小/中方阵    |
| S08  | 4096 | 4096 |  512 | K 扫描补点   |
| S09  |  256 | 4096 | 4096 | 小 M、大 N   |
| S10  | 4096 |  256 | 4096 | 大 M、小 N   |
| S11  | 1024 | 8192 | 4096 | 非方阵，偏 N |
| S12  | 8192 | 1024 | 4096 | 非方阵，偏 M |

如果某自研实现暂不支持特定 Shape，必须记录 `UNSUPPORTED` 和原因，不得默默跳过或让其返回错误结果。

### 3.3 可选参数扫描（先完成公平基线，再执行）

- CUSTOM_V2 在 S01/S02/S03 上扫描 `PIPE=2,3`（若资源及正确性允许，可追加 PIPE=4）。
- 对候选 CTA Tile（如 `128×128×64` 与其他有意义的 Tile）做单变量比较；有真实算法可供运行才测试。
- 仅在已有正确实现情况下追加 Warp-specialized 版本；**不得把尚未完成的新架构重写作为本项目验收的前提**。
- 对以上每个实验同时记录实际 `SMEM/CTA`、Registers/Thread、Threads/CTA、理论/实际驻留 CTA 等信息，不能只记录 TFLOPS。

## 4. 计时模式（必须明确分开）

### 4.1 Warm-cache / steady-state 主性能测试

主排行榜采用正常 Benchmark，**不用 NCU 的 Profile Duration 充当基准耗时**。

- 先初始化、编译、分配内存和创建库句柄/工作区；这部分不计时。
- 每个 Kernel 先充分预热，例如 `warmup=20` 次；执行多轮独立测量，例如每轮 `100` 次 Kernel 调用，至少 `5` 轮；次数由配置项调整。
- 优先实现适用的 GPU 端 Kernel 耗时计时（如 CUDA Events）。考虑有些目标环境不可用 CUDA Events，也必须提供 **CPU `std::chrono` wall-time + `cudaDeviceSynchronize()` 前后同步** 的可选路径，按“批量调用总时长/次数”输出。此时应标注**包含 Host 发射/库调用开销**，不能冒充纯 GPU Kernel 时间，也不能把不同计时方法放在同一个排名。
- 可追加 CUDA Graph replay 作为第二种受控模式，以减轻频繁 Host Launch 的影响；必须与普通 launch 分开报告，禁止混用。
- 默认同一 Shape 上交替运行（例如 V1→cuBLAS→V2→CUTLASS，然后轮换顺序），避免把长时间升温、频率变化系统性地分配给某个实现。
- 每轮记录耗时，汇总中位数、最小值、p10/p90、波动范围；不要只保存一次最优值。

### 4.2 Cold-cache 测试（独立榜单）

- 明确 cold-cache 定义与实现手段，例如在测量前执行独立缓存冲刷操作；冲刷不计入 Kernel 时间，但必须在日志中保留执行记录。缓存冲刷应覆盖足够大的数据工作集，并尽量验证缓存状态。
- 仅对核心 Shape S01、S02 的 TN/NT 做 cold-cache 重点对比即可；必要时扩展。
- 如果由于计时工具限制，只能测量包含 Host Launch 的单次 wall time，必须独立标记测量口径，避免与 GPU Kernel latency 混淆。
- NCU 的 `--cache-control all` 是工具的 replay 机制，不等同于自定义 Benchmark 的 cold-cache 计时；两者不得直接合并比较。

### 4.3 Nsight Compute：诊断，不替代 Benchmark

- 先跑完普通 Benchmark，再对 S01 的 TN/NT 和最关键的两三个实现采集 NCU 报告；必要时加 S02。
- NCU 关注：Duration、Tensor Active、SM Issue Active、Eligible Warps、Warp Stall 分类（Long Scoreboard / Barrier / GMMA 等）、DRAM/L2、Registers、SMEM、Occupancy、SASS 热点和 TMA 相关统计。
- 不同版本的 NCU 指标名称可能不同；优先使用工具实际支持的 metric 或 section，自动保存 `ncu --version`、可用指标和失败原因，**不能填入假数据**。
- 若已通过外部方式锁定 GPU 频率，NCU 应采用 `--clock-control none`；必须记录 `--cache-control` 与 `--replay-mode`。
- NCU 内部会发生 replay/缓存/频率控制，其 Duration 与普通 benchmark 的计时条件不同；报告必须独立展示。

## 5. 硬件、时钟、功耗与环境记录

### 5.1 环境快照

每轮测试保存：GPU 名称、GPU UUID、SM 数量、PCIe/SXM、MIG 状态、Driver、CUDA Runtime/Toolkit、NVCC、cuBLAS/cuBLASLt/CUTLASS 版本、Git commit、编译参数、时区/日期，以及完整执行命令。

### 5.2 GPU 时钟

- 保存测试前后 GPU SM/Graphics Clock、Memory Clock、功耗、温度、功率限制及 Throttle/Clock Events 原因。
- 如有管理员权限，允许脚本在用户明确指定目标频率时进行 `nvidia-smi -lgc` 锁定并验证；**不要自动假设 1200 MHz 或其他固定值适合这台 GPU**。目标频率必须先从硬件支持列表获取。
- 无权限锁频时继续测试并标记 `clock_locked=false`，记录实际观测频率与波动，不能宣称“固定频率对比”。
- 若脚本主动改变时钟，必须使用 trap/finally 在正常结束或中断时恢复；严禁影响其他 GPU 或静默修改服务器的长期设置。
- H100 Hopper 显存频率不能简单套用一般 GPU 的即时 `-lmc` 锁频路径；先查设备支持能力，显存频率不改变也必须记录。

### 5.3 功耗与温度

- 采用 `nvidia-smi` 周期性采样并附时间戳，记录整个稳态批量运行区间的平均/峰值/最低功耗和时钟、温度。
- 对约几百微秒的单次 GEMM，普通 `nvidia-smi` 采样不能精确归因到每个 Kernel；**应使用持续运行批次估算稳态功耗**，不能把某个瞬时瓦数声称为单 Kernel 功耗。
- 运行前检查其他 GPU 进程与显存使用量。如果存在干扰进程，记录并提示，默认不杀进程。

## 6. 正确性验证（强制）

- 先测正确性，后测性能；错误或非有限输出的版本不得进入排名。
- 使用固定 seed 生成包含正、负和小数的非平凡 FP16 输入，并对所有实现在同一输入上比对。另设 all-zero/ones 等简易用例用于调试，但不得作为唯一性能输入。
- 使用已知可靠的参考结果，明确累加精度和允许误差（绝对误差、相对误差与最大错误比例），实际阈值必须由输出/累加精度及 K 长度决定且在实验前固定。
- 保存 `max_abs_error`、`max_rel_error`、`mismatch_count`、`NaN/Inf_count`、`correctness_pass`；如果不同累加精度导致误差范围不同，分组报告。
- 主要验证 S01/S02/S05，并包含不规则边界或明确说明 Kernel 只支持整 Tile Shape；不能为通过验证而偷偷修改运算定义。

## 7. 指标、计算和判断标准

基础公式：

$$
\mathrm{FLOPs}=2MNK;\qquad \mathrm{TFLOPS}=\frac{2MNK}{t_{sec}\times10^{12}}.
$$

主结果至少包含：

| 字段                                                        | 含义                                         |
| ----------------------------------------------------------- | -------------------------------------------- |
| `plan_id,run_id,case_id`                                    | 计划、单次运行、测试 Case 的唯一标识         |
| `shape_id,m,n,k,layout`                                     | Shape 与 TN/NT 定义                          |
| `implementation,kernel_name`                                | V1、V2、cuBLAS、cuBLASLt 或 CUTLASS 具体算法 |
| `a_dtype,b_dtype,accum_dtype,out_dtype,alpha,beta`          | 实际数学语义                                 |
| `timing_mode,cache_mode`                                    | 测时手段与冷热缓存口径                       |
| `duration_us_median,duration_us_p10,duration_us_p90`        | 多轮统计                                     |
| `tflops_median`                                             | 由真实耗时计算的吞吐                         |
| `sm_clock_mhz_avg,mem_clock_mhz_avg,power_w_avg,temp_c_max` | 运行状态；无法可靠获得的记为 N/A             |
| `clock_locked,clock_target_mhz`                             | 是否主动锁频及目标                           |
| `correctness_pass,max_abs_error,max_rel_error`              | 正确性结果                                   |
| `workspace_bytes,algorithm_id,build_git_sha,config_hash`    | 可复现信息；`config_hash` 对应实际生效配置   |
| `status,notes`                                              | OK / FAIL / UNSUPPORTED / NOT_RUN 及原因     |

需要额外计算：

1. **相对同条件最佳库实现的吞吐比**：`custom_tflops / best_matched_library_tflops × 100%`。最佳基线仅从**同 Shape、同布局、同计算精度、同计时/缓存模式**且正确的 cuBLAS/cuBLASLt/CUTLASS 结果中选择。
2. **相对理论峰值利用率**：`tflops / 756 × 100%`，清楚标注“按 H100 PCIe 官方稠密 FP16 标称峰值估算”；若 GPU 身份不是 H100 PCIe，则不可沿用此分母。
3. **按实测时钟粗略归一的理论峰值利用率（可选）**：必须注明标称峰值对应的时钟假设及其来源，不应将该估算称为实测硬件上限。
4. **差距按 Shape 分析**：区分“自研差距主要存在于短 K”“主要存在于 TN/NT 某一布局”“所有大规模 GEMM 都落后”等情形。

验收时重点输出：**S01 上 V1-TN/V2-TN、V1-NT/V2-NT 对最佳公平库基线的百分比，以及建议先优化什么。**

## 8. 工程目录与交付物（**按测试计划分类；本次必须遵守**）

### 8.1 目录设计原则

1. **公共代码与计划脚本分离**：`src/`、顶层 `CMakeLists.txt` 负责统一测试逻辑、cuBLAS/cuBLASLt/CUTLASS 和自研 Kernel 适配；`scripts/plan1/` 负责本次具体测试的参数、命令与结果汇总。以后新增 plan2 时复用公共代码，不拷贝整份 `src/`。
2. **计划与结果一一对应**：`scripts/plan1/` 的所有运行结果必须落在 `output/plan1/<timestamped_run>/`；不得混入其他计划目录。`planN` 是实验**主题/方案**，`<timestamped_run>` 是同一方案的某次实际执行。
3. **配置可追溯**：每个计划维护一个可版本控制的 `plan.yaml`；执行时存储完整的 `resolved_plan.yaml` 快照、环境与命令。修改源 `plan.yaml` 不得改变历史记录。
4. **原始数据只有一份**：`raw_runs.csv` 的唯一规范路径是 `csvs/raw_runs.csv`，不再于运行目录根部重复放置。
5. **不覆盖历史**：每次 quick/full 默认生成新的唯一运行目录。即使失败也保存已完成的原始记录和错误日志；任何自动汇总不得覆盖其他运行。
6. **计划可扩展但不预创建**：此次实际生成 `plan1` 即可；`plan2`、`plan3`、`plan n` 只是未来命名示例，**不要建立空目录/空文件**。

### 8.2 目标目录结构

```text
best-matmul/cuBLAS_test/
├── README.md                       # 项目整体介绍、目录索引、各计划导航
├── CMakeLists.txt                  # 统一、可复现构建；源码与公共库适配
├── .gitignore                      # build/、临时文件及大型报告的管理规则
├── src/                            # 公共测试程序、库适配、正确性与计时组件
│   ├── ...                         # 根据仓库现状组织，不强制具体文件名
├── scripts/
│   ├── README.md                   # 各计划目的/差异/适用 GPU/执行入口总览
│   ├── common/                     # 可复用工具：环境、计时参数、日志、GPU 监控等
│   │   └── ...                     # 只放跨计划复用代码，不放 plan1 专属逻辑
│   └── plan1/                      # 本次计划：H100 PCIe FP16 GEMM 公平基线
│       ├── README.md               # 测试任务、方案、编译/运行命令、结果目录
│       ├── plan.yaml               # 本计划的所有测试参数（建议新增）
│       ├── build.sh                # 一键构建；可复用顶层 CMake
│       ├── run_quick.sh            # S01 + S02：正确性与基础性能
│       ├── run_full.sh             # S01～S12：全布局、全可用实现
│       ├── run_ncu.sh              # 重点 Case 的 NCU 采集
│       └── summarize.py            # 单次运行的原始 CSV -> 汇总 CSV/报告
├── build/                          # 编译产物，不是实验结果；一般不提交版本库
│   └── plan1/                      # 计划相关构建配置；允许按配置哈希细分
└── output/
    ├── plan1/
    │   └── <timestamped_run>/      # 例：20261105_113000_a1b2c3（仅示意）
    │       ├── resolved_plan.yaml  # 执行时真正生效的完整配置快照
    │       ├── run_manifest.json   # 本轮状态、Case 列表、成功/失败/跳过原因
    │       ├── environment.json    # GPU/驱动/CUDA/库版本/Git SHA/锁频状态
    │       ├── commands.log        # 所有构建、测试、NCU 命令及返回码
    │       ├── run.log             # 脚本执行过程、错误与关键事件
    │       ├── correctness.log     # 测试输入与误差、PASS/FAIL/UNSUPPORTED
    │       ├── clock_power.csv     # 带时间戳的 GPU 频率/功耗/温度采样
    │       ├── csvs/
    │       │   ├── raw_runs.csv    # 唯一原始记录：每轮真实测量
    │       │   ├── summary.csv     # 中位数、性能比、正确性、测量口径
    │       │   └── ncu_summary.csv # NCU 指标汇总，与 benchmark 分开
    │       ├── reports/            # .ncu-rep；可含 NCU 辅助导出
    │       ├── figures/            # 可选：由实际 CSV 生成的图表
    │       └── analysis.md         # 本轮结论、证据、限制与下一步建议
    └── ...                         # 未来运行才生成 plan2、plan3 等目录
```

**与原目录方案的关系：**保留了你指定的 `scripts/plan1/`、`scripts/README.md`、`output/plan1/<timestamped_run>/` 的核心结构；只新增了必要的配置快照、状态清单与共享工具，并删除了重复的根目录 `raw_runs.csv`。`build/`、`figures/` 可按需生成；**不要建立占位的 `plan n`**。



### 8.3 plan1 的含义与 README 要求

`plan1` 就是本需求第 1～7 章描述的**第一次基线测量计划**：在 H100 PCIe 上比较 CUSTOM_V1、CUSTOM_V2、cuBLAS、cuBLASLt、CUTLASS 的 FP16 TN/NT GEMM 性能。后续计划可以是新的 Kernel 优化实验，但不要把下一次重复运行误命名为 `plan2`；**同一计划的重复运行只增加新的 `<timestamped_run>`**。

三份 README 应当各司其职，不要重复整篇需求文档：

| 文件                      | 必须包含的内容                                               |
| ------------------------- | ------------------------------------------------------------ |
| 顶层 `README.md`          | 项目目标、完整目录说明、依赖与公共编译方式、如何找到每个计划、结果数据约定 |
| `scripts/README.md`       | 计划索引表：plan ID、计划名称、测试目标、主要改动、关联脚本、输出路径；新增计划时更新 |
| `scripts/plan1/README.md` | **本次测试任务、S01～S12、对比实现、输入/精度约束、构建命令、quick/full/NCU 运行命令、参数说明、输出路径、如何汇总与解读结果、常见失败** |

### 8.4 `plan.yaml`：让每次测试可复现

为 plan1 增加 `scripts/plan1/plan.yaml`，**具体字段可按已有代码接口适配**，但至少保存：

- `plan_id`、名称、目标 GPU/架构、测试目的；
- Shape 集合、TN/NT 布局、V1/V2/cuBLAS/cuBLASLt/CUTLASS 实现选择；
- FP16 输入、实际累加精度/输出类型、`alpha/beta`、输入种子、正确性阈值；
- `warmup`、每轮重复次数、独立轮数、计时口径、warm/cold 定义、算法搜索预算；
- `quick/full/ncu` 的 Case 选择，NCU section/metric、replay/cache/clock-control 参数；
- GPU 频率策略（默认不主动更改）、监控采样间隔，以及失败处理选项。

配置的命令行覆盖项（若支持）也必须写入本轮的 `resolved_plan.yaml`。`environment.json` 记录硬件与软件环境，二者作用不同，不能相互替代。建议为最终配置计算 `config_hash` 并写入各结果记录。

### 8.5 脚本与运行目录的行为约定

- `scripts/plan1/build.sh`：从任意当前工作目录调用都能定位项目根目录；复用顶层 CMake，构建到 `build/plan1/`（或按配置哈希建立子目录），不把编译产物放进 `output/`。
- `run_quick.sh`：先做正确性验证，再运行 S01/S02 的 TN/NT 基础性能；默认**创建一个新** `output/plan1/<timestamped_run>/`，输出环境与日志。
- `run_full.sh`：覆盖全部要求的 Shape/布局/实现，记录所有有效与无效 Case；默认创建**另一个新**运行目录，绝不修改 quick 的结果。
- `run_ncu.sh`：默认单独创建新运行目录；同时支持显式指定现有 `--run-dir`，将同一轮的 NCU 报告**附加**到指定目录的 `reports/` 和 `csvs/ncu_summary.csv`。不提供 `--run-dir` 时不得随意猜测“最近一次”运行。
- `summarize.py`：以明确的 `--run-dir` 为输入，只分析这一运行目录的真实原始数据，生成/更新 `summary.csv`、`ncu_summary.csv`、`analysis.md`；不扫描其他计划、不凭空补齐缺失 Case。不主动创建新运行 ID。
- `run_id` 应由时间戳与短随机后缀或唯一标识组成，保证并发/同秒执行不会冲突；使用统一时区，记录精确开始/结束时间。
- 运行状态建议包括 `RUNNING`、`COMPLETED`、`PARTIAL`、`FAILED`，Case 状态包括 `OK`、`FAIL`、`UNSUPPORTED`、`NOT_RUN`；部分失败必须留下原因和已有数据。
- 如果脚本主动锁频、修改功耗或其他系统状态，必须在正常退出和异常中断时恢复；监控进程亦需清理。不要默认改变共享服务器上的设备设置。

建议将下面命令写入 `scripts/plan1/README.md`，并确保可以直接执行（具体参数按实现调整，但操作语义不能改变）：

```bash
cd matmul/best-matmul/cuBLAS_test
bash scripts/plan1/build.sh
bash scripts/plan1/run_quick.sh
bash scripts/plan1/run_full.sh
RUN_DIR="output/plan1/20261105_113000_a1b2c3" # 示例：替换为脚本实际打印的运行目录
bash scripts/plan1/run_ncu.sh --run-dir "$RUN_DIR"
python3 scripts/plan1/summarize.py --run-dir "$RUN_DIR"
```

### 8.6 输出数据的唯一真相与跨计划对比

1. `csvs/raw_runs.csv` 是唯一的每轮实测数据来源，每一行必须有 `plan_id`、`run_id`、`case_id`、实现、Shape、布局、正确性状态、计时口径、耗时和必要的环境关联字段。
2. `csvs/summary.csv` 由 `raw_runs.csv` 生成，必须保留准确的算法选择和统计方法；`ncu_summary.csv` 是诊断表，不得把 NCU Duration 混入普通 benchmark 排名。
3. `run_manifest.json` 列出预期 Case 和最终状态，帮助判断是**没有运行**、**没有数据**还是**真实性能较低**。未测数据填 `NOT_RUN`/`N/A`，禁止插值、猜测和伪造。
4. `analysis.md` 必须引用同目录的真实 CSV/NCU 结果。可选生成图片放在 `figures/`，图片不能替代原始数据。
5. 为将来的 plan2/plan3 预留**跨计划比较规范**：必须按同一 GPU、Shape、布局、计算精度、Cache Mode、Timing Mode 和时钟策略筛选可比结果；比较时明确指出代码版本与配置差异。日后如确需自动对比，可以新增公共的 `scripts/compare_plans.py`，结果另存 `output/_comparisons/<timestamped_run>/`，**本次不要求创建该脚本及目录**。
6. 原始测试历史不覆盖；大型 `.ncu-rep`、编译产物是否纳入 Git 由项目规则决定，不要为了版本控制而意外删除真实实验数据。

### 8.7 `analysis.md` 必须回答的 6 个问题

1. S01 上自研 V1/V2 与每个公平库基线分别达到多少 TFLOPS？
2. 性能差距是 5%、15% 还是 30% 以上？是否在重复测试中稳定？
3. TN/NT 差异是否只存在于自研实现，还是基线同样存在？
4. K 从 256 增大到 4096 时，哪个实现最早进入高吞吐区？
5. 实际 GPU 时钟和功耗是否解释了一部分理论峰值差距？
6. 如果只能选 **3 个下一步优化实验**，应该选择什么？每项必须引用测试指标支撑。

## 9. 验收标准

### 9.1 测试功能

- [ ] 在目标 H100 上能够构建并运行；如果没有目标 GPU，生成可执行完整代码并明确标记未验证。
- [ ] S01/S02 同时覆盖 TN、NT，以及可用的 V1、V2、cuBLAS、cuBLASLt、CUTLASS。
- [ ] 数学语义、累加精度、矩阵物理布局、输出精度在对比表中可核查。
- [ ] 每条排名结果通过正确性检查；失败不能参与基线选优。
- [ ] Warm 与 Cold、CPU wall 与 GPU events、普通 benchmark 与 NCU 完全分栏。
- [ ] 自动记录 GPU 频率/功耗/温度、环境版本、测试命令、算法 ID、每次原始结果。
- [ ] 输出逐轮 CSV、汇总 CSV、重点 NCU 原始报告和中文分析报告。

### 9.2 按计划组织与可复现性（本版新增）

- [ ] **本次仅创建 plan1**：`scripts/plan1/` 对应 `output/plan1/<timestamped_run>/`；不创建空 plan2 等目录。
- [ ] 三层 README（顶层、scripts 总览、plan1 说明）齐全，且职责清楚。
- [ ] 所有计划配置保存在 `plan.yaml`；每次运行保存实际生效的 `resolved_plan.yaml`、`environment.json` 与命令日志。
- [ ] `run_quick.sh` 与 `run_full.sh` 每次生成不同运行目录，旧结果不能被静默覆盖。
- [ ] `run_ncu.sh` 可独立新建运行，也可显式添加到指定 `--run-dir`，不能误写其他计划。
- [ ] `raw_runs.csv` 仅位于 `csvs/`；Summary 与分析来自真实原始记录。
- [ ] `run_manifest.json` 可以核对测试完成情况、失败状态与对应输出文件。
- [ ] 构建代码和公共复用逻辑不在各 `planN` 下重复复制；从任意工作目录调用脚本仍能正常解析路径。
- [ ] 不修改现有自研 Kernel 的数学逻辑，不覆盖旧结果，不编造数据。

## 10. qoder 实施说明（请完整执行）

请先扫描已有 CUDA/CuTe/CUTLASS GEMM 代码、当前 `cuBLAS_test` 目录和现有 Benchmark 接口，再按第 8 章的**计划分组架构**构建统一测试框架。优先保留用户指定的目录结构；补充的 `plan.yaml`、`resolved_plan.yaml`、`run_manifest.json`、`scripts/common/` 用来确保后续新增 plan2/plan3 时仍可复用、可追溯。**不要创建未来计划的占位目录。**

不要只写说明或占位代码。若某项 API 与当前 CUDA/CUTLASS 版本不兼容，请在代码中提供实际可用的替代路径并记录差异；在没有硬件时完成编译检查及能够离线执行的测试，但不要声称已验证 GPU 性能。统一使用 `quick → full → NCU` 的操作入口，但**一次性生成完整功能**，不要拆成多轮交付。

最终回复必须列出：（1）实际生成的文件树；（2）plan1 的构建、quick/full/NCU/汇总命令；（3）真实执行并通过的检查；（4）未验证的功能及原因；（5）至少一个真实生成的 `output/plan1/<timestamped_run>/` 路径（若未运行，只说明预期路径，不能虚构结果）。

## 11. 参考资料

- NVIDIA H100 规格：<https://www.nvidia.com/en-us/data-center/h100/>
- cuBLAS/cuBLASLt API：<https://docs.nvidia.com/cuda/cublas/>
- CUTLASS Profiler：<https://github.com/NVIDIA/cutlass/wiki/Performance-Profiling>
- Nsight Compute Profiling Guide：<https://docs.nvidia.com/nsight-compute/ProfilingGuide/>
- Nsight Compute CLI：<https://docs.nvidia.com/nsight-compute/NsightComputeCli/>
- NVIDIA System Management Interface：<https://docs.nvidia.com/deploy/nvidia-smi/index.html>