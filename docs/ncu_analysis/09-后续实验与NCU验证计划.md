# 09 · 后续实验与 NCU 验证计划

> 本文把 [07 · 根因总结](07-关键性能瓶颈与根因总结.md) 中的 R-01～R-07 与 [08 · 优化方案](08-Kernel优化方案与优先级.md) 中的 OPT-01～OPT-08 转成可执行、可判定的实验。每个实验都先做正确性回归，再做计时，最后才用 NCU 解释；**任何 NCU replay 时长都不进入成绩**。

---

## 1. 执行纪律

1. **一个实验只改一个变量。** 源码变体、PIPE、布局、形状、编译选项不得同时变化；多变量改动只能作为探索，不能作为结论。
2. **正确性先于性能。** 新变体必须通过 `--mode validate`（NP 边界用例）和必要的 `racecheck/synccheck`，才能进入 bench。
3. **benchmark 与 NCU 分离。** TFLOPS 只取 `--mode bench` 的 p50；NCU 只用于解释机制，不与 benchmark 绝对值混排。
4. **冷/热缓存不可混用。** focused cold 用于单 kernel 机理；需要接近稳态时用 warm application replay，并明确标注。
5. **计时前检查机器独占。** 4096³ 的 PIPE/M256 数据已被外部 `./test` 污染（R-07）；每次 bench 前用 `nvidia-smi --query-compute-apps` 确认无其他进程，运行中保留 telemetry。
6. **判定阈值。** 现有 3 进程 p50 的批间波动约 1–3%；以 **同形状、同布局、跨进程 p50 中位数变化 >3%** 作为“有实际收益”，1–3% 视为噪声需重复，<1% 不作为结论。NCU 指标只做方向性判据。
7. **不改依赖。** 任何实验都不得删除 full/empty barrier 或 `warpgroup_wait`；只能改变“谁等、何时等、等多少”，数据依赖必须保持。

### 1.1 统一环境设置

```bash
REPO=/home/myCode/cutlass_study/peak_flops/src/sm90/nsight_compute/matmul/best-matmul
BUILD=$REPO/build/h100_pcie                     # manual_20261020 构建，含 v1–v5 及 PIPE/M256 变体
NCU=/opt/nvidia/nsight-compute/2024.3.2/ncu     # 或 /usr/local/cuda/bin/ncu
STAMP=$(date +%Y%m%d_%H%M%S)
EXP=$REPO/reports/h100_pcie/verify_$STAMP       # run.sh 要求 output-dir 不存在
cd "$REPO"

# 每次计时前：确认无外部进程；实验结束后保存 before/after
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu \
  --format=csv,noheader
```

`--telemetry-ms 1000` 会生成运行期 GPU 状态采样；分析时用它确认没有降频或其他进程抢占。新源码变体需在 `build.sh` 中增加配置项，并用 `--with-validation` 构建；下文以 `$BUILD_V6` 指代新构建目录。

---

## 2. 实验总表

| 实验 ID | 目标问题 | 修改变量 | 保持不变的条件 | 必要 NCU 指标 | 预期判断 |
|---|---|---|---|---|---|
| **EXP-01** | NT 是否应直接调度 V3 | 版本 V3 vs V4/V5 | 形状、布局、编译选项、机器 | Tensor%、duration、DRAM/L2、NANOSLEEP/UTMALDG 计数 | V3 在 NT 全形状 ≥ V4，确立布局相关调度 |
| **EXP-02** | V4/V5 NT 是否被“单 lane 串行 TMA 发射”卡住 | producer 1 lane → 4 warp 并行 + election 外提 | TMA atom、PIPE、CTA tile、consumer 代码 | consumer 自旋/tile、UTMALDG 分布、Tensor、TMA/L1TEX%、duration | 自旋 ≤3/tile 且 Tensor ≥65% → 机制成立；仅自旋降而 Tensor 不升 → 另有瓶颈 |
| **EXP-03** | NT 的 1 KB TMA 粒度是否是结构性代价 | 预转置走 TN 路径（含转置成本）/ 更大 TMA box | 端到端问题、M/N/K、输出布局 | TMA 请求数、crossbar sectors、Tensor、duration | 预转置端到端仍胜 V3 NT >5% → 粒度是主因；否则否证 R-03 的定量版本 |
| **EXP-04** | 4096³ 的 PIPE 结论是否被污染 | PIPE=2/3/4（V3/V4/V5） | 形状、布局、独占机器 | Tensor%、occupancy limit、duration、SMEM/regs | 得到干净的 PIPE 最优点；与 K=256 结论对照 |
| **EXP-05** | V5 能否通过 ≤56 regs + PIPE=2 达到 3 CTAs | 寄存器上限、PIPE | CTA tile、线程数、TMA/consumer 结构 | regs/thread、4 个 occupancy limit、achieved occupancy、spill 指令、Tensor | 3 CTAs 且无 spill，K=256 +8%，4096³ 不回退 |
| **EXP-06** | V3 的 PIPE=2 适用 K 范围 | K=256…8192（M=N=4096） | 版本 V3、布局、PIPE 变体 | Tensor%、duration、occupancy limit | 找到 PIPE=2→3 的 K 阈值，形成按 K 调度 |
| **EXP-07** | 消费者自旋本身是否消耗吞吐 | full-barrier 等待：128 线程 → 每 warp 1 线程/单线程广播 | 数据依赖、barrier 语义、其余代码 | NANOSLEEP 指令数、Tensor、duration、barrier stall | 自旋降 4–32× 而 Tensor 不变 → 自旋是症状不是根因 |
| **EXP-08** | 合并写回的 SMEM 往返/STS 开销能否降低 | STS.U16 → 向量化 STS/STMatrix；短 K 跳过 SMEM | alpha=1/beta=0、C 布局、PIPE | STS 计数、shared wavefronts/excessive、BAR.SYNC、duration | 短 K +2%，长 K 持平；否则回退 epilogue 改动 |
| **EXP-09** | 小/中形状是否存在波次尾部与提交空隙损失 | 常规 grid → persistent tile scheduler | 计算/写回逻辑、PIPE、占用 | duration、achieved occupancy、nsys 时间线间隙 | 1024³ +5% 且 4096³ 不回退 |
| **EXP-10** | cold replay 是否夸大 NT 的 TMA 饥饿 | cold → warm application replay；按 PC 分角色 | 版本 V4/V5、NT/TN、形状 | Tensor%、DRAM/L2、TMA/L1TEX%、stall、NANOSLEEP | warm 下差距消失 → 冷缓存伪影；差距仍在 → 真实瓶颈 |
| **EXP-11** | full-barrier 等待与 TMA 发射的真实时长 | 加 `clock64()` 计时（仅诊断构建） | 其余逻辑不变 | 不依赖 NCU；每 tile 等待/发射周期直方图 | 等待中位数 >1 个 k-tile 计算时间 → 饥饿实锤 |

优先级：**EXP-01 → EXP-04 → EXP-06 → EXP-02 → EXP-10 → EXP-03 → EXP-07 → EXP-08 → EXP-05 → EXP-09 → EXP-11**。前三个只使用现有二进制，可立即执行。

---

## 3. 各实验细则

### EXP-01 · NT 布局调度 A/B

| 字段 | 内容 |
|---|---|
| **假设** | 在 NT 布局下，V3（4 warp 全员计算、inflight=2、无独立 producer）比 V4/V5 的 WS 路径更快；TN 下 V4 仍有优势 |
| **变量** | 版本。V3/V4/V5 与 `--baseline v3` |
| **控制** | 同一 `$BUILD`、同一 GPU、NT 与 TN 分别聚合；形状覆盖 1024³、4096³、8192³、K=256、16384×1024×4096、1024×16384×4096 |
| **正确性** | 先跑 `--mode validate --variants v3,v4,v5 --layouts NT,TN` |
| **命令** | 见下方代码块 |
| **NCU 指标** | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`、`gpu__time_duration.avg`、`gpu__dram_throughput...`、`lts__throughput...`、`smsp__warps_eligible.avg.per_cycle_active`、`smsp__issue_active...`、`smsp__average_warps_issue_stalled_{barrier,gmma,long_scoreboard}_per_issue_active.ratio`；source page 的 NANOSLEEP/UTMALDG 计数 |
| **判定** | NT：V3 在 4096³、8192³ 领先 ≥5%，且 1024³/K=256 不回退 >2% → 采用“NT→V3、TN→V4”的布局相关调度。若小形状 V4 反而赢 >3%，改为“布局 + 形状”二维调度 |
| **止损** | 若 V3 仅在部分形状领先且差异 <3%，不下调度结论，只保留数据 |

```bash
bash scripts/run.sh --mode validate --build-dir "$BUILD" --gpu 0 \
  --variants v3,v4,v5 --layouts NT,TN \
  --output-dir "$EXP/exp01_validate"

bash scripts/run.sh --mode bench --build-dir "$BUILD" --gpu 0 \
  --variants v3,v4,v5 --baseline v3 --layouts NT,TN \
  --shapes 1024x1024x1024,4096x4096x4096,8192x8192x8192,4096x4096x256,16384x1024x4096,1024x16384x4096 \
  --repeats 3 --per-batch 20 --min-warmup 300 --batches 100 --telemetry-ms 1000 \
  --output-dir "$EXP/exp01_bench"

bash scripts/run.sh --mode ncu --build-dir "$BUILD" --gpu 0 \
  --variants v3,v4,v5 --layouts NT --profile-shape 4096x4096x4096 \
  --ncu "$NCU" --ncu-set focused --ncu-cache cold --ncu-clock none --launch-skip 50 \
  --metrics l1tex__tmain_requests.sum,l1tex__m_xbar2l1tex_read_sectors.sum,l1tex__t_sector_hit_rate.pct \
  --output-dir "$EXP/exp01_ncu"
```

> `--metrics` 中的名称来自本机 `env_ncu_metrics_gh100.txt`；换机器前先用 `"$NCU" --query-metrics` 复核。

---

### EXP-02 · V4/V5 producer 并行发射 TMA + election 外提

| 字段 | 内容 |
|---|---|
| **假设** | V4 NT 消费者饥饿来自 producer 单 lane 每 tile 突发 32 个 1 KB TMA；4 个 producer warp 分担后，full barrier 完成更早，消费者每 tile 自旋从 14.5 次降到 ≤3 次 |
| **变量** | 新变体 `v6_ws_par`：producer 改为 4 warp；full barrier count 1→4；每个 producer warp 的 elected lane 对本 warp 的 atom 子集调用 `arrive_and_expect_tx(bytes/4)` 与 `copy`；election 移出 atom 循环（若 CuTe `copy` 内部逐 atom election） |
| **控制** | 与 V4 完全相同的 CTA tile、PIPE=3、consumer 代码、TMA atom、SMEM layout；TN 同步构建以检查副作用 |
| **正确性约束** | full barrier 必须在**全部** TMA 字节到达后才翻转；barrier 的 arrive 计数必须等于参与发射的 warp 数；每个 atom 只能由一个 warp 发射，覆盖率必须为 100%；不得改变 empty barrier 的 consumer 释放语义 |
| **命令** | 先在 `kernels/detail/tma_wgmma_warp_specialized_device.cuh:64`（`is_producer_leader`）与 `:79-90`（producer 循环）、`:68`（full barrier 初始化）实现变体，`build.sh` 增加 `v6_ws_par`，然后：见下方代码块 |
| **NCU 指标** | source page 按 SASS PC 统计 NANOSLEEP、UTMALDG、ELECT；`sm__pipe_tensor_cycles_active...`、`sm__pipe_tma_cycles_active...`、`l1tex__tmain_requests.sum`、`l1tex__m_xbar2l1tex_read_sectors.sum`、`gpu__time_duration.avg`；consumer 自旋次数/tile |
| **判定** | 自旋 ≤3/tile 且 Tensor 从 51.8% 升到 ≥65% → 接受；自旋下降但 Tensor <60% → producer 不是唯一限制，转 EXP-10/EXP-03；TN 回退 >2% → 拒绝 |
| **止损** | 若 4 warp 并行后 UTMALDG 总周期不变且 full barrier 完成时间不提前，说明 TMA 引擎按请求串行，停止该方向 |

```bash
bash scripts/build.sh --suite all --output-dir "$BUILD_V6" --with-validation
bash scripts/run.sh --mode validate --build-dir "$BUILD_V6" --gpu 0 \
  --variants v4,v6_ws_par --layouts NT,TN --output-dir "$EXP/exp02_validate"
bash scripts/run.sh --mode bench --build-dir "$BUILD_V6" --gpu 0 \
  --variants v4,v6_ws_par --baseline v4 --layouts NT,TN \
  --shapes 4096x4096x4096,4096x4096x256 \
  --repeats 3 --per-batch 20 --min-warmup 300 --batches 100 --telemetry-ms 1000 \
  --output-dir "$EXP/exp02_bench"
bash scripts/run.sh --mode ncu --build-dir "$BUILD_V6" --gpu 0 \
  --variants v4,v6_ws_par --layouts NT --profile-shape 4096x4096x4096 \
  --ncu "$NCU" --ncu-set full --ncu-cache cold --ncu-clock none --launch-skip 50 \
  --metrics l1tex__tmain_requests.sum,l1tex__m_xbar2l1tex_read_sectors.sum \
  --output-dir "$EXP/exp02_ncu"
```

---

### EXP-03 · NT 预转置 / 更大 TMA box 的因果检验

| 字段 | 内容 |
|---|---|
| **假设** | NT 的 1 KB TMA atom（32 请求/tile）是 V4 NT 崩溃的结构条件；如果换成 TN 的 16 KB atom，即使支付一次预转置，端到端也应不差于 V3 NT |
| **变量（两臂）** | 3A：在计时区间内先用 transpose kernel 把 A/B 转为 TN 布局，再跑 V4 TN；3B：实现大 box NT 变体（TMA box 覆盖 128×64 的连续段，配合匹配的 SMEM swizzle），不预转置 |
| **控制** | 同一逻辑 M/N/K、同一输出布局、同一 GPU；3A 的转置 kernel 必须计入端到端时间，且单独报告其耗时；3B 必须保持 SMEM 无 bank conflict |
| **正确性** | 3A 需校验转置后矩阵逐元素一致；3B 需 validate + `racecheck`，并检查 TMA 坐标与 swizzle 边界 |
| **命令** | 3A 的转置入口需新增 host 路径；3B 需新变体。构建后按下方模板跑 bench/NCU |
| **NCU 指标** | `l1tex__tmain_requests.sum`（原始请求数）、`l1tex__m_xbar2l1tex_read_sectors.sum`、`sm__pipe_tma_cycles_active...`、Tensor%、DRAM/L2%、duration；3B 另看 shared wavefronts/excessive |
| **判定** | 3A：转置 + V4 TN 的端到端比 V3 NT 快 >5% → TMA 粒度的定量代价成立；若转置成本吃掉收益 → 3A 否证，转 3B。3B：Tensor ≥75% 且无 bank conflict → 接受大 box |
| **止损** | 大 box 需要改变 SMEM 布局并引入 conflict，收益 <5% → 停止 |

```bash
# 以新变体 v6_nt_bigbox 为例；3A 使用现有 v4 + TN 布局和独立 transpose 计时
bash scripts/run.sh --mode validate --build-dir "$BUILD_V6" --gpu 0 \
  --variants v6_nt_bigbox --layouts NT --output-dir "$EXP/exp03_validate"
bash scripts/run.sh --mode bench --build-dir "$BUILD_V6" --gpu 0 \
  --variants v3,v4,v6_nt_bigbox --baseline v3 --layouts NT,TN \
  --shapes 4096x4096x4096,8192x8192x8192 \
  --repeats 3 --telemetry-ms 1000 --output-dir "$EXP/exp03_bench"
bash scripts/run.sh --mode ncu --build-dir "$BUILD_V6" --gpu 0 \
  --variants v4,v6_nt_bigbox --layouts NT --profile-shape 4096x4096x4096 \
  --ncu "$NCU" --ncu-set focused --ncu-cache cold --ncu-clock none \
  --metrics l1tex__tmain_requests.sum,l1tex__m_xbar2l1tex_read_sectors.sum \
  --output-dir "$EXP/exp03_ncu"
```

> 3A 的公平比较必须把 transpose 的时间和 DRAM 流量计入；只比较转置后的 GEMM 会系统性高估该方案。

---

### EXP-04 · 干净的 PIPE=2/3/4 扫描

| 字段 | 内容 |
|---|---|
| **假设** | 现有 4096³ PIPE/M256 数据被外部进程污染（R-07），无法判断 PIPE 最优点；K=256 的干净数据表明 PIPE=2 在短 K 有 +13–16% 优势 |
| **变量** | V3/V4/V5 各自的 PIPE=2/3/4（现有 `v3_p2,v3,v3_p4,...`） |
| **控制** | 独占 GPU；同形状同布局；以各版本 PIPE=3 为 baseline；运行前记录 `nvidia-smi` 进程列表，运行中 telemetry |
| **正确性** | 现有构建已含 `.validate`，仍先跑 validate |
| **命令** | 见下方代码块 |
| **NCU 指标** | `gpu__time_duration.avg`、Tensor%、`sm__warps_active...`、`launch__occupancy_limit_{shared_mem,registers,warps,blocks}`、`launch__shared_mem_per_block_dynamic`、`launch__registers_per_thread` |
| **判定** | 4096³：PIPE=3 优于 2 且 2 优于 4 → 保持默认 3；若 PIPE=2 在 4096³ 也不差（±1%），结合 EXP-06 直接按 K 选择；PIPE=4 若全面最差则确认占用损失 |
| **止损** | telemetry 显示降频或外部进程，整轮作废重跑 |

```bash
bash scripts/run.sh --mode bench --build-dir "$BUILD" --gpu 0 \
  --variants v3_p2,v3,v3_p4,v4_p2,v4,v4_p4,v5_p2,v5,v5_p4 \
  --baseline v3 --layouts NT,TN \
  --shapes 4096x4096x4096,4096x4096x256 \
  --repeats 3 --per-batch 20 --min-warmup 300 --batches 100 --telemetry-ms 1000 \
  --output-dir "$EXP/exp04_pipe_clean"

bash scripts/run.sh --mode ncu --build-dir "$BUILD" --gpu 0 \
  --variants v3_p2,v3,v3_p4,v4_p2,v4,v4_p4,v5_p2,v5,v5_p4 \
  --layouts NT --profile-shape 4096x4096x4096 \
  --ncu "$NCU" --ncu-set focused --ncu-cache cold --ncu-clock none --launch-skip 50 \
  --output-dir "$EXP/exp04_ncu"
```

---

### EXP-05 · V5 ≤56 regs + PIPE=2 → 3 CTAs

| 字段 | 内容 |
|---|---|
| **假设** | V5（384 线程、58 regs）当前 2 CTAs/SM 受 `launch__occupancy_limit_registers=2` 限制；PIPE=2 的 SMEM 允许 3 CTAs，若把 regs 压到 ≤56，V5 可在短 K 获得 3 CTAs 的延迟隐藏 |
| **变量** | 新变体 `v6_v5_occ3`：`__launch_bounds__(384, 3)` 或 `--maxrregcount=56` + `PIPE=2` |
| **控制** | 与 V5 相同的 2 个 consumer warpgroup、epilogue、TMA atom；同时构建 PIPE=3 的 3-CTA 版本作为对照（SMEM 98,432 B × 3 > 上限，预计失败，仅验证 limit） |
| **正确性** | validate；若出现 spill，SASS 检查 LDL/STL；`racecheck` 验证 3 CTA 并发下 barrier 无越界 |
| **命令** | 构建后在下方模板中替换 variants |
| **NCU 指标** | `launch__registers_per_thread`、`launch__occupancy_limit_registers/shared_mem/warps/blocks`、`sm__warps_active.avg.pct_of_peak_sustained_active`、Tensor%、duration；SASS 的 local load/store |
| **判定** | 达成 3 CTAs、无 spill、K=256 +8%、4096³ 回退 ≤1% → 采用为短 K 专用；若 spill 或 4096³ 回退 >3% → 放弃 |
| **止损** | 编译器为压寄存器引入重复计算/spill，Tensor 不升反降 → 停止 |

```bash
bash scripts/build.sh --suite all --output-dir "$BUILD_V6" --with-validation
bash scripts/run.sh --mode bench --build-dir "$BUILD_V6" --gpu 0 \
  --variants v5,v5_p2,v6_v5_occ3 --baseline v5 --layouts NT,TN \
  --shapes 4096x4096x256,4096x4096x1024,4096x4096x4096 \
  --repeats 3 --telemetry-ms 1000 --output-dir "$EXP/exp05_bench"
bash scripts/run.sh --mode ncu --build-dir "$BUILD_V6" --gpu 0 \
  --variants v5,v6_v5_occ3 --layouts NT --profile-shape 4096x4096x256 \
  --ncu "$NCU" --ncu-set focused --ncu-cache cold --ncu-clock none \
  --output-dir "$EXP/exp05_ncu"
bash scripts/run.sh --mode disasm --build-dir "$BUILD_V6" \
  --variants v6_v5_occ3 --output-dir "$EXP/exp05_disasm"
```

---

### EXP-06 · V3 的 PIPE=2 适用 K 阈值

| 字段 | 内容 |
|---|---|
| **假设** | PIPE=2 在短 K 因 3 CTAs/SM 获益；随 K 增大，PIPE=3 的深度更重要，存在切换阈值 |
| **变量** | `v3_p2` vs `v3`，K=256/512/1024/2048/4096/8192（M=N=4096） |
| **控制** | 只改 K；同布局、同版本；每组 3 进程 |
| **正确性** | validate 已覆盖；抽查 K=512/2048 的数值校验 |
| **命令** | 见下方代码块 |
| **NCU 指标** | Tensor%、duration、occupancy、`launch__occupancy_limit_*`、barrier/gmma stall |
| **判定** | 若 PIPE=2 在 K≤512 领先 ≥5%、K≥2048 落后 ≥3%，阈值取 512–1024；形成 `K<=T ? PIPE2 : PIPE3` 调度 |
| **止损** | 曲线非单调且差异 <3% → 不做阈值调度 |

```bash
bash scripts/run.sh --mode bench --build-dir "$BUILD" --gpu 0 \
  --variants v3_p2,v3 --baseline v3 --layouts NT,TN \
  --shapes 4096x4096x256,4096x4096x512,4096x4096x1024,4096x4096x2048,4096x4096x4096,4096x4096x8192 \
  --repeats 3 --per-batch 20 --min-warmup 300 --batches 100 --telemetry-ms 1000 \
  --output-dir "$EXP/exp06_kthreshold"
bash scripts/run.sh --mode ncu --build-dir "$BUILD" --gpu 0 \
  --variants v3_p2,v3 --layouts NT --profile-shape 4096x4096x512 \
  --ncu "$NCU" --ncu-set focused --ncu-cache cold --ncu-clock none \
  --output-dir "$EXP/exp06_ncu_k512"
```

---

### EXP-07 · 消费者自旋的消因实验

| 字段 | 内容 |
|---|---|
| **假设** | V4 NT 的 3.81M NANOSLEEP 是“等 TMA 数据”的表现，而不是吞吐损失的原因；减少自旋指令本身不应显著改变 Tensor |
| **变量** | 新变体 `v6_wait1`：full barrier 由每 warp 1 个 elected lane 等待，随后 `__syncwarp()` 广播；或 1 个 consumer 线程等待 + 128 线程 named barrier |
| **控制** | 其余代码与 V4 相同；数据依赖不变；必须补 `fence.proxy.async.shared::cta` 或 CUTLASS 对应的 acquire fence，保证非等待线程看到 TMA 写入 |
| **正确性** | 高风险的同步改动：validate + `racecheck` + `synccheck`；与 V4 输出逐元素比对；小形状多槽位回绕 |
| **命令** | 构建后同 EXP-02 模板 |
| **NCU 指标** | NANOSLEEP 指令数（source page）、`smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`、Tensor%、duration |
| **判定** | 自旋降 4× 以上而 Tensor 变化 <1% → 自旋是症状，停止在同步指令上做优化；Tensor +≥2% → 自旋确实占用发射，保留 |
| **止损** | 任何正确性失败立即回退，不做性能解释 |

> 该实验的价值主要是**否证**：如果结果符合“症状”判断，08 文档中的 OPT-08 应降级或删除。

---

### EXP-08 · Epilogue STS 向量化 / 短 K 跳过 SMEM

| 字段 | 内容 |
|---|---|
| **假设** | V3+ 的合并写回引入 524,288 条 STS.U16、shared excessive 1,572,864（6%）、BAR.SYNC 8,192；短 K 下这些固定成本导致 −2.3%/−3.7% |
| **变量** | 8A：STS.U16 → 128-bit STS（需按 swizzle 重排每线程元素）；8B：K≤512 时使用 V2 的 direct STG epilogue（按 K 调度） |
| **控制** | alpha=1/beta=0；C 全局布局不变；8A 只改 SMEM 写，8B 只改路径选择 |
| **正确性** | validate；C 逐元素比对；`racecheck` 检查 SMEM 写回与 WGMMA 完成之间的依赖 |
| **命令** | 8B 可先用调度层实现（无需改 kernel）；8A 需新变体 |
| **NCU 指标** | `smsp__inst_executed_op_shared_st.sum`（或 source page 的 STS 计数）、shared wavefronts/excessive、`memory_l2_theoretical_sectors_global/_ideal/_excessive`、Tensor%、duration |
| **判定** | 短 K +2% 且长 K 持平 → 接受 8A；若 8A 无收益，采用 8B 的按 K 调度；两者都无收益 → 保留 V3 epilogue |
| **止损** | 向量化 STS 造成 bank conflict（excessive 上升）→ 回退 |

```bash
# 8B：若实现为运行期选择，则 v3 短 K 与 v2 对比即可验证上限
bash scripts/run.sh --mode bench --build-dir "$BUILD" --gpu 0 \
  --variants v2,v3 --baseline v2 --layouts NT,TN \
  --shapes 4096x4096x256,4096x4096x512,4096x4096x1024 \
  --repeats 3 --telemetry-ms 1000 --output-dir "$EXP/exp08_shortk"
```

---

### EXP-09 · Persistent tile scheduler

| 字段 | 内容 |
|---|---|
| **假设** | 1024³（1024 CTAs、2 CTAs/SM、4.5 波）存在尾部与提交空隙；persistent kernel 可减少波次量化与 launch/调度开销 |
| **变量** | 常规 grid vs persistent（grid = SM 数 × 2，内部循环 tile）；调度顺序按 N-major 或 swizzle 以保留 L2 复用 |
| **控制** | 同一 tile 计算与 epilogue；persistent 版本必须保证每个 tile 的 barrier 状态独立、无跨 tile 残留 |
| **正确性** | validate + 全形状数值校验；K=256 多槽位回绕；`synccheck` |
| **命令** | 新变体 `v6_persist`；bench 后用 nsys 看提交间隙 |
| **NCU 指标** | `gpu__time_duration.avg`、achieved occupancy、Tensor%、DRAM/L2；nsys 时间线的 kernel 间隙与尾部 |
| **判定** | 1024³ +5% 且 4096³ 不回退 → 对小形状启用；若仅 launch 开销受益而 kernel duration 不变 → 收益来自主机侧，需单独报告 |
| **止损** | persistent 导致 L2 复用下降、4096³ 回退 >2% → 仅对小形状启用或放弃 |

```bash
bash scripts/run.sh --mode bench --build-dir "$BUILD_V6" --gpu 0 \
  --variants v3,v6_persist --baseline v3 --layouts NT,TN \
  --shapes 1024x1024x1024,4096x4096x4096 \
  --repeats 3 --telemetry-ms 1000 --output-dir "$EXP/exp09_bench"
bash scripts/run.sh --mode nsys --build-dir "$BUILD_V6" --gpu 0 \
  --variants v3,v6_persist --layouts NT --profile-shape 1024x1024x1024 \
  --output-dir "$EXP/exp09_nsys"
```

---

### EXP-10 · Cold vs warm 与按角色归因

| 字段 | 内容 |
|---|---|
| **假设** | cold kernel replay 会清空 L2 并重置 TMA 前置状态，可能夸大 V4/V5 NT 的 starvation；若 warm 下 NT 差距仍在，则瓶颈是真实的 |
| **变量** | 缓存控制 cold vs warm（application replay）；source page 按 SASS PC 区间把 producer/consumer 分开 |
| **控制** | 同一变体、同一形状、同一时钟策略（`--clock-control none`） |
| **正确性** | NCU 的 `--check-exit-code yes` 已保证数值校验通过；warm 需更长 timeout |
| **命令** | 见下方代码块 |
| **NCU 指标** | Tensor%、DRAM/L2%、`l1tex__tmain_requests.sum`、crossbar sectors、barrier/gmma/long_scoreboard stall、NANOSLEEP 计数 |
| **判定** | warm 下 V4 NT 与 V3 NT 的 Tensor 差距缩小 ≥10pp → cold 伪影，结论需限定；差距仍 ≥3pp → R-02 在稳态也成立 |
| **止损** | warm 运行超时或无法稳定复现，保留 cold 结论并明确标注 |

```bash
bash scripts/run.sh --mode ncu --build-dir "$BUILD" --gpu 0 \
  --variants v3,v4,v5 --layouts NT --profile-shape 4096x4096x4096 \
  --ncu "$NCU" --ncu-set full --ncu-cache warm --ncu-clock none --launch-skip 50 \
  --timeout 7200 --output-dir "$EXP/exp10_warm"

# 冷报告按 PC 分角色（producer/consumer 的 SASS 地址区间需从 disasm 确认）
"$NCU" --import "$EXP/exp01_ncu/ncu_v4_4096x4096x4096_NT_cold.ncu-rep" \
  --page source --csv > "$EXP/exp10_v4_nt_source.csv"
```

> NCU 不提供“按 warp 角色分组”的硬件计数器；角色归因只能依赖 PC 区间与采样分布，这是本系列所有 WS 结论的边界。

---

### EXP-11 · `clock64()` 直接测量等待与发射时长

| 字段 | 内容 |
|---|---|
| **假设** | PC 采样只能给“哪里等”，不能给“等多久”；在 consumer full-barrier 等待前后、producer 每 tile 发射前后插入 `clock64()`，可直接量化饥饿 |
| **变量** | 诊断构建：每个 CTA 的 consumer warp 0 lane 0 记录每 tile 的 full wait 周期；producer lane 0 记录每 tile 的 TMA 发射周期；写入全局缓冲，host 汇总中位数/P95 |
| **控制** | 只在单 CTA（grid 1×1 或按 blockIdx 过滤）记录，避免写带宽污染；诊断构建的 TFLOPS 不作成绩 |
| **正确性** | 插桩不得改变 barrier 调用次数与顺序；仅读取 `clock64()`，不引入额外同步 |
| **命令** | 需新诊断变体；运行 4096³ 与 K=256 各一次，host 打印直方图 |
| **判据** | consumer 等待中位数 > 一个 k-tile 的 WGMMA 执行时间（可由 Tensor% 与 duration 估算）→ 数据饥饿实锤；producer 发射中位数随 atom 数线性增长 → 串行发射实锤 |
| **止损** | 插桩本身改变寄存器/占用导致行为变化 → 只把它当定性证据 |

---

## 4. 缺失数据与补充方式

| 缺失项 | 现状 | 补充方式 | 影响的结论 |
|---|---|---|---|
| TMA 原始请求数 | focused 报告只有 `...sum.pct...`；原始计数未采集 | EXP-01/02 用 `--metrics l1tex__tmain_requests.sum` 重采 | R-03 的“16× 请求”目前由 UTMALDG 指令数推得，需原始计数确认 |
| L1 sector 命中率（V3+） | 报告值为 0，判定为未采集/不支持 | 重采 `l1tex__t_sector_hit_rate.pct`，并与 `l1tex__m_xbar2l1tex_read_sectors.sum` 联用 | 不影响当前主结论，但限制对 L1 路径的解释 |
| 每 warp/每角色 stall | NCU 只给 SM 级聚合与 PC 采样 | PC 区间归因（EXP-10）+ `clock64()`（EXP-11） | V4/V5 的 consumer starvation 目前是“高置信现象 + 机制假设” |
| TMA 单请求延迟 | 无直接指标 | EXP-11 的 producer 发射时长 + full wait 时长 | 无法区分“TMA 引擎排队”与“DRAM/L2 延迟” |
| NCU 期间的时钟/功耗 | `--clock-control none`，仅报告时长 | 报告前后 `nvidia-smi` + telemetry；用 `sm__cycles_elapsed.avg.per_second` 估算实际时钟（先 query 确认） | 防止把降频误读为优化收益 |
| benchmark 稳态功耗/时钟 | 现有 env 只有前后快照 | 每轮 bench 保留 telemetry CSV 并纳入分析 | 4096³ PIPE 数据污染事件需要这种证据 |
| 小形状主机提交间隙 | 只有批均 Event 时间 | EXP-09 的 nsys 时间线 | 1024³ 的 TFLOPS 损失可能部分来自主机侧 |

---

## 5. 结果记录与版本扩展

每个实验完成后，在对应 kernel 笔记中追加一行“实验结论”，格式固定：

```text
EXP-xx | 日期 | 变体/布局 | 形状 | bench p50 (ms, 3 进程) | TFLOPS | Δ vs baseline | NCU 关键证据 | 判定（接受/否证/待定） | 备注
```

新增 V6/V7 时保持以下字段不变，以便直接并入 [06 · 横向对比](06-五个Kernel横向对比分析.md)：

- 版本 ID、CTA tile、PIPE、inflight、producer/compute 线程数、epilogue 类型；
- Tensor%、duration、DRAM/L2/L1TEX、eligible/issue/WCPI、barrier/gmma/long_scoreboard、regs/SMEM/occupancy；
- 报告路径、构建 manifest、源码 commit；
- 数据质量标记：独占/污染、cold/warm、clock control。

已否证的假设不删除，在 [07 · 根因总结](07-关键性能瓶颈与根因总结.md) 中把状态改为“已否证”并写明实验 ID；已证实的结论保留原始证据链接。

---

## 6. 最小执行集（如果时间有限）

只做三个实验即可覆盖最大的不确定性：

1. **EXP-01**：立即确定 NT 是否回退 V3（预期收益最大，零代码改动）。
2. **EXP-04 + EXP-06**：用干净数据确定 PIPE 策略（零代码改动，直接影响短 K）。
3. **EXP-02**：验证 V4/V5 NT 的 producer 串行假设（唯一能解释 WS 在 NT 失效的机制）。

EXP-03/05/07/08/09 属于结构性改造，只有在上述三项确认方向后再投入。
