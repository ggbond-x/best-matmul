# 06 · 五个 Kernel 横向对比分析

> 数据来源：`bench_core/summary.csv`（干净基准，A 级）+ 4096³/K=256 的 NCU 报告（B 级，仅机制对照）
> 排名口径：同 shape、同 layout、同 CTA tile、同 PIPE、同 GMMA_N；`TFLOPS = 2MNK / median_p50_ms / 1e9`
> 变化率：`(new/old − 1) × 100%`

---

## 1. 可比较的 TFLOPS 排名

### 1.1 全形状表（单位 TFLOPS）

| Shape | V1-NT | V2-NT | V3-NT | V4-NT | V5-NT | V1-TN | V2-TN | V3-TN | V4-TN | V5-TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024³ | 116.8 | 147.2 | **150.2** | 92.8 | 93.0 | 141.2 | 184.0 | 180.2 | **190.0** | 190.3 |
| 4096³ | 420.2 | **480.0** | 478.3 | 367.5 | 364.1 | 470.1 | 470.7 | 480.1 | **482.4** | 478.6 |
| 8192³（不稳定） | 400.6 | 424.6 | **429.9** | 380.9 | 356.8 | 278.0 | 364.8 | 374.5 | 360.8 | **383.4** |
| K=256 | 176.5 | **191.3** | 186.9 | 184.3 | 181.4 | 191.1 | 195.5 | 188.2 | **203.3** | 203.0 |
| 16384×1024×4096 | 285.0 | 331.1 | **331.4** | 316.2 | 312.9 | **335.4** | 329.1 | 329.1 | 330.6 | 329.9 |
| 1024×16384×4096 | 405.4 | 460.7 | **466.0** | 361.6 | 354.8 | 449.3 | 455.9 | 469.2 | **473.8** | 471.1 |

**排名（仅 A 级：1024³、4096³、K=256、两个不规则 shape）**：

| 维度 | 第 1 | 第 2 | 第 3 | 第 4 | 第 5 |
|---|---|---|---|---|---|
| NT 综合 | **V3**（3 个 shape 第一，4096³ 与 V2 持平） | V2（K=256 与 4096³ 第一） | V1 | V4 | V5 |
| TN 综合 | **V4**（4 个 shape 第一/并列） | V5（1024³ 与 8192³ 略高） | V3 | V2 | V1 |
| 峰值单点 | **V4-TN 482.4 @4096³** | V3-TN 480.1 | V2-NT 480.0 | V5-TN 478.6 | V3-NT 478.3 |

> 8192³ 的 V1-TN 278.0、V5-NT 356.8 明显异常，`flagged` 与运行不稳定相关；不参与排名。

### 1.2 版本间 delta（4096³，干净基准）

| 对比 | NT TFLOPS | Δ | TN TFLOPS | Δ |
|---|---:|---:|---:|---:|
| V2 / V1 | 480.0 / 420.2 | **+14.2%** | 470.7 / 470.1 | +0.1% |
| V3 / V2 | 478.3 / 480.0 | −0.36% | 480.1 / 470.7 | **+1.99%** |
| V4 / V3 | 367.5 / 478.3 | **−23.16%** | 482.4 / 480.1 | +0.50% |
| V5 / V4 | 364.1 / 367.5 | −0.94% | 478.6 / 482.4 | −0.79% |
| V3 / V1 | 478.3 / 420.2 | **+13.8%** | 480.1 / 470.1 | +2.1% |

### 1.3 K 依赖性：V4 NT 的缺陷是“稳态”而非“启动”

| Shape | V3-NT | V4-NT | Δ TFLOPS | NCU Duration Δ |
|---|---:|---:|---:|---:|
| K=256 | 186.9 | 184.3 | **−1.4%** | +2.6%（47.2→48.4 µs） |
| 4096³ | 478.3 | 367.5 | **−23.2%** | **+38.6%**（270.9→375.6 µs） |

只有 4 个 k-tile 时 V4 几乎不落后；k-tile 增到 64 时差距拉到 23–39%。这排除了“prologue/epilogue 固定开销”，把 V4 NT 的问题锁定在 **主循环稳态的 TMA 供给**。

---

## 2. 跨版本 NCU 指标（4096³）

| 指标 | V1-NT | V2-NT | V3-NT | V4-NT | V5-NT | V1-TN | V2-TN | V3-TN | V4-TN | V5-TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Duration µs | 311.7 | 275.4 | **270.9** | 375.6 | 382.9 | 275.5 | 278.5 | 275.5 | **271.3** | 269.3 |
| Tensor % | 66.2 | 76.4 | 77.0 | 51.8 | 50.6 | 74.8 | 75.2 | 77.2 | 78.1 | **78.6** |
| Compute % | 67.2 | 69.7 | 68.9 | 50.7 | 52.7 | 73.3 | 69.8 | 65.6 | 70.9 | 71.7 |
| DRAM % | 37.9 | 43.9 | 44.6 | 32.3 | 31.8 | 44.5 | 43.4 | 44.1 | 44.7 | 44.8 |
| L2 % | 66.8 | 70.3 | 68.8 | 51.4 | 53.4 | 73.6 | 69.7 | 66.7 | 70.0 | 71.4 |
| L2 hit % | 80.6 | 78.2 | 80.5 | 77.2 | 80.2 | 76.7 | 77.2 | 75.7 | 76.9 | 78.3 |
| Occ % | 12.1 | 12.3 | 12.4 | 23.1 | 34.2 | 12.3 | 12.4 | 12.3 | 24.5 | 36.9 |
| Eligible/SMSP | 0.22 | 0.29 | 0.29 | 0.25 | 0.35 | 0.20 | 0.21 | 0.22 | 0.24 | 0.36 |
| Issue % | 20.9 | 25.9 | 25.8 | 21.4 | 24.4 | 18.7 | 19.7 | 20.7 | 19.9 | 24.1 |
| WCPI | 9.31 | 7.67 | 7.43 | 17.4 | 22.5 | 10.7 | 9.79 | 9.39 | 19.8 | 24.7 |
| Barrier | 3.07 | 0.10 | 0.10 | 6.52 | 5.75 | 4.11 | 0.14 | 0.17 | 7.55 | 6.56 |
| Wait | 0.88 | 0.98 | 0.97 | 2.00 | 2.25 | 1.04 | 1.31 | 1.23 | 1.33 | 1.77 |
| Long SB | 1.16 | 4.07 | 3.90 | 6.93 | 12.0 | 2.93 | 6.05 | 5.49 | 7.13 | 12.1 |
| Registers | 116 | 111 | 111 | 109 | **58** | 112 | 111 | 111 | 110 | **60** |
| SMEM B | 98,432 | 98,432 | 98,432 | 98,432 | 98,432 | 同 | 同 | 同 | 同 | 同 |
| CTAs/SM | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |

### 2.1 聚合与单位说明

- 所有 Tensor/DRAM/L2/Occupancy/Issue 均为 `pct_of_peak_sustained_*`，单位 %；
- `Eligible/SMSP` 与 `Active warps/SMSP` 为每周期平均 warp 数；
- `WCPI` 为 `smsp__average_warps_active_per_inst_executed.ratio`，无量纲；
- Stall 为 `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`，无量纲；
- Duration 为 NCU replay 单次 kernel，不是 benchmark 批平均；
- CTAs/SM 由 `launch__occupancy_limit_shared_mem` 与 `limit_registers` 取小值；V1–V5 均为 2。

---

## 3. 源码差异链：改了什么 → 意图 → NCU 结果 → 性能 → 为什么

### 3.1 V2：`wait<0>` → `wait<1>` + 延迟释放

| 项目 | 内容 |
|---|---|
| 改动 | 从第 2 批起 `warpgroup_wait<1>()`；用 `release_state` 释放**前一批**槽位（`tma_wgmma_device.cuh:193-198`） |
| 意图 | 让相邻 WGMMA 批次重叠，不增加 SMEM |
| NCU | NT Tensor 66.2→76.4%（+10.1 pp）；barrier 3.07→0.10；DEPBAR 采样 32.5%→0；long_sb 1.16→4.07 |
| 性能 | NT +14.2%；TN +0.1% |
| 为什么 | NT 的 TMA 交付慢，`wait<0>` 让 Tensor 在每个 tile 边界排空；inflight=2 把等待与上一批 WGMMA 重叠。TN 的 TMA 快，且 2 CTAs 的跨 CTA 交错已掩盖排空，故收益小 |
| 归因 | **单变量**（只改等待/释放），因果清晰 |

### 3.2 V3：合并写回

| 项目 | 内容 |
|---|---|
| 改动 | epilogue 先寄存器→SMEM，再 uint4 连续写；两次 `__syncthreads`（`tma_wgmma_device.cuh:209-239`） |
| 意图 | 把 C 的 16-bit 分散写变成 128-bit 合并写 |
| NCU | C sectors 2,097,152→1,048,576，excessive 1,048,576→**0**；新增 STS.U16 524,288、shared excessive 1,572,864、BAR.SYNC 8,192；Tensor 76.4→77.0/75.2→77.2 |
| 性能 | NT −0.36%（噪声内）；TN +1.99%；K=256 NT −2.3%、TN −3.7% |
| 为什么 | C 仅占 ~3% 流量，sector 减半不改变 Tensor 供给；短 K 下 SMEM 往返 + 两次 barrier 占比高 |
| 归因 | **单变量**（只改 epilogue），但收益与 K 强相关 |

### 3.3 V4：WS producer

| 项目 | 内容 |
|---|---|
| 改动 | producer 128 threads（仅 thread 0 执行循环）+ consumer 128；block 256；per-atom ELECT；epilogue 同 V3（`tma_wgmma_warp_specialized_device.cuh`） |
| 意图 | 解耦 TMA 与计算，让 producer 独立推进 |
| NCU | NT Tensor 77.0→**51.8%**；consumer NANOSLEEP 262k→3.81M；barrier 0.10→6.52；DRAM 910→659 GB/s；TN Tensor 77.2→78.1%，barrier 0.17→7.55 |
| 性能 | NT **−23.16%**；TN +0.50%；K=256 NT −1.4%、TN **+8.0%** |
| 为什么 | 同代码下 NT/TN 分化，锁定“WS producer × 32×1 KB TMA”的交互；NT 的 TMA 完成尾部延迟超过 3-stage 覆盖；TN 请求少，producer 解耦反而有利 |
| 归因 | **多变量**：WS + block 256 + leader-only loop + per-atom ELECT + epilogue 与 V3 相同。V4 vs V3 的 NT 崩溃不能只归给“WS”，需 ablation 分离 ELECT/发射结构/队列深度 |

### 3.4 V5：双 consumer warpgroup

| 项目 | 内容 |
|---|---|
| 改动 | consumer 256（2 WG）+ producer 128；block 384；TiledMMA warpgroup layout `Shape<2,1,1>`；每 WG 64 行；寄存器 58/60 |
| 意图 | 降寄存器、加 warp、提升 Tensor 占用 |
| NCU | NT Tensor 51.8→50.6%；consumer NANOSLEEP 3.81M→8.42M；WARPGROUP 524k→1,048k；SYNCS 8.10M→17.57M；TN Tensor 78.1→78.6%，duration 271.3→269.3 µs |
| 性能 | NT −0.94%；TN −0.79%（benchmark，噪声内）；K=256 TN 203.0 vs V4 203.3（持平） |
| 为什么 | 供给未变，消费者翻倍只增加等待；双 WG 的 empty count 256 与同步成本上升；TN 的 NCU 提升未转化为稳定 benchmark 收益 |
| 归因 | **多变量**：consumer 数 + block + 寄存器 + TMA 布局不变。PIPE=2 的对照证明 occupancy 被寄存器锁死（`limit_registers=2`） |

---

## 4. 横向矛盾与解释

### 4.1 “用了 WS，为什么 NT 反而更慢？”

- V4 NT −23.2%，V5 NT −0.9%；V4 TN +0.5%，V5 TN −0.8%。
- 若 WS 本身有害，TN 也应回退；实际 TN 不变或略好。
- 差异在 TMA 请求数：NT 32/tile、TN 2/tile。WS 改变了 producer 的**发射节奏**（从与 WGMMA 交错的“平滑”发射，变成单线程“突发”发射），把 32 个小请求的完成尾部集中暴露。
- **结论**：不是“WS 一定重叠”，也不是“WS 一定有害”；是 **WS 与 TMA 粒度的交互**。证据等级：较高可能性，需 ablation。

### 4.2 “PIPE=3 为什么比 PIPE=2 慢？”

K=256 的干净数据：

| 配置 | CTAs/SM | Tensor | TFLOPS | 结论 |
|---|---:|---:|---:|---|
| V3 PIPE=2 | **3** | 31.9/32.2% | **211.4/216.1** | stage 少但 CTA 多，短 K 更快 |
| V3 PIPE=3 | 2 | 28.1/28.9% | 186.4/186.7 | 基准 |
| V3 PIPE=4 | 1 | 21.8/24.0% | 148.9/154.1 | CTA 少，最慢 |
| V4 PIPE=2 | 2 | 24.3/29.6% | 165.1/194.7 | 寄存器限 2 CTAs，只损失深度 |
| V5 PIPE=2 | 2 | 24.0/30.8% | 161.4/196.9 | 寄存器 58×384×3 差 1,280 个到 3 CTAs |

**不是“PIPE=2 优于 PIPE=3”**，而是“**PIPE=2 在 V3 上把 CTAs/SM 从 2 提到 3**”。V4/V5 因为寄存器/线程 footprint 拿不到第 3 个 CTA，PIPE=2 变成纯损失。这个对照把“stage 数”与“occupancy”两个变量分离开了。

### 4.3 “barrier 很高，Tensor 也很高，矛盾吗？”

V4 TN：`stall_barrier=7.55`，Tensor 78.1%。不矛盾：

- barrier 采样包含 producer 的 3 个非 leader warp 在 epilogue `__syncthreads` 上空转（BSSY 36%）；
- 2 CTAs/SM 的跨 CTA 交错让一个 CTA 等 barrier 时另一个 CTA 继续发射 HGMMA；
- warp stall 是“warp 在等”，不是“Tensor 在空”。
- 判断必须看 `sm__pipe_tensor_cycles_active` 与 duration；单独看 barrier 会误判。

### 4.4 “occupancy 高了，为什么 TFLOPS 不涨甚至降？”

V4/V5 的 occupancy 23%/34–37%，V3 只有 12%，但 NT TFLOPS V4/V5 更低：

- occupancy 统计的是 **resident/active warps**，包含 4 个 producer warp（其中 3 个全程空闲在 epilogue）；
- `eligible` 只有 0.25/0.35，说明多出来的 warp 大多不可发射；
- CTAs/SM 仍是 2，真正做 WGMMA 的 consumer warp 数没有增加（V4 4/CTA，V5 8/CTA 但供给不变）；
- 所以“occupancy 高”是**线程数增加**的假象，不是“有效并行度增加”。

### 4.5 “内存指标改善了，为什么 GEMM 不变？”

V3 把 C 的 L2 sector 减半（excessive 1,048,576→0），DRAM 43.9→44.6%、L2 70.3→68.8%，TFLOPS 478.3 vs 480.0（−0.36%）。原因：C 只占 ~3% 流量，主循环的 A/B 供给与 Tensor 占用未变。**优化了非瓶颈指标**。

### 4.6 “NCU 单次更快，benchmark 却没有”

| 对比 | NCU | Benchmark | 解释 |
|---|---|---|---|
| V3-NT vs V2-NT | 270.9 vs 275.4 µs（V3 快） | 478.3 vs 480.0（V3 慢 0.36%） | 冷缓存单次 vs 热缓存批平均；差异在噪声内 |
| V5-TN vs V4-TN | 269.3 vs 271.3 µs（V5 快） | 478.6 vs 482.4（V5 慢 0.79%） | 同上；V5 的 NCU 优势不构成稳定收益 |

**排名以 benchmark 为准，NCU 只用于机制解释。**

---

## 5. 本地指标改善但 TFLOPS 不涨的清单

| 改动 | 改善的本地指标 | TFLOPS 结果 | 为什么无效 |
|---|---|---|---|
| V3 合并写回 | C L2 sector −50%，excessive→0 | NT −0.36%，TN +2.0% | C 只占 3% 流量；主循环未变 |
| V4 WS | Occupancy 12→23%，issue 21%，barrier 被独立 producer 承担 | NT −23.2% | 供给链变差；多出的 warp 不解决 TMA 尾部 |
| V5 双 WG | Occupancy 23→34–37%，寄存器 109→58，TN Tensor +0.4 pp | NT −0.9%，TN −0.8% | consumer 翻倍不增加 TMA 供给；同步成本上升 |
| V2 inflight=2 | Tensor +10 pp（NT） | NT +14.2%，TN +0.1% | **有效**：改了正确的瓶颈（批次串行） |

---

## 6. 混淆变量与建议 ablation

| 对比 | 同时变化的变量 | 建议 ablation | 目的 |
|---|---|---|---|
| V4 vs V3 | WS、block 256、leader-only producer、per-atom ELECT、epilogue 相同 | ① V4 NT 用 4 个 producer warp 并行发 8 atom；② election 提到循环外；③ V4 NT 输入预转置为 TN | 分离“WS 结构”与“TMA 发射粒度/节奏” |
| V4 vs V2 | WS、block、epilogue（V2 无合并写回） | 用 V3 作为基线（已做）；再跑 V4 with direct epilogue | 排除 epilogue 影响 |
| V5 vs V4 | consumer 数、block、寄存器、TMA 不变 | 固定 1 WG、只改寄存器/线程；或 2 WG + PIPE=2 + ≤56 regs | 分离“更多 consumer”与“occupancy/寄存器” |
| PIPE=2 vs 3 | stage 数、CTAs/SM（仅 V3）、流水深度 | 在 V3 上固定 2 CTAs（人为占 SMEM）对照 | 分离 stage 深度与 occupancy |
| 4096³ PIPE/M256 | 外部进程污染 | 重跑干净基准 + telemetry | 恢复绝对 TFLOPS 可信度 |

---

## 7. 设计优劣总结

| 设计 | 优势 | 代价 | 适用 |
|---|---|---|---|
| **V2/V3 非 WS + inflight=2** | 结构简单；NT 480 TFLOPS；3 CTAs 可调（PIPE=2）；C sector 可合并 | 单 warpgroup 同时管 TMA 与 WGMMA；短 K epilogue 有开销 | **NT 首选 V3**（短 K 用 V2） |
| **V4 单 consumer WG + 独立 producer** | TN 最高 482.4 TFLOPS；producer 解耦；短 K TN +8% | NT 因 TMA 粒度交互崩溃；producer 127 线程空转；per-atom ELECT | **TN 首选 V4** |
| **V5 双 consumer WG** | TN Tensor 78.6% 最高；寄存器 58；occupancy 37% | 同步成本翻倍；NT 无救；benchmark 无稳定收益 | 探索平台；不建议生产默认 |
| **V1 无 inflight** | 最简单 | Tensor 66–75%；NT 420 | 仅 baseline |

**最终推荐（当前证据）**：

```text
NT → V3（K 很大时）；K ≤ 512 可试 V2 或 V3 PIPE=2
TN → V4
V5 → 仅作为低寄存器/多 WG 实验分支
```

---

## 8. 报告不能回答的问题

- 4096³ 下 PIPE=2/4 的干净排序（现有基准污染）；
- V4 NT 的 TMA 尾部延迟来源（请求率/队列深度/请求大小）；
- V5 降到 56 regs 后 3 CTAs 的真实收益；
- 热缓存与冷缓存的差异；
- 功耗/时钟对长 kernel 的影响（benchmark 无同步 telemetry）。
