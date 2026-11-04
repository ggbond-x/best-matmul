# 04 · Kernel V4（v4_warp_specialized）性能分析

> 报告：`ncu_v4_4096x4096x4096_{NT,TN}_cold.ncu-rep`、`ncu_v4_4096x4096x256_{NT,TN}_cold.ncu-rep`
> 源码：`kernels/wgmma_tma_fp16_warp_specialized_v4.cuh` + `kernels/detail/tma_wgmma_warp_specialized_device.cuh`
> 配置：CTA 128×128×64，PIPE=3，GMMA_N=128，**producer 128 threads + consumer 128 threads**，inflight=2，coalesced epilogue=1
> 基准：NT **367.5 TFLOPS（较 V3 −23.16%）** / TN **482.4 TFLOPS（+0.50%）** @4096³

---

## 1. 设计意图与执行流

V4 把 TMA 生产从计算 warpgroup 中拆出来，形成 classic warp specialization：

```text
Producer warpgroup（128 threads，但只有 thread 0 执行循环）
  for k_tile:
    wait empty[pipe]（第 1 轮后）
    arrive_and_expect_tx(full[pipe], 32 KB)
    TMA A ×16 atoms + TMA B ×16 atoms

Consumer warpgroup（128 threads）
  for k_tile:
    wait full[pipe]
    warpgroup_arrive → gemm(8×HGMMA) → commit
    if k_tile>=1: warpgroup_wait<1> → arrive empty[old pipe]
  warpgroup_wait<0>
```

关键源码：

| 行为 | 位置 |
|---|---|
| producer/consumer 判定 | `tma_wgmma_warp_specialized_device.cuh:63-64` |
| **只有 thread 0 执行 producer 循环** | `:79` `if (is_producer_leader)` |
| producer 等空槽 + TMA | `:79-90` |
| producer tail waits | `:94-100` |
| consumer full barrier wait | `:122` |
| WGMMA + commit | `:123-125` |
| `wait<1>` / 释放空槽 | `:136-139` |
| 最终 `wait<0>` / 释放 | `:142-147` |
| 合并写回 | `:153-187` |

**与 V3 的差异不只是“多了一个 producer warpgroup”**：

1. producer 循环由 **单个线程** 执行，其余 127 个 producer 线程不参与主循环，直接在 epilogue 的 `__syncthreads` 等待；
2. producer 独立运行，不受 consumer 的 WGMMA 节奏约束，会尽早把 3 个 stage 全部填满并保持满队列；
3. SASS 中每个 TMA atom 前都生成 `ELECT`（`ELECT` 数 = `UTMALDG` 数 = 2,097,152），而 V1–V3 只有 4,096 条 `ELECT`；
4. block 从 128 线程变为 256 线程，寄存器 109/110，SMEM 仍是 98,432 B，仍是 2 CTAs/SM。

---

## 2. 关键 NCU 指标（4096³）

| 指标 | V3-NT | **V4-NT** | V3-TN | **V4-TN** | 解读 |
|---|---:|---:|---:|---:|---|
| Duration | 270.91 µs | **375.58 µs** | 275.49 µs | **271.33 µs** | NT **+38.6% 时间**；TN −1.5% |
| Tensor active | 77.04% | **51.78%** | 77.18% | **78.13%** | NT **−25.3 pp**；TN +0.95 pp |
| Compute | 68.93% | 50.66% | 65.56% | 70.87% | NT 全面下降 |
| DRAM | 44.64% | 32.31% | 44.07% | 44.68% | NT 绝对带宽 909.98→658.82 GB/s |
| L2 | 68.75% | 51.40% | 66.69% | 70.00% | NT 未饱和且更低 |
| L2 hit | 80.52% | 77.21% | 75.67% | 76.90% | 基本不变 |
| L1TEX read sectors | 35.09% | **24.77%** | 34.42% | **35.11%** | NT 数据路径利用率仅 TN 的 0.71× |
| TMA pipe active | 9.63% | 6.47% | 0.60% | 0.61% | NT 请求多，但管线未饱和 |
| TMA requests | 4.39% | 3.10% | 0.27% | 0.27% | NT/TN ≈ **11.5×**（百分比之比；原始请求数为 16×） |
| Achieved Occupancy | 12.35% | **23.09%** | 12.28% | **24.51%** | 线程翻倍，CTA 数不变 |
| Active warps / scheduler | 1.92 | **3.72** | 1.94 | **3.90** | 主要是空闲 producer warp |
| Eligible / scheduler | 0.29 | **0.25** | 0.22 | 0.24 | 可用 warp 没有增加 |
| Issue active | 25.84% | 21.40% | 20.67% | 19.91% | 发射率下降 |
| WCPI | 7.43 | **17.37** | 9.39 | **19.81** | 每条指令伴随更多等待 |
| Stall barrier | 0.10 | **6.52** | 0.17 | **7.55** | full barrier 自旋 |
| Stall wait | 0.97 | 2.00 | 1.23 | 1.33 | 固定延迟 |
| Stall gmma | 0.44 | **0.01** | 0.12 | 0.07 | WGMMA 完成等待被 inflight 掩盖 |
| Stall long scoreboard | 3.90 | **6.93** | 5.49 | **7.13** | barrier/TMA 依赖 |
| Stall sleeping | 0.00 | 0.01 | 0.00 | **0.90** | TN 的 NANOSLEEP 被计为 sleeping |
| Registers | 111 | 109 | 111 | 110 | 2 CTAs/SM 不变 |
| SMEM | 98,432 B | 98,432 B | 相同 | 相同 | 占用不变 |

**第一层结论**：V4 的 NT 回退不是带宽、不是 L2、不是 Tensor 算力不够，而是 **消费者在 full barrier 上等 TMA 数据**。DRAM 只有 659 GB/s（32%），L2 51%，Tensor 52%，三者同时低，只有“供给链断开”能解释。

---

## 3. 证据链：谁在等，等什么

### 3.1 Source/SASS 采样

| 项目 | V4-NT | V4-TN |
|---|---:|---:|
| 总采样 | 63,365 | 41,920 |
| `NANOSLEEP.SYNCS` 总量 | **24,950** | 14,352 |
| 其中 consumer full barrier 自旋 | **24,773（39.1%）** | 13,873（33.1%） |
| `BSSY`（epilogue 等待） | **22,915（36.2%）** | 15,101（36.0%） |
| `HGMMA` | 1,312（2.1%） | ~1,485（3.5%） |
| `stall_barrier` 占比 | 37.6% | 38.3% |
| `stall_long_scoreboard` 占比 | 40.2% | 36.2% |

### 3.2 消费者饥饿的量化

V4-NT 的 consumer spin PC：

```text
0x...6030  @!P0 NANOSLEEP.SYNCS 0x989680     samples 24,773 (39.1%)
           ie = 3,814,406
0x...6010  SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [R7+URZ+0x18000], R5
0x...6050  @!P0 BRA 0x...6010
```

- `3,814,406 / (1024 CTA × 64 tile × 4 consumer warp) = 14.55` 次自旋 / warp / tile；
- 对比 V2-NT：`262,144 / (1024 × 64 × 4) = 1.0` 次 / warp / tile；
- barrier 地址 `0x18000` 是动态 SMEM 末尾的 `full[pipe]`（与源码 `smem.tma_barrier` 一致）；
- 说明消费者每个 tile 都要等 **约 14.5 个 NANOSLEEP 周期** 才看到 TMA 完成。

### 3.3 生产者没有卡在空槽

V4-NT producer empty barrier wait：

```text
0x...3f50  @!P0 NANOSLEEP.SYNCS 0x989680     samples 158
           ie = 62,464 = 1024 CTA × 61 tile
```

- 生产者每个 tile 只探测一次空槽就成功，采样仅 158；
- `UTMALDG.2D` 的 32 个 PC 各执行 65,536 次（=1024×64），总 2,097,152，采样每个仅 ~10；
- 生产者不是被 empty barrier 卡住，而是把 TMA 请求发出去了；消费者却在等这些请求完成。

### 3.4 不是 TMA 指令管线饱和

- `sm__pipe_tma_cycles_active = 6.47%`（NT），`6.32%`（V5 NT）；
- `l1tex__m_xbar2l1tex_read_sectors = 24.77%`（NT）vs 35.11%（TN）；
- DRAM 659 GB/s、L2 51%，都低于 V3；
- 所以不是“TMA 指令发射满了”或“L1/DRAM 带宽满了”，而是 **单个 tile 的 TMA 完成延迟超过了 3-stage 流水能覆盖的范围**。

### 3.5 BSSY 36% 不是主循环 barrier

- `BSSY @0x...6770` 只执行 8,192 次 = 1024 CTA × 8 warp（4 producer + 4 consumer 的 epilogue 分支）；
- 它位于 epilogue 的 `__syncthreads` 前后，producer 的 3 个非 leader warp 从 kernel 一开始就跳到这里等待；
- 这 36% 的采样是“空闲 producer warp”造成的统计放大，**不代表主循环有 36% 的 barrier 开销**；
- 也解释了“barrier stall 高 + Tensor 高”的表象：TN 的 barrier 7.55 很高，但 Tensor 78.13%，因为等待的是空闲 warp，不是 Tensor。

> 如果只看 `stall_barrier` 或 `BSSY`，会错误地得出“V4 barrier 太多”的结论；必须结合 PC 地址、执行次数和 Tensor active 判断。

---

## 4. 为什么 TN 没事，NT 崩了？

| 项目 | NT | TN |
|---|---|---|
| TMA atom | 1 KB（64×8） | 16 KB（64×128） |
| UTMALDG / tile | 32 | 2 |
| 总 UTMALDG | 2,097,152 | 131,072 |
| V4 Tensor | **51.78%** | **78.13%** |
| V4 duration | 375.58 µs | 271.33 µs |
| V3 Tensor | 77.04% | 77.18% |
| V4 vs V3 | **−25.3 pp** | +0.95 pp |

同一个 V4 代码、同一个 producer 结构，TN 甚至略有提升，NT 却回退 25 pp。**唯一的结构差异是每 tile 的 TMA 请求数与单请求大小**。这把问题从“WS 本身”收窄到“**WS producer 与细粒度 TMA 的交互**”。

### 4.1 候选机制（按证据强度排序）

**H1（较高可能性）：单线程 producer 的突发式 TMA 发射，放大了 32 个小请求的完成尾部延迟。**
- V4 producer 独立运行，3 个 stage 填满后仍以最快速度补槽；每个 tile 的 32 个 1 KB 请求在很短时间内全部发出；
- 消费者必须等该 tile 的 **全部 32 个请求** 完成才能翻转 full barrier，尾部一个慢请求就会拖住整个 tile；
- V2 的 producer 与 consumer 在同一 warpgroup，TMA 发射被 WGMMA 节奏“摊平”，32 个请求分散在 tile 时间内，尾部延迟不被集中暴露；
- 证据：V4 消费者每 tile 14.5 次自旋 vs V2 1 次；producer 不卡空槽；UTMALDG 本身几乎不 stall；TMA pipe 未饱和。

**H2（较高可能性）：per-atom `ELECT` 使单线程 producer 的 TMA 发射路径变长。**
- SASS：V4/V5 的 `ELECT` 数 = `UTMALDG` 数 = 2,097,152；V1–V3 仅 4,096；
- 每个 atom 都执行一次 warp election，32 次/tile；在单 active lane 的 warp 上，这仍是 32 次额外的 warp 指令与依赖；
- 它不会单独造成 23% 的回退，但会放大 H1 的突发长度；
- 需 ablation（把 election 提到循环外 / 用 4 个 producer warp 并行发 8 个 atom）验证。

**H3（待验证）：3-stage 的 outstanding 请求数不足。**
- 每 CTA 3 stage × 32 atom = 96 个 1 KB 请求在途，2 CTAs/SM = 192；
- 若 TMA/L1TEX 的每 SM outstanding 容量小于该值，完成会出现排队；V2 的在途数被节奏压低，反而不排队；
- 现有报告没有 TMA 队列深度/延迟计数器，无法直接证实；需 PIPE/atoms 的 ablation。

**H4（较低可能性）：producer warpgroup 的空闲线程/寄存器影响。**
- 127 个 producer 线程不执行主循环，只在 epilogue 等 `__syncthreads`；它们不发射指令，但占用 warp slot 与寄存器（109×256×2 = 55,808 regs/SM）；
- occupancy 仍是 2 CTAs/SM，consumer warp 数不变；没有证据表明它们直接拖慢 TMA；
- BSSY 采样高是统计假象，不应作为瓶颈证据。

**H5（已排除）：带宽/L2/Tensor 饱和。** DRAM 32%、L2 51%、Tensor 52%，同时低。

### 4.2 为什么 V5 也没有救回 NT？

V5 把 consumer 扩到 2 个 warpgroup（256 threads），WGMMA 指令总数仍 2,097,152，但 consumer warp 数翻倍：

| 指标 | V4-NT | V5-NT |
|---|---:|---:|
| Tensor | 51.78% | **50.56%** |
| Duration | 375.58 µs | **382.94 µs** |
| Consumer NANOSLEEP 指令 | 3,814,406 | **8,418,228** |
| TMA requests | 3.10% | 3.06% |
| DRAM | 32.31% | 31.75% |

**增加消费者不增加供给**：TMA 交付没有变快，消费者越多，等得越久，Tensor 反而略降。这反向支持“瓶颈在 producer/TMA 交付，而不在 consumer 算力”。

---

## 5. 诊断表

| 优先级 | 性能问题 | NCU 证据 | 对应代码 | 根因假设 | 置信度 |
|---|---|---|---|---|---|
| **P0** | NT 消费者在 full barrier 上饥饿 | Tensor 51.8%（−25.3 pp）；consumer NANOSLEEP 3,814,406 次（14.5/warp/tile）；barrier stall 6.52；DRAM 659 GB/s、L2 51% | `tma_wgmma_warp_specialized_device.cuh:122` | TMA 数据交付延迟超过 3-stage 覆盖 | **已证实（现象）** |
| **P0** | WS producer + NT 细粒度 TMA 的交互 | NT/TN 同代码，Tensor 51.8 vs 78.1；TMA 请求 3.10% vs 0.27%；UTMALDG 2,097,152 vs 131,072；producer empty wait 仅 158 samples | `:79-90`；`tma_wgmma_adapter.cuh:77-78` | 单线程突发式发射 32×1 KB，完成尾部延迟被 full barrier 放大 | **较高可能性** |
| **P1** | per-atom `ELECT` | `ELECT=2,097,152`（=UTMALDG），V1–V3 仅 4,096；每个 UTMALDG 前一条 ELECT | TMA copy 路径（CuTe copy atom） | 单 lane 每 atom 选举，拉长发射路径、放大突发 | **较高可能性（机制）**，性能贡献**待验证** |
| P2 | epilogue BSSY 36% | BSSY ie=8,192；位于 epilogue `__syncthreads`；producer 3 个非 leader warp 全程空闲 | `:79`、`:153-187` | 空闲 producer warp 的采样放大，不是主循环 barrier | **已证实** |
| P2 | producer warpgroup 127 线程空转 | block 256，producer 仅 thread 0 工作；occupancy 23% 但 eligible 仅 0.25 | `:79` | 资源占用但不直接拖慢主循环 | **已证实（现象）**，影响**待验证** |
| P2 | TN 的 WS 收益有限 | Tensor 77.18→78.13%；TFLOPS +0.50%；barrier 7.55 | `:114-149` | TN TMA 请求少，producer 不成为瓶颈；收益来自解耦 | **已证实** |

---

## 6. Top 3 问题

1. **NT 的消费者饥饿**：full barrier 每 tile 14.5 次自旋，Tensor 只有 51.8%，是 V4 的唯一主因。
2. **WS producer 与细粒度 TMA 的交互**：同代码 TN 78.1%、NT 51.8%，差异锁定在每 tile 32×1 KB 请求与单线程突发发射。
3. **per-atom ELECT 与 producer 单线程结构**：发射路径被拉长，且 127 个 producer 线程空转；是否是主因需 ablation。

## 7. Top 3 优化方向

1. **NT 直接回退到 V3**（非 WS，inflight=2）：这是当前证据下最稳的决策，预计 NT 从 367.5 恢复到 478 TFLOPS（+30%）。
2. **改造 TMA 发射结构**：把 32 个 atom 分给 4 个 producer warp 并行发射，或把 election 提到循环外，减少单线程突发长度；目标是降低每 tile 完成尾部延迟。
3. **改变 NT 的数据供给形态**：更大 TMA box（换 SMEM atom/swizzle）、输入预转置为 TN、或用 cp.async 批量搬运；从根上把请求数降下来。

## 8. 报告不能回答的问题

- TMA 从发出到 full barrier 翻转的延迟分布与队列深度；
- 32 个 1 KB 请求的“尾部延迟”具体由请求率、请求大小还是 mbarrier transaction 更新导致；
- per-atom ELECT 的实际周期成本（需要 instruction-level 计时/ablation）；
- 2 CTAs/SM 的跨 CTA 交错对 NT 的影响（需 1 CTA 对照）；
- warm-cache 下消费者饥饿是否缓解；
- 生产者 4 个 warp 并行发射是否能提升（需新 kernel）。

---

**证据等级汇总**：消费者饥饿与 NT/TN 分化已证实；突发式发射 + 细粒度 TMA 为较高可能性；ELECT 与队列深度为待验证假设；BSSY 为已证实的统计假象。
