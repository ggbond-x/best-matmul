# 05 · Kernel V5（v5_multi_warpgroup）性能分析

> 报告：`ncu_v5_4096x4096x4096_{NT,TN}_cold.ncu-rep`、`ncu_v5_4096x4096x256_{NT,TN}_cold.ncu-rep`
> 源码：`kernels/wgmma_tma_fp16_multi_warpgroup_v5.cuh` + `kernels/detail/tma_wgmma_warp_specialized_device.cuh` + `kernels/detail/warp_specialized_config.cuh`
> 配置：CTA 128×128×64，PIPE=3，GMMA_N=128，**producer 128 + consumer 256（2 个 warpgroup）**，inflight=2，coalesced epilogue=1
> 基准：NT 364.1 TFLOPS（较 V4 −0.94%）/ TN 478.6 TFLOPS（−0.79%）@4096³

---

## 1. 相对 V4 的改动

V5 在 V4 的 producer/consumer 分离之上，把 consumer 扩成 **2 个 warpgroup**：

| 项目 | V4 | V5 |
|---|---|---|
| compute threads | 128（1 WG） | **256（2 WG）** |
| block threads | 256 | **384** |
| TiledMMA warpgroup layout | `Shape<1,1,1>` | `Shape<2,1,1>` |
| CTA_M 分配 | 1 WG 覆盖 128 行（2 个 64 行 atom） | 2 WG 各覆盖 64 行 |
| 每线程 accumulator | 128×128 / 128 = 128 half | 64×128 / 128 = 64 half |
| 寄存器/线程 | 109/110 | **58/60** |
| WGMMA 指令总数 | 2,097,152 | 2,097,152（不变） |
| `WARPGROUP` 指令 | 524,288 | **1,048,576** |
| empty barrier count | 128 | **256** |
| 动态 SMEM | 98,432 B | 98,432 B |
| CTAs/SM | 2 | 2 |

源码约束（`kernels/detail/warp_specialized_config.cuh`）：

- `kComputeWarpgroups = kComputeThreads / 128`；
- `static_assert`：compute threads 必须是 128 的倍数、≤896；
- `static_assert`：`CTA_M` 必须能被 `64 × warpgroups` 整除（128 可被 128 整除）；
- `make_tiled_mma` 的 warpgroup layout 由 `kComputeWarpgroups` 决定（`tma_wgmma_adapter.cuh:65-66`）。

**设计意图**：用两个 warpgroup 分担 128×128 的 CTA tile，降低每线程寄存器压力，增加可用于延迟隐藏的 warp 数；理论上可让 Tensor 流水更满。

---

## 2. 关键 NCU 指标（4096³）

| 指标 | V4-NT | **V5-NT** | V4-TN | **V5-TN** |
|---|---:|---:|---:|---:|
| Duration | 375.58 µs | **382.94 µs** | 271.33 µs | **269.31 µs** |
| Tensor active | 51.78% | **50.56%** | 78.13% | **78.56%** |
| Compute | 50.66% | 52.70% | 70.87% | 71.67% |
| DRAM | 32.31% | 31.75% | 44.68% | 44.82% |
| L2 | 51.40% | 53.37% | 70.00% | 71.35% |
| L2 hit | 77.21% | 80.23% | 76.90% | 78.28% |
| L1TEX read sectors | 24.77% | 24.50% | 35.11% | 35.40% |
| TMA pipe active | 6.47% | 6.32% | 0.61% | 0.61% |
| TMA requests | 3.10% | 3.06% | 0.27% | 0.28% |
| Achieved Occupancy | 23.09% | **34.19%** | 24.51% | **36.90%** |
| Active warps / scheduler | 3.72 | **5.49** | 3.90 | **5.90** |
| Eligible / scheduler | 0.25 | **0.35** | 0.24 | 0.36 |
| Issue active | 21.40% | 24.37% | 19.91% | 24.06% |
| WCPI | 17.37 | **22.54** | 19.81 | **24.73** |
| Stall barrier | 6.52 | 5.75 | 7.55 | 6.56 |
| Stall wait | 2.00 | 2.25 | 1.33 | 1.77 |
| Stall long scoreboard | 6.93 | **12.03** | 7.13 | **12.13** |
| Stall sleeping | 0.01 | 0.01 | 0.90 | 0.68 |
| Registers | 109 | **58** | 110 | **60** |
| SMEM | 98,432 B | 98,432 B | 相同 | 相同 |
| CTAs/SM | 2 | 2 | 2 | 2 |

**第一层结论**：
- NT 仍然只有 50.6% Tensor，且比 V4 更慢；**加倍 consumer 没有增加供给**；
- TN 达到全系列最高的 78.56% Tensor、最短 NCU duration 269.31 µs，但 benchmark 反而比 V4 低 0.79%（在噪声内）；
- Occupancy 从 23% 升到 34–37%，但 Tensor 没有同步上升——又一次证明 **occupancy 高 ≠ 更快**。

### 2.1 Source/SASS 采样

| 项目 | V5-NT | V5-TN |
|---|---:|---:|
| 总采样 | 65,233 | 42,128 |
| `NANOSLEEP.SYNCS`（consumer full barrier） | **33,995（52.1%）** | 20,067（47.6%） |
| 其中主自旋 PC | 33,868（51.9%），ie=**8,418,228** | — |
| `BSSY`（epilogue 空闲） | 16,399（25.1%） | 10,399（24.7%） |
| `HGMMA` | 1,281（2.0%） | ~1.4% |
| `stall_long_scoreboard` 占比 | **52.9%** | **49.5%** |
| `stall_barrier` 占比 | 26.3% | 27.1% |

Consumer 自旋次数：

```text
8,418,228 / (1024 CTA × 64 tile × 8 consumer warp) = 16.06 次 / warp / tile
```

对比 V4 NT 的 14.55 次、V2 NT 的 1.0 次。**consumer 越多，每个 warp 等得越久**，总自旋指令从 3.81M 涨到 8.42M。这是“供给未变、需求翻倍”的直接表现。

---

## 3. 为什么加 consumer 没用？

### 3.1 供给侧没有变化

| 指标 | V4-NT | V5-NT |
|---|---:|---:|
| TMA atom 数 / tile | 32 | 32 |
| UTMALDG 总数 | 2,097,152 | 2,097,152 |
| TMA requests % | 3.10% | 3.06% |
| TMA pipe active | 6.47% | 6.32% |
| DRAM GB/s | 658.82 | 647.41 |
| Consumer NANOSLEEP | 3.81M | 8.42M |

V5 没有改变 producer、TMA atom、stage 数、full barrier 语义；唯一变化是消费者从 4 warp 变成 8 warp。**供给链的瓶颈在 TMA 交付，增加消费者只会让更多 warp 一起等**。

### 3.2 两个 warpgroup 的额外同步成本

- `EmptyBarrier` count 从 128 变为 256：producer 必须等 **两个 warpgroup 的 256 个线程** 全部 arrive 才能覆盖槽位；
- 每个 consumer warpgroup 各自执行 `warpgroup_wait<1>()`，两个 WG 的完成时间不一致时，empty barrier 的翻转由较慢者决定；
- `WARPGROUP` 指令从 524,288 翻倍到 1,048,576（每 WG 各发一半 M），但 HGMMA 总数不变；
- `SYNCS` 指令从 V4 NT 8,096,418 涨到 17,565,734，barrier/MIO 压力翻倍。

这些成本解释了为什么 V5 的 issue active 更高（24.37%）、WCPI 更高（22.54），但 Tensor 反而略低：多出来的发射主要是在轮询 barrier，不是在算。

---

## 4. PIPE 扫描：V5 的 occupancy 被寄存器锁死

K=256 的 PIPE 扫描（干净基准）给出一个关键反例：

| 配置 | K=256 NT TFLOPS | K=256 TN TFLOPS | Achieved Occ | Registers | `limit_registers` | `limit_shared_mem` |
|---|---:|---:|---:|---:|---:|---:|
| V5 PIPE=2 | 161.4 | 196.9 | 33.34% / 32.74% | 58/60 | **2** | 2 |
| V5 PIPE=3 | **180.3** | **202.3** | 33.04% / 33.79% | 58/60 | 2 | 2 |
| V5 PIPE=4 | 119.7 | 167.7 | 18.27% / 18.07% | 58/60 | 2 | 1 |

对比 V3（非 WS，128 threads）：

| 配置 | K=256 NT TFLOPS | K=256 TN TFLOPS | Achieved Occ | CTAs/SM |
|---|---:|---:|---:|---:|
| V3 PIPE=2 | **211.4** | **216.1** | 17.66% / 17.65% | **3** |
| V3 PIPE=3 | 186.4 | 186.7 | 11.52% / 11.44% | 2 |
| V3 PIPE=4 | 148.9 | 154.1 | 6.19% / 6.20% | 1 |

**机制**：

- V3 PIPE=2 的 SMEM 降到 65,664 B，`limit_shared_mem=3`，寄存器 111×128×3 = 42,624 < 65,536，于是 **3 CTAs/SM**，短 K 直接 +13–16%；
- V4/V5 PIPE=2 虽然 SMEM 也降到 65,664 B，但 `limit_registers=2`：
  - V4：109 × 256 × 3 = 83,712 > 65,536；
  - V5：58 × 384 × 3 = 66,816 > 65,536（只差 1,280 个寄存器）；
- 因此 V4/V5 的 PIPE=2 **只减少流水深度，不增加 CTA 数**，所以更慢（V5 NT −10.5%，TN −2.7%）。

> 这是一个可以改变优化方向的发现：**V5 的寄存器已经降到 58，只要再降到 ≤56，PIPE=2 就可能拿到 3 CTAs/SM**。但 SMEM PIPE=3 仍限制 2 CTAs，所以必须同时用 PIPE=2。短 K 可能有收益；长 K 的深度损失需要实测。见 [08 · OPT-06](08-Kernel优化方案与优先级.md) 与 [09 · EXP-05](09-后续实验与NCU验证计划.md)。

---

## 5. TN：V5 的“最好”与 benchmark 的“没赢”

| 数据 | V4-TN | V5-TN | 判断 |
|---|---:|---:|---|
| NCU Tensor | 78.13% | **78.56%** | V5 略好 |
| NCU Duration | 271.33 µs | **269.31 µs** | V5 略好 |
| Benchmark 4096³ | **482.4 TFLOPS** | 478.6 TFLOPS | V4 高 0.79%（噪声内） |
| Benchmark K=256 | **203.3** | 203.0 | 持平 |
| Benchmark 1024³ | 190.0 | **190.3** | 持平 |
| Registers | 110 | 60 | V5 更省 |
| Barrier stall | 7.55 | 6.56 | V5 略好 |
| Consumer 自旋 | 337k | 605k | V5 翻倍 |

结论：
- V5 在 TN 上把 Tensor 推到全系列最高，但 **benchmark 没有可复现的提升**（0.79% 在运行噪声内）；
- 额外 128 个 consumer 线程、翻倍的 WARPGROUP/SYNCS、256-count empty barrier 换来的是“NCU 单次更好看”，不是稳定吞吐；
- 因此 **TN 的推荐仍是 V4**（更简单的 1 WG consumer），V5 只作为“低寄存器/多 WG”探索平台。

---

## 6. 诊断表

| 优先级 | 性能问题 | NCU 证据 | 对应代码 | 根因假设 | 置信度 |
|---|---|---|---|---|---|
| **P0** | NT 仍被 TMA 供给限制，加 consumer 无效 | Tensor 50.56%（−1.2 pp vs V4）；consumer NANOSLEEP 8.42M（16.1/warp/tile）；TMA requests 3.06%、DRAM 647 GB/s | `tma_wgmma_warp_specialized_device.cuh:114-149` | 供给链未变，消费者翻倍只增加等待 | **已证实** |
| **P0** | 双 WG 的 empty barrier 与 WGMMA 同步成本 | WARPGROUP 524,288→1,048,576；SYNCS 8.10M→17.57M；empty count 128→256；WCPI 17.37→22.54 | `warp_specialized_config.cuh`；`:136-147` | 两个 WG 完成时间不一致，较慢者决定槽位释放 | **已证实（成本）**，对 Tensor 的定量影响**较高可能性** |
| **P1** | PIPE=2 不增加 occupancy | K=256：PIPE=2 33.34% vs PIPE=3 33.04%；`limit_registers=2`；58×384×3=66,816 > 65,536 | `wgmma_config.cuh:16-17`；block 384 | 寄存器/线程 footprint 使 3 CTAs 不可达；仅损失流水深度 | **已证实** |
| **P1** | TN 的 NCU 提升未转化为 benchmark | NCU Tensor 78.56%、269.31 µs；benchmark 478.6 vs V4 482.4（−0.79%） | — | 差异在噪声内；多 WG 开销抵消 | **已证实（无稳定收益）** |
| P2 | 低寄存器未兑现为 occupancy | 58 regs 但 SMEM PIPE=3 仍限 2 CTAs；PIPE=2 时差 1,280 regs 到 3 CTAs | 同上 | 需要 ≤56 regs + PIPE=2 才能 3 CTAs | **待验证** |
| P2 | epilogue BSSY 25% | BSSY ie 对应 epilogue 分支；producer 非 leader warp 空闲 | `:79`、`:153-187` | 空闲 warp 采样放大 | **已证实** |

---

## 7. Top 3 问题

1. **NT 的供给瓶颈没有被触动**：加 consumer 只让自旋从 3.81M 涨到 8.42M，Tensor 反而降。
2. **双 WG 的同步/barrier 成本翻倍**：empty count 256、WARPGROUP/SYNCS 翻倍，换来 TN 不足 1% 的 NCU 提升。
3. **PIPE=2 的 occupancy 红利被寄存器锁死**：V5 58 regs 已很接近 56 的门槛，但当前配置拿不到 3 CTAs。

## 8. Top 3 优化方向

1. **NT 用 V3，不用 V4/V5**：当前证据下这是唯一稳定的 NT 方案。
2. **V5 降寄存器到 ≤56 + PIPE=2**：尝试解锁 3 CTAs/SM，验证短 K 是否复制 V3 PIPE=2 的 +13–16%；若长 K 因深度不足回退，则按 K 分派。
3. **减少双 WG 的同步开销**：让每个 WG 独立使用自己的 empty barrier/槽位（按 M 半区拆分 pipeline），避免“较慢 WG 决定释放”；这是结构性改造，优先级低于前两项。

## 9. 报告不能回答的问题

- 两个 consumer WG 的完成时间差分布（需要 per-warp/per-WG 时间线）；
- 56 regs 是否可通过 `maxrregcount` 达成而不 spill；
- 3 CTAs/SM 下 NT 的 TMA 请求竞争是否恶化；
- TN 的 0.79% 差异是噪声还是真实（需要更多 repeats 的 A/B）；
- 长 K 下 V5 PIPE=2 + 3 CTAs 是否仍能维持 Tensor。

---

**证据等级汇总**：NT 供给限制与双 WG 成本已证实；PIPE=2 occupancy 被寄存器锁死已证实；降寄存器解锁 3 CTAs 为待验证方案；TN 无稳定收益已证实。
