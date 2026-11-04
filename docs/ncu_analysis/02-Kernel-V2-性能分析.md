# 02 · Kernel V2（v2_pipeline）性能分析

> 报告：`ncu_v2_4096x4096x4096_{NT,TN}_cold.ncu-rep`、`ncu_v2_4096x4096x256_{NT,TN}_cold.ncu-rep`
> 源码：`kernels/wgmma_tma_fp16_pipeline_v2.cuh` + `kernels/detail/tma_wgmma_device.cuh`
> 配置：CTA 128×128×64，PIPE=3，GMMA_N=128，128 threads，**inflight=2**，无合并写回
> 基准：NT **480.0 TFLOPS**（较 V1 **+14.2%**）/ TN **470.7 TFLOPS**（+0.1%）@4096³

---

## 1. 相对 V1 的改动

V2 只改流水等待策略，不改 tile、SMEM、线程数、TMA atom：

| 改动 | V1 | V2 |
|---|---|---|
| 每 tile 完成等待 | `warpgroup_wait<0>()`（`tma_wgmma_device.cuh:167`） | 从第 2 批起 `warpgroup_wait<1>()`（`:194`） |
| 释放槽位 | 本批全部完成后释放当前槽 | 等前一批完成后释放**前一批**槽（`release_state`，`:196`） |
| refill | 释放后立即补下一 tile（`:174`） | 同样在释放后补（`:198`） |
| 末尾 | 无额外等待 | `warpgroup_wait<0>()` 收尾（`:202`） |
| SMEM / occupancy | 98,432 B / 2 CTAs | 相同 |

执行流（稳态）：

```text
tile i:   wait full[i] → arrive → gemm(i) → commit
          if i>=1: wait<1>  ← 等 tile i-1 完成，tile i 仍在 Tensor 中
                   arrive empty[i-1] → TMA refill(i-1 的槽)
```

**含义**：Tensor 中始终允许 1 个批次在途，tile i 的 HGMMA 可以与 tile i−1 的完成等待重叠。这没有增加 SMEM，也没有改变数据依赖正确性：槽位 i−1 只在其 WGMMA 完成后才被覆盖。

---

## 2. 关键 NCU 指标（4096³）

| 指标 | V1-NT | **V2-NT** | V1-TN | **V2-TN** | 变化解读 |
|---|---:|---:|---:|---:|---|
| Duration | 311.71 µs | **275.36 µs** | 275.49 µs | **278.50 µs** | NT −11.7%；TN +1.1% |
| Tensor active | 66.23% | **76.37%** | 74.77% | **75.21%** | NT **+10.14 pp**；TN +0.44 pp |
| Compute | 67.21% | 69.69% | 73.29% | 69.83% | NT 综合吞吐反而略降 |
| DRAM | 37.87% | 43.92% | 44.51% | 43.44% | 均未饱和 |
| L2 | 66.79% | 70.28% | 73.63% | 69.67% | 未饱和 |
| L2 hit | 80.64% | 78.17% | 76.74% | 77.18% | 基本不变 |
| Eligible / scheduler | 0.22 | **0.29** | 0.20 | 0.21 | NT 调度余量改善 |
| Issue active | 20.92% | **25.88%** | 18.72% | 19.68% | NT 发射率提升 |
| WCPI | 9.31 | **7.67** | 10.69 | 9.79 | NT 指令间等待缩短 |
| Stall barrier | 3.07 | **0.10** | 4.11 | **0.14** | **几乎消失** |
| Stall wait | 0.88 | 0.98 | 1.04 | 1.31 | 略升 |
| Stall gmma | 2.41 | **0.47** | 0.50 | 0.13 | 完成等待被掩盖 |
| Stall long scoreboard | 1.16 | **4.07** | 2.93 | **6.05** | 新的主因：等 barrier/TMA |
| Occupancy | 12.09% | 12.31% | 12.33% | 12.38% | 不变 |
| Registers | 116 | 111 | 112 | 111 | 不变 |
| SMEM | 98,432 B | 98,432 B | 相同 | 相同 | 2 CTAs/SM |

### 2.1 Source/SASS 采样

| 项目 | V2-NT | V2-TN |
|---|---:|---:|
| 总采样 | 42,978 | 43,462 |
| `NANOSLEEP.SYNCS`（full barrier 自旋） | **18,900（44.0%）** | **23,919（55.0%）** |
| `HGMMA` | 4,340（10.1%） | ~8.2% |
| `stall_long_scoreboard` 占比 | **52.7%** | **61.1%** |
| `stall_wait` 占比 | 12.4% | 12.9% |
| `stall_barrier` 占比 | **1.3%** | **1.4%** |

SASS：

- `NANOSLEEP.SYNCS 0x989680 @0x...6580`：18,900 samples，指令执行 **262,144** 次 = 1024 CTA × 64 tile × 4 warp，即 **每 warp 每 tile 恰好一次自旋**；
- 后续 `SYNCS.PHASECHK.TRANS64.TRYWAIT P1, [R13+0x18000]` 读取 full barrier；
- `HGMMA` 每 tile 8 条，执行 262,144 次/PC，Tensor 批次内部仍是背靠背。

**对比 V1**：barrier stall 从 3.07/4.11 降到 0.10/0.14，说明 V1 的“等本批完成再释放”确实造成了 barrier/完成等待；V2 把这些等待转成了 full barrier 的 long-scoreboard 自旋（等 TMA 数据）。

---

## 3. 为什么 NT 提升 14.2%，TN 只提升 0.1%？

### 3.1 同代码下的不同响应

| 指标 | NT 变化 | TN 变化 |
|---|---:|---:|
| Benchmark TFLOPS | **+14.2%** | +0.1% |
| NCU Tensor | **+10.14 pp** | +0.44 pp |
| NCU Duration | −11.7% | +1.1% |
| V1 时 TMA atom | 1 KB ×32/tile | 16 KB ×2/tile |
| V1 时 Tensor | 66.2% | **74.8%** |

TN 在 V1 时 Tensor 已经 74.8%，且 2 CTAs/SM 的**跨 CTA 交错**已经掩盖了单个 CTA 的 `wait<0>` 排空：当一个 CTA 在 DEPBAR 等待时，同 SM 的另一个 CTA 可以发射 HGMMA。V1-TN 的 DEPBAR 采样占 38%，但 Tensor 仍高，正说明这些 warp 级等待没有等比例转化为 Tensor 空闲。

NT 在 V1 时只有 66.2%，因为每个 tile 的 32 个 1 KB TMA 请求交付更慢，`wait<0>` 之后的 full barrier 等待更长；跨 CTA 交错不足以填满空洞。V2 的 inflight=2 把“等 TMA”的时间与上一批 WGMMA 重叠，因此收益集中在 NT。

> 这是“**warp stall 占比高 ≠ Tensor 空闲同比例**”的典型例子：TN 的 DEPBAR 更高，但 Tensor 更高；判断必须回到 `sm__pipe_tensor_cycles_active` 与 duration。

### 3.2 V2 仍未解决的问题

- Tensor 仅 76%，仍有约 24% 空闲；
- 每 warp 每 tile 仍有一次 full barrier 自旋（262,144 次），`long_scoreboard` 占 53–61%；
- TMA atom 数未变（NT 2,097,152 条 UTMALDG）；
- PIPE=3、2 CTAs/SM 的预取深度未变；
- C 写回仍是 16-bit 标量（`STG.E.U16` 524,288，excessive 1,048,576）。

---

## 4. PIPE=2 的短 K 证据（干净基准）

K=256 的 PIPE 扫描没有外部进程污染，可直接比较：

| 配置 | K=256 NT TFLOPS | vs PIPE=3 | K=256 TN TFLOPS | vs PIPE=3 |
|---|---:|---:|---:|---:|
| V3 PIPE=2 | **211.4** | **+13.4%** | **216.1** | **+15.7%** |
| V3 PIPE=3 | 186.4 | — | 186.7 | — |
| V3 PIPE=4 | 148.9 | −20.1% | 154.1 | −17.5% |

原因不是“stage 越多越好”，而是 **occupancy**：

| PIPE | 动态 SMEM | CTAs/SM | K=256 Tensor | K=256 Occupancy |
|---|---:|---:|---:|---:|
| 2 | ≈65.6 KB | **3** | 31.90% / 32.16% | 17.66% / 17.65% |
| 3 | 98.4 KB | 2 | 28.10% / 28.88% | 11.52% / 11.44% |
| 4 | ≈131 KB | **1** | 21.77% / 24.04% | 6.19% / 6.20% |

短 K 下 prologue/epilogue 占比高，**3 CTAs/SM 的并行度**比第 3 个 stage 的预取深度更有价值；PIPE=4 掉到 1 CTA/SM，直接损失 1/3 的并行度。V2 的收益来自 inflight，不来自 stage 数。

> 注意：这只证明 K=256 的 PIPE=2 更好，**不能外推到 K=4096**。长 K 下 3 个 stage 是否必要，需要干净的 4096³ 重测（见 [09](09-后续实验与NCU验证计划.md)）。

---

## 5. 诊断表

| 优先级 | 性能问题 | NCU 证据 | 对应代码 | 根因假设 | 置信度 |
|---|---|---|---|---|---|
| **已解决** | V1 的 `wait<0>` 批次串行 | barrier 3.07→0.10；gmma 2.41→0.47；Tensor 66.2→76.4%；benchmark +14.2% | `tma_wgmma_device.cuh:193-198` | inflight=2 允许相邻批次重叠 | **已证实** |
| **P0** | 残余 24% Tensor 空闲，消费者等 TMA | long_scoreboard 4.07/6.05（占采样 53/61%）；NANOSLEEP 262,144 次（每 warp 每 tile 1 次） | `tma_wgmma_device.cuh:185` full barrier wait；TMA 在 `:143-146` | 3-stage 预取 + 2 CTAs/SM 不足以完全覆盖 NT 的 TMA 交付延迟 | **较高可能性** |
| **P1** | NT 每 tile 32 个 1 KB TMA 请求 | UTMALDG 2,097,152；tmain 4.30%；TMA pipe 9.55% | `tma_wgmma_adapter.cuh:77-78` | MN-major TMA box 64×8；请求数 16× TN | **已证实**（粒度）；对 V2 的定量影响**待验证** |
| P1 | TN 的 V2 收益被跨 CTA 交错掩盖 | TN Tensor 74.8→75.2%；duration +1.1%；barrier 4.11→0.14 | 同上 | 2 CTAs/SM 已提供足够的批次间重叠 | **较高可能性** |
| P2 | C 写回 2× sector | STG.E.U16 524,288；excessive 1,048,576 | epilogue `axpby` | 16-bit 标量写 | **已证实**，影响小 |
| P2 | 长 K 的 PIPE 选择未验证 | 4096³ PIPE 基准受污染 | `wgmma_config.cuh:16-17` | PIPE=3 是折中 | **待验证** |

---

## 6. Top 3 问题

1. **残余的 TMA 交付等待**：Tensor 76%，full barrier 自旋每 tile 一次，long-scoreboard 占 53–61%。
2. **NT 的 TMA 请求粒度过细**：32 个 1 KB 请求/tile，是 NT/TN 结构差异的根源。
3. **流水深度与 occupancy 的折中未系统验证**：PIPE=3 在长 K 下是否最优没有干净数据；短 K 已证明 PIPE=2 更好。

## 7. Top 3 优化方向

1. **在不增加 SMEM 的前提下继续提高在途批次数**（inflight 2 已用满 3 槽的安全上限；要再提高需 PIPE≥4，但会掉 occupancy，需实验权衡）。
2. **减少 NT 的 TMA 请求数**（更大 box / 单 lane 批量 / 预转置），把每 tile 32 请求降下来。
3. **合并写回**（V3）：虽然对 K=4096 只 +0.3%，但能消除 2× sector，并为后续 epilogue 优化铺路。

## 8. 报告不能回答的问题

- NT 的 TMA 交付延迟具体是多少个周期、由请求率还是单请求延迟主导；
- 2 CTAs/SM 的跨 CTA 交错对 Tensor 的贡献比例（需要 1 CTA/SM 对照）；
- 4096³ 下 PIPE=2/4 的真实排序（现有基准被污染）；
- warm-cache 下 full barrier 等待是否消失；
- producer lane 与 consumer warp 的分 warp stall 归因。

---

**证据等级汇总**：V2 的 inflight 收益已证实；残余 TMA 等待为较高可能性；NT 粒度对 V2 的定量贡献待验证。
