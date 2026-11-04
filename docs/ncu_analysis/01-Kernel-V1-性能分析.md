# 01 · Kernel V1（v1_baseline）性能分析

> 报告：`ncu_v1_4096x4096x4096_{NT,TN}_cold.ncu-rep`、`ncu_v1_4096x4096x256_{NT,TN}_cold.ncu-rep`
> 源码：报告快照为 `kernels/wgmma_tma_fp16_v1.cuh` + `kernels/detail/tma_wgmma_device.cuh`；当前 worktree 已内联于 `kernels/wgmma_tma_fp16_v1.cuh:163-181`
> 配置：CTA 128×128×64，PIPE=3，GMMA_N=128，128 threads，inflight=1，无合并写回
> 基准：NT 420.2 TFLOPS / TN 470.1 TFLOPS @4096³（`bench_core/summary.csv`）

---

## 1. 设计意图与执行流

V1 是最小可工作的 TMA + WGMMA 流水：

- **一个 warpgroup（128 threads）同时承担 TMA 生产与 WGMMA 计算**；
- 生产者是 `warp 0` 的 elected lane（`tma_wgmma_device.cuh:89`）；
- 3 个 SMEM 槽位，每个槽位放 A/B 各 128×64×FP16（合计 32 KB），加 barrier 后动态 SMEM 98,432 B；
- full barrier（`ClusterTransactionBarrier`，count=1）由 TMA 完成 transaction 后翻转；
- empty barrier（`ClusterBarrier`，count=128）由 128 个消费者线程各 arrive 一次后翻转；
- 每个 K tile 的时序：

```text
producer lane:  wait empty[pipe] → arrive_and_expect_tx(32 KB) → TMA A → TMA B
consumer 128T:  wait full[pipe] → warpgroup_arrive → gemm(8×HGMMA) → commit_batch
                → warpgroup_wait<0>  ← 等本批全部完成
                → arrive empty[pipe] → refill 下一 tile
```

关键源码：

| 行为 | 位置 |
|---|---|
| 生产者判定 | `kernels/detail/tma_wgmma_device.cuh:89` |
| 首次填满 3 槽 | `tma_wgmma_device.cuh:110-120` |
| 每 tile 等待空槽并 TMA | `tma_wgmma_device.cuh:141-153` |
| **每 tile `warpgroup_wait<0>`** | `tma_wgmma_device.cuh:167` |
| 释放槽位 | `tma_wgmma_device.cuh:171` |
| 再次 refill | `tma_wgmma_device.cuh:174` |
| 标量写回 `axpby` | `tma_wgmma_device.cuh` epilogue 分支（`kCoalescedEpilogue=false`） |

> 表中的行号对应报告快照 `source.tar.gz`（当时 V1 包含 `tma_wgmma_device.cuh`）。当前 worktree 已把设备实现内联到各版本头文件，该文件不再被引用；对应位置为 `kernels/wgmma_tma_fp16_v1.cuh:163-181`（`wait<0>` :173、释放 :177、refill :180）。

**设计含义**：`wait<0>` 使相邻 K tile 的 WGMMA 批次严格串行——本批 8 条 HGMMA 全部完成后，才释放槽位、补充 TMA、进入下一 tile。TMA 与 Tensor 只在“3 个预取槽”范围内重叠，Tensor 流水在每个 tile 边界被排空。

---

## 2. 关键 NCU 指标（4096³）

| 指标 | V1-NT | V1-TN | 解读 |
|---|---:|---:|---|
| Duration `gpu__time_duration.sum` | **311.71 µs** | **275.49 µs** | NT 慢 13.1% |
| Tensor active | **66.23%** | **74.77%** | 算力未打满，TN 更接近上限 |
| Compute throughput | 67.21% | 73.29% | 与 Tensor 同量级，不是内存受限 |
| DRAM throughput | 37.87% | 44.51% | 远未饱和 |
| L2 throughput | 66.79% | 73.63% | 与 Tensor 接近，是“陪跑”而非瓶颈 |
| L2 hit rate | 80.64% | 76.74% | NT 复用更高，但没换来更快 |
| L1TEX read sectors | 30.37% | 34.48% | 数据路径有余量 |
| TMA pipe active | 8.28% | 0.58% | 仅指令发射占用；NT 请求多 16× |
| TMA requests | 3.80% | 0.27% | NT 1 KB atom ×32/tile，TN 16 KB ×2/tile |
| Achieved Occupancy | 12.09% | 12.33% | 2 CTAs/SM，7.74 warps/SM |
| Active warps / scheduler | 1.95 | 2.00 | 每 SMSP 约 2 warps |
| Eligible warps / scheduler | **0.22** | **0.20** | 调度器大多数周期无 warp 可发 |
| Issue active | 20.92% | 18.72% | 发射槽利用率低 |
| WCPI | 9.31 | 10.69 | 每条指令平均伴随 ~9–11 个活跃 warp 的等待 |
| Stall barrier | **3.07** | **4.11** | 等 full/empty barrier |
| Stall wait | 0.88 | 1.04 | 固定延迟等待 |
| Stall gmma | **2.41** | 0.50 | NT 的 WGMMA 完成等待更突出 |
| Stall long scoreboard | 1.16 | **2.93** | TN 的 TMA/mbarrier 依赖更突出 |
| Stall short scoreboard | 0.20 | 0.26 | 可忽略 |
| Registers / thread | 116 | 112 | 不构成 occupancy 限制 |
| Dynamic SMEM | 98,432 B | 98,432 B | **2 CTAs/SM 的唯一限制** |
| Grid / waves | 1024 / 2.25（NCU 理论值） | 1024 / 2.25（NCU 理论值） | 按寄存器上限 4 blocks/SM 计算；实际被 SMEM 限到 2 CTAs/SM，等效 **4.49 波**，尾损约 10% |

> **Waves 口径**：`launch__waves_per_multiprocessor=2.25` 是 NCU 的理论值（1024/(114×4)，分母使用寄存器允许的 4 blocks/SM）。实际 `launch__occupancy_limit_shared_mem=2`，并发 CTA 数为 228，因此**实际是 4.49 波**。最后一波只有 112 个 CTA、116 个槽位空转，按等长 tile 估算上限损失约 10%。固定 tile 的 persistent 调度无法消除这个整数尾（1024 = 228×4 + 112），只能通过改 tile 粒度或提高 CTAs/SM 缓解；见第 8 节。

### 2.1 Source/SASS 采样

| 项目 | V1-NT | V1-TN |
|---|---:|---:|
| 总采样 | 49,289 | 43,504 |
| `WARPGROUP.DEPBAR.LE`（wait<0>） | **32.5%** | **38.0%** |
| `HGMMA.64x128x16.F16` | 28.4% | 8.2% |
| `NANOSLEEP.SYNCS`（mbarrier 自旋） | 1,876 条 | 9,062 条 |
| `stall_barrier` 占比 | 32.7% | 38.3% |
| `stall_long_scoreboard` 占比 | 12.6% | 27.3% |

SASS 关键 PC（V1-TN）：

- `WARPGROUP.DEPBAR.LE gsb0, 0x0 @0x...5090`：16,528 samples，指令执行 262,144 次（= 1024 CTA × 64 tile × 4 warp），对应 `warpgroup_wait<0>()`；
- `NANOSLEEP.SYNCS 0x989680 @0x...4b80`：8,814 samples，对应消费者在 full barrier 上的自旋；
- `SYNCS.PHASECHK.TRANS64.TRYWAIT P1, [R13+0x18000]`：barrier 位于动态 SMEM 末尾 0x18000，与源码 `producer_barrier/consumer_barrier` 一致。

> 重要：DEPBAR 是 **WGMMA 异步完成等待**，不是“barrier 指令过多”。它高说明每个 tile 都在等 Tensor 流水排空，属于数据依赖，不能删除，只能通过增加 inflight 批次来掩盖。

---

## 3. 核心瓶颈：每 tile `wait<0>` 把 WGMMA 批次串行化

### 3.1 现象

- Tensor 只有 66–75%，而 DRAM 38–45%、L2 67–74%，不存在带宽或 L2 饱和；
- Eligible warps 仅 0.20–0.22，issue 19–21%，说明 warp 经常全部在等；
- DEPBAR + barrier 合计占采样 65–76%；
- NT 比 TN 慢 13.1%，但 NT 的 L2 命中更高、DRAM 更低，排除“NT 更吃带宽”。

### 3.2 NCU → SASS → 源码 → 机制

1. NCU：`stall_gmma=2.41`（NT）、DEPBAR 32.5%；`smsp__warps_eligible=0.22`。
2. SASS：`WARPGROUP.DEPBAR.LE gsb0, 0x0` 每 tile 每 warp 执行一次，等待 `gsb0` 批次全部完成。
3. 源码：`tma_wgmma_device.cuh:167`（当前 `wgmma_tma_fp16_v1.cuh:173`）的 `warpgroup_wait<0>()`。
4. 机制：`wait<0>` 要求本 tile 的 8 条 HGMMA 全部完成。真正的依赖是 **SMEM 槽位复用**：V1 在等待之后才 `arrive` 释放槽位，于是释放、refill、full barrier、下一批 HGMMA 全部排在本批完成之后。累加器本身不要求批次间等待——同一 accumulator 的多个 WGMMA commit group 由硬件按序执行，标准 CUTLASS 主循环只在读累加器或复用 SMEM 前等待。Tensor 流水因此在批次边界排空；Tensor 利用率 = 批次内 HGMMA 背靠背效率 × 批次间无重叠比例。
5. 为什么 TN 仍有 74.8%：TN 的 TMA 是 2 个 16 KB atom，TMA 完成更快，full barrier 等待更短；但 DEPBAR 仍占 38%，说明 **wait<0> 是两种 layout 共同的上限**，只是 NT 叠加了更长的 TMA 交付。

### 3.3 正常等待还是瓶颈？

- 必须满足的依赖只有一条：**覆盖某个 SMEM 槽位前，使用该槽位的 WGMMA 批次必须已读完操作数**。
- 累加器不构成批次间依赖：同一 accumulator 的后续 WGMMA 可以在前一批未完成时发射（硬件按序执行），只需在读累加器前 `wait<0>`。
- 因此“每批都等到 0”是**调度选择**，不是正确性要求。V2 的 `wait<1>` 把“等本批”与“释放上一槽位”解耦，正是利用这一点。
- 所以这是**可优化的流水深度不足**，不是同步 bug。

---

## 4. NT vs TN：同代码、不同 TMA 粒度

| 项目 | NT（Major::MN） | TN（Major::K） |
|---|---|---|
| TMA atom（kernel 名） | `Copy_Traits<SM90_TMA_LOAD, C<8192>>`，box 64×8 = **1 KB** | `Copy_Traits<SM90_TMA_LOAD, C<131072>>`，box 64×128 = **16 KB** |
| atom 数 / 128×64 operand tile | 16 | 1 |
| UTMALDG / k-tile / CTA | 32 | 2 |
| UTMALDG 总数（4096³） | 2,097,152 | 131,072 |
| `l1tex__tmain_requests` | 3.80% | 0.27% |
| TMA pipe active | 8.28% | 0.58% |
| Tensor | 66.23% | 74.77% |
| Duration | 311.71 µs | 275.49 µs |

TMA atom 由 `kernels/detail/tma_wgmma_adapter.cuh:77-78`（当前 worktree 为 `:90-91`）的 `make_tma_atom(..., make_shape(bM,bK))` 与 SMEM layout 共同决定。NT 的 SW128 MN-major atom 只能生成 64×8 的 box，导致 **每 tile 32 个 1 KB 请求**；TN 的 K-major atom 生成 64×128 的 16 KB 请求。

**这不是 V1 的主瓶颈**（V2 用同样的 32 atom 仍把 NT 提到 76.4%），但它是 NT 始终比 TN 低 8.5 pp Tensor 的结构性原因之一。详见 [10 · 专题](10-专题-NT布局TMA粒度与WS交互.md)。

---

## 5. C 写回：V1 的 16-bit 标量写是“已证实但影响小”的问题

| 证据 | V1（NT/TN 相同） |
|---|---|
| SASS | `STG.E.U16` 524,288 条 |
| `memory_l2_theoretical_sectors_global` | 2,097,152 |
| `_ideal` | 1,048,576 |
| `_excessive` | **1,048,576（2×）** |
| C 写回流量 @4096³ | 32 MiB |
| C 读取流量 @4096³ | 32 MiB（`axpby` 在 beta=0 时仍读 C） |
| 占全局流量比例（A+B 读 2 GiB + C 读/写 64 MiB） | ≈3% |

`axpby(alpha, tCrC, beta, tCgC)` 的 `beta` 是运行时参数，基准固定 beta=0 也不能在编译期消除对 C 的读取，因此 V1 的 C 流量是 **读 32 MiB + 写 32 MiB**。NCU 证明 16-bit 写回把每个 32-bit sector 写成了 2 个 sector；但 C 读写合计只占约 3% 流量，且 DRAM 仅 38–45%，所以**这不是 V1 的主要瓶颈**。V3 的 fast path 同时消除了冗余读与 2× sector，TFLOPS 仅 +0.3%（见 [03](03-Kernel-V3-性能分析.md)），反向验证了这一点。不要仅凭 Excessive Sectors 就判定 C 写回是瓶颈。

---

## 6. 诊断表

| 优先级 | 性能问题 | NCU 证据 | 对应代码 | 根因假设 | 置信度 |
|---|---|---|---|---|---|
| **P0** | 每 tile 等 WGMMA 全部完成，Tensor 流水排空 | Tensor 66.2/74.8%；DEPBAR 32.5/38.0%；`stall_gmma=2.41/0.50`；eligible 0.22/0.20 | `tma_wgmma_device.cuh:167`（当前 `wgmma_tma_fp16_v1.cuh:173`）`warpgroup_wait<0>()` | inflight=1；等待与槽位释放绑定，释放/refill 被推迟到本批完成之后 | **已证实** |
| **P1** | TMA 交付在 full barrier 上暴露，NT 更严重 | NANOSLEEP 1,876/9,062；barrier stall 3.07/4.11；NT duration +13.1% | `tma_wgmma_device.cuh:171-174`（先 wait<0> 再 release/refill） | 释放槽位与 refill 被绑在 WGMMA 完成之后，TMA 启动偏晚 | **较高可能性** |
| **P1** | NT 的 1 KB TMA 粒度使请求数 16× | UTMALDG 2,097,152 vs 131,072；tmain 3.80% vs 0.27%；TMA pipe 8.28% vs 0.58% | `tma_wgmma_adapter.cuh:77-78`（当前 :90-91）；`make_operand_smem_layout` | MN-major SW128 atom 限制 TMA box 为 64×8；每 tile 32 请求 | **已证实**（粒度）；对性能的定量贡献**待验证** |
| P2 | C 写回 2× sector | STG.E.U16 524,288；excessive 1,048,576 | epilogue `axpby` 分支 | 逐元素 16-bit store | **已证实**，但影响小 |
| P2 | Occupancy 仅 2 CTAs/SM，实际 4.49 波 | SMEM 98,432 B；occupancy 12.09%；limit_shared_mem=2；NCU waves 2.25 为理论值 | PIPE=3 的 3×32 KB + barriers | SMEM 容量限制；最后一波 112/228 个槽，尾损约 10% | **已证实** |

---

## 7. Top 3 问题

1. **`warpgroup_wait<0>` 每 tile 排空 Tensor 流水**：Tensor 66–75%，DEPBAR 占采样 32–38%，是 V1 最主要的上限。
2. **TMA 与计算的重叠窗口被“先等完成、再释放、再 refill”压缩**：full barrier 自旋与 barrier stall 高，NT 尤其明显。
3. **NT 的 TMA 请求粒度是 TN 的 16×**：同代码下 NT 始终低 8.5 pp Tensor、慢 13%，是结构性差异。

## 8. Top 4 优化方向

1. **把 inflight 提到 2，使用 `wait<1>`**：在不增加 SMEM 的前提下让相邻 WGMMA 批次重叠（V2，已验证 NT +14.2%）。
2. **把“释放旧槽位”与“本批全部完成”解耦**：释放第 i−1 个槽位只需等第 i−1 批完成，而不是第 i 批；这正是 V2 的 `wait<1>` + `release_state`。
3. **针对 NT 的 TMA 粒度做 ablation**：更大 TMA box（改 SMEM atom/布局）、单 lane 批量发射、或预转置输入，目标是降低每 tile 请求数；在未验证前不要直接改 kernel 主体。
4. **在长 K 上干净复测 PIPE=2（V3）**：PIPE=2 时动态 SMEM 为 65,664 B，允许 3 CTAs/SM；4096³ 的 1024 CTA / (114×3) = **2.99 波**，整数尾几乎消失，同时多 50% 的 CTA 并发来掩盖 TMA 延迟。现有 4096³ PIPE 基准被外部进程污染而弃用，长 K 下 PIPE=2 vs PIPE=3 尚无干净结论。实验需同时对照“固定 2 CTAs”与“3 CTAs”，把 stage 深度与 occupancy/尾波分开。

## 9. 报告不能回答的问题

- TMA 从 `UTMALDG` 到 full barrier 翻转的**实际延迟分布**（缺 TMA 数据引擎计数器）；
- V1 的 3 槽预取是否足够覆盖 L2/DRAM 延迟（需要 PIPE 扫描 + 干净基准）；
- 2 CTAs/SM 与 1 CTA/SM 的尾波影响（需要固定 tile 的 occupancy 实验）；
- 冷缓存 NCU 与 benchmark 重复同数据的热缓存差异；
- producer lane 与 consumer warp 的**分 warp stall 归因**（当前采样混合）；
- 长 K（4096³）下 PIPE=2 × 3 CTAs 的干净对照（现有 4096³ PIPE 基准被污染）；
- C 冗余读取的定量贡献（beta=0 仍读 C，约占流量 3%）；
- 尾波的实际损失（实际 4.49 波，最后一波仅 112/228 个槽）。

---

## 10. 复核与勘误记录（2026-11-03）

| 项目 | 原文档 | 复核结论 |
|---|---|---|
| 源码引用 | `tma_wgmma_device.cuh:167/171/174` | 对报告快照 `source.tar.gz` 正确；当前 worktree 已内联，映射为 `wgmma_tma_fp16_v1.cuh:173/177/180` |
| TMA atom 引用 | `tma_wgmma_adapter.cuh:77-78` | 报告快照正确；当前文件为 `:90-91` |
| Waves | 2.25 波，尾效应敏感 | 2.25 是 NCU 理论值（按寄存器 4 blocks/SM）；实际 2 CTAs/SM → 4.49 波，尾损约 10% |
| `wait<0>` 依赖 | accumulator 在本批完成前不能进入下一批 | 不准确；依赖来自 SMEM 槽位复用，累加器由硬件按序保证 |
| C 流量 | 写回 32 MiB，占 ≈3% | 读写合计 64 MiB；`axpby` 在 beta=0 时仍读 C |
| 长 K PIPE | 未回答（基准污染） | 新增 PIPE=2 × 3 CTAs 的待验证推断（第 8 节第 4 条） |

**证据等级汇总**：P0 已证实；P1 释放时序为较高可能性；NT 粒度为已证实但定量贡献待验证；C 写回与 occupancy 已证实但非主因。
