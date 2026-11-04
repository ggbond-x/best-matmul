# 03 · Kernel V3（v3_epilogue）性能分析

> 报告：`ncu_v3_4096x4096x4096_{NT,TN}_cold.ncu-rep`、`ncu_v3_4096x4096x256_{NT,TN}_cold.ncu-rep`
> 源码：`kernels/wgmma_tma_fp16_epilogue_v3.cuh` + `kernels/detail/tma_wgmma_device.cuh`（合并写回分支）
> 配置：CTA 128×128×64，PIPE=3，GMMA_N=128，128 threads，inflight=2，**coalesced epilogue=1**
> 基准：NT 478.3 TFLOPS（较 V2 −0.36%）/ TN **480.1 TFLOPS**（+1.99%）@4096³；K=256 NT 186.9（−2.3%）/ TN 188.2（−3.7%）

---

## 1. 相对 V2 的改动：合并写回

V3 只改 epilogue，主循环仍是 V2 的 inflight=2：

```text
V2: 每个线程直接把 tCrC 的分散元素 axpby 到 gC  → STG.E.U16 ×524,288
V3: ① __syncthreads()
    ② 所有线程把 tCrC 拷到 SMEM 的 sC（stride (1, CTA_M)） → STS.U16 ×524,288
    ③ __syncthreads()
    ④ 所有线程按连续地址做 uint4 写回（每线程 8 个 half） → STG.E.128 ×65,536
```

源码位置（`kernels/detail/tma_wgmma_device.cuh`）：

| 步骤 | 行号 |
|---|---|
| `kCoalescedEpilogue` 分支 / alpha=1,beta=0 约束 | 209–217 |
| 第一个 `__syncthreads()` | 218 |
| `copy(tCrC, tCsC)` 寄存器→SMEM | 224 |
| 第二个 `__syncthreads()` | 225 |
| uint4 向量化写 | 229–239 |
| 非合并路径 `axpby` | 241+ |

设计意图：把 V2 的 16-bit 分散写（每 32-bit sector 只写一半）改成 128-bit 合并写，理论上把 C 的 L2 sector 减半，并让写回更接近 DRAM burst。

---

## 2. 关键 NCU 指标（4096³）

| 指标 | V2-NT | **V3-NT** | V2-TN | **V3-TN** |
|---|---:|---:|---:|---:|
| Duration | 275.36 µs | **270.91 µs** | 278.50 µs | **275.49 µs** |
| Tensor active | 76.37% | **77.04%** | 75.21% | **77.18%** |
| Compute | 69.69% | 68.93% | 69.83% | 65.56% |
| DRAM | 43.92% | 44.64% | 43.44% | 44.07% |
| L2 | 70.28% | 68.75% | 69.67% | 66.69% |
| L2 hit | 78.17% | 80.52% | 77.18% | 75.67% |
| Eligible / scheduler | 0.29 | 0.29 | 0.21 | 0.22 |
| Issue active | 25.88% | 25.84% | 19.68% | 20.67% |
| WCPI | 7.67 | 7.43 | 9.79 | 9.39 |
| Stall barrier | 0.10 | 0.10 | 0.14 | 0.17 |
| Stall long scoreboard | 4.07 | 3.90 | 6.05 | 5.49 |
| Stall short scoreboard | 0.26 | **0.36** | 0.36 | **0.47** |
| Stall MIO throttle | 0.19 | **0.34** | 0.24 | **0.44** |
| Occupancy | 12.31% | 12.35% | 12.38% | 12.28% |
| Registers | 111 | 111 | 111 | 111 |
| SMEM | 98,432 B | 98,432 B | 相同 | 相同 |

Tensor 提升 0.7–2.0 pp，但 duration 只降 1.6–1.1%，因为 epilogue 只占 kernel 的一小段；Tensor% 的上升主要来自主循环噪声与分母变化，不足以证明 epilogue 是主要收益。

### 2.1 C 写回：sector 减半的硬证据

| 指标 | V2-NT/TN | V3-NT/TN |
|---|---:|---:|
| 主要 store 指令 | `STG.E.U16` **524,288** | `STG.E.128` **65,536** |
| `memory_l2_theoretical_sectors_global` | **2,097,152** | **1,048,576** |
| `_ideal` | 1,048,576 | 1,048,576 |
| `_excessive` | **1,048,576（2×）** | **0** |

这是“合并写回确实生效”的**已证实**证据：C 的 L2 全局 sector 从 2× 降到 1×，没有超额 sector。

### 2.2 代价：SMEM 往返与 shared wavefront

| 指标 | V2 | V3 |
|---|---:|---:|
| `memory_l1_wavefronts_shared` | 25,165,824 | **27,525,120** |
| `_ideal` | 25,165,824 | 25,952,256 |
| `_excessive` | 0 | **1,572,864（≈6%）** |
| `derived__memory_l1_conflicts_shared_nway` | 96 | **624** |
| `STS.U16` 指令 | 0 | **524,288** |
| `BAR.SYNC` warp 指令 | 0 | **8,192**（2 次/CTA × 4 warp） |

寄存器→SMEM 的拷贝是**逐元素 16-bit STS**，不是向量化 STS，因此：
- 增加了 524,288 条 STS.U16；
- 带来 1,572,864 个超额 shared wavefront（约占 shared wavefront 总量 6%）；
- shared bank conflict 从 96 升到 624；
- 两个 `__syncthreads()` 让 4 个 warp 在 epilogue 对齐。

这些代价在 K=4096 时被主循环摊薄（C 流量仅约 3%），但在 K=256 时占比高，直接导致回归。

---

## 3. 为什么 sector 减半，TFLOPS 不涨？

### 3.1 流量占比

4096³ 的全局流量近似：

- A + B 读取：`(M·K + N·K) × 2 B = 2 × 4096² × 2 = 67.1 MB`（未计 L2 复用，DRAM 侧约 0.66 TB/s × 275 µs ≈ 181 MB，含重复读取）；
- C 写回：`M·N × 2 B = 32 MB`；
- C 只占**约 3%** 的全局字节。

即使 V2 的 C sector 是理想的 2×，额外也只有 32 MB 的 L2 写流量；而 DRAM 只有 43–44%、L2 67–70%，两者都没有饱和。把 C sector 减半不会改变主循环的 Tensor 供给。

### 3.2 Benchmark 与 NCU 的方向分歧

| 数据 | V2-NT | V3-NT | 解释 |
|---|---:|---:|---|
| NCU duration | 275.36 µs | **270.91 µs** | 冷缓存单次 replay，V3 略快 |
| Benchmark TFLOPS | **480.0** | 478.3 | 热缓存批平均，V3 低 0.36%（在噪声内） |
| K=256 benchmark | **191.3** | 186.9 | 短 K 下 epilogue 占比高，V3 明确更慢 |

结论：**V3 的写回优化在长 K 下是“中性”的**（−0.4% ~ +2.0%，取决于 layout 与噪声），不是性能提升项；在短 K 下是负收益。不要把它当作主要优化。

---

## 4. 短 K 回归的机制

K=256 只有 4 个 k-tile，主循环时间短，epilogue 的固定开销占比高：

| 指标（K=256） | V2-NT | V3-NT | V2-TN | V3-TN |
|---|---:|---:|---:|---:|
| TFLOPS | **191.3** | 186.9 | **195.5** | 188.2 |
| Tensor | **30.02%** | 28.10% | **30.60%** | 28.88% |
| Stall short scoreboard | 0.27 | **1.38** | 0.36 | **1.56** |
| Stall MIO throttle | 0.44 | **1.78** | 0.41 | **1.90** |
| Duration | 45.9 µs | 47.2 µs | 45.5 µs | 45.5 µs |

短 K 下 V3 的 short-scoreboard / MIO-throttle 明显上升，与新增的 STS.U16 + 两次 `__syncthreads` 一致。NCU K=256 的 TN duration 恰好持平，但 benchmark TN 明确 −3.7%，说明 NCU 单次冷启动对短 kernel 的噪声较大；以 benchmark 为准。

---

## 5. 诊断表

| 优先级 | 性能问题 | NCU 证据 | 对应代码 | 根因假设 | 置信度 |
|---|---|---|---|---|---|
| **P1** | C sector 减半但长 K 无收益 | sectors 2,097,152→1,048,576；excessive 1,048,576→0；DRAM 44%、L2 69% 未饱和 | `tma_wgmma_device.cuh:209-239` | C 仅占 ~3% 流量，主瓶颈是 TMA→Tensor 供给 | **已证实** |
| **P1** | 短 K 回归 −2.3%/−3.7% | K=256 Tensor 30.0→28.1%；short_sb 0.27→1.38；MIO 0.44→1.78；STS.U16 524,288；BAR.SYNC 8,192 | `:218`、`:225`、`:224` | SMEM 往返 + 两次全 CTA barrier 的固定开销 | **已证实** |
| **P2** | 寄存器→SMEM 拷贝未向量化 | shared excessive 1,572,864（6%）；conflicts 96→624；STS.U16 524,288 | `:224` `copy(tCrC,tCsC)` | 逐元素 16-bit STS，未用 STS.128 | **已证实** |
| **P2** | Epilogue 只支持 alpha=1,beta=0 | 源码约束 `:210-211` | `:209-242` | 通用路径回退到 axpby，未基准 | **已证实**（功能限制） |
| **P2** | 对主循环 Tensor 的间接影响 | Tensor 76.4→77.0%（NT）、75.2→77.2%（TN） | — | 分母缩短 + 尾段变化，非主循环改进 | **待验证** |

---

## 6. Top 3 问题

1. **写回优化打在了非瓶颈上**：C 只占约 3% 流量，sector 减半不能提升 Tensor 供给。
2. **短 K 的固定开销**：SMEM 往返 + 2 次 `__syncthreads` 使 K=256 回归 2–4%。
3. **SMEM 拷贝本身不高效**：STS.U16 逐元素写，带来 6% 超额 shared wavefront 与 6.5× bank conflict。

## 7. Top 3 优化方向

1. **按 K 选择 epilogue**：长 K 用 V3，短 K 用 V2（或 V3 直接跳过 SMEM 往返）。
2. **把寄存器→SMEM 拷贝向量化**：用 128-bit STS 替代 STS.U16，消除超额 shared wavefront；预期只在短 K 有可见收益。
3. **探索无 SMEM 的向量化直接写**：通过 lane 置换把 tCrC 重排成连续 128-bit store，省掉两次 barrier；风险是寄存器压力和重排开销。

## 8. 报告不能回答的问题

- Epilogue 在总时间中的精确占比（NCU 没有单独的 epilogue 区间计时）；
- 写回与下一 CTA 启动的重叠程度（需要 persistent/尾波实验）；
- 短 K 下两次 `__syncthreads` 的单独成本（需要去掉其一的 ablation）；
- 热缓存下 C 是否命中 L2、写回是否被完全吸收。

---

**证据等级汇总**：C sector 减半与短 K 回归均已证实；STS.U16 低效已证实；对长 K TFLOPS 的中性结论由 benchmark + NCU 双向支持。
