# 08 · Kernel 优化方案与优先级

> 排序依据：**预期收益 × 证据强度 ÷（实现复杂度 × 正确性风险 × 测试成本）**。
> 原则：先做“已验证瓶颈”的低风险改动；不在分析未闭环前重写主循环；任何删除/弱化同步的方案必须证明依赖仍被保留。
>
> 优先级：P0 = 立即执行；P1 = 本轮迭代；P2 = 有余力/后续。

---

## 总览

| ID | 方案 | 目标 | 优先级 | 预期收益 | 置信度 |
|---|---|---|---|---|---|
| **OPT-01** | NT 调度回退 V3 | NT 全 shape | **P0** | NT +30%（367→478 TFLOPS @4096³） | **高** |
| **OPT-02** | V4/V5 producer 并行发射 TMA + election 外提 | V4/V5 NT | **P0** | Tensor 52%→65–75%（目标） | 中 |
| **OPT-03** | NT 更大 TMA box / 预转置 ablation | V4/V5 NT 根因 | **P1** | 验证粒度因果；若成立可复制 V2 水平 | 中高（机制） |
| **OPT-04** | 按 K 选择 PIPE=2/3（V3） | 短 K 全 layout | **P1** | K≤512 +13–16% | **高** |
| **OPT-05** | Epilogue 向量化 STS / 短 K 跳过 SMEM | V3/V4/V5 短 K | P2 | +1–3%（短 K） | 中 |
| **OPT-06** | V5 ≤56 regs + PIPE=2 → 3 CTAs | V5 TN 短 K | **P1** | 短 K +10–15%（若达成） | 中低 |
| **OPT-07** | Persistent tile scheduler | 4096²/不规则 shape 尾波 | P2 | +2–5% | 中低 |
| **OPT-08** | Consumer 单线程等待 + warpgroup 广播 | V4/V5 同步开销 | P2 | +1–3% | 中低 |

---

## OPT-01 · NT 调度回退 V3（P0）

| 字段 | 内容 |
|---|---|
| **目标 kernel** | 所有 NT（Major::MN）问题 |
| **当前问题** | V4/V5 NT 消费者饥饿，Tensor 51–52%，比 V3 低 25 pp |
| **NCU 证据** | V4 NT：Tensor 51.78%、DRAM 658.82 GB/s、L2 51.40%、consumer NANOSLEEP 3,814,406 次；V3 NT：Tensor 77.04%、duration 270.91 µs、478.3 TFLOPS |
| **源码位置** | 调度层：`kernels/detail/tma_wgmma_adapter.cuh:145-155`（`is_nt()` 分支）；基准/驱动选择版本 |
| **根因** | WS producer 与 NT 的 32×1 KB TMA 突发交互（R-02/R-03） |
| **建议改动** | 在 NT 上选择 V3（非 WS，inflight=2，coalesced epilogue）；不修改 kernel 主体。短期可在运行脚本/基准层按 `is_nt()` 分派，长期把 V4 的 producer 改造完成后再评估 |
| **预期效果** | @4096³ NT 367.5→478.3 TFLOPS（**+30.1%**）；K=256 NT 184.3→186.9（+1.4%）；1024³ 92.8→150.2（+61.9%） |
| **副作用** | K=256 时 V3 比 V2 慢 2.3%（186.9 vs 191.3）；短 K 可再按 OPT-04 选 V2 |
| **验证方法** | 同 shape A/B：`v3` vs `v4` NT；检查 Tensor≥75%、duration 回到 ~271 µs；确认 C 结果一致（validate 已 PASS） |
| **正确性风险** | 无（不改变数学/同步） |
| **测试成本** | 低（配置/脚本级） |
| **置信度** | **高**（benchmark + NCU 双证据） |

**分派建议**：

```text
NT:
  K <= 512           → V2（191.3 @K=256）或 V3 PIPE=2（OPT-04，211.4）
  K > 512            → V3
TN:
  K <= 512           → V4（203.3 @K=256）
  K > 512            → V4（482.4 @4096³）
```

---

## OPT-02 · V4/V5 producer 并行发射 TMA + election 外提（P0）

| 字段 | 内容 |
|---|---|
| **目标 kernel** | V4/V5 NT（以及所有 WS + 细粒度 TMA 场景） |
| **当前问题** | 单线程 producer 每 tile 突发 32 个 1 KB 请求；消费者每 tile 自旋 14.5 次；producer 不受 empty 限制但 TMA 完成晚 |
| **NCU 证据** | consumer NANOSLEEP 3.81M（V4）/8.42M（V5）；TMA pipe 6.47%、L1TEX 24.77%、DRAM 659 GB/s（均未饱和）；producer empty wait 仅 158 samples；`ELECT=UTMALDG=2,097,152` |
| **源码位置** | `kernels/detail/tma_wgmma_warp_specialized_device.cuh:79-90`（producer 循环、`is_producer_leader`）；TMA copy 路径（CuTe copy atom） |
| **根因** | 单 lane 串行发射 + per-atom election，形成长突发；full barrier 等全部 32 个完成，尾部延迟被放大 |
| **建议改动** | 方案 A（推荐先做）：**4 个 producer warp 各负责 8 个 atom**，每个 warp 的 elected lane 发 8 条 TMA；full barrier count 改为 4，每个 warp `arrive_and_expect_tx(bytes/4)`。方案 B（更小改动）：把 election 提到 atom 循环外，避免每条 UTMALDG 前一次 ELECT。方案 C：每发 8 个 atom 插入一次轻量让出/等待，平滑突发 |
| **预期效果** | 目标把 NT consumer 自旋从 14.5 次/tile 降到 ≤3 次，Tensor 52%→65–75%；TN 不应回退（请求少，改动影响小） |
| **副作用** | barrier count 从 1→4，consumer 的等待语义不变但初始化需同步修改；4 个 producer warp 的地址计算/发射增加 issue 压力；若 TMA 引擎本身按请求串行，收益有限 |
| **验证方法** | 先只改 barrier count/初始化并跑 validate；NCU 看 consumer NANOSLEEP 次数、Tensor、duration；benchmark A/B；对比 producer 各 warp 的 UTMALDG 分布 |
| **正确性约束** | **不能**在 TMA 未全部完成前翻转 full barrier；**不能**在 consumer 释放槽位前覆盖 SMEM；barrier count 必须等于参与 `arrive_and_expect_tx` 的 warp 数 |
| **测试成本** | 中（kernel 改造 + 正确性回归） |
| **置信度** | 中（机制合理，但 TMA 引擎行为未知） |

**方案 A 的 CuTe 草图**（仅示意，需按实际 `tAgA/tAsA` 形状切分）：

```cpp
// producer: 4 个 warp，每个 warp 负责 1/4 的 atom
int const warp_in_producer = canonical_warp_idx_sync();      // 0..3
int const lane = elected_lane;
if (lane) {
  FullBarrier::arrive_and_expect_tx(&full[pipe], kTransactionBytes / 4);
  // atom_begin = warp_in_producer * (kAtomsPerTile / 4)
  // 只 copy 本 warp 的 atom 子集；所有子集仍写入同一 full[pipe]
  copy(tma_a.with(full[pipe]), tAgA(_, k_tile, atom_range), tAsA(_, pipe, atom_range));
  copy(tma_b.with(full[pipe]), tBgB(_, k_tile, atom_range), tBsB(_, pipe, atom_range));
}
```

> 注意：`copy` 的 atom 切分依赖 TMA tensor 的坐标布局；若 CuTe 不能直接按 atom 切片，可改为按 **K 的四分之一** 切分（每个 warp 负责 16 个 K 元素），但 A/B 的 K 维切分必须与 SMEM layout 的 atom 边界对齐。

**方案 B 的改动点**：确认 `copy` 内部是否对每个 atom 调用 `elect_one_sync`；若是，改用 `cute::copy_if` 或直接调用 `SM90_TMA_LOAD` 的 `copy_unpack`，把 election 放在循环外。SASS 验收标准：`ELECT` 数从 2,097,152 降到 ~65,536（每 tile 一次）。

---

## OPT-03 · NT 更大 TMA box / 预转置 ablation（P1）

| 字段 | 内容 |
|---|---|
| **目标 kernel** | V4/V5 NT；也是 R-03 的因果实验 |
| **当前问题** | NT 的 TMA atom 为 64×8=1 KB，TN 为 64×128=16 KB；NT 每 tile 32 请求 |
| **NCU 证据** | `l1tex__tmain_requests` 3.10% vs 0.27%；UTMALDG 2,097,152 vs 131,072；V4 NT Tensor 51.8% vs TN 78.1% |
| **源码位置** | `kernels/detail/tma_wgmma_adapter.cuh:25-41`（SMEM layout atom）、`:77-78`（`make_tma_atom`） |
| **根因** | MN-major SW128 atom 限制 TMA box 为 64×8；请求数 16× |
| **建议改动** | 三条并行路线：① **预转置**：host 侧把 A/B 转成 TN 布局，用 V4 TN 路径验证“请求数降低后 NT 是否恢复”；② **换 SMEM atom/swizzle**：尝试 SW64/SW32 或更大的 MN atom，找到能生成更大 TMA box 的组合；③ **cp.async.bulk 1D**：按行连续搬运，绕过 2D TMA box 限制（需自行处理 swizzle） |
| **预期效果** | 若预转置后 V4 NT 恢复到 ~78% Tensor，则证明粒度是因果；生产上需把转置成本计入（当前不计入 GEMM 计时，不能直接宣称收益） |
| **副作用** | 预转置增加 host 时间与内存；换 atom 可能改变 SMEM bank 行为、WGMMA descriptor 合法性；cp.async 需要额外的 barrier/描述符管理 |
| **验证方法** | 预转置 A/B 后跑 V4 NT：Tensor、NANOSLEEP、UTMALDG 数；对比原 NT。若恢复 → 粒度因果成立；若不恢复 → 问题在 producer 发射结构（OPT-02） |
| **正确性风险** | 中（layout 转置/描述符） |
| **测试成本** | 中高（需要 host 侧支持与正确性验证） |
| **置信度** | 机制高；端到端收益中（预转置成本可能抵消） |

---

## OPT-04 · 按 K 选择 PIPE=2/3（V3）（P1）

| 字段 | 内容 |
|---|---|
| **目标 kernel** | V3（NT/TN）、短 K |
| **当前问题** | K≤512 时 PIPE=3 只有 2 CTAs/SM，短 K 并行度不足 |
| **NCU 证据** | K=256：V3 PIPE=2 3 CTAs/SM、Tensor 31.90/32.16%、TFLOPS 211.4/216.1；PIPE=3 2 CTAs、28.10/28.88%、186.4/186.7（+13.4%/+15.7%）；PIPE=4 1 CTA、−20% |
| **源码位置** | `kernels/detail/wgmma_config.cuh:16-17`（`PIPE` 宏）；`:61-62` SMEM layout |
| **根因** | 短 K 下 occupancy（CTAs/SM）比 stage 深度更重要；PIPE=2 的 65,664 B 允许 3 CTAs |
| **建议改动** | 编译两个 V3 变体（PIPE=2/3），host 按 `K <= threshold` 选择；threshold 先取 512，用 256/512/1024 三点扫描确定。不要用运行时改 SMEM 大小（编译期常量） |
| **预期效果** | K≤512 +13–16%；K=4096 保持 PIPE=3（避免深度不足） |
| **副作用** | 二进制/编译时间增加；threshold 选择错误会在中等 K 回退；短 K 的 3 CTAs 可能增加 L2 压力 |
| **验证方法** | K=256/512/1024/2048 四点 benchmark；NCU 检查 CTAs/SM=3、Tensor、duration；确认长 K 不回退 |
| **正确性风险** | 低（只改 stage 数，依赖关系不变） |
| **测试成本** | 低 |
| **置信度** | **高**（K=256 干净数据） |

---

## OPT-05 · Epilogue 向量化 STS / 短 K 跳过 SMEM（P2）

| 字段 | 内容 |
|---|---|
| **目标 kernel** | V3/V4/V5 的 coalesced epilogue |
| **当前问题** | 寄存器→SMEM 是逐元素 `STS.U16`，shared excessive 1,572,864（6%）、conflicts 96→624；短 K 回归 2–4% |
| **NCU 证据** | `STS.U16` 524,288；`memory_l1_wavefronts_shared_excessive=1,572,864`；K=256 Tensor 30.0→28.1%；short_sb 0.27→1.38、MIO 0.44→1.78 |
| **源码位置** | `kernels/detail/tma_wgmma_device.cuh:224`（`copy(tCrC,tCsC)`）；`:218`、`:225` 两次 `__syncthreads()` |
| **根因** | SMEM 中转未向量化；两次全 CTA barrier 在短 K 占比高 |
| **建议改动** | ① 用 `stmatrix.sync.aligned.m8n8.x4.shared.b16` 或 128-bit `STS` 替代逐元素拷贝；② 短 K 直接走 V2 的 `axpby` 路径（跳过 SMEM 与两次 barrier）；③ 长 K 保留 V3 合并写回 |
| **预期效果** | 短 K +1–3%；shared excessive 降到 0；长 K 中性 |
| **副作用** | `stmatrix` 的寄存器布局约束强，可能需要重排 accumulator；跳过 SMEM 会回到 2× C sector（短 K 影响小） |
| **验证方法** | SASS 确认 `STS.128`/`STMATRIX` 数；NCU shared excessive、short_sb、MIO；K=256/4096 benchmark |
| **正确性风险** | 中（stmatrix 布局） |
| **测试成本** | 中 |
| **置信度** | 中（收益小，但方向明确） |

---

## OPT-06 · V5 ≤56 regs + PIPE=2 → 3 CTAs（P1）

| 字段 | 内容 |
|---|---|
| **目标 kernel** | V5（尤其 TN）短 K |
| **当前问题** | PIPE=2 时 V5 仍是 2 CTAs/SM，只损失流水深度；`limit_registers=2` |
| **NCU 证据** | V5 58/60 regs；58×384×3 = 66,816 > 65,536；PIPE=2 K=256 TFLOPS 161.4/196.9 vs PIPE=3 180.3/202.3；V3 PIPE=2 达 3 CTAs 后 +13–16% |
| **源码位置** | `kernels/detail/tma_wgmma_warp_specialized_device.cuh` 的 `__launch_bounds__`（当前只有线程数）；`kernels/detail/warp_specialized_config.cuh` |
| **根因** | 寄存器 footprint 使 3 CTAs 差 1,280 个寄存器；PIPE=2 的 SMEM 已足够 3 CTAs |
| **建议改动** | 在 PIPE=2 变体上：`__launch_bounds__(kProducerThreads + size(TiledMma), 3)`，让 ptxas 把 regs 压到 ≤56；或 `-maxrregcount=56`。只对 V5 PIPE=2 生效，避免影响 PIPE=3 |
| **预期效果** | 若达成 3 CTAs：TN 短 K +10–15%；NT 可能因 TMA 竞争无收益甚至回退 |
| **副作用** | 56 regs 可能导致 spill（local memory），反而更慢；3 CTAs 增加 SMEM/TMA 竞争；长 K 的 PIPE=2 深度不足 |
| **验证方法** | 编译后看 `launch__registers_per_thread`、`limit_registers`、CTAs/SM；检查 spill（`local memory` 指标）；K=256/4096 benchmark；TN/NT 分开评估 |
| **正确性风险** | 低（只改资源上限） |
| **测试成本** | 低 |
| **置信度** | 中低（算术成立，spill 风险未知） |

**代码草图**：

```cpp
// 仅 PIPE==2 的 V5 变体
__launch_bounds__(kProducerThreads + size(TiledMma), 3)
__global__ void gemm_device(...) { ... }
```

---

## OPT-07 · Persistent tile scheduler（P2）

| 字段 | 内容 |
|---|---|
| **目标 kernel** | 所有版本；4096²/不规则 shape |
| **当前问题** | grid 1024 CTA、2 CTAs/SM、228 个并发槽；最后一波 CTA 数不足，尾波利用率低 |
| **NCU 证据** | `launch__grid_size=1024`、`launch__waves_per_multiprocessor=2.25`、CTAs/SM=2；4096³ 的 duration 与 2.25 波的理论值有差距 |
| **源码位置** | `kernels/detail/tma_wgmma_adapter.cuh:91-92`（grid = ceil_div）、kernel 入口的 CTA→tile 映射 |
| **根因** | 非 persistent 调度：每个 CTA 处理一个 tile，尾波 CTA 少，SM 空闲 |
| **建议改动** | 用 persistent kernel：grid = SM_count × CTAs/SM，循环取 tile；tile 顺序按 L2 友好 swizzle（先 N 后 M 或 serpentine） |
| **预期效果** | 尾波改善 +2–5%（4096²），不规则 shape 更明显；对 8192³（waves 多）影响小 |
| **副作用** | 需要处理 tile 计数/边界；persistent 循环增加寄存器/指令；可能改变 L2 命中 |
| **验证方法** | 对比 grid/persistent 的 duration、Tensor、尾段利用率；用 nsys 看最后 10% 时间的 SM 活跃；benchmark 多 shape |
| **正确性风险** | 中（tile 映射/边界） |
| **测试成本** | 中 |
| **置信度** | 中低（尾波收益受 L2/调度影响） |

---

## OPT-08 · Consumer 单线程等待 + warpgroup 广播（P2）

| 字段 | 内容 |
|---|---|
| **目标 kernel** | V4/V5（以及 V1–V3 的 full barrier 等待） |
| **当前问题** | 128/256 个 consumer 线程各自 `mbarrier.try_wait` 自旋；V4 NT SYNCS 8.10M、V5 NT 17.57M，NANOSLEEP 3.81M/8.42M |
| **NCU 证据** | SYNCS、NANOSLEEP 指令数；issue active 21–24%、MIO throttle 低但同步流量大；TN 的 SYNCS 仅 1.01M（V4） |
| **源码位置** | `tma_wgmma_warp_specialized_device.cuh:122`（`FullBarrier::wait` 被所有 consumer 执行）；`tma_wgmma_device.cuh:162/185` |
| **根因** | 每个线程独立轮询，重复读取同一 mbarrier；在 NT 供给慢时放大自旋流量 |
| **建议改动** | 每个 warpgroup 只让 elected lane 执行 `FullBarrier::wait`，然后用 `bar.sync`/`warpgroup` 级同步广播；或在 `wait` 成功后用 `__syncwarp`/named barrier 唤醒同 WG |
| **预期效果** | 同步指令减少 ~128×；MIO/issue 压力下降；对 Tensor 的直接收益可能只有 1–3%（因为等待本身仍是必要的） |
| **副作用** | 增加一次 WG 内同步；若 elected lane 被延迟，整个 WG 受影响；需保证所有线程在看到数据后才发 WGMMA |
| **验证方法** | 对比 SYNCS/NANOSLEEP 指令数、issue active、Tensor、duration；确认无 race（validate + compute-sanitizer） |
| **正确性风险** | 中（同步语义） |
| **测试成本** | 低中 |
| **置信度** | 中低（省的是轮询，不是等待） |

**不要做**：直接删除 `warpgroup_wait`/barrier 来“减少 stall”；那会破坏 accumulator 与 SMEM 的依赖。

---

## 实施顺序建议

```text
第 1 步（当天）：OPT-01 + OPT-04 的 dispatch 验证
        → NT 恢复 478 TFLOPS；K≤512 用 PIPE=2
第 2 步（本轮）：OPT-02 方案 B（election 外提）→ 方案 A（4 warp 并行）
        → 用 NCU 判定 NT 自旋是否下降
第 3 步（本轮）：OPT-03 预转置 ablation
        → 若 V4 NT 恢复，说明粒度因果成立，再决定生产化
第 4 步（下一轮）：OPT-06 寄存器/occupancy；OPT-05 epilogue；OPT-07 persistent
第 5 步：OPT-08 同步优化（收益最小，最后做）
```

---

## 验证与止损标准

| 方案 | 继续条件 | 止损条件 |
|---|---|---|
| OPT-02 | NT consumer 自旋 ≤3 次/tile，Tensor ≥65% | 自旋不降或 TN 回退 >2% |
| OPT-03 | 预转置后 V4 NT Tensor ≥75% | 转置成本使端到端无收益 |
| OPT-04 | K≤512 +10% 且长 K 不回退 | 中等 K（1024）回退 >3% |
| OPT-06 | 3 CTAs 达成且无 spill，短 K +8% | 出现 spill 或长 K 回退 |
| OPT-07 | 尾波改善 ≥2% | L2 命中下降抵消收益 |
| OPT-08 | 同步指令显著下降且 Tensor +1% | 正确性风险或无可测收益 |
