#pragma once
#include "wgmma_config.cuh"

// 阅读顺序：CTA 视图 -> TMA 预取 -> WGMMA 主循环 -> C 写回。
// 三个入口文件只选择版本；这里保留一份完整设备流程，避免修改同步逻辑时
// 还要在 v1/v2/v3 之间复制代码。if constexpr 在编译时选定版本，没有运行时开销。
namespace gemm_backend {

inline constexpr int kProducerThreads = 0;
inline constexpr int kComputeThreads = 128;
inline constexpr int kComputeWarpgroups = 1;

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
  // 每个操作数包含 PIPE 份 tile。128 字节对齐与 SW128 swizzle 保持一致。
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
  // 每个槽位各有两道屏障：数据搬完后可读，计算完成后才可覆盖。
  uint64_t tma_barrier[size<2>(SmemLayoutA{})];
  uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};

// 当前每个 CTA 只有一个 128 线程 warpgroup，线程数由 TiledMma 给出。
// CUTLASS_GRID_CONSTANT 让整个 grid 共用 CPU 已编码好的 TMA 描述符。
// A/B 的地址已包含在描述符中；仍保留这两个指针参数以保持 launch ABI 一致。
template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
                 TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
                 TC* C, CStride dC, TiledMma mma, Alpha alpha, Beta beta)
{
  // 1. 构造当前 CTA 的全局内存、共享内存视图。
  // 模板类型决定布局；这些断言在编译时检查 tile 与布局能否配合。
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});
  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));

  // CUTE 命名：m 是完整矩阵，g 是当前 CTA 的全局 tile，s 是共享内存 tile。
  // A 的逻辑形状始终为 (M,K)，B 为 (N,K)；NT/TN 体现在各自 stride 中。
  auto [M, N, K] = shape_MNK;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{}); // (BLK_M,BLK_N)

  // v3 的向量写回也会复用这块内存，因此基址保持 128 字节对齐。
  extern __shared__ __align__(128) char shared_memory[];
  using Storage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
  Storage& smem = *reinterpret_cast<Storage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});

  // 2. TMA 生产者：将全局 K tile 搬入环形共享内存槽。
  // tAgA/tBgB 的后维是 K tile，tAsA/tBsB 的后维是 PIPE 槽位。
  auto tma_views_a = tma_partition(
      tma_a, Int<0>{}, Layout<_1>{}, group_modes<0,2>(sA), group_modes<0,2>(gA));
  auto tma_views_b = tma_partition(
      tma_b, Int<0>{}, Layout<_1>{}, group_modes<0,2>(sB), group_modes<0,2>(gB));
  // 使用普通局部变量，后面的 lambda 可以按 C++17 规则捕获它们。
  Tensor tAgA = get<0>(tma_views_a);
  Tensor tAsA = get<1>(tma_views_a);
  Tensor tBgB = get<0>(tma_views_b);
  Tensor tBsB = get<1>(tma_views_b);
  constexpr int kPipeCount = size<2>(SmemLayoutA{});
  // 一个槽位同时搬 A 和 B；屏障只等待这一个槽位的总字节数。
  constexpr int kTransactionBytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                  + sizeof(make_tensor_like(tensor<0>(tBsB)));
  using ProducerBarrier = cutlass::arch::ClusterTransactionBarrier;
  using ConsumerBarrier = cutlass::arch::ClusterBarrier;
  uint64_t* producer_barrier = smem.tma_barrier;
  uint64_t* consumer_barrier = smem.mma_barrier;
  int warp_index = cutlass::canonical_warp_idx_sync();
  int elected_lane = cute::elect_one_sync();
  bool is_producer = (warp_index == 0) && elected_lane;

  CUTE_UNROLL
  for (int pipe = 0; pipe < kPipeCount; ++pipe) {
    if (is_producer) {
      ProducerBarrier::init(&producer_barrier[pipe], 1);   // 一个 lane 提交 TMA。
      ConsumerBarrier::init(&consumer_barrier[pipe], 128); // 全 warpgroup 释放槽位。
    }
  }
  cluster_sync(); // 先让屏障初始化可见，再允许任何异步搬运。

  // 只由生产者 lane 调用。先声明预期字节数，再提交两份异步 TMA。
  auto load_tile = [&](int k_tile, int pipe) {
    ProducerBarrier::arrive_and_expect_tx(&producer_barrier[pipe], kTransactionBytes);
    copy(tma_a.with(producer_barrier[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
    copy(tma_b.with(producer_barrier[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
  };

  int tiles_remaining = size<1>(tAgA); // 尚未提交 TMA 的 K tile 数。
  int next_k_tile = 0;
  int prefetch_tiles = tiles_remaining < kPipeCount ? tiles_remaining : kPipeCount;
  CUTE_UNROLL
  for (int pipe = 0; pipe < kPipeCount; ++pipe) {
    // 短 K 只预取实际存在的 tile；首次使用槽位，无需等 consumer barrier。
    if (pipe < prefetch_tiles) {
      if (is_producer) {
        load_tile(next_k_tile, pipe);
      }
      --tiles_remaining;
      ++next_k_tile;
    }
  }

  // 3. 为每个线程分配 WGMMA 操作数和寄存器累加器。
  // SS 的 A/B fragment 是共享内存描述符；只有 C fragment 存储计算结果。
  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  // PipelineState 同时记录槽位 index 和 phase。每绕环一周，phase 翻转，
  // 使屏障区分该槽位的前后两轮使用。读、释放、写不能共用一个状态。
  auto write_state = cutlass::PipelineState<kPipeCount>();
  auto read_state = cutlass::PipelineState<kPipeCount>();
  int pending_tiles = prefetch_tiles; // 已提交 TMA、尚未提交 WGMMA 的 tile 数。

  // 释放一个槽位后，尝试补充一个 tile。所有线程都调用本函数：
  // 仅生产者等待/搬运/推进 write_state；逻辑计数必须由所有线程一致更新。
  auto refill_one_tile = [&]() {
    if (tiles_remaining > 0) {
      if (is_producer) {
        int pipe = write_state.index();
        ConsumerBarrier::wait(&consumer_barrier[pipe], write_state.phase());
        load_tile(next_k_tile, pipe);
        ++write_state;
      }
      --tiles_remaining;
      ++next_k_tile;
      ++pending_tiles;
    }
  };

  // 4. 沿 K 维计算。每个 K tile 对应一个 WGMMA batch；gemm 内部可能
  // 发出多条指令。两种调度的核心区别是何时等待计算完成并释放共享内存。
  if constexpr (kMaxInflight == 1) {
    // v1：当前批次全部完成后才释放槽位，时序最直观。
    CUTE_NO_UNROLL
    while (pending_tiles > 0) {
      int pipe = read_state.index();
      ProducerBarrier::wait(&producer_barrier[pipe], read_state.phase());
      warpgroup_fence_operand(tCrC);
      warpgroup_arrive();
      gemm(mma, tCrA(_,_,_,pipe), tCrB(_,_,_,pipe), tCrC);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tCrC);

      // 128 个线程各到达一次；此后生产者才可覆盖本槽位。
      ConsumerBarrier::arrive(&consumer_barrier[pipe]);
      ++read_state;
      --pending_tiles;
      refill_one_tile();
    }
  } else {
    // v2/v3：提交第 i 批后，wait<1> 保证第 i-1 批已完成，
    // 第 i 批可以继续计算。释放旧槽位，让 TMA 与当前计算重叠。
    auto release_state = cutlass::PipelineState<kPipeCount>();
    int issued_batches = 0;
    warpgroup_fence_operand(tCrC); // 将 clear 的结果交给首批异步 WGMMA。
    CUTE_NO_UNROLL
    while (pending_tiles > 0) {
      int pipe = read_state.index();
      ProducerBarrier::wait(&producer_barrier[pipe], read_state.phase());
      warpgroup_arrive();
      gemm(mma, tCrA(_,_,_,pipe), tCrB(_,_,_,pipe), tCrC);
      warpgroup_commit_batch();
      ++read_state;
      --pending_tiles;
      ++issued_batches;

      if (issued_batches >= 2) {
        warpgroup_wait<1>();
        warpgroup_fence_operand(tCrC);
        ConsumerBarrier::arrive(&consumer_barrier[release_state.index()]);
        ++release_state;
        refill_one_tile();
      }
    }
    // 最后一批还没有被 wait<1> 排空；读累加器前必须等待全部计算完成。
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);
  }

  // 5. 写回 C = alpha * accumulator + beta * C。
  // 基准固定 alpha=1、beta=0，每次启动覆盖 C。尺寸整除约束由 host 检查，
  // 因而这里无需逐元素边界判断。
  if constexpr (kCoalescedEpilogue) {
    if (alpha == Alpha(1.0f) && beta == Beta(0.0f)) {
      static_assert(sizeof(TC) == 2, "V3 vector epilogue expects 16-bit C elements");
      static_assert(PEAK_CTA_M % 8 == 0, "V3 epilogue requires CTA_M divisible by 8");
      constexpr int kElementsPerVector = 8; // 8 个 FP16 = 16 字节 = 一个 uint4。
      constexpr int kVectorsPerColumn = PEAK_CTA_M / kElementsPerVector;
      constexpr int kVectorsPerTile = PEAK_CTA_N * kVectorsPerColumn;

      // 先等整块 CTA 不再访问 A/B，再复用原有动态共享内存作为 C 暂存区。
      __syncthreads();
      Tensor sC = make_tensor(
          make_smem_ptr(reinterpret_cast<TC*>(shared_memory)),
          make_layout(make_shape(Int<PEAK_CTA_M>{}, Int<PEAK_CTA_N>{}),
                      make_stride(Int<1>{}, Int<PEAK_CTA_M>{})));
      Tensor tCsC = thr_mma.partition_C(sC);
      copy(tCrC, tCsC); // 按原 MMA 线程映射，将分散的寄存器结果放回 (m,n)。
      __syncthreads(); // 所有结果就绪后，才按新的连续映射读取。

      TC const* smem_C = reinterpret_cast<TC const*>(shared_memory);
      int const ldc = int(get<1>(dC)); // C 为列主序，M 维连续。
      for (int vec = int(threadIdx.x); vec < kVectorsPerTile; vec += int(blockDim.x)) {
        int tile_col = vec / kVectorsPerColumn;
        int tile_row = (vec % kVectorsPerColumn) * kElementsPerVector;
        int global_col = int(blockIdx.y) * PEAK_CTA_N + tile_col;
        int global_row = int(blockIdx.x) * PEAK_CTA_M + tile_row;
        TC* dst = C + global_col * ldc + global_row;
        // 相邻线程负责相邻 16 字节向量；CTA_M 是 8 的倍数，向量不会跨列。
        // cudaMalloc、ldC 和共享内存基址满足此处的 16 字节对齐要求。
        uint4 word = *reinterpret_cast<uint4 const*>(smem_C + vec * kElementsPerVector);
        *reinterpret_cast<uint4*>(dst) = word;
      }
    } else {
      axpby(alpha, tCrC, beta, tCgC); // 非基准 alpha/beta 使用通用语义。
    }
  } else {
    axpby(alpha, tCrC, beta, tCgC);
  }
}
}  // namespace gemm_backend
