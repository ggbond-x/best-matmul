#pragma once

// teach：一个 warpgroup（128 线程）同时负责 TMA 与 WGMMA。
// 默认是 v3：PIPE=3，两批 WGMMA 在途，合并写回。这些数来自 config.cuh 的宏。
// -DWGMMA_MAX_INFLIGHT=1 改成算完再释放；-DEPILOGUE_COALESCED=0 改成直接 axpby。
// WGMMA_COMPUTE_THREADS 不作用于 teach，计算线程固定 128。
// 线程 0 所在 warp 的 elected lane 提交 TMA；128 个线程都执行 WGMMA，并在释放槽位时各到达一次。
#include "config.cuh"
#include "epilogue.cuh"
#include "smem_tma.cuh"

namespace gemm_backend {

inline constexpr int kInterfaceVersion = 1;
inline constexpr char kName[] = "teach";
inline constexpr int kProducerThreads = 0;
inline constexpr int kComputeWarpgroups = kTeachComputeWarpgroups;
inline constexpr int kComputeThreads = kWarpgroupThreads * kComputeWarpgroups;
inline constexpr int kCtaThreads = kProducerThreads + kComputeThreads;

static_assert(kComputeThreads == 128, "teach is a single compute warpgroup");
static_assert(kCtaThreads == 128, "teach does not reserve a producer warpgroup");

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(kCtaThreads)
void gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
                 TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
                 TC* C, CStride dC, TiledMma mma, Alpha alpha, Beta beta)
{
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});
  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<2>(SmemLayoutA{}) == Int<kPipe>{});
  CUTE_STATIC_ASSERT_V(size(TiledMma{}) == Int<kComputeThreads>{});
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));
  (void)A;
  (void)B;

  auto [M, N, K] = shape_MNK;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

  extern __shared__ __align__(128) char shared_memory[];
  using Storage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
  Storage& smem = *reinterpret_cast<Storage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});

  // C++17 的 lambda 不能捕获结构化绑定，所以这里用普通局部变量。
  auto tma_views_a = tma_partition(
      tma_a, Int<0>{}, Layout<_1>{}, group_modes<0,2>(sA), group_modes<0,2>(gA));
  auto tma_views_b = tma_partition(
      tma_b, Int<0>{}, Layout<_1>{}, group_modes<0,2>(sB), group_modes<0,2>(gB));
  Tensor tAgA = get<0>(tma_views_a);
  Tensor tAsA = get<1>(tma_views_a);
  Tensor tBgB = get<0>(tma_views_b);
  Tensor tBsB = get<1>(tma_views_b);

  constexpr int kPipeCount = kPipe;
  constexpr int kTransactionBytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                  + sizeof(make_tensor_like(tensor<0>(tBsB)));
  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;
  uint64_t* full = smem.tma_barrier;
  uint64_t* empty = smem.mma_barrier;

  int warp_index = cutlass::canonical_warp_idx_sync();
  int elected_lane = cute::elect_one_sync();
  bool is_producer = (warp_index == 0) && elected_lane;

  CUTE_UNROLL
  for (int pipe = 0; pipe < kPipeCount; ++pipe) {
    if (is_producer) {
      FullBarrier::init(&full[pipe], 1);                  // 一个 lane 提交 TMA。
      EmptyBarrier::init(&empty[pipe], kComputeThreads);  // 整个 warpgroup 释放槽位。
    }
  }
  cluster_sync();

  auto load_tile = [&](int k_tile, int pipe) {
    FullBarrier::arrive_and_expect_tx(&full[pipe], kTransactionBytes);
    copy(tma_a.with(full[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
    copy(tma_b.with(full[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
  };

  int tiles_remaining = size<1>(tAgA);
  int next_k_tile = 0;
  int prefetch_tiles = tiles_remaining < kPipeCount ? tiles_remaining : kPipeCount;
  CUTE_UNROLL
  for (int pipe = 0; pipe < kPipeCount; ++pipe) {
    if (pipe < prefetch_tiles) {
      if (is_producer) {
        load_tile(next_k_tile, pipe);
      }
      --tiles_remaining;
      ++next_k_tile;
    }
  }

  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  // tCgC 必须活到 kernel 结束：fragment 可能引用它的布局。
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  // read 等 full，release 等计算完成后再 arrive empty，write 是下一次 TMA 要复用的槽。
  // 三者不能共用一个 PipelineState：phase 记录的是各自绕环的次数。
  auto write_state = cutlass::PipelineState<kPipeCount>();
  auto read_state = cutlass::PipelineState<kPipeCount>();
  int pending_tiles = prefetch_tiles;

  // 所有线程都要更新计数；只有 producer lane 等待 empty 并提交 TMA。
  auto refill_one_tile = [&]() {
    if (tiles_remaining > 0) {
      if (is_producer) {
        int pipe = write_state.index();
        EmptyBarrier::wait(&empty[pipe], write_state.phase());
        load_tile(next_k_tile, pipe);
        ++write_state;
      }
      --tiles_remaining;
      ++next_k_tile;
      ++pending_tiles;
    }
  };

  if constexpr (kMaxInflight == 1) {
    // 当前批次全部完成后才释放槽位。128 个线程各到达一次。
    CUTE_NO_UNROLL
    while (pending_tiles > 0) {
      int pipe = read_state.index();
      FullBarrier::wait(&full[pipe], read_state.phase());
      warpgroup_fence_operand(tCrC);
      warpgroup_arrive();
      gemm(mma, tCrA(_,_,_,pipe), tCrB(_,_,_,pipe), tCrC);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tCrC);
      EmptyBarrier::arrive(&empty[pipe]);
      ++read_state;
      --pending_tiles;
      refill_one_tile();
    }
  } else {
    // 提交第 i 批后，wait<1> 保证第 i-1 批已经结束，于是可以释放旧槽、与当前计算重叠。
    // 主循环结束后不再复用槽位，最后一批只 wait<0>，不必再 arrive empty。
    auto release_state = cutlass::PipelineState<kPipeCount>();
    int issued_batches = 0;
    warpgroup_fence_operand(tCrC);
    CUTE_NO_UNROLL
    while (pending_tiles > 0) {
      int pipe = read_state.index();
      FullBarrier::wait(&full[pipe], read_state.phase());
      warpgroup_arrive();
      gemm(mma, tCrA(_,_,_,pipe), tCrB(_,_,_,pipe), tCrC);
      warpgroup_commit_batch();
      ++read_state;
      --pending_tiles;
      ++issued_batches;

      if (issued_batches >= 2) {
        warpgroup_wait<1>();
        warpgroup_fence_operand(tCrC);
        EmptyBarrier::arrive(&empty[release_state.index()]);
        ++release_state;
        refill_one_tile();
      }
    }
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);
  }

  // tCgC 必须活到 kernel 结束：fragment 可能引用它的布局。
  if constexpr (kCoalescedEpilogue) {
    (void)alpha;
    (void)beta;
    (void)tCgC;
    store_c_coalesced(C, dC, thr_mma, tCrC, shared_memory, true);
  } else {
    axpby(alpha, tCrC, beta, tCgC);
  }
}

}  // namespace gemm_backend

#include "launch.cuh"
