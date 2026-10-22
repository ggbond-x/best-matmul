#pragma once

// fast：thread 0..127 是 producer warpgroup，其中只有 thread 0 提交 TMA。
// 计算线程从 128 开始，个数是 WGMMA_COMPUTE_THREADS（默认 256，沿 M 维分工）。
// empty 屏障的到达计数等于计算线程数，不包括 producer。
// 主循环结束后 producer 要排空每个用过的槽位。
// inflight=2 时，最后一批在 wait<0> 之后再 arrive 一次 empty；inflight=1 每个 tile 已经到达过。
#include "config.cuh"
#include "epilogue.cuh"
#include "smem_tma.cuh"

namespace gemm_backend {

inline constexpr int kInterfaceVersion = 1;
inline constexpr char kName[] = "fast";
inline constexpr int kProducerThreads = kProducerWarpgroupThreads;
inline constexpr int kComputeWarpgroups = kFastComputeWarpgroups;
inline constexpr int kComputeThreads = kWarpgroupThreads * kComputeWarpgroups;
inline constexpr int kCtaThreads = kProducerThreads + kComputeThreads;

static_assert(kProducerThreads == 128, "fast reserves one producer warpgroup");
static_assert(kComputeThreads == kFastComputeThreads, "fast compute threads come from WGMMA_COMPUTE_THREADS");
static_assert(kComputeThreads >= 128 && kComputeThreads % 128 == 0,
              "fast compute threads are whole warpgroups");
static_assert(kCtaThreads <= 1024, "fast CTA threads must fit in one block");
static_assert(kCtaM % (kInstM * kComputeWarpgroups) == 0,
              "CTA_M must be divisible by 64 * compute warpgroups");

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
  static_assert(is_static<SmemLayoutA>::value && is_static<SmemLayoutB>::value);
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<2>(SmemLayoutA{}) == size<2>(SmemLayoutB{}));
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
  auto [tAgA, tAsA] = tma_partition(
      tma_a, Int<0>{}, Layout<_1>{}, group_modes<0,2>(sA), group_modes<0,2>(gA));
  auto [tBgB, tBsB] = tma_partition(
      tma_b, Int<0>{}, Layout<_1>{}, group_modes<0,2>(sB), group_modes<0,2>(gB));

  constexpr int kPipeCount = kPipe;
  constexpr int kTransactionBytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                  + sizeof(make_tensor_like(tensor<0>(tBsB)));
  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;
  uint64_t* full = smem.tma_barrier;
  uint64_t* empty = smem.mma_barrier;
  bool const is_consumer = int(threadIdx.x) >= kProducerThreads;
  bool const is_producer_leader = threadIdx.x == 0;
  int const k_tile_count = size<1>(tAgA);

  if (is_producer_leader) {
    CUTE_UNROLL
    for (int pipe = 0; pipe < kPipeCount; ++pipe) {
      FullBarrier::init(&full[pipe], 1);
      // 只计算线程 arrive。producer 不执行 WGMMA，不能算进这个计数。
      EmptyBarrier::init(&empty[pipe], kComputeThreads);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();

  if (is_producer_leader) {
    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
      int pipe = k_tile % kPipeCount;
      int round = k_tile / kPipeCount;
      if (round > 0) {
        EmptyBarrier::wait(&empty[pipe], (round - 1) & 1);
      }
      FullBarrier::arrive_and_expect_tx(&full[pipe], kTransactionBytes);
      copy(tma_a.with(full[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
      copy(tma_b.with(full[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
    }

    // 排空：等待每个用过的槽位的最后一轮 empty。短 K 不能等从未写过的槽。
    // 合并写回会复用共享内存，producer 必须先离开这些 barrier。
    CUTE_UNROLL
    for (int pipe = 0; pipe < kPipeCount; ++pipe) {
      if (pipe < k_tile_count) {
        int last_round = (k_tile_count - 1 - pipe) / kPipeCount;
        EmptyBarrier::wait(&empty[pipe], last_round & 1);
      }
    }
  }

  // producer 使用 slice 0 占位，不写累加器。计算线程连续编号 0..kComputeThreads-1，
  // 每 128 个物理线程对齐成一个 warpgroup。
  int mma_thread = is_consumer ? int(threadIdx.x) - kProducerThreads : 0;
  ThrMMA thr_mma = mma.get_thread_slice(mma_thread);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  // tCgC 必须活到 kernel 结束：fragment 可能引用它的布局。
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);

  if (is_consumer) {
    clear(tCrC);
    warpgroup_fence_operand(tCrC);
    auto read_state = cutlass::PipelineState<kPipeCount>();
    auto release_state = cutlass::PipelineState<kPipeCount>();
    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
      int pipe = read_state.index();
      FullBarrier::wait(&full[pipe], read_state.phase());
      warpgroup_arrive();
      gemm(mma, tCrA(_,_,_,pipe), tCrB(_,_,_,pipe), tCrC);
      warpgroup_commit_batch();
      ++read_state;

      if constexpr (kMaxInflight == 1) {
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrC);
        EmptyBarrier::arrive(&empty[release_state.index()]);
        ++release_state;
      } else if (k_tile >= 1) {
        // 本 warpgroup 的前一批已完成。每个计算线程到达一次；
        // 全部计算 warpgroup 都到达后，producer 才能覆盖这个 A/B 槽。
        warpgroup_wait<1>();
        warpgroup_fence_operand(tCrC);
        EmptyBarrier::arrive(&empty[release_state.index()]);
        ++release_state;
      }
    }
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);
    // inflight=2 时最后一批没有被 wait<1> 配对释放。producer 的排空等待这一次 arrive。
    if constexpr (kMaxInflight == 2) {
      if (k_tile_count > 0) {
        EmptyBarrier::arrive(&empty[release_state.index()]);
      }
    }
  }

  // tCgC 必须活到 kernel 结束：fragment 可能引用它的布局。
  if constexpr (kCoalescedEpilogue) {
    (void)alpha;
    (void)beta;
    (void)tCgC;
    store_c_coalesced(C, dC, thr_mma, tCrC, shared_memory, is_consumer);
  } else if (is_consumer) {
    axpby(alpha, tCrC, beta, tCgC);
  } else {
    (void)alpha;
    (void)beta;
    (void)tCgC;
  }
}

}  // namespace gemm_backend

#include "launch.cuh"
