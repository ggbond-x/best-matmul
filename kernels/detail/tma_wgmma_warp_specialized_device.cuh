#pragma once
#include "warp_specialized_config.cuh"

namespace gemm_backend {

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
  // full：TMA 完成；empty：所有计算 warpgroup 都不再读取该槽位。
  uint64_t tma_barrier[size<2>(SmemLayoutA{})];
  uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(kProducerThreads + decltype(size(TiledMma{}))::value)
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
  CUTE_STATIC_ASSERT_V(size(TiledMma{}) == Int<kComputeThreads>{});
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));

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
  constexpr int kPipeCount = size<2>(SmemLayoutA{});
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
      EmptyBarrier::init(&empty[pipe], kComputeThreads);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads(); // cluster=(1,1,1)，将初始化发布给整个 CTA 和 TMA。

  // Producer 独立推进 K：第一轮直接填满，之后等待所有 consumer 释放槽位。
  // 同一个 empty barrier 第 r 轮释放后翻转 phase；下一轮等待旧的 phase r。
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

    // Tail：等待每个已使用槽位的最后一轮释放。防止合并写回重用共享内存时，
    // producer 仍在访问其中的 barrier；短 K 不能等待从未使用过的槽位。
    CUTE_UNROLL
    for (int pipe = 0; pipe < kPipeCount; ++pipe) {
      if (pipe < k_tile_count) {
        int last_round = (k_tile_count - 1 - pipe) / kPipeCount;
        EmptyBarrier::wait(&empty[pipe], last_round & 1);
      }
    }
  }

  // Producer 的占位 slice 不参与计算或写入；实际 MMA 编号连续映射为
  // 0..kComputeThreads-1，每连续 128 个物理线程构成一个对齐 warpgroup。
  int mma_thread = is_consumer ? int(threadIdx.x) - kProducerThreads : 0;
  ThrMMA thr_mma = mma.get_thread_slice(mma_thread);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
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
        // 本 warpgroup 已完成前一批。所有计算线程各到达一次；最后一个
        // warpgroup 完成后，producer 才能覆盖整个 A/B 槽位。
        warpgroup_wait<1>();
        warpgroup_fence_operand(tCrC);
        EmptyBarrier::arrive(&empty[release_state.index()]);
        ++release_state;
      }
    }
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);
    if constexpr (kMaxInflight == 2) {
      if (k_tile_count > 0) {
        EmptyBarrier::arrive(&empty[release_state.index()]);
      }
    }
  }

  // 保留 v3 的合并写回。两个 CTA barrier 必须由 producer 和所有 consumer
  // 共同参加：先等 producer tail/所有 WGMMA 结束，再等完整 C tile 就绪。
  if constexpr (kCoalescedEpilogue) {
    if (alpha == Alpha(1.0f) && beta == Beta(0.0f)) {
      static_assert(sizeof(TC) == 2, "Vector epilogue expects 16-bit C elements");
      static_assert(PEAK_CTA_M % 8 == 0, "Vector epilogue requires CTA_M divisible by 8");
      constexpr int kElementsPerVector = 8;
      constexpr int kVectorsPerColumn = PEAK_CTA_M / kElementsPerVector;
      constexpr int kVectorsPerTile = PEAK_CTA_N * kVectorsPerColumn;
      __syncthreads();
      if (is_consumer) {
        Tensor sC = make_tensor(
            make_smem_ptr(reinterpret_cast<TC*>(shared_memory)),
            make_layout(make_shape(Int<PEAK_CTA_M>{}, Int<PEAK_CTA_N>{}),
                        make_stride(Int<1>{}, Int<PEAK_CTA_M>{})));
        Tensor tCsC = thr_mma.partition_C(sC);
        copy(tCrC, tCsC);
      }
      __syncthreads();
      TC const* smem_C = reinterpret_cast<TC const*>(shared_memory);
      int const ldc = int(get<1>(dC));
      // 搬运组也可协助连续写回，所有 CTA 线程共同覆盖各个 16 字节向量。
      for (int vec = int(threadIdx.x); vec < kVectorsPerTile; vec += int(blockDim.x)) {
        int tile_col = vec / kVectorsPerColumn;
        int tile_row = (vec % kVectorsPerColumn) * kElementsPerVector;
        int global_col = int(blockIdx.y) * PEAK_CTA_N + tile_col;
        int global_row = int(blockIdx.x) * PEAK_CTA_M + tile_row;
        TC* dst = C + global_col * ldc + global_row;
        uint4 word = *reinterpret_cast<uint4 const*>(smem_C + vec * kElementsPerVector);
        *reinterpret_cast<uint4*>(dst) = word;
      }
    } else if (is_consumer) {
      axpby(alpha, tCrC, beta, tCgC);
    }
  } else if (is_consumer) {
    axpby(alpha, tCrC, beta, tCgC);
  }
}
}  // namespace gemm_backend
