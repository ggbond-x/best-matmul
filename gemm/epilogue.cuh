#pragma once

// 唯一的写回：寄存器按 MMA 线程映射进共享内存，再以 16 字节向量连续写回 C。
// alpha=1、beta=0。调用时 CTA 内每个线程都必须进入本函数（里面有 __syncthreads）。
#include "config.cuh"

namespace gemm_backend {

template <class TC, class CStride, class ThrMma, class FragC>
__device__ __forceinline__ void store_c_coalesced(TC* C, CStride dC,
                                                   ThrMma const& thr_mma, FragC const& accum,
                                                   char* shared_memory, bool writes_accumulator)
{
  static_assert(sizeof(TC) == 2, "coalesced epilogue stores 16-bit C elements");
  static_assert(kCtaM % 8 == 0, "coalesced epilogue needs CTA_M divisible by 8");
  constexpr int kElementsPerVector = 8;  // 8 个 FP16 = 16 字节 = 一个 uint4。
  constexpr int kVectorsPerColumn = kCtaM / kElementsPerVector;
  constexpr int kVectorsPerTile = kCtaN * kVectorsPerColumn;

  // 先等全部线程结束对 A/B 槽位的访问，再把同一块动态共享内存当作 C 暂存。
  __syncthreads();
  if (writes_accumulator) {
    Tensor sC = make_tensor(
        make_smem_ptr(reinterpret_cast<TC*>(shared_memory)),
        make_layout(make_shape(Int<kCtaM>{}, Int<kCtaN>{}),
                    make_stride(Int<1>{}, Int<kCtaM>{})));
    copy(accum, thr_mma.partition_C(sC));
  }
  __syncthreads();

  TC const* smem_C = reinterpret_cast<TC const*>(shared_memory);
  int const ldc = int(get<1>(dC));  // C 为列主序，M 维连续。
  for (int vec = int(threadIdx.x); vec < kVectorsPerTile; vec += int(blockDim.x)) {
    int tile_col = vec / kVectorsPerColumn;
    int tile_row = (vec % kVectorsPerColumn) * kElementsPerVector;
    int global_col = int(blockIdx.y) * kCtaN + tile_col;
    int global_row = int(blockIdx.x) * kCtaM + tile_row;
    TC* dst = C + global_col * ldc + global_row;
    // CTA_M 是 8 的倍数，向量不跨列。cudaMalloc 与 ldC 保证 16 字节对齐。
    uint4 word = *reinterpret_cast<uint4 const*>(smem_C + vec * kElementsPerVector);
    *reinterpret_cast<uint4*>(dst) = word;
  }
}

}  // namespace gemm_backend
