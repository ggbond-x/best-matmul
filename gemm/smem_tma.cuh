#pragma once

// 两份设备代码共用的共享内存和 TMA 布局。launch.cuh 直接包含本文件，
// 不依赖“先包含哪个 kernel”来看到 SharedStorage。
#include "config.cuh"

#include <cstdint>

namespace gemm_backend {

// 每个 PIPE 槽一份 A tile 和一份 B tile。full/empty 屏障与槽位一一对应。
// 128 字节对齐与 SW128 swizzle 一致，写回阶段也会复用这块基址。
template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
  uint64_t tma_barrier[size<2>(SmemLayoutA{})];
  uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};

// Major::MN：M/N 维连续，stride=(1, ld)，对应 NT。
// Major::K ：K 维连续，stride=(ld, 1)，对应 TN。
template <GMMA::Major Major>
auto make_operand_stride(int leading_dimension)
{
  static_assert(Major == GMMA::Major::MN || Major == GMMA::Major::K);
  if constexpr (Major == GMMA::Major::MN) {
    return make_stride(Int<1>{}, leading_dimension);
  } else {
    return make_stride(leading_dimension, Int<1>{});
  }
}

// 将 SW128 原子铺满 (M或N, K, PIPE)。布局、WGMMA Major 与 TMA 描述符必须一致。
template <GMMA::Major Major, class T, class Shape>
auto make_operand_smem_layout(Shape const& shape)
{
  static_assert(Major == GMMA::Major::MN || Major == GMMA::Major::K);
  if constexpr (Major == GMMA::Major::MN) {
    constexpr int mn = decltype(size<0>(Shape{}))::value;
    if constexpr ((mn & (mn - 1)) != 0) {
      // 非 2 次幂的 M/N：先铺 K，再铺 M/N，避免把 3/5/7 插进 swizzle 的连续地址。
      return tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, shape, Step<_2, _1, _3>{});
    } else {
      return tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, shape);
    }
  } else {
    return tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, shape);
  }
}

}  // namespace gemm_backend
