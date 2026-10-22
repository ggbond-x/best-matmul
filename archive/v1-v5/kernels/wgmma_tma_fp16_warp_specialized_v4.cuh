#pragma once
#include "detail/wgmma_config.cuh"

// v4：保留 v3 的计算流水/合并写回，另设一个 producer warpgroup 专门搬运。
// 线程 0..127 为 producer，128..255 为唯一的计算 warpgroup。
#ifndef WGMMA_COMPUTE_THREADS
#define WGMMA_COMPUTE_THREADS 128
#endif
static_assert(WGMMA_COMPUTE_THREADS == 128,
              "v4 has exactly 128 compute threads; use v5 for multiple compute warpgroups");

namespace gemm_backend {
inline constexpr int kInterfaceVersion = 1;
inline constexpr char kName[] = "v4_warp_specialized";
}  // namespace gemm_backend

#include "detail/warp_specialized_config.cuh"
#include "detail/tma_wgmma_warp_specialized_device.cuh"
#include "detail/tma_wgmma_adapter.cuh"
