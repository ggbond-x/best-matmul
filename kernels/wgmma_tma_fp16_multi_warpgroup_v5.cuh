#pragma once
#include "detail/wgmma_config.cuh"

// v5：沿用 v4 的 producer/consumer 分工，计算线程数以 128 为单位配置。
// 默认 256 个计算线程（两个 warpgroup），各负责 CTA 输出的不同 M 区域。
#ifndef WGMMA_COMPUTE_THREADS
#define WGMMA_COMPUTE_THREADS 256
#endif

namespace gemm_backend {
inline constexpr int kInterfaceVersion = 1;
inline constexpr char kName[] = "v5_multi_warpgroup";
}  // namespace gemm_backend

#include "detail/warp_specialized_config.cuh"
#include "detail/tma_wgmma_warp_specialized_device.cuh"
#include "detail/tma_wgmma_adapter.cuh"
