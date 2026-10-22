#pragma once
#include "detail/wgmma_config.cuh"

// v2：允许两个 WGMMA 批次在途，等待较早一批完成后立即复用其槽位。
// 设为 1 可切回 v1 的调度，用同一份代码做对照实验。
#ifndef WGMMA_MAX_INFLIGHT
#define WGMMA_MAX_INFLIGHT 2
#endif
static_assert(WGMMA_MAX_INFLIGHT == 1 || WGMMA_MAX_INFLIGHT == 2,
              "WGMMA_MAX_INFLIGHT must be 1 or 2");
static_assert(WGMMA_MAX_INFLIGHT == 1 || PIPE >= 2,
              "WGMMA_MAX_INFLIGHT=2 requires at least two TMA buffers");

namespace gemm_backend {
inline constexpr int kInterfaceVersion = 1;
inline constexpr char kName[] = "v2_pipeline";
inline constexpr int kMaxInflight = WGMMA_MAX_INFLIGHT;
inline constexpr bool kCoalescedEpilogue = false;
}  // namespace gemm_backend

#include "detail/tma_wgmma_device.cuh"
#include "detail/tma_wgmma_adapter.cuh"
