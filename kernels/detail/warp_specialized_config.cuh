#pragma once
#include "wgmma_config.cuh"

// v4/v5 保留 v3 的两个独立开关。
#ifndef WGMMA_MAX_INFLIGHT
#define WGMMA_MAX_INFLIGHT 2
#endif
#ifndef EPILOGUE_COALESCED
#define EPILOGUE_COALESCED 1
#endif
static_assert(WGMMA_MAX_INFLIGHT == 1 || WGMMA_MAX_INFLIGHT == 2,
              "WGMMA_MAX_INFLIGHT must be 1 or 2");
static_assert(WGMMA_MAX_INFLIGHT == 1 || PIPE >= 2,
              "WGMMA_MAX_INFLIGHT=2 requires at least two TMA buffers");
static_assert(EPILOGUE_COALESCED == 0 || EPILOGUE_COALESCED == 1,
              "EPILOGUE_COALESCED must be 0 or 1");
static_assert(WGMMA_COMPUTE_THREADS >= 128 && WGMMA_COMPUTE_THREADS % 128 == 0,
              "WGMMA_COMPUTE_THREADS must be a positive multiple of 128");
static_assert(WGMMA_COMPUTE_THREADS <= 896,
              "128 producer threads plus compute threads must not exceed 1024 CTA threads");

namespace gemm_backend {
inline constexpr int kMaxInflight = WGMMA_MAX_INFLIGHT;
inline constexpr bool kCoalescedEpilogue = (EPILOGUE_COALESCED != 0);
inline constexpr int kProducerThreads = 128;
inline constexpr int kComputeThreads = WGMMA_COMPUTE_THREADS;
// 下限使非法线程数也能给出上面的明确断言，而不是先触发除零错误。
inline constexpr int kComputeWarpgroups = kComputeThreads >= 128 ? kComputeThreads / 128 : 1;
static_assert(PEAK_CTA_M % (64 * kComputeWarpgroups) == 0,
              "PEAK_CTA_M must be divisible by 64 * (WGMMA_COMPUTE_THREADS / 128)");
}  // namespace gemm_backend
