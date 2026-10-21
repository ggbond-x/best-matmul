#pragma once
#include "detail/wgmma_config.cuh"

// v3：沿用 v2 的主循环，再将寄存器结果经共享内存重排，连续写回 C。
// 两个开关分别控制计算流水和输出重排，便于单独比较它们的影响。
#ifndef WGMMA_MAX_INFLIGHT
#define WGMMA_MAX_INFLIGHT 2
#endif
static_assert(WGMMA_MAX_INFLIGHT == 1 || WGMMA_MAX_INFLIGHT == 2,
              "WGMMA_MAX_INFLIGHT must be 1 or 2");
static_assert(WGMMA_MAX_INFLIGHT == 1 || PIPE >= 2,
              "WGMMA_MAX_INFLIGHT=2 requires at least two TMA buffers");

#ifndef EPILOGUE_COALESCED
#define EPILOGUE_COALESCED 1
#endif
static_assert(EPILOGUE_COALESCED == 0 || EPILOGUE_COALESCED == 1,
              "EPILOGUE_COALESCED must be 0 or 1");

namespace gemm_backend {
inline constexpr int kInterfaceVersion = 1;
inline constexpr char kName[] = "v3_epilogue";
inline constexpr int kMaxInflight = WGMMA_MAX_INFLIGHT;
inline constexpr bool kCoalescedEpilogue = (EPILOGUE_COALESCED != 0);
}  // namespace gemm_backend

#include "detail/tma_wgmma_device.cuh"
#include "detail/tma_wgmma_adapter.cuh"
