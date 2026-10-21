#pragma once
#include "detail/wgmma_config.cuh"

// v1：每提交一个 K tile，就等待该批 WGMMA 全部完成。
// 适合先理解「TMA 搬入 -> WGMMA 计算 -> 释放槽位」的基本流程。
namespace gemm_backend {
inline constexpr int kInterfaceVersion = 1;
inline constexpr char kName[] = "v1_baseline";
inline constexpr int kMaxInflight = 1;
inline constexpr bool kCoalescedEpilogue = false;
}  // namespace gemm_backend

// 三个版本使用同一套准备代码，通过上面的两个常量选择主循环和写回。
#include "detail/tma_wgmma_device.cuh"
#include "detail/tma_wgmma_adapter.cuh"
