#pragma once

// 唯一的默认配置。teach 与 fast、单内核构建与 README 都使用这些常量。
// 不使用 #ifndef 宏：改 tile / PIPE / 指令 N 时只改这里，并让下面的断言通过。
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/mma_sm90.h>
#include <cutlass/device_kernel.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>

namespace gemm_backend {
using namespace cute;

// CTA tile、流水槽数、WGMMA 指令。指令形状固定为 64 x 128 x 16，SS，FP16。
inline constexpr int kCtaM = 128;
inline constexpr int kCtaN = 128;
inline constexpr int kCtaK = 64;
inline constexpr int kPipe = 3;
inline constexpr int kInstM = 64;
inline constexpr int kInstN = 128;
inline constexpr int kInstK = 16;
// 每个计算 warpgroup 最多两批 WGMMA 在途，因此环形缓冲至少两个槽。
inline constexpr int kMaxInflight = 2;
inline constexpr int kWarpgroupThreads = 128;

// teach：一个计算 warpgroup，线程自己发起 TMA。
inline constexpr int kTeachComputeWarpgroups = 1;
// fast：一个 producer warpgroup + 两个计算 warpgroup。
inline constexpr int kFastComputeWarpgroups = 2;
inline constexpr int kProducerWarpgroupThreads = 128;

// INST_N 已固定为 128，直接使用这一条 SS 指令，不再按 N 做类型选择。
template <GMMA::Major MajorA, GMMA::Major MajorB>
using Fp16SsAtom = SM90_64x128x16_F16F16F16_SS<MajorA, MajorB>;

static_assert(kCtaM > 0 && kCtaN > 0 && kCtaK > 0, "CTA tile must be positive");
static_assert(kPipe >= 2, "two in-flight WGMMA batches need at least two TMA stages");
static_assert(kMaxInflight == 2, "both kernels use two in-flight WGMMA batches");
static_assert(kInstN == 128, "the main path is the 64x128x16 SS instruction");
static_assert(kCtaM % kInstM == 0, "CTA_M must be a multiple of the WGMMA M dimension");
static_assert(kCtaN % kInstN == 0, "CTA_N must be a multiple of the WGMMA N dimension");
// 同一二进制同时实例化 NT 与 TN 的 SW128 布局，K 与 N 都要是 64 的倍数。
static_assert(kCtaN % 64 == 0 && kCtaK % 64 == 0,
              "SW128 NT/TN layouts need CTA_N and CTA_K to be multiples of 64");
static_assert(kCtaM % (kInstM * kTeachComputeWarpgroups) == 0,
              "teach CTA_M must cover its compute warpgroups");
static_assert(kCtaM % (kInstM * kFastComputeWarpgroups) == 0,
              "fast CTA_M must be divisible by 64 * compute warpgroups");
static_assert(kProducerWarpgroupThreads == kWarpgroupThreads,
              "the producer reservation is one warpgroup");
static_assert(kProducerWarpgroupThreads + kWarpgroupThreads * kFastComputeWarpgroups <= 1024,
              "fast CTA threads must fit in one block");
}  // namespace gemm_backend
