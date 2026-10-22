#pragma once

// 不传 -D 时的唯一默认值。teach 与 fast 都读这些宏；命令行 -D 必须先于本头文件生效。
// 旧套件用过的名字保持不变：PEAK_CTA_*、PIPE、GMMA_INST_N、WGMMA_MAX_INFLIGHT、
// EPILOGUE_COALESCED、WGMMA_COMPUTE_THREADS。producer 线程数没有宏，fast 固定一个 warpgroup。
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/mma_sm90.h>
#include <cutlass/device_kernel.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>

#ifndef PEAK_CTA_M
#define PEAK_CTA_M 128
#endif
#ifndef PEAK_CTA_N
#define PEAK_CTA_N 128
#endif
#ifndef PEAK_CTA_K
#define PEAK_CTA_K 64
#endif
#ifndef PIPE
#define PIPE 3
#endif
#ifndef GMMA_INST_N
#define GMMA_INST_N 128
#endif
#ifndef WGMMA_MAX_INFLIGHT
#define WGMMA_MAX_INFLIGHT 2
#endif
#ifndef EPILOGUE_COALESCED
#define EPILOGUE_COALESCED 1
#endif
// 只被 fast 读取。teach 始终是一个计算 warpgroup，忽略这个宏。
#ifndef WGMMA_COMPUTE_THREADS
#define WGMMA_COMPUTE_THREADS 256
#endif

static_assert(PEAK_CTA_M > 0 && PEAK_CTA_N > 0 && PEAK_CTA_K > 0 && PIPE >= 1,
              "CTA dimensions must be positive and PIPE must be at least 1");
static_assert(WGMMA_MAX_INFLIGHT == 1 || WGMMA_MAX_INFLIGHT == 2,
              "WGMMA_MAX_INFLIGHT must be 1 or 2");
static_assert(WGMMA_MAX_INFLIGHT == 1 || PIPE >= 2,
              "WGMMA_MAX_INFLIGHT=2 requires at least two TMA stages");
static_assert(EPILOGUE_COALESCED == 0 || EPILOGUE_COALESCED == 1,
              "EPILOGUE_COALESCED must be 0 or 1");
static_assert(GMMA_INST_N == 8 || GMMA_INST_N == 16 || GMMA_INST_N == 32 ||
                  GMMA_INST_N == 64 || GMMA_INST_N == 128 || GMMA_INST_N == 256,
              "GMMA_INST_N must be one of {8, 16, 32, 64, 128, 256}");
static_assert(PEAK_CTA_M % 64 == 0,
              "PEAK_CTA_M must be a multiple of the WGMMA M dimension 64");
static_assert(PEAK_CTA_N % GMMA_INST_N == 0,
              "PEAK_CTA_N must be a multiple of GMMA_INST_N");
// 同一二进制同时实例化 NT 与 TN 的 SW128 布局，N 和 K 都要是 64 的倍数。
static_assert(PEAK_CTA_N % 64 == 0 && PEAK_CTA_K % 64 == 0,
              "SW128 NT/TN layouts need PEAK_CTA_N and PEAK_CTA_K to be multiples of 64");
static_assert(WGMMA_COMPUTE_THREADS >= 128 && WGMMA_COMPUTE_THREADS % 128 == 0,
              "WGMMA_COMPUTE_THREADS must be a positive multiple of 128");
static_assert(WGMMA_COMPUTE_THREADS <= 896,
              "128 producer threads plus compute threads must not exceed 1024");

namespace gemm_backend {
using namespace cute;

inline constexpr int kCtaM = PEAK_CTA_M;
inline constexpr int kCtaN = PEAK_CTA_N;
inline constexpr int kCtaK = PEAK_CTA_K;
inline constexpr int kPipe = PIPE;
inline constexpr int kInstM = 64;
inline constexpr int kInstN = GMMA_INST_N;
inline constexpr int kInstK = 16;
inline constexpr int kMaxInflight = WGMMA_MAX_INFLIGHT;
inline constexpr bool kCoalescedEpilogue = (EPILOGUE_COALESCED != 0);
inline constexpr int kWarpgroupThreads = 128;

// teach：一个计算 warpgroup，线程自己发起 TMA。不读 WGMMA_COMPUTE_THREADS。
inline constexpr int kTeachComputeWarpgroups = 1;
// fast：一个 producer warpgroup，计算线程数来自 WGMMA_COMPUTE_THREADS。
inline constexpr int kFastComputeThreads = WGMMA_COMPUTE_THREADS;
inline constexpr int kFastComputeWarpgroups = kFastComputeThreads / kWarpgroupThreads;
inline constexpr int kProducerWarpgroupThreads = 128;

// 按指令 N 选 SS FP16 atom。Major 仍由 NT/TN 在实例化时决定。
template <int InstN, GMMA::Major MajorA, GMMA::Major MajorB>
struct Fp16SsAtomSelector;

template <GMMA::Major MajorA, GMMA::Major MajorB>
struct Fp16SsAtomSelector<8, MajorA, MajorB> {
  using type = SM90_64x8x16_F16F16F16_SS<MajorA, MajorB>;
};
template <GMMA::Major MajorA, GMMA::Major MajorB>
struct Fp16SsAtomSelector<16, MajorA, MajorB> {
  using type = SM90_64x16x16_F16F16F16_SS<MajorA, MajorB>;
};
template <GMMA::Major MajorA, GMMA::Major MajorB>
struct Fp16SsAtomSelector<32, MajorA, MajorB> {
  using type = SM90_64x32x16_F16F16F16_SS<MajorA, MajorB>;
};
template <GMMA::Major MajorA, GMMA::Major MajorB>
struct Fp16SsAtomSelector<64, MajorA, MajorB> {
  using type = SM90_64x64x16_F16F16F16_SS<MajorA, MajorB>;
};
template <GMMA::Major MajorA, GMMA::Major MajorB>
struct Fp16SsAtomSelector<128, MajorA, MajorB> {
  using type = SM90_64x128x16_F16F16F16_SS<MajorA, MajorB>;
};
template <GMMA::Major MajorA, GMMA::Major MajorB>
struct Fp16SsAtomSelector<256, MajorA, MajorB> {
  using type = SM90_64x256x16_F16F16F16_SS<MajorA, MajorB>;
};

template <GMMA::Major MajorA, GMMA::Major MajorB>
using Fp16SsAtom = typename Fp16SsAtomSelector<kInstN, MajorA, MajorB>::type;

static_assert(kCtaM % (kInstM * kTeachComputeWarpgroups) == 0,
              "teach CTA_M must cover its one compute warpgroup");
static_assert(kProducerWarpgroupThreads == kWarpgroupThreads,
              "the producer reservation is one warpgroup");
}  // namespace gemm_backend
