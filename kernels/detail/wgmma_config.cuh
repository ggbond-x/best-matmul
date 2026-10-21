#pragma once
// 各版本共用的编译期配置：CTA tile、PIPE 槽数、WGMMA 指令。
// 宏便于从 build_benchmark.sh 传参；它们在编译时确定，不产生设备分支。
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/arch/mma_sm90.h>
#include <cutlass/device_kernel.h>

// 一条指令的形状为 64 x GMMA_INST_N x 16，输入、累加和输出均为 FP16。
// SS 表示 A/B 均通过共享内存描述符提供操作数。
#ifndef GMMA_INST_N
#define GMMA_INST_N 64
#endif

#ifndef PIPE
#define PIPE 3
#endif

// CTA tile 是一个线程块负责的 (M,N,K) 分块，可以包含多条 WGMMA 指令。
// 每个 PIPE 槽位存放一对 A/B tile，例如 PIPE=3 表示三槽环形缓冲。
#ifndef PEAK_CTA_M
#define PEAK_CTA_M 128
#endif

#ifndef PEAK_CTA_N
#define PEAK_CTA_N 128
#endif

#ifndef PEAK_CTA_K
#define PEAK_CTA_K 64
#endif

static_assert(PEAK_CTA_M > 0 && PEAK_CTA_N > 0 && PEAK_CTA_K > 0 && PIPE > 0,
              "CTA dimensions and PIPE must be positive");

static_assert(GMMA_INST_N == 8   ||
              GMMA_INST_N == 16  ||
              GMMA_INST_N == 32  ||
              GMMA_INST_N == 64  ||
              GMMA_INST_N == 128 ||
              GMMA_INST_N == 256,
              "GMMA_INST_N must be one of {8, 16, 32, 64, 128, 256}");

static_assert(PEAK_CTA_M % 64 == 0,
              "PEAK_CTA_M must be a multiple of GMMA instruction M dimension 64");
static_assert(PEAK_CTA_N % GMMA_INST_N == 0,
              "PEAK_CTA_N must be a multiple of GMMA_INST_N");
static_assert(PEAK_CTA_K % 16 == 0,
              "PEAK_CTA_K must be a multiple of GMMA instruction K dimension 16");

namespace gemm_backend {
using namespace cute;

// 从指令 N 维度映射到 CUTLASS 类型。这里只做直接的类型映射；
// MajorA/MajorB 由适配层依据 NT/TN 选择，不需要运行时指令选择逻辑。
template <int InstN, GMMA::Major MajorA, GMMA::Major MajorB>
struct Sm90F16F16F16SsSelector;

template <GMMA::Major MajorA, GMMA::Major MajorB>
struct Sm90F16F16F16SsSelector<8, MajorA, MajorB> {
  using type = SM90_64x8x16_F16F16F16_SS<MajorA, MajorB>;
};

template <GMMA::Major MajorA, GMMA::Major MajorB>
struct Sm90F16F16F16SsSelector<16, MajorA, MajorB> {
  using type = SM90_64x16x16_F16F16F16_SS<MajorA, MajorB>;
};

template <GMMA::Major MajorA, GMMA::Major MajorB>
struct Sm90F16F16F16SsSelector<32, MajorA, MajorB> {
  using type = SM90_64x32x16_F16F16F16_SS<MajorA, MajorB>;
};

template <GMMA::Major MajorA, GMMA::Major MajorB>
struct Sm90F16F16F16SsSelector<64, MajorA, MajorB> {
  using type = SM90_64x64x16_F16F16F16_SS<MajorA, MajorB>;
};

template <GMMA::Major MajorA, GMMA::Major MajorB>
struct Sm90F16F16F16SsSelector<128, MajorA, MajorB> {
  using type = SM90_64x128x16_F16F16F16_SS<MajorA, MajorB>;
};

template <GMMA::Major MajorA, GMMA::Major MajorB>
struct Sm90F16F16F16SsSelector<256, MajorA, MajorB> {
  using type = SM90_64x256x16_F16F16F16_SS<MajorA, MajorB>;
};
}  // namespace gemm_backend
