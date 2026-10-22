#pragma once

// 主机侧准备：布局、TMA 描述符、启动配置都在 prepare() 里完成。
// 返回的对象只负责把一次 GEMM 提交到测量流。描述符保存在对象里，活到测量结束。
//
// 本文件由 gemm_teach.cuh / gemm_fast.cuh 在定义 gemm_device 与下列常量之后包含：
//   kName, kProducerThreads, kComputeThreads, kComputeWarpgroups, kCtaThreads
// SharedStorage 与 TMA 布局来自 smem_tma.cuh，不依赖包含顺序。
#include "smem_tma.cuh"

#include "../host/common.cuh"

#include <cutlass/cluster_launch.hpp>

#include <algorithm>
#include <memory>
#include <utility>

namespace gemm_backend {

static_assert(kProducerThreads == 0 || kProducerThreads == kProducerWarpgroupThreads,
              "producer threads are either none or one warpgroup");
static_assert(kComputeThreads == kWarpgroupThreads * kComputeWarpgroups,
              "compute threads must match the warpgroup count");
static_assert(kCtaThreads == kProducerThreads + kComputeThreads,
              "CTA size is producer threads plus compute threads");

inline void validate_problem(gemm_bench::GemmProblem const& problem)
{
  if (problem.m % kCtaM != 0 || problem.n % kCtaN != 0 || problem.k % kCtaK != 0) {
    std::cerr << "Selected kernel requires M % " << kCtaM
              << " == 0, N % " << kCtaN << " == 0, K % " << kCtaK << " == 0\n";
    std::exit(EXIT_FAILURE);
  }
}

inline void print_configuration()
{
  std::printf("Kernel role: %s\n", kName);
  std::printf("WGMMA: SM90_64x%dx16_F16F16F16_SS, CTA %dx%dx%d, PIPE=%d, inflight=%d\n",
              kInstN, kCtaM, kCtaN, kCtaK, kPipe, kMaxInflight);
  std::printf("Producer threads=%d, compute threads=%d, compute warpgroups=%d, CTA threads=%d\n",
              kProducerThreads, kComputeThreads, kComputeWarpgroups, kCtaThreads);
  std::printf("Epilogue: %s, alpha=1, beta=0\n",
              kCoalescedEpilogue ? "coalesced 16-byte store" : "direct axpby");
}

// 类型擦除只包在准备好的状态外面。operator() 不再分支 NT/TN，也不再建描述符。
struct LaunchState {
  virtual ~LaunchState() = default;
  __host__ virtual void launch() const = 0;
};

struct PreparedGemm {
  gemm_bench::LaunchConfig config{};
  std::unique_ptr<LaunchState> state;

  __host__ void operator()() const { state->launch(); }
};

// 把一次启动需要的值装进可移动的包。NT/TN 的 TMA 类型不同，所以包是模板。
// kernel 保存成 void const*：__global__ 函数指针不能写成普通的 void(*)(...)。
template <class Shape, class CtaTiler, class TmaA, class TmaB,
          class StrideC, class TiledMma, class Alpha, class Beta>
struct GemmPack {
  using Element = gemm_bench::Element;

  Shape prob_shape;
  CtaTiler cta_tiler;
  Element const* A;
  TmaA tmaA;
  Element const* B;
  TmaB tmaB;
  Element* C;
  StrideC dC;
  TiledMma tiled_mma;
  Alpha alpha;
  Beta beta;
  void const* kernel;
  cutlass::ClusterLaunchParams params;

  __host__ void launch() const {
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        params, kernel,
        prob_shape, cta_tiler, A, tmaA, B, tmaB, C, dC, tiled_mma, alpha, beta);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Error: Failed at kernel launch\n";
      std::exit(EXIT_FAILURE);
    }
  }
};

template <class Pack>
struct LaunchModel final : LaunchState {
  Pack pack;
  explicit LaunchModel(Pack pack_) : pack(std::move(pack_)) {}
  __host__ void launch() const override { pack.launch(); }
};

template <GMMA::Major Major>
__host__ PreparedGemm prepare_major(gemm_bench::GemmProblem const& problem,
                                    gemm_bench::GemmBuffers& buffers,
                                    cudaStream_t stream)
{
  using Element = gemm_bench::Element;

  auto prob_shape = make_shape(problem.m, problem.n, problem.k);
  auto bM = Int<kCtaM>{};
  auto bN = Int<kCtaN>{};
  auto bK = Int<kCtaK>{};
  auto bP = Int<kPipe>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto dA = make_operand_stride<Major>(problem.ldA());
  auto dB = make_operand_stride<Major>(problem.ldB());
  auto dC = make_stride(Int<1>{}, problem.ldC());
  auto sA = make_operand_smem_layout<Major, Element>(make_shape(bM, bK, bP));
  auto sB = make_operand_smem_layout<Major, Element>(make_shape(bN, bK, bP));
  TiledMMA tiled_mma = make_tiled_mma(
      Fp16SsAtom<Major, Major>{},
      Layout<Shape<Int<kComputeWarpgroups>, _1, _1>>{});
  CUTE_STATIC_ASSERT_V(size(tiled_mma) == Int<kComputeThreads>{});

  auto const* A = buffers.d_A.data().get();
  auto const* B = buffers.d_B.data().get();
  auto* C = buffers.d_C.data().get();
  Tensor mA = make_tensor(A, make_shape(problem.m, problem.k), dA);
  Tensor mB = make_tensor(B, make_shape(problem.n, problem.k), dB);
  // 描述符只编码一个槽位；设备代码选择 K tile 和 PIPE 槽。
  auto tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
  auto tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));
  Element alpha(1.0f);
  Element beta(0.0f);

  using Storage = SharedStorage<Element, Element, decltype(sA), decltype(sB)>;
  gemm_bench::LaunchConfig config;
  int mainloop_bytes = int(sizeof(Storage));
  int epilogue_bytes = kCoalescedEpilogue ? kCtaM * kCtaN * int(sizeof(Element)) : 0;
  config.smem_bytes = std::max(mainloop_bytes, epilogue_bytes);
  config.block = dim3(kProducerThreads + size(tiled_mma));
  config.grid = dim3(round_up(size(ceil_div(problem.m, bM)), config.cluster.x),
                     round_up(size(ceil_div(problem.n, bN)), config.cluster.y));

  using Pack = GemmPack<decltype(prob_shape), decltype(cta_tiler), decltype(tmaA), decltype(tmaB),
                        decltype(dC), decltype(tiled_mma), decltype(alpha), decltype(beta)>;
  auto* kernel = &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                             Element, decltype(sA), decltype(tmaA),
                             Element, decltype(sB), decltype(tmaB),
                             Element, decltype(dC), decltype(tiled_mma),
                             decltype(alpha), decltype(beta)>;
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.smem_bytes));

  cutlass::ClusterLaunchParams params = {config.grid, config.block, config.cluster, config.smem_bytes};
  params.cuda_stream = stream;

  Pack pack{prob_shape, cta_tiler, A, tmaA, B, tmaB, C, dC,
            tiled_mma, alpha, beta, reinterpret_cast<void const*>(kernel), params};

  PreparedGemm prepared;
  prepared.config = config;
  prepared.state = std::make_unique<LaunchModel<Pack>>(std::move(pack));
  return prepared;
}

inline PreparedGemm prepare(gemm_bench::BenchmarkOptions const& options,
                            gemm_bench::GemmBuffers& buffers)
{
  if (options.problem.is_nt()) {
    return prepare_major<GMMA::Major::MN>(
        options.problem, buffers, options.measurement.stream);
  }
  return prepare_major<GMMA::Major::K>(
      options.problem, buffers, options.measurement.stream);
}

}  // namespace gemm_backend
