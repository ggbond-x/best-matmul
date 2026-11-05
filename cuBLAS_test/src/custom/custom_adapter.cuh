#pragma once
// 自研 Hopper kernel 的 host 适配层。
//
// 每个 .cu 只包含一个 kernel 头文件，因此本文件中的 gemm_backend 常量
// （PIPE、kComputeThreads、SharedStorage 等）在各自翻译单元内解析。
// 这里不改动 kernels/ 下的任何数学逻辑，只负责：
//   - 依据 NT/TN 选择 CUTE stride 与共享内存布局；
//   - 构造 TMA 描述符与 WGMMA TiledMMA（一次性，不计时）；
//   - 把启动参数封存成可重复调用的 std::function。
#include "gemm/backend.hpp"

#include <cutlass/cluster_launch.hpp>
#include <cutlass/half.h>

#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#include <algorithm>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace gemm {
namespace custom_detail {

using Element = cutlass::half_t;

// Major::MN：M/N 维连续（NT）；Major::K：K 维连续（TN）。
// 两种 stride 的类型不同，if constexpr 只实例化所选类型。
template <cute::GMMA::Major Major>
auto make_operand_stride(int leading_dimension) {
  static_assert(Major == cute::GMMA::Major::MN || Major == cute::GMMA::Major::K);
  if constexpr (Major == cute::GMMA::Major::MN) {
    return cute::make_stride(cute::Int<1>{}, leading_dimension);
  } else {
    return cute::make_stride(leading_dimension, cute::Int<1>{});
  }
}

// 与 kernels/detail/tma_wgmma_adapter.cuh 完全一致的 SW128 共享内存布局。
template <cute::GMMA::Major Major, class T, class Shape>
auto make_operand_smem_layout(Shape const& shape) {
  static_assert(Major == cute::GMMA::Major::MN || Major == cute::GMMA::Major::K);
  if constexpr (Major == cute::GMMA::Major::MN) {
    constexpr int mn = decltype(cute::size<0>(Shape{}))::value;
    if constexpr ((mn & (mn - 1)) != 0) {
      return cute::tile_to_shape(cute::GMMA::Layout_MN_SW128_Atom<T>{}, shape,
                                 cute::Step<cute::_2, cute::_1, cute::_3>{});
    } else {
      return cute::tile_to_shape(cute::GMMA::Layout_MN_SW128_Atom<T>{}, shape);
    }
  } else {
    return cute::tile_to_shape(cute::GMMA::Layout_K_SW128_Atom<T>{}, shape);
  }
}

struct CustomState {
  std::function<void()> launch;
  int smem_bytes = 0;
  dim3 grid{1, 1, 1};
  dim3 block{1, 1, 1};
  dim3 cluster{1, 1, 1};
  int pipe = 0;
  int inflight = 0;
  int threads = 0;
  int registers_per_thread = 0;
  int static_smem_bytes = 0;
  int theoretical_ctas_per_sm = 0;
};

class CustomPlan final : public GemmPlan {
 public:
  CustomPlan(CustomState state, std::string implementation, std::string kernel_name,
             std::string details)
      : state_(std::move(state)) {
    info.implementation = std::move(implementation);
    info.kernel_name = std::move(kernel_name);
    info.accum = Accum::FP16;
    info.algorithm_id = "custom_sm90_tma_wgmma";
    info.workspace_bytes = 0;
    info.heuristic_candidates = 1;
    info.details = std::move(details);
  }

  void run() override { state_.launch(); }

 private:
  CustomState state_;
};

template <cute::GMMA::Major Major>
CustomState prepare_impl(Options const& options, DeviceBuffers const& buffers,
                         cudaStream_t stream, std::string const& kernel_label) {
  Problem const& problem = options.problem;
  if (problem.m % PEAK_CTA_M != 0 || problem.n % PEAK_CTA_N != 0 ||
      problem.k % PEAK_CTA_K != 0) {
    throw Skip("custom kernel requires M%CTA_M==0, N%CTA_N==0, K%CTA_K==0 (CTA " +
               std::to_string(PEAK_CTA_M) + "x" + std::to_string(PEAK_CTA_N) + "x" +
               std::to_string(PEAK_CTA_K) + ")");
  }

  auto prob_shape = cute::make_shape(problem.m, problem.n, problem.k);
  auto b_m = cute::Int<PEAK_CTA_M>{};
  auto b_n = cute::Int<PEAK_CTA_N>{};
  auto b_k = cute::Int<PEAK_CTA_K>{};
  auto b_pipe = cute::Int<PIPE>{};
  auto cta_tiler = cute::make_shape(b_m, b_n, b_k);
  auto d_a = make_operand_stride<Major>(problem.ldA());
  auto d_b = make_operand_stride<Major>(problem.ldB());
  auto d_c = cute::make_stride(cute::Int<1>{}, problem.ldC());
  auto s_a = make_operand_smem_layout<Major, Element>(cute::make_shape(b_m, b_k, b_pipe));
  auto s_b = make_operand_smem_layout<Major, Element>(cute::make_shape(b_n, b_k, b_pipe));

  using GmmaAtom = typename gemm_backend::Sm90F16F16F16SsSelector<GMMA_INST_N, Major, Major>::type;
  auto tiled_mma = cute::make_tiled_mma(
      GmmaAtom{}, cute::Layout<cute::Shape<cute::Int<gemm_backend::kComputeWarpgroups>, cute::_1,
                                            cute::_1>>{});

  auto const* a = reinterpret_cast<Element const*>(buffers.a);
  auto const* b = reinterpret_cast<Element const*>(buffers.b);
  auto* c = reinterpret_cast<Element*>(buffers.c);
  auto m_a = cute::make_tensor(a, cute::make_shape(problem.m, problem.k), d_a);
  auto m_b = cute::make_tensor(b, cute::make_shape(problem.n, problem.k), d_b);
  auto tma_a = cute::make_tma_atom(cute::SM90_TMA_LOAD{}, m_a, s_a(cute::_, cute::_, 0),
                                         cute::make_shape(b_m, b_k));
  auto tma_b = cute::make_tma_atom(cute::SM90_TMA_LOAD{}, m_b, s_b(cute::_, cute::_, 0),
                                         cute::make_shape(b_n, b_k));
  Element alpha(options.alpha);
  Element beta(options.beta);

  using Storage = gemm_backend::SharedStorage<Element, Element, decltype(s_a), decltype(s_b)>;
  int const mainloop_bytes = int(sizeof(Storage));
  int const epilogue_bytes = gemm_backend::kCoalescedEpilogue
                                 ? PEAK_CTA_M * PEAK_CTA_N * int(sizeof(Element))
                                 : 0;
  int const smem_bytes = std::max(mainloop_bytes, epilogue_bytes);

  dim3 const block(gemm_backend::kProducerThreads + gemm_backend::kComputeThreads, 1, 1);
  dim3 const grid((problem.m + PEAK_CTA_M - 1) / PEAK_CTA_M,
                  (problem.n + PEAK_CTA_N - 1) / PEAK_CTA_N, 1);
  dim3 const cluster(1, 1, 1);
  auto* kernel = &gemm_backend::gemm_device<
      decltype(prob_shape), decltype(cta_tiler), Element, decltype(s_a), decltype(tma_a),
      Element, decltype(s_b), decltype(tma_b), Element, decltype(d_c), decltype(tiled_mma),
      decltype(alpha), decltype(beta)>;
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  cudaFuncAttributes attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  int resident_ctas = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident_ctas, kernel, int(block.x), smem_bytes));

  cutlass::ClusterLaunchParams params = {grid, block, cluster, smem_bytes};
  params.cuda_stream = stream;
  auto launch = [params, kernel, prob_shape, cta_tiler, a, tma_a, b, tma_b, c, d_c, tiled_mma,
                 alpha, beta]() {
    cutlass::Status const status = cutlass::launch_kernel_on_cluster(
        params, reinterpret_cast<void const*>(kernel), prob_shape, cta_tiler, a, tma_a, b, tma_b,
        c, d_c, tiled_mma, alpha, beta);
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("custom kernel launch failed (status " +
                               std::to_string(int(status)) + ")");
    }
  };

  CustomState state;
  state.launch = launch;
  state.smem_bytes = smem_bytes;
  state.grid = grid;
  state.block = block;
  state.cluster = cluster;
  state.pipe = PIPE;
  state.inflight = gemm_backend::kMaxInflight;
  state.threads = gemm_backend::kProducerThreads + gemm_backend::kComputeThreads;
  state.registers_per_thread = attributes.numRegs;
  state.static_smem_bytes = int(attributes.sharedSizeBytes);
  state.theoretical_ctas_per_sm = resident_ctas;
  (void)kernel_label;
  return state;
}

inline CustomState prepare(Options const& options, DeviceBuffers const& buffers,
                           cudaStream_t stream) {
  int major = 0;
  int minor = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, options.device));
  CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, options.device));
  if (major != 9) {
    throw Skip("custom WGMMA kernel requires compute capability 9.0 (Hopper); device is " +
               std::to_string(major) + "." + std::to_string(minor));
  }
  if (options.problem.is_nt()) {
    return prepare_impl<cute::GMMA::Major::MN>(options, buffers, stream, "NT");
  }
  return prepare_impl<cute::GMMA::Major::K>(options, buffers, stream, "TN");
}

inline std::unique_ptr<GemmPlan> make_plan(Options const& options, DeviceBuffers const& buffers,
                                           cudaStream_t stream, char const* implementation,
                                           char const* kernel_name, char const* source_header) {
  CustomState state = prepare(options, buffers, stream);
  std::ostringstream details;
  details << "{\"source\":\"" << source_header << "\",\"tile\":\"" << PEAK_CTA_M << "x"
          << PEAK_CTA_N << "x" << PEAK_CTA_K << "\",\"pipe\":" << PIPE
          << ",\"wgmma_max_inflight\":" << gemm_backend::kMaxInflight
          << ",\"gmma_n\":" << GMMA_INST_N << ",\"threads\":" << state.threads
          << ",\"smem_bytes\":" << state.smem_bytes << ",\"grid\":\"" << state.grid.x << "x"
          << state.grid.y << "x" << state.grid.z << "\",\"cluster\":\"" << state.cluster.x << "x"
          << state.cluster.y << "x" << state.cluster.z << "\",\"accum\":\"FP16\""
          << ",\"registers_per_thread\":" << state.registers_per_thread
          << ",\"static_smem_bytes\":" << state.static_smem_bytes
          << ",\"theoretical_ctas_per_sm\":" << state.theoretical_ctas_per_sm
          << ",\"actual_ctas_per_sm\":\"N/A (requires profiling)\""
          << ",\"epilogue_coalesced\":" << (gemm_backend::kCoalescedEpilogue ? "true" : "false")
          << "}";
  return std::make_unique<CustomPlan>(std::move(state), implementation, kernel_name,
                                      details.str());
}

}  // namespace custom_detail
}  // namespace gemm
