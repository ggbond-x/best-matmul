// CUTLASS SM90 (Hopper) FP16 Tensor Core GEMM 后端。
//
// 只实例化有限候选，避免编译整个 CUTLASS kernel 库：
//   tile  : 128x128x64 (cluster 1x1x1) 与 128x256x64 (cluster 2x1x1)
//   accum : FP16 与 FP32
//   布局  : NT 与 TN
// 每个组合都记录 CTA tile、cluster、stage 数、调度器与 CUTLASS 版本。
//
// 布局映射（与 kernels/host/common.cuh 相同）：
//   NT -> A ColumnMajor(M 连续)，B ColumnMajor(N 连续)；
//   TN -> A RowMajor(K 连续)，B RowMajor(K 连续)；
//   C/D 始终 ColumnMajor。
#include "gemm/backend.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cutlass/version.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace gemm {
namespace {

using Element = cutlass::half_t;
using LayoutC = cutlass::layout::ColumnMajor;
constexpr int kAlignment = 128 / cutlass::sizeof_bits<Element>::value;

template <class LayoutA, class LayoutB, class ElementAccumulator, class TileShape,
          class ClusterShape>
class CutlassSm90Plan final : public GemmPlan {
 public:
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, Element, LayoutC, kAlignment, Element, LayoutC, kAlignment,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, Element, LayoutA, kAlignment, Element, LayoutB, kAlignment,
          ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<cutlass::Shape<int, int, int>, CollectiveMainloop,
                                           CollectiveEpilogue>;
  // details 中记录的调度器必须与 CUTLASS 默认解析结果一致。
  static_assert(std::is_same_v<typename GemmKernel::TileScheduler,
                               cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90>,
                "CUTLASS default scheduler changed; update the scheduler string in details");
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  static constexpr int kTileM = decltype(cute::size<0>(TileShape{}))::value;
  static constexpr int kTileN = decltype(cute::size<1>(TileShape{}))::value;
  static constexpr int kTileK = decltype(cute::size<2>(TileShape{}))::value;
  static constexpr int kClusterX = decltype(cute::size<0>(ClusterShape{}))::value;
  static constexpr int kClusterY = decltype(cute::size<1>(ClusterShape{}))::value;
  static constexpr int kClusterZ = decltype(cute::size<2>(ClusterShape{}))::value;
  // StageCountAutoCarveout 会在 builder 内解析成静态 stage 数。
  static constexpr int kStages =
      decltype(cute::size<2>(typename CollectiveMainloop::SmemLayoutA{}))::value;

  CutlassSm90Plan(Options const& options, DeviceBuffers const& buffers, cudaStream_t stream,
                  Accum accum, std::string const& candidate)
      : options_(options),
        buffers_(buffers),
        stream_(stream),
        accum_(accum),
        candidate_(candidate) {
    int major = 0;
    int minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, options.device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, options.device));
    if (major != 9) {
      throw Skip("CUTLASS SM90 kernel requires compute capability 9.0 (Hopper); device is " +
                 std::to_string(major) + "." + std::to_string(minor));
    }

    Problem const& p = options.problem;
    if (p.m % kTileM != 0 || p.n % kTileN != 0 || p.k % kTileK != 0) {
      throw Skip("CUTLASS candidate requires M%tile_M==0, N%tile_N==0, K%tile_K==0");
    }

    auto const* a = reinterpret_cast<Element const*>(buffers.a);
    auto const* b = reinterpret_cast<Element const*>(buffers.b);
    auto* c = reinterpret_cast<Element*>(buffers.c);

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {p.m, p.k, 1});
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {p.n, p.k, 1});
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {p.m, p.n, 1});
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {p.m, p.n, 1});

    using ElementCompute = typename Gemm::EpilogueOutputOp::ElementCompute;
    ElementCompute const alpha = ElementCompute(options.alpha);
    ElementCompute const beta = ElementCompute(options.beta);

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {p.m, p.n, p.k},
        {a, stride_a, b, stride_b},
        {{alpha, beta}, c, stride_c, c, stride_d},
    };

    cutlass::Status const implementable = Gemm::can_implement(arguments);
    if (implementable != cutlass::Status::kSuccess) {
      throw Skip("CUTLASS can_implement failed for this shape/layout (status " +
                 std::to_string(int(implementable)) + ")");
    }
    workspace_size_ = Gemm::get_workspace_size(arguments);
    if (workspace_size_ > options.lt_workspace) {
      throw Skip("CUTLASS workspace exceeds the common workspace budget: " +
                 std::to_string(workspace_size_) + " > " + std::to_string(options.lt_workspace));
    }
    workspace_ = std::make_unique<Buffer>(workspace_size_);
    cutlass::Status const initialized =
        gemm_.initialize(arguments, workspace_->value, stream_);
    if (initialized != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS initialize failed (status " +
                               std::to_string(int(initialized)) + ")");
    }

    info.implementation = accum_ == Accum::FP16 ? implementation_name("F16ACC")
                                                 : implementation_name("F32ACC");
    info.accum = accum_;
    info.kernel_name = "cutlass_sm90_" + candidate_ + "_" + layout_name() + "_" +
                       (accum_ == Accum::FP16 ? "F16ACC" : "F32ACC");
    info.algorithm_id = "cutlass_sm90_tma_wgmma";
    info.workspace_bytes = workspace_size_;
    info.heuristic_candidates = 1;
    std::ostringstream details;
    details << "{\"library\":\"CUTLASS\",\"version\":\"" << CUTLASS_MAJOR << "."
            << CUTLASS_MINOR << "." << CUTLASS_PATCH << "\",\"candidate\":\"" << candidate_
            << "\",\"tile\":\"" << kTileM << "x" << kTileN << "x" << kTileK << "\",\"cluster\":\""
            << kClusterX << "x" << kClusterY << "x" << kClusterZ << "\",\"stages\":" << kStages
            << ",\"scheduler\":\"PersistentTileSchedulerSm90\",\"layout_a\":\"" << layout_a_name()
            << "\",\"layout_b\":\"" << layout_b_name() << "\",\"epilogue\":\"EpilogueScheduleAuto\""
            << "}";
    info.details = details.str();
  }

  void run() override {
    cutlass::Status const status = gemm_.run(stream_);
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS run failed (status " + std::to_string(int(status)) +
                               ")");
    }
  }

 private:
  static std::string layout_name() {
    return std::is_same_v<LayoutA, cutlass::layout::RowMajor> ? "TN" : "NT";
  }
  static char const* layout_a_name() {
    return std::is_same_v<LayoutA, cutlass::layout::RowMajor> ? "RowMajor(K-contiguous)"
                                                              : "ColumnMajor(M-contiguous)";
  }
  static char const* layout_b_name() {
    return std::is_same_v<LayoutB, cutlass::layout::RowMajor> ? "RowMajor(K-contiguous)"
                                                              : "ColumnMajor(N-contiguous)";
  }
  std::string implementation_name(char const* suffix) const {
    return "CUTLASS_SM90_" + candidate_ + "_" + suffix;
  }

  Options options_;
  DeviceBuffers buffers_;
  cudaStream_t stream_ = nullptr;
  Accum accum_ = Accum::FP32;
  std::string candidate_;
  std::size_t workspace_size_ = 0;
  std::unique_ptr<Buffer> workspace_;
  Gemm gemm_;
};

// NT: A/B 都按 M/N 连续；TN: A/B 都按 K 连续。
template <class ElementAccumulator, class TileShape, class ClusterShape>
std::unique_ptr<GemmPlan> make_nt(Options const& options, DeviceBuffers const& buffers,
                                  cudaStream_t stream, Accum accum, std::string const& candidate) {
  return std::make_unique<CutlassSm90Plan<cutlass::layout::ColumnMajor,
                                          cutlass::layout::ColumnMajor, ElementAccumulator,
                                          TileShape, ClusterShape>>(
      options, buffers, stream, accum, candidate);
}

template <class ElementAccumulator, class TileShape, class ClusterShape>
std::unique_ptr<GemmPlan> make_tn(Options const& options, DeviceBuffers const& buffers,
                                  cudaStream_t stream, Accum accum, std::string const& candidate) {
  return std::make_unique<
      CutlassSm90Plan<cutlass::layout::RowMajor, cutlass::layout::RowMajor, ElementAccumulator,
                      TileShape, ClusterShape>>(options, buffers, stream, accum, candidate);
}

template <class ElementAccumulator, class TileShape, class ClusterShape>
std::unique_ptr<GemmPlan> make_layout(Options const& options, DeviceBuffers const& buffers,
                                      cudaStream_t stream, Accum accum,
                                      std::string const& candidate) {
  if (options.problem.is_nt()) {
    return make_nt<ElementAccumulator, TileShape, ClusterShape>(options, buffers, stream, accum,
                                                                candidate);
  }
  return make_tn<ElementAccumulator, TileShape, ClusterShape>(options, buffers, stream, accum,
                                                              candidate);
}

}  // namespace

std::unique_ptr<GemmPlan> make_cutlass_candidate(Options const& options,
                                                 DeviceBuffers const& buffers,
                                                 cudaStream_t stream, Accum accum,
                                                 std::string const& candidate) {
  if (candidate == "128x128x64_C1x1x1") {
    if (accum == Accum::FP16) {
      return make_layout<cutlass::half_t, cute::Shape<cute::_128, cute::_128, cute::_64>,
                         cute::Shape<cute::_1, cute::_1, cute::_1>>(options, buffers, stream,
                                                                   accum, candidate);
    }
    return make_layout<float, cute::Shape<cute::_128, cute::_128, cute::_64>,
                       cute::Shape<cute::_1, cute::_1, cute::_1>>(options, buffers, stream,
                                                                 accum, candidate);
  }
  if (candidate == "128x256x64_C2x1x1") {
    if (accum == Accum::FP16) {
      return make_layout<cutlass::half_t, cute::Shape<cute::_128, cute::_256, cute::_64>,
                         cute::Shape<cute::_2, cute::_1, cute::_1>>(options, buffers, stream,
                                                                   accum, candidate);
    }
    return make_layout<float, cute::Shape<cute::_128, cute::_256, cute::_64>,
                       cute::Shape<cute::_2, cute::_1, cute::_1>>(options, buffers, stream,
                                                                 accum, candidate);
  }
  throw std::invalid_argument("Unknown CUTLASS candidate: " + candidate);
}

}  // namespace gemm
