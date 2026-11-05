// CUSTOM_V1：仓库 kernels/wgmma_tma_fp16_v1.cuh 的原始实现。
#include "wgmma_tma_fp16_v1.cuh"

#include "custom_adapter.cuh"

namespace gemm {

std::unique_ptr<GemmPlan> make_custom_v1(Options const& options, DeviceBuffers const& buffers,
                                         cudaStream_t stream) {
  return custom_detail::make_plan(options, buffers, stream, "CUSTOM_V1_F16ACC",
                                  "wgmma_tma_fp16_v1::gemm_device",
                                  "kernels/wgmma_tma_fp16_v1.cuh");
}

}  // namespace gemm
