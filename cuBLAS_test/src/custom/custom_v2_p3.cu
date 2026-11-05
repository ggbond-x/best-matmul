// CUSTOM_V2 PIPE=3：仓库 kernels/wgmma_tma_fp16_pipeline_v2.cuh。
#include "wgmma_tma_fp16_pipeline_v2.cuh"

#include "custom_adapter.cuh"

namespace gemm {

std::unique_ptr<GemmPlan> make_custom_v2(Options const& options, DeviceBuffers const& buffers,
                                         cudaStream_t stream) {
  return custom_detail::make_plan(options, buffers, stream, "CUSTOM_V2_PIPE3_F16ACC",
                                  "wgmma_tma_fp16_pipeline_v2::gemm_device",
                                  "kernels/wgmma_tma_fp16_pipeline_v2.cuh (PIPE=3)");
}

}  // namespace gemm
