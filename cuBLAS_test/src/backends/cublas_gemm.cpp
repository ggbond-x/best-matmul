#include "gemm/backend.hpp"

#include <cuda_fp16.h>

#include <sstream>

namespace gemm {
namespace {

class CublasExPlan final : public GemmPlan {
 public:
  CublasExPlan(Options const& options, DeviceBuffers const& buffers, cudaStream_t stream,
               Accum accum)
      : options_(options), buffers_(buffers), accum_(accum), workspace_(options.lt_workspace) {
    BLAS_CHECK(cublasSetStream(handle_.value, stream));
    // cublasSetStream resets the workspace, so set the common budget afterwards.
    BLAS_CHECK(cublasSetWorkspace(handle_.value, workspace_.value, workspace_.bytes));
    BLAS_CHECK(cublasSetPointerMode(handle_.value, CUBLAS_POINTER_MODE_HOST));
    if (accum_ == Accum::FP32) {
      BLAS_CHECK(cublasSetMathMode(
          handle_.value,
          static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH |
                                    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    } else {
      BLAS_CHECK(cublasSetMathMode(handle_.value, CUBLAS_DEFAULT_MATH));
    }
    alpha_h_ = __float2half(options.alpha);
    beta_h_ = __float2half(options.beta);
    alpha_f_ = options.alpha;
    beta_f_ = options.beta;

    info.implementation = accum_ == Accum::FP16 ? "CUBLAS_EX_F16ACC" : "CUBLAS_EX_F32ACC";
    info.accum = accum_;
    info.kernel_name = "cublasGemmEx";
    info.algorithm_id = "CUBLAS_GEMM_DEFAULT";
    info.workspace_bytes = workspace_.bytes;  // supplied budget; actual use is not exposed
    std::ostringstream details;
    details << "{\"api\":\"cublasGemmEx\",\"compute_type\":\""
            << (accum_ == Accum::FP16 ? "CUBLAS_COMPUTE_16F" : "CUBLAS_COMPUTE_32F")
            << "\",\"algorithm\":\"CUBLAS_GEMM_DEFAULT\""
            << ",\"reduced_precision_reduction\":\""
            << (accum_ == Accum::FP16 ? "library_default" : "disallowed")
            << "\",\"workspace_supplied_bytes\":" << workspace_.bytes
            << ",\"workspace_actual_bytes\":\"N/A\"}";
    info.details = details.str();
  }

  void run() override {
    cublasOperation_t const op_a = options_.problem.ta ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t const op_b = options_.problem.tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    Problem const& p = options_.problem;
    if (accum_ == Accum::FP16) {
      BLAS_CHECK(cublasGemmEx(handle_.value, op_a, op_b, p.m, p.n, p.k, &alpha_h_, buffers_.a,
                              CUDA_R_16F, p.ldA(), buffers_.b, CUDA_R_16F, p.ldB(), &beta_h_,
                              buffers_.c, CUDA_R_16F, p.ldC(), CUBLAS_COMPUTE_16F,
                              CUBLAS_GEMM_DEFAULT));
    } else {
      BLAS_CHECK(cublasGemmEx(handle_.value, op_a, op_b, p.m, p.n, p.k, &alpha_f_, buffers_.a,
                              CUDA_R_16F, p.ldA(), buffers_.b, CUDA_R_16F, p.ldB(), &beta_f_,
                              buffers_.c, CUDA_R_16F, p.ldC(), CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
    }
  }

 private:
  BlasHandle handle_;
  Options options_;
  DeviceBuffers buffers_;
  Accum accum_ = Accum::FP32;
  Buffer workspace_;
  __half alpha_h_{};
  __half beta_h_{};
  float alpha_f_ = 1.0f;
  float beta_f_ = 0.0f;
};

}  // namespace

std::unique_ptr<GemmPlan> make_cublas_ex(Options const& options, DeviceBuffers const& buffers,
                                         cudaStream_t stream, Accum accum) {
  return std::make_unique<CublasExPlan>(options, buffers, stream, accum);
}

}  // namespace gemm
