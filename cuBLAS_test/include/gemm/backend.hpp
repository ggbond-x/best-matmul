#pragma once
#include "gemm/config.hpp"
#include "gemm/cuda_utils.hpp"
#include <memory>
#include <string>
#include <vector>
namespace gemm {
struct AlgorithmInfo {
  std::string description="not_selected";
  int heuristic_index=-1, candidates=0;
  std::size_t workspace_used=0;
};
// Constructors prepare handles, descriptors, workspace and selection.
// run() only enqueues GEMM on the supplied stream. Both backends use C==D.
class GemmPlan {
 public:
  virtual ~GemmPlan()=default;
  virtual void run()=0;
  AlgorithmInfo info;
};
std::unique_ptr<GemmPlan> make_cublas(const Config&,const void*,const void*,void*,cudaStream_t);
std::unique_ptr<GemmPlan> make_cublaslt(const Config&,const void*,const void*,void*,const void*,std::size_t,cudaStream_t);
inline cudaDataType_t input_type(const Config& c) {
  return c.dtype==Dtype::FP16 ? CUDA_R_16F : c.dtype==Dtype::BF16 ? CUDA_R_16BF : CUDA_R_32F;
}
inline cublasComputeType_t compute_type(const Config& c) {
  return c.dtype==Dtype::FP32 ? CUBLAS_COMPUTE_32F_PEDANTIC : c.dtype==Dtype::TF32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
}
inline cudaDataType_t output_type(const Config& c) { return output_name(c)=="FP32" ? CUDA_R_32F : input_type(c); }
inline std::size_t type_size(cudaDataType_t t) { return t==CUDA_R_32F ? 4 : 2; }
void upload(void* dst,std::vector<float>& data,cudaDataType_t type,cudaStream_t stream);
std::vector<float> download(const void* src,std::size_t count,cudaDataType_t type,cudaStream_t stream);
}



