#include "gemm/backend.hpp"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
namespace gemm {
template<class T> void upload_t(void* dst,std::vector<float>& values,cudaStream_t s) {
  std::vector<T> packed(values.size());
  for(std::size_t i=0;i<values.size();++i) { packed[i]=T(values[i]); values[i]=static_cast<float>(packed[i]); }
  CUDA_CHECK(cudaMemcpyAsync(dst,packed.data(),packed.size()*sizeof(T),cudaMemcpyHostToDevice,s));
  CUDA_CHECK(cudaStreamSynchronize(s)); // Keep staging memory alive until copy completes.
}
void upload(void* dst,std::vector<float>& v,cudaDataType_t t,cudaStream_t s) {
  if(t==CUDA_R_16F) upload_t<__half>(dst,v,s);
  else if(t==CUDA_R_16BF) upload_t<__nv_bfloat16>(dst,v,s);
  else { CUDA_CHECK(cudaMemcpyAsync(dst,v.data(),v.size()*sizeof(float),cudaMemcpyHostToDevice,s)); CUDA_CHECK(cudaStreamSynchronize(s)); }
}
template<class T> std::vector<float> download_t(const void* src,std::size_t count,cudaStream_t s) {
  std::vector<T> packed(count);
  CUDA_CHECK(cudaMemcpyAsync(packed.data(),src,count*sizeof(T),cudaMemcpyDeviceToHost,s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  std::vector<float> result(count);
  for(std::size_t i=0;i<count;++i) result[i]=static_cast<float>(packed[i]);
  return result;
}
std::vector<float> download(const void* src,std::size_t n,cudaDataType_t t,cudaStream_t s) {
  if(t==CUDA_R_16F) return download_t<__half>(src,n,s);
  if(t==CUDA_R_16BF) return download_t<__nv_bfloat16>(src,n,s);
  return download_t<float>(src,n,s);
}

} // namespace gemm
