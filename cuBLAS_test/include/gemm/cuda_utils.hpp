#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
namespace gemm {
struct Skip : std::runtime_error { using std::runtime_error::runtime_error; };
struct CudaError : std::runtime_error { using std::runtime_error::runtime_error; };
inline void check_cuda(cudaError_t s, const char* expr, const char* file, int line) {
  if(s==cudaSuccess) return;
  std::string msg=std::string(file)+":"+std::to_string(line)+" "+expr+": "+cudaGetErrorString(s);
  if(s==cudaErrorMemoryAllocation) { (void)cudaGetLastError(); throw Skip(msg); }
  throw CudaError(msg);
}
inline void check_blas(cublasStatus_t s, const char* expr, const char* file, int line) {
  if(s==CUBLAS_STATUS_SUCCESS) return;
  std::string msg=std::string(file)+":"+std::to_string(line)+" "+expr+": cuBLAS status "+std::to_string(s);
  if(s==CUBLAS_STATUS_NOT_SUPPORTED || s==CUBLAS_STATUS_ARCH_MISMATCH || s==CUBLAS_STATUS_ALLOC_FAILED) throw Skip(msg);
  throw std::runtime_error(msg);
}
#define CUDA_CHECK(expr) ::gemm::check_cuda((expr), #expr, __FILE__, __LINE__)
#define BLAS_CHECK(expr) ::gemm::check_blas((expr), #expr, __FILE__, __LINE__)
struct Stream {
  cudaStream_t value=nullptr;
  Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking)); }
  ~Stream() { if(value) cudaStreamDestroy(value); }
  Stream(const Stream&)=delete; Stream& operator=(const Stream&)=delete;
};
struct Event {
  cudaEvent_t value=nullptr;
  Event() { CUDA_CHECK(cudaEventCreate(&value)); }
  ~Event() { if(value) cudaEventDestroy(value); }
  Event(const Event&)=delete; Event& operator=(const Event&)=delete;
};
struct Buffer {
  void* value=nullptr;
  explicit Buffer(std::size_t bytes) { if(bytes) CUDA_CHECK(cudaMalloc(&value,bytes)); }
  ~Buffer() { if(value) cudaFree(value); }
  Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
};
struct BlasHandle {
  cublasHandle_t value=nullptr;
  BlasHandle() { BLAS_CHECK(cublasCreate(&value)); }
  ~BlasHandle() { if(value) cublasDestroy(value); }
  BlasHandle(const BlasHandle&)=delete; BlasHandle& operator=(const BlasHandle&)=delete;
};
struct Graph {
  cudaGraph_t graph=nullptr; cudaGraphExec_t executable=nullptr;
  ~Graph() { if(executable) cudaGraphExecDestroy(executable); if(graph) cudaGraphDestroy(graph); }
};
}

