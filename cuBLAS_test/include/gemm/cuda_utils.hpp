#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace gemm {

// Skip 表示环境/形状不支持，而不是实现错误；CSV 中记为 UNSUPPORTED。
struct Skip : std::runtime_error {
  using std::runtime_error::runtime_error;
};
// CudaError 表示真实的 CUDA 执行错误；CSV 中记为 FAIL。
struct CudaError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline void check_cuda(cudaError_t status, char const* expr, char const* file, int line) {
  if (status == cudaSuccess) return;
  std::string message = std::string(file) + ":" + std::to_string(line) + " " + expr + ": " +
                        cudaGetErrorString(status);
  if (status == cudaErrorMemoryAllocation) {
    (void)cudaGetLastError();
    throw Skip(message);
  }
  throw CudaError(message);
}

inline void check_blas(cublasStatus_t status, char const* expr, char const* file, int line) {
  if (status == CUBLAS_STATUS_SUCCESS) return;
  std::string message = std::string(file) + ":" + std::to_string(line) + " " + expr +
                        ": cuBLAS status " + std::to_string(int(status));
  if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH ||
      status == CUBLAS_STATUS_ALLOC_FAILED) {
    throw Skip(message);
  }
  throw std::runtime_error(message);
}

#define CUDA_CHECK(expr) ::gemm::check_cuda((expr), #expr, __FILE__, __LINE__)
#define BLAS_CHECK(expr) ::gemm::check_blas((expr), #expr, __FILE__, __LINE__)

struct Stream {
  cudaStream_t value = nullptr;
  Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() {
    if (value) cudaStreamDestroy(value);
  }
  Stream(Stream const&) = delete;
  Stream& operator=(Stream const&) = delete;
};

struct Event {
  cudaEvent_t value = nullptr;
  Event() { CUDA_CHECK(cudaEventCreate(&value)); }
  ~Event() {
    if (value) cudaEventDestroy(value);
  }
  Event(Event const&) = delete;
  Event& operator=(Event const&) = delete;
};

struct Buffer {
  void* value = nullptr;
  std::size_t bytes = 0;
  explicit Buffer(std::size_t size) : bytes(size) {
    if (size) CUDA_CHECK(cudaMalloc(&value, size));
  }
  ~Buffer() {
    if (value) cudaFree(value);
  }
  Buffer(Buffer const&) = delete;
  Buffer& operator=(Buffer const&) = delete;
};

struct BlasHandle {
  cublasHandle_t value = nullptr;
  BlasHandle() { BLAS_CHECK(cublasCreate(&value)); }
  ~BlasHandle() {
    if (value) cublasDestroy(value);
  }
  BlasHandle(BlasHandle const&) = delete;
  BlasHandle& operator=(BlasHandle const&) = delete;
};

struct Graph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  ~Graph() {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
  }
  Graph() = default;
  Graph(Graph const&) = delete;
  Graph& operator=(Graph const&) = delete;
};

inline void stream_sync(cudaStream_t stream) { CUDA_CHECK(cudaStreamSynchronize(stream)); }

}  // namespace gemm
