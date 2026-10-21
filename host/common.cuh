#pragma once

// 公共数据契约：FP16 输入/输出、C = A * B（alpha=1、beta=0）。
// 此处只描述问题和数据，不绑定某个 kernel、tile 或 pipeline。
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cutlass/half.h>
#include <cute/util/debug.hpp>

namespace gemm_bench {

using Element = cutlass::half_t;

// 这些整数都能被 FP16 精确表示，便于复现和抽样校验。
enum class InputMode {
  Zero = 0,       // 全 0：检查清零与写回。
  One = 1,        // 全 1：每个 C 元素的数学结果为 K。
  RandomPm1 = 2,  // 随机 -1/+1：默认输入。
  Pattern = 3,    // 周期性 -2..2：便于检查布局。
};

enum class TimerMode {
  Wall,   // CPU 单调时钟：提交 kernel 到该流完成的总时间。
  Event,  // CUDA Event：GPU 时间线上的起止间隔，可能包含提交间隙。
};

inline char const* timer_mode_name(TimerMode mode)
{
  switch (mode) {
    case TimerMode::Wall: return "wall";
    case TimerMode::Event: return "event";
  }
  return "unknown";
}

// 统一使用逻辑形状 A(M,K)、B(N,K)、C(M,N)：
//   C(m,n) = sum_k A(m,k) * B(n,k)
// B 的逻辑形状与 BLAS 常见的 B(K,N) 不同，NT/TN 应结合下方 stride 理解。
// NT：A/B 的 M/N 维连续；TN：A/B 的 K 维连续；C 始终是列主序。
struct GemmProblem {
  int m = 4096;
  int n = 4096;
  int k = 4096;
  char transA = 'N';
  char transB = 'T';

  bool is_nt() const { return transA == 'N' && transB == 'T'; }
  int ldA() const { return transA == 'N' ? m : k; }
  int ldB() const { return transB == 'T' ? n : k; }
  int ldC() const { return m; }

  // 乘法前转成 size_t，避免先发生 int 溢出。
  std::size_t a_elements() const { return std::size_t(m) * k; }
  std::size_t b_elements() const { return std::size_t(n) * k; }
  std::size_t c_elements() const { return std::size_t(m) * n; }

  // CPU 参考计算和 GPU Tensor 必须遵循同一套物理索引。
  std::size_t a_index(int row, int p) const {
    return transA == 'N' ? std::size_t(row) + std::size_t(p) * ldA()
                         : std::size_t(row) * ldA() + p;
  }
  std::size_t b_index(int col, int p) const {
    return transB == 'T' ? std::size_t(col) + std::size_t(p) * ldB()
                         : std::size_t(col) * ldB() + p;
  }
  std::size_t c_index(int row, int col) const {
    return std::size_t(row) + std::size_t(col) * ldC();
  }
};

struct MeasurementConfig {
  TimerMode timer_mode = TimerMode::Event;
  int launches_per_batch = 20;
  int min_warmup_launches = 20;   // 最少预热次数，实际还会等待耗时进入平台期。
  int batches = 200;             // 正式采样批数，不包含预热和校验。
  cudaStream_t stream = nullptr; // kernel、Event 和同步都使用这一条流。
};

struct BenchmarkOptions {
  GemmProblem problem;
  InputMode input_mode = InputMode::RandomPm1;
  MeasurementConfig measurement;
};

struct LaunchConfig {
  int smem_bytes = 0;
  dim3 grid{1, 1, 1};
  dim3 block{1, 1, 1};
  dim3 cluster{1, 1, 1};
};

// 经验阈值集中命名，便于调整；它们不是硬件保证或数值误差理论上界。
namespace benchmark_policy {
constexpr std::size_t kWarmupWindowBatches = 5;
constexpr int kStableWindowCount = 2;
constexpr int kExtraWarmupBatches = 20;
constexpr double kPlateauRelativeTolerance = 0.02;
constexpr double kDriftWarningPct = 5.0;
constexpr int kValidationAxisSamples = 4;
constexpr double kValidationAbsTolerance = 1.0;
constexpr double kValidationRelativeTolerance = 0.02;
}  // namespace benchmark_policy

inline Element make_input_value(std::size_t index, int matrix_id, InputMode mode)
{
  switch (mode) {
    case InputMode::Zero: return Element(0);
    case InputMode::One: return Element(1);
    case InputMode::RandomPm1: return Element(std::rand() % 2 ? 1 : -1);
    case InputMode::Pattern:
      // 先把取模结果转成有符号数，再减 2，避免 size_t 下溢。
      return Element(int((index + std::size_t(matrix_id) * 17) % 5) - 2);
  }
  return Element(0);
}

// vector 自动管理内存。分配、初始化和 Host -> Device 拷贝全部发生在测量前。
struct GemmBuffers {
  thrust::host_vector<Element> h_A, h_B, h_C;
  thrust::device_vector<Element> d_A, d_B, d_C;

  explicit GemmBuffers(BenchmarkOptions const& options)
      : h_A(options.problem.a_elements()),
        h_B(options.problem.b_elements()),
        h_C(options.problem.c_elements(), Element(0)) {
    // A、B 共用固定随机序列；保持同一构建下相同参数的输入可复现。
    std::srand(0);
    for (std::size_t i = 0; i < h_A.size(); ++i) {
      h_A[i] = make_input_value(i, 0, options.input_mode);
    }
    for (std::size_t i = 0; i < h_B.size(); ++i) {
      h_B[i] = make_input_value(i, 1, options.input_mode);
    }
    d_A = h_A;
    d_B = h_B;
    d_C = h_C;
  }
};

}  // namespace gemm_bench
