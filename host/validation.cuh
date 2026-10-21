#pragma once
#include "common.cuh"

#include <cmath>

// 校验独立于计时：回读全部 C，并在 CPU 上抽样计算参考值。
// 当前约定固定为 FP16 输入/输出、alpha=1、beta=0。
namespace gemm_bench {

// 全量消费输出，同时拒绝 NaN/Inf。checksum 只是结果摘要：
// 不保证计算正确，且 checksum == 0 完全可能是合法结果。
inline double copy_output_and_checksum(thrust::host_vector<Element>& output,
                                       Element const* device_output)
{
  CUTE_CHECK_ERROR(cudaMemcpy(output.data(), device_output, output.size() * sizeof(Element),
                             cudaMemcpyDeviceToHost));
  double checksum = 0.0;
  for (std::size_t i = 0; i < output.size(); ++i) {
    double value = double(float(output[i]));
    if (!std::isfinite(value)) {
      std::cerr << "Non-finite GEMM output at " << i << '\n';
      std::exit(EXIT_FAILURE);
    }
    checksum += value;
  }
  return checksum;
}

// 只计算一个 C(row,col)，用 FP64 累加减少参考结果自身的舍入误差。
// a_index/b_index 处理 NT/TN 的物理布局，点积公式本身保持一致。
inline double reference_gemm_entry(GemmProblem const& problem,
                                   thrust::host_vector<Element> const& A,
                                   thrust::host_vector<Element> const& B,
                                   int row, int col)
{
  double reference = 0.0;
  for (int p = 0; p < problem.k; ++p) {
    reference += double(float(A[problem.a_index(row, p)])) *
                 double(float(B[problem.b_index(col, p)]));
  }
  return reference;
}

// 在行、列方向各取 4 个均匀位置（含首尾），检查布局和明显计算错误。
// 允许误差为 1 + 2%*abs(reference)，是 FP16 的经验阈值；
// 抽样通过不等同于全矩阵验证，也不提供严格数值误差保证。
inline void validate_gemm_output(char const* phase, GemmProblem const& problem,
                                 thrust::host_vector<Element> const& A,
                                 thrust::host_vector<Element> const& B,
                                 thrust::host_vector<Element>& C, Element const* device_C)
{
  double checksum = copy_output_and_checksum(C, device_C);
  double max_abs_error = 0.0;
  constexpr int axis_samples = benchmark_policy::kValidationAxisSamples;
  for (int r = 0; r < axis_samples; ++r) {
    int row = int(static_cast<long long>(problem.m - 1) * r / (axis_samples - 1));
    for (int c = 0; c < axis_samples; ++c) {
      int col = int(static_cast<long long>(problem.n - 1) * c / (axis_samples - 1));
      double reference = reference_gemm_entry(problem, A, B, row, col);
      double actual = double(float(C[problem.c_index(row, col)]));
      double error = std::abs(actual - reference);
      max_abs_error = std::max(max_abs_error, error);
      double tolerance = benchmark_policy::kValidationAbsTolerance +
                         benchmark_policy::kValidationRelativeTolerance * std::abs(reference);
      if (error > tolerance) {
        std::cerr << "Sampled GEMM validation failed at (" << row << ',' << col
                  << "): actual=" << actual << ", reference=" << reference << '\n';
        std::exit(EXIT_FAILURE);
      }
    }
  }
  std::cout << "Validation [" << phase << "]: " << axis_samples * axis_samples
            << " sampled entries PASS, max_abs_error=" << max_abs_error
            << ", checksum=" << checksum << std::endl;
}

}  // namespace gemm_bench
