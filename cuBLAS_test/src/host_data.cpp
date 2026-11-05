// 输入生成、FP64 参考与正确性比较。
//
// 关键约束：输入值都是 FP16 可精确表示的十进制小数（i/32），所以
//   - 所有后端看到的是逐元素完全相同的输入；
//   - FP64 参考结果不包含输入舍入误差；
//   - 不需要依赖任何框架 RNG。
#include "gemm/bench.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gemm {
namespace {

constexpr std::uint64_t kSaltBase = 0xD1B54A32D192ED03ull;
constexpr std::uint64_t kSaltStep = 0x9E3779B97F4A7C15ull;
constexpr std::size_t kReferenceSamples = 4096;

std::uint64_t mix(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// 63 个值：-31/32 ... +31/32，全部在 FP16 中精确表示。
float grid_value(std::size_t index, std::uint64_t salt) {
  std::uint64_t const h = mix(std::uint64_t(index) ^ salt) >> 40;
  return float(int(h % 63u) - 31) / 32.0f;
}

std::size_t sample_index(Problem const& p, std::size_t n, std::uint64_t seed) {
  // 先放固定边界样本，其余由哈希决定，保证可复现且覆盖角落/中间。
  if (n < 5) {
    switch (n) {
      case 0: return 0;
      case 1: return p.c_index(0, 0);
      case 2: return p.c_index(p.m - 1, p.n - 1);
      case 3: return p.c_index(0, p.n - 1);
      default: return p.c_index(p.m - 1, 0);
    }
  }
  switch (n) {
    case 0: return p.c_index(0, 0);
    case 1: return p.c_index(0, p.n - 1);
    case 2: return p.c_index(p.m - 1, 0);
    case 3: return p.c_index(p.m - 1, p.n - 1);
    case 4: return p.c_index(p.m / 2, p.n / 2);
    default: return std::size_t(mix(seed + n) % p.c_elements());
  }
}

bool enough_memory(std::size_t bytes) {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (status != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  // 给 cuBLAS 内部工作区和上下文留出余量。
  std::size_t const reserve = std::min<std::size_t>(512ull << 20, total_bytes / 8);
  return free_bytes > bytes && free_bytes - bytes > reserve;
}

void upload_doubles(Buffer& destination, HostOperands const& host, bool is_a,
                    std::size_t count, cudaStream_t stream) {
  std::vector<double> chunk(std::min<std::size_t>(count, 1ull << 20));
  for (std::size_t offset = 0; offset < count; offset += chunk.size()) {
    std::size_t const n = std::min(chunk.size(), count - offset);
    for (std::size_t i = 0; i < n; ++i) {
      chunk[i] = is_a ? double(host.value_at_a(offset + i)) : double(host.value_at_b(offset + i));
    }
    CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(destination.value) + offset * sizeof(double),
                               chunk.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
}

Reference sampled_reference(Problem const& problem, HostOperands const& host,
                            std::string const& reason) {
  Reference result;
  result.mode = "sampled_cpu_fp64";
  result.reason = reason;
  result.full = false;
  result.indices.reserve(kReferenceSamples);
  result.sampled.reserve(kReferenceSamples);

  std::uint64_t const seed = 0xC0FFEE123456789ull;
  for (std::size_t n = 0; n < kReferenceSamples; ++n) {
    std::size_t index = sample_index(problem, n, seed);
    result.indices.push_back(index);
  }
  // 去重，保持顺序稳定。
  std::sort(result.indices.begin(), result.indices.end());
  result.indices.erase(std::unique(result.indices.begin(), result.indices.end()),
                       result.indices.end());
  for (std::size_t index : result.indices) {
    int const row = int(index % std::size_t(problem.m));
    int const col = int(index / std::size_t(problem.m));
    double sum = 0.0;
    for (int p = 0; p < problem.k; ++p) {
      sum += double(host.value_at_a(problem.a_index(row, p))) *
             double(host.value_at_b(problem.b_index(col, p)));
    }
    result.sampled.push_back(sum);
  }
  return result;
}

}  // namespace

HostOperands make_host_operands(Problem const& problem, std::uint64_t seed) {
  HostOperands host;
  host.a.resize(problem.a_elements());
  host.b.resize(problem.b_elements());

  // 两个盐相差一个完整 64 位常数，避免行索引被低比特置换。
  std::uint64_t const salt_a = kSaltBase ^ (seed * kSaltStep + 0x123456789ABCDEFull);
  std::uint64_t const salt_b = salt_a ^ kSaltStep;

  for (std::size_t i = 0; i < host.a.size(); ++i) host.a[i] = __float2half(grid_value(i, salt_a));
  for (std::size_t i = 0; i < host.b.size(); ++i) host.b[i] = __float2half(grid_value(i, salt_b));
  return host;
}

HostOperands pack_operands(Problem const& problem, HostOperands const& logical) {
  HostOperands packed;
  packed.a.resize(problem.a_elements());
  packed.b.resize(problem.b_elements());
  // 规范 A：a[row + p*M]；规范 B：b[col + p*N]。
  for (std::size_t i = 0; i < packed.a.size(); ++i) {
    int row = 0;
    int p = 0;
    if (problem.ta) {
      row = int(i / std::size_t(problem.k));
      p = int(i % std::size_t(problem.k));
    } else {
      row = int(i % std::size_t(problem.m));
      p = int(i / std::size_t(problem.m));
    }
    packed.a[i] = logical.a[std::size_t(row) + std::size_t(p) * std::size_t(problem.m)];
  }
  // 规范 B：b[col + p*N]。
  // tb=true（NT）：物理布局为 N×K 列主序，b_phys[col + p*N]；
  // tb=false（TN）：物理布局为 N×K 行主序（K 连续），b_phys[col*K + p]。
  for (std::size_t i = 0; i < packed.b.size(); ++i) {
    int col = 0;
    int p = 0;
    if (problem.tb) {
      col = int(i % std::size_t(problem.n));
      p = int(i / std::size_t(problem.n));
    } else {
      col = int(i / std::size_t(problem.k));
      p = int(i % std::size_t(problem.k));
    }
    packed.b[i] = logical.b[std::size_t(col) + std::size_t(p) * std::size_t(problem.n)];
  }
  return packed;
}

void upload_operands(DeviceBuffers const& buffers, HostOperands const& host, cudaStream_t stream) {
  if (buffers.a_bytes != host.a.size() * sizeof(__half) ||
      buffers.b_bytes != host.b.size() * sizeof(__half)) {
    throw std::runtime_error("Internal error: host/device operand size mismatch");
  }
  CUDA_CHECK(cudaMemcpyAsync(buffers.a, host.a.data(), buffers.a_bytes, cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(buffers.b, host.b.data(), buffers.b_bytes, cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void initialize_output(DeviceBuffers const& buffers, cudaStream_t stream) {
  // 用 FP16 NaN 填充；任何未写到的元素都会被正确性检查当作非有限值捕获。
  CUDA_CHECK(cudaMemsetAsync(buffers.c, 0x7f, buffers.c_bytes, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

std::vector<float> download_half_as_float(void const* source, std::size_t count,
                                          cudaStream_t stream) {
  std::vector<__half> packed(count);
  CUDA_CHECK(cudaMemcpyAsync(packed.data(), source, count * sizeof(__half),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<float> result(count);
  for (std::size_t i = 0; i < count; ++i) result[i] = __half2float(packed[i]);
  return result;
}

Reference build_reference(Problem const& problem, HostOperands const& host, int device,
                          cudaStream_t stream, bool allow_full) {
  std::size_t const na = problem.a_elements();
  std::size_t const nb = problem.b_elements();
  std::size_t const nc = problem.c_elements();
  std::size_t const required = (na + nb + nc) * sizeof(double);
  if (!allow_full) {
    return sampled_reference(problem, host, "full FP64 reference disabled by command line");
  }
  if (!enough_memory(required)) {
    return sampled_reference(problem, host,
                             "insufficient free device memory for full FP64 reference");
  }

  try {
    Buffer a64(na * sizeof(double));
    Buffer b64(nb * sizeof(double));
    Buffer c64(nc * sizeof(double));
    upload_doubles(a64, host, true, na, stream);
    upload_doubles(b64, host, false, nb, stream);

    BlasHandle handle;
    BLAS_CHECK(cublasSetStream(handle.value, stream));
    double alpha = 1.0;
    double beta = 0.0;
    cublasOperation_t const op_a = problem.ta ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t const op_b = problem.tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    BLAS_CHECK(cublasDgemm(handle.value, op_a, op_b, problem.m, problem.n, problem.k, &alpha,
                           static_cast<double const*>(a64.value), problem.ldA(),
                           static_cast<double const*>(b64.value), problem.ldB(), &beta,
                           static_cast<double*>(c64.value), problem.ldC()));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    Reference result;
    result.mode = "full_gpu_fp64";
    result.full = true;
    result.values.resize(nc);
    CUDA_CHECK(cudaMemcpyAsync(result.values.data(), c64.value, nc * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    (void)device;
    return result;
  } catch (std::exception const& error) {
    (void)cudaGetLastError();
    return sampled_reference(problem, host,
                             std::string("full FP64 reference failed: ") + error.what());
  }
}

Correctness validate_output(Problem const& problem, Reference const& reference,
                            std::vector<float> const& output, Thresholds const& thresholds) {
  Correctness result;
  result.thresholds = thresholds;
  result.mode = reference.mode;
  if (reference.mode == "unavailable") {
    result.reason = reference.reason.empty() ? "reference unavailable" : reference.reason;
    return result;
  }
  result.checked = true;
  std::size_t const total = problem.c_elements();
  if (output.size() != total) {
    result.reason = "output size mismatch";
    return result;
  }

  double error_sq = 0.0;
  double reference_sq = 0.0;
  auto check = [&](std::size_t index, double expected) {
    double const got = double(output[index]);
    ++result.checked_elements;
    if (!std::isfinite(got)) {
      ++result.nonfinite_count;
      return;
    }
    double const error = std::abs(got - expected);
    result.max_abs = std::max(result.max_abs, error);
    result.max_rel = std::max(result.max_rel, error / std::max(std::abs(expected), 1e-6));
    error_sq += error * error;
    reference_sq += expected * expected;
    double const bound = result.thresholds.atol + result.thresholds.rtol * std::abs(expected);
    if (error > bound) ++result.mismatch_count;
  };

  if (reference.full) {
    for (std::size_t i = 0; i < total; ++i) check(i, reference.values[i]);
  } else {
    // 抽样参考只比对固定索引，但仍需扫描全部输出，确保未被 kernel 写入的
    // 区域（初始化时填 NaN）不会被漏掉。
    for (std::size_t i = 0; i < total; ++i) {
      if (!std::isfinite(double(output[i]))) ++result.nonfinite_count;
    }
    for (std::size_t i = 0; i < reference.indices.size(); ++i) {
      check(reference.indices[i], reference.sampled[i]);
    }
  }

  result.rel_l2 = std::sqrt(error_sq / std::max(reference_sq, 1e-30));
  double const fraction = result.checked_elements
                              ? double(result.mismatch_count) / double(result.checked_elements)
                              : 1.0;
  result.pass = result.nonfinite_count == 0 &&
                fraction <= result.thresholds.mismatch_fraction_max &&
                result.rel_l2 <= result.thresholds.rel_l2_max;
  if (!result.pass) {
    char text[512];
    std::snprintf(text, sizeof(text),
                  "nonfinite=%zu mismatch=%zu/%zu relL2=%.6g (limits relL2<=%.6g, "
                  "mismatch_fraction<=%.6g)",
                  result.nonfinite_count, result.mismatch_count, result.checked_elements,
                  result.rel_l2, result.thresholds.rel_l2_max,
                  result.thresholds.mismatch_fraction_max);
    result.reason = text;
  }
  return result;
}

void flush_caches(Buffer& flush_buffer, cudaStream_t stream) {
  if (!flush_buffer.value || flush_buffer.bytes == 0) return;
  CUDA_CHECK(cudaMemsetAsync(flush_buffer.value, 0, flush_buffer.bytes, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace gemm
