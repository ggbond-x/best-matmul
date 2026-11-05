#pragma once
// 数据生成、FP64 参考、正确性检查与计时统计。
#include "gemm/config.hpp"
#include "gemm/cuda_utils.hpp"

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gemm {

// 每张矩阵使用独立的 64 位盐，避免 M/N 为 2 的幂时行索引与种子低比特
// 产生行置换相关（这会让 A 的某行与 B 的某行完全相同，制造虚假的极端值）。
struct HostOperands {
  std::vector<__half> a;
  std::vector<__half> b;
  // 用于记录实际生成的数学值；与 __half 值完全一致。
  float value_at_a(std::size_t i) const { return __half2float(a[i]); }
  float value_at_b(std::size_t i) const { return __half2float(b[i]); }
};

// 生成规范逻辑矩阵：A 按列主序 MxK，B 按列主序 NxK（即 B^T 的 KxN 视图）。
// 与具体 TN/NT 物理存储无关；不同布局使用同一份逻辑值。
HostOperands make_host_operands(Problem const& problem, std::uint64_t seed);
// 把规范逻辑矩阵打包成 problem 指定的物理布局（TN 需要转置成 K 连续）。
HostOperands pack_operands(Problem const& problem, HostOperands const& logical);
void upload_operands(DeviceBuffers const& buffers, HostOperands const& host, cudaStream_t stream);
void initialize_output(DeviceBuffers const& buffers, cudaStream_t stream);
std::vector<float> download_half_as_float(void const* source, std::size_t count, cudaStream_t stream);

struct Reference {
  std::string mode = "unavailable";  // full_gpu_fp64 | sampled_cpu_fp64
  std::string reason;
  bool full = false;
  std::vector<double> values;         // full=true 时有效，长度 M*N
  std::vector<std::size_t> indices;   // full=false 时有效
  std::vector<double> sampled;        // full=false 时与 indices 对齐
};

Reference build_reference(Problem const& problem, HostOperands const& host, int device,
                          cudaStream_t stream, bool allow_full);

struct Correctness {
  bool checked = false;
  bool pass = false;
  std::string mode = "not_checked";
  std::string reason;
  std::size_t checked_elements = 0;
  std::size_t mismatch_count = 0;
  std::size_t nonfinite_count = 0;
  double max_abs = 0.0;
  double max_rel = 0.0;
  double rel_l2 = 0.0;
  Thresholds thresholds;
};

Correctness validate_output(Problem const& problem, Reference const& reference,
                            std::vector<float> const& output, Thresholds const& thresholds);

// 冷缓存模式：在计时区间外写满一个大于 L2 的缓冲区。
void flush_caches(Buffer& flush_buffer, cudaStream_t stream);

}  // namespace gemm
