#pragma once
// plan1 的公共数学/运行配置定义。
//
// 统一逻辑问题（与仓库 kernels/host/common.cuh 完全一致）：
//   A: (M, K), B: (N, K), C: (M, N), C = alpha * A * B^T + beta * C
//   - 输入 A/B 为 FP16，输出 C 为 FP16，alpha=1、beta=0 为主测试。
//   - C 始终按列主序存储：C(row, col) 位于 row + col * M。
//   - TN/NT 只改变 A/B 的物理连续方向，不改变数学运算：
//       TN: A 按 K 连续（行主序 MxK），B 按 K 连续（行主序 NxK）；
//       NT: A 按 M 连续（列主序 MxK），B 按 N 连续（列主序 NxK）。
//   这与 kernels/host/common.cuh 的 GemmProblem 定义一致，报告中必须原样引用。
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace gemm {

enum class Accum { FP16, FP32 };

inline char const* accum_name(Accum a) {
  return a == Accum::FP16 ? "FP16" : "FP32";
}

struct Problem {
  int m = 0;
  int n = 0;
  int k = 0;
  // 与 kernels/host/common.cuh 相同：ta/tb 表示 A/B 的存储方向。
  bool ta = false;  // false='N'（M/N 连续），true='T'（K 连续）
  bool tb = false;

  bool is_nt() const { return !ta && tb; }
  bool is_tn() const { return ta && !tb; }

  char trans_a() const { return ta ? 'T' : 'N'; }
  char trans_b() const { return tb ? 'T' : 'N'; }
  std::string layout() const { return is_tn() ? "TN" : "NT"; }

  int ldA() const { return ta ? k : m; }
  int ldB() const { return tb ? n : k; }
  int ldC() const { return m; }

  // 物理存储索引；A/B/C 都是列主序存储（与 BLAS 的 leading dimension 一致）。
  std::size_t a_index(int row, int p) const {
    return ta ? std::size_t(row) * std::size_t(k) + std::size_t(p)
              : std::size_t(row) + std::size_t(p) * std::size_t(m);
  }
  // tb=true（NT）：B 为 N 连续（N×K 列主序），b[col + p*N]；
  // tb=false（TN）：B 为 K 连续（N×K 行主序），b[col*K + p]。
  std::size_t b_index(int col, int p) const {
    return tb ? std::size_t(col) + std::size_t(p) * std::size_t(n)
              : std::size_t(col) * std::size_t(k) + std::size_t(p);
  }
  std::size_t c_index(int row, int col) const {
    return std::size_t(row) + std::size_t(col) * std::size_t(m);
  }
  std::size_t a_elements() const { return std::size_t(m) * std::size_t(k); }
  std::size_t b_elements() const { return std::size_t(n) * std::size_t(k); }
  std::size_t c_elements() const { return std::size_t(m) * std::size_t(n); }
  double flops() const { return 2.0 * double(m) * double(n) * double(k); }
};

// 正确性阈值在实验前固定。它们随累加精度和 K 长度变化，但不随实现变化。
// FP16 累加：报告参考值 C 的量级约 sqrt(K)*0.1，FP16 舍入误差随 K 增长；
//           使用 0.5 + 1e-4*K 的绝对误差 + 1% 相对误差，rel-L2 <= 1%。
// FP32 累加：输出仍为 FP16，最大误差主要来自最终一次 FP16 舍入；
//           使用 0.125 绝对误差 + 0.2% 相对误差，rel-L2 <= 0.1%。
struct Thresholds {
  double atol = 0.0;
  double rtol = 0.0;
  double rel_l2_max = 0.0;
  double mismatch_fraction_max = 0.0;
};

inline Thresholds thresholds_for(Accum a, int k) {
  if (a == Accum::FP16) return {0.5 + 1e-4 * double(k), 0.01, 0.01, 0.0};
  return {0.125, 0.002, 0.001, 0.0};
}

struct AlgorithmTrial {
  int index = -1;
  std::string description;
  std::string status;
  std::size_t workspace = 0;
  double median_ms = 0.0;
};

struct PlanInfo {
  std::string implementation;
  std::string kernel_name;
  Accum accum = Accum::FP32;
  std::string algorithm_id = "not_selected";
  std::size_t workspace_bytes = 0;
  int heuristic_candidates = 0;
  std::string details;  // 单行 JSON，便于写入原始 CSV
  std::vector<AlgorithmTrial> trials;
};

struct DeviceBuffers {
  void* a = nullptr;
  void* b = nullptr;
  void* c = nullptr;
  std::size_t a_bytes = 0;
  std::size_t b_bytes = 0;
  std::size_t c_bytes = 0;
};

struct Options {
  std::string plan_id = "plan1";
  std::string run_id;
  std::string mode = "quick";
  std::string impl;
  std::string case_id;
  std::string shape_id;
  Problem problem;

  std::string timing_mode = "event";  // event | wall | graph
  std::string cache_mode = "warm";    // warm | cold
  int warmup = 20;
  int inner = 100;
  int samples = 5;
  int rounds = 5;
  std::size_t flush_bytes = 256ull << 20;

  bool validate = true;
  bool compute_reference = true;
  bool skip_timing = false;
  bool ncu_mode = false;

  int device = 0;
  std::string csv_path;
  std::string manifest_path;
  std::string correctness_log_path;

  std::size_t lt_workspace = 64ull << 20;
  int lt_candidates = 32;
  std::string lt_selection = "tune";  // tune | first | index
  int lt_index = 0;
  std::string lt_algo_hex;
  std::string lt_algo_version;

  std::uint64_t seed = 20261105ull;
  float alpha = 1.0f;
  float beta = 0.0f;

  std::string config_hash;
  std::string build_git_sha;
  bool clock_locked = false;
  int clock_target_mhz = 0;

  std::string notes;
};

inline std::string quote_csv(std::string const& value) {
  std::string out = "\"";
  for (char ch : value) {
    if (ch == '"') out += '"';
    out += ch;
  }
  out += '"';
  return out;
}

inline std::string json_escape(std::string const& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (unsigned char ch : value) {
    switch (ch) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (ch < 0x20) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x", unsigned(ch));
          out += buf;
        } else {
          out += char(ch);
        }
    }
  }
  return out;
}

}  // namespace gemm
