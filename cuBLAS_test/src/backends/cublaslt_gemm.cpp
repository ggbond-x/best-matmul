// cuBLASLt 后端：枚举 heuristic 候选，逐个检查合法性，可选地实测挑选，
// 并记录算法 ID、workspace 与搜索范围。所有候选都在构造阶段完成选择，
// 正式计时只调用一次 cublasLtMatmul。
//
// 布局映射（与 kernels/host/common.cuh 相同）：
//   A 存储 M x K，ta=false 时列主序（M 连续），ta=true 时等价于 K x M 列主序（K 连续）；
//   B 存储 N x K，tb=true 时列主序（N 连续），tb=false 时等价于 K x N 列主序（K 连续）；
//   C 恒定列主序 M x N。
#include "gemm/backend.hpp"
#include "gemm/bench.hpp"

#include <cublasLt.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gemm {
namespace {

struct LtHandle {
  cublasLtHandle_t value = nullptr;
  LtHandle() { BLAS_CHECK(cublasLtCreate(&value)); }
  ~LtHandle() {
    if (value) cublasLtDestroy(value);
  }
  LtHandle(LtHandle const&) = delete;
  LtHandle& operator=(LtHandle const&) = delete;
};

struct LtDescriptor {
  cublasLtMatmulDesc_t value = nullptr;
  ~LtDescriptor() {
    if (value) cublasLtMatmulDescDestroy(value);
  }
};

struct LtLayout {
  cublasLtMatrixLayout_t value = nullptr;
  ~LtLayout() {
    if (value) cublasLtMatrixLayoutDestroy(value);
  }
};

struct LtPreference {
  cublasLtMatmulPreference_t value = nullptr;
  ~LtPreference() {
    if (value) cublasLtMatmulPreferenceDestroy(value);
  }
};

struct AlgoRecord {
  cublasLtMatmulHeuristicResult_t heuristic{};
  int algo_id = -1;
  std::string summary;
  std::string status = "not_checked";
  double median_ms = 0.0;
};

int algo_attribute_int(cublasLtMatmulAlgo_t const& algo, cublasLtMatmulAlgoConfigAttributes_t attribute) {
  int value = -1;
  std::size_t written = 0;
  cublasStatus_t const status = cublasLtMatmulAlgoConfigGetAttribute(
      &algo, attribute, &value, sizeof(value), &written);
  return status == CUBLAS_STATUS_SUCCESS ? value : -1;
}

std::string algo_description(cublasLtMatmulAlgo_t const& algo, int id) {
  std::ostringstream stream;
  stream << "algo_id=" << id << " tile=" << algo_attribute_int(algo, CUBLASLT_ALGO_CONFIG_TILE_ID)
         << " splitk="
         << algo_attribute_int(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM)
         << " reduction="
         << algo_attribute_int(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
  return stream.str();
}

std::string algo_hex(cublasLtMatmulAlgo_t const& algo) {
  static char const digits[] = "0123456789abcdef";
  auto bytes = reinterpret_cast<unsigned char const*>(&algo);
  std::string out;
  out.reserve(2 * sizeof(algo));
  for (std::size_t i = 0; i < sizeof(algo); ++i) {
    out += digits[bytes[i] >> 4]; out += digits[bytes[i] & 15];
  }
  return out;
}

cublasLtMatmulAlgo_t decode_algo(std::string const& hex) {
  if (hex.size() != 2 * sizeof(cublasLtMatmulAlgo_t)) throw std::runtime_error("invalid cuBLASLt algorithm blob length");
  cublasLtMatmulAlgo_t algo{};
  auto bytes = reinterpret_cast<unsigned char*>(&algo);
  auto digit = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    throw std::runtime_error("invalid cuBLASLt algorithm blob");
  };
  for (std::size_t i = 0; i < sizeof(algo); ++i) bytes[i] = unsigned((digit(hex[2*i]) << 4) | digit(hex[2*i+1]));
  return algo;
}

class CublasLtPlan final : public GemmPlan {
 public:
  CublasLtPlan(Options const& options, DeviceBuffers const& buffers, cudaStream_t stream,
               Accum accum, Validator const& validator)
      : options_(options),
        buffers_(buffers),
        stream_(stream),
        accum_(accum),
        validator_(validator),
        workspace_(options.lt_workspace) {
    build_descriptors();
    enumerate_and_select();
  }

  void run() override {
    cublasLtMatmulHeuristicResult_t const& result = selected_.heuristic;
    void* workspace = result.workspaceSize ? workspace_.value : nullptr;
    BLAS_CHECK(cublasLtMatmul(handle_.value, desc_.value, alpha_ptr(), buffers_.a,
                              layout_a_.value, buffers_.b, layout_b_.value, beta_ptr(),
                              buffers_.c, layout_c_.value, buffers_.c, layout_c_.value,
                              &result.algo, workspace, result.workspaceSize, stream_));
  }

 private:
  void build_descriptors() {
    Problem const& p = options_.problem;
    cudaDataType_t const scale_type = accum_ == Accum::FP16 ? CUDA_R_16F : CUDA_R_32F;
    cublasComputeType_t const compute_type =
        accum_ == Accum::FP16 ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F;

    BLAS_CHECK(cublasLtMatmulDescCreate(&desc_.value, compute_type, scale_type));
    cublasOperation_t const op_a = p.ta ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t const op_b = p.tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    BLAS_CHECK(cublasLtMatmulDescSetAttribute(desc_.value, CUBLASLT_MATMUL_DESC_TRANSA, &op_a,
                                              sizeof(op_a)));
    BLAS_CHECK(cublasLtMatmulDescSetAttribute(desc_.value, CUBLASLT_MATMUL_DESC_TRANSB, &op_b,
                                              sizeof(op_b)));
    // 明确不启用 FAST_ACCUM，避免改变累加语义。

    // 描述符描述的是“转置前”的存储矩阵。
    int const a_rows = p.ta ? p.k : p.m;
    int const a_cols = p.ta ? p.m : p.k;
    int const b_rows = p.tb ? p.n : p.k;
    int const b_cols = p.tb ? p.k : p.n;
    BLAS_CHECK(cublasLtMatrixLayoutCreate(&layout_a_.value, CUDA_R_16F, a_rows, a_cols, p.ldA()));
    BLAS_CHECK(cublasLtMatrixLayoutCreate(&layout_b_.value, CUDA_R_16F, b_rows, b_cols, p.ldB()));
    BLAS_CHECK(cublasLtMatrixLayoutCreate(&layout_c_.value, CUDA_R_16F, p.m, p.n, p.ldC()));

    BLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference_.value));
    std::size_t const workspace_limit = options_.lt_workspace;
    BLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference_.value, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit,
        sizeof(workspace_limit)));

    alpha_h_ = __float2half(options_.alpha);
    beta_h_ = __float2half(options_.beta);
    alpha_f_ = options_.alpha;
    beta_f_ = options_.beta;
  }

  void* alpha_ptr() {
    return accum_ == Accum::FP16 ? static_cast<void*>(&alpha_h_) : static_cast<void*>(&alpha_f_);
  }
  void* beta_ptr() {
    return accum_ == Accum::FP16 ? static_cast<void*>(&beta_h_) : static_cast<void*>(&beta_f_);
  }

  bool legal(AlgoRecord const& record) const {
    cublasLtMatmulHeuristicResult_t checked{};
    cublasStatus_t const status =
        cublasLtMatmulAlgoCheck(handle_.value, desc_.value, layout_a_.value, layout_b_.value,
                                layout_c_.value, layout_c_.value, &record.heuristic.algo, &checked);
    return status == CUBLAS_STATUS_SUCCESS && checked.state == CUBLAS_STATUS_SUCCESS;
  }

  // 只用于构造阶段的候选选择，不计入正式计时。
  double probe_ms(AlgoRecord const& record) {
    void* workspace = record.heuristic.workspaceSize ? workspace_.value : nullptr;
    auto launch = [&]() {
      BLAS_CHECK(cublasLtMatmul(
          handle_.value, desc_.value, alpha_ptr(), buffers_.a, layout_a_.value, buffers_.b,
          layout_b_.value, beta_ptr(), buffers_.c, layout_c_.value, buffers_.c,
          layout_c_.value, &record.heuristic.algo, workspace, record.heuristic.workspaceSize,
          stream_));
    };
    launch();
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    constexpr int kProbeRuns = 5;
    Event start;
    Event stop;
    CUDA_CHECK(cudaEventRecord(start.value, stream_));
    for (int i = 0; i < kProbeRuns; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop.value, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop.value));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.value, stop.value));
    return double(ms) / double(kProbeRuns);
  }

  void enumerate_and_select() {
    if (!options_.lt_algo_hex.empty()) {
      if (options_.lt_algo_version != std::to_string(cublasLtGetVersion()))
        throw std::runtime_error("saved cuBLASLt algorithm version differs from current library");
      auto algo = decode_algo(options_.lt_algo_hex);
      cublasLtMatmulHeuristicResult_t checked{};
      BLAS_CHECK(cublasLtMatmulAlgoCheck(handle_.value, desc_.value, layout_a_.value, layout_b_.value,
                                         layout_c_.value, layout_c_.value, &algo, &checked));
      if (checked.state != CUBLAS_STATUS_SUCCESS || checked.workspaceSize > options_.lt_workspace)
        throw std::runtime_error("saved cuBLASLt algorithm is unsupported or exceeds workspace budget");
      checked.algo = algo;
      selected_.heuristic = checked;
      selected_.algo_id = algo_attribute_int(algo, CUBLASLT_ALGO_CONFIG_ID);
      info.implementation = accum_ == Accum::FP16 ? "CUBLAS_LT_F16ACC" : "CUBLAS_LT_F32ACC";
      info.accum = accum_;
      info.kernel_name = "cublasLtMatmul";
      info.algorithm_id = "cublasLt/algo_id=" + std::to_string(selected_.algo_id);
      info.workspace_bytes = checked.workspaceSize;
      info.details = "{\"selection\":\"saved_exact\",\"algo_blob\":\"" + options_.lt_algo_hex + "\"}";
      return;
    }
    int requested = std::max(1, options_.lt_candidates);
    std::vector<cublasLtMatmulHeuristicResult_t> heuristic{std::size_t(requested)};
    int returned = 0;
    BLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        handle_.value, desc_.value, layout_a_.value, layout_b_.value, layout_c_.value,
        layout_c_.value, preference_.value, requested, heuristic.data(), &returned));
    if (returned <= 0) {
      throw Skip("cuBLASLt heuristic returned no algorithm for this shape/dtype/layout");
    }
    heuristic.resize(std::size_t(returned));

    std::vector<AlgoRecord> records;
    records.reserve(heuristic.size());
    // selection=first 用于 NCU：只取第一个合法候选，不做 probe，避免在 NCU 进程里
    // 额外启动一批候选 kernel 干扰采集。
    bool const probe_candidates = options_.lt_selection != "first";
    for (auto const& item : heuristic) {
      AlgoRecord record;
      record.heuristic = item;
      record.algo_id = algo_attribute_int(item.algo, CUBLASLT_ALGO_CONFIG_ID);
      record.summary = algo_description(item.algo, record.algo_id);
      record.status = legal(record) ? "legal" : "illegal";
      if (record.status == "legal" && probe_candidates) {
        try {
          record.median_ms = probe_ms(record);
          record.status = "timed";
        } catch (std::exception const& error) {
          record.status = std::string("probe_failed: ") + error.what();
          record.median_ms = std::numeric_limits<double>::infinity();
        }
      }
      records.push_back(record);
    }

    auto eligible = [](AlgoRecord const& record) {
      return record.status == "timed" || record.status == "legal";
    };
    std::vector<std::size_t> order;
    for (std::size_t i = 0; i < records.size(); ++i) {
      if (eligible(records[i])) order.push_back(i);
    }
    if (probe_candidates) {
      std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return records[a].median_ms < records[b].median_ms;
      });
    }
    if (order.empty()) {
      throw Skip("cuBLASLt: no legal algorithm for this configuration");
    }

    auto selection = options_.lt_selection;
    std::size_t chosen = order.front();
    if (selection == "first") {
      for (std::size_t i = 0; i < records.size(); ++i) {
        if (eligible(records[i])) {
          chosen = i;
          break;
        }
      }
    } else if (selection == "index") {
      int const requested_index = std::max(0, options_.lt_index);
      chosen = order[std::min<std::size_t>(std::size_t(requested_index), order.size() - 1)];
    }

    // 只对最终候选做一次正确性检查；若失败，按耗时顺序尝试下一个。
    // validator 由调用方提供参考结果；NCU 模式下为空，跳过验证。
    bool validated = false;
    if (validator_) {
      std::vector<std::size_t> candidates = order;
      if (selection == "first" || selection == "index") {
        candidates.clear();
        candidates.push_back(chosen);
        for (std::size_t i : order) {
          if (i != chosen) candidates.push_back(i);
        }
      }
      for (std::size_t i : candidates) {
        initialize_output(buffers_, stream_);
        try {
          AlgoRecord& record = records[i];
          void* workspace = record.heuristic.workspaceSize ? workspace_.value : nullptr;
          BLAS_CHECK(cublasLtMatmul(
              handle_.value, desc_.value, alpha_ptr(), buffers_.a, layout_a_.value, buffers_.b,
              layout_b_.value, beta_ptr(), buffers_.c, layout_c_.value, buffers_.c,
              layout_c_.value, &record.heuristic.algo, workspace, record.heuristic.workspaceSize,
              stream_));
          CUDA_CHECK(cudaStreamSynchronize(stream_));
        } catch (std::exception const& error) {
          records[i].status = std::string("validation_launch_failed: ") + error.what();
          continue;
        }
        if (validator_()) {
          chosen = i;
          validated = true;
          break;
        }
        records[i].status = "validation_failed";
      }
      if (!validated) {
        throw std::runtime_error(
            "cuBLASLt: no heuristic candidate produced a finite, correct output");
      }
    }

    selected_ = records[chosen];
    selected_.status = validated ? "selected_validated" : "selected";

    info.implementation = accum_ == Accum::FP16 ? "CUBLAS_LT_F16ACC" : "CUBLAS_LT_F32ACC";
    info.accum = accum_;
    info.kernel_name = "cublasLtMatmul";
    info.algorithm_id = "cublasLt/algo_id=" + std::to_string(selected_.algo_id);
    info.workspace_bytes = selected_.heuristic.workspaceSize;
    info.heuristic_candidates = returned;
    for (std::size_t i = 0; i < records.size(); ++i) {
      AlgorithmTrial trial;
      trial.index = int(i);
      trial.description = records[i].summary;
      trial.status = records[i].status;
      trial.workspace = records[i].heuristic.workspaceSize;
      trial.median_ms = records[i].median_ms;
      info.trials.push_back(trial);
    }
    std::ostringstream details;
    details << "{\"api\":\"cublasLtMatmul\",\"compute_type\":\""
            << (accum_ == Accum::FP16 ? "CUBLAS_COMPUTE_16F" : "CUBLAS_COMPUTE_32F")
            << "\",\"scale_type\":\"" << (accum_ == Accum::FP16 ? "CUDA_R_16F" : "CUDA_R_32F")
            << "\",\"selection\":\"" << options_.lt_selection << "\",\"requested_candidates\":"
            << requested << ",\"returned_candidates\":" << returned << ",\"legal_candidates\":"
            << order.size() << ",\"workspace_limit_bytes\":" << options_.lt_workspace
            << ",\"workspace_required_bytes\":" << selected_.heuristic.workspaceSize
            << ",\"selected\":\"" << selected_.summary
            << "\",\"algo_blob\":\"" << algo_hex(selected_.heuristic.algo)
            << "\",\"cublaslt_version\":\"" << cublasLtGetVersion()
            << "\",\"fast_accum\":false}";
    info.details = details.str();
  }

  Options options_;
  DeviceBuffers buffers_;
  cudaStream_t stream_ = nullptr;
  Accum accum_ = Accum::FP32;
  Validator validator_;

  LtHandle handle_;
  LtDescriptor desc_;
  LtLayout layout_a_;
  LtLayout layout_b_;
  LtLayout layout_c_;
  LtPreference preference_;
  Buffer workspace_;

  __half alpha_h_{};
  __half beta_h_{};
  float alpha_f_ = 1.0f;
  float beta_f_ = 0.0f;
  AlgoRecord selected_;
};

}  // namespace

std::unique_ptr<GemmPlan> make_cublas_lt(Options const& options, DeviceBuffers const& buffers,
                                         cudaStream_t stream, Accum accum,
                                         Validator const& validator) {
  return std::make_unique<CublasLtPlan>(options, buffers, stream, accum, validator);
}

}  // namespace gemm
