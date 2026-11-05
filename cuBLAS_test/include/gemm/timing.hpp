#pragma once
#include "gemm/backend.hpp"
#include "gemm/config.hpp"
#include "gemm/cuda_utils.hpp"
#include "gemm/telemetry.hpp"

#include <string>
#include <vector>

namespace gemm {

struct RoundStats {
  double mean_us = 0.0;
  double median_us = 0.0;
  double p10_us = 0.0;
  double p90_us = 0.0;
  double min_us = 0.0;
  double max_us = 0.0;
};

struct RoundMeasurement {
  int round = 0;
  std::string started_utc;
  std::string finished_utc;
  RoundStats stats;
  double tflops = 0.0;
  TelemetrySample telemetry;
};

struct TimingResult {
  std::vector<RoundMeasurement> rounds;
  std::string effective_mode;  // event | wall | graph
  std::string note;
};

// 每个 round 内执行 samples 次采样，每次采样 inner 次 kernel 调用。
// warm：数据驻留缓存；cold：每次采样前冲刷 L2，且强制 inner=1。
TimingResult benchmark_plan(GemmPlan& plan, Problem const& problem, Options const& options,
                            cudaStream_t stream, Buffer* flush_buffer,
                            TelemetryReader* telemetry);

}  // namespace gemm
