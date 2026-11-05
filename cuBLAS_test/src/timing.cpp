#include "gemm/timing.hpp"
#include "gemm/bench.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>
#include <stdexcept>

namespace gemm {
namespace {

std::string utc_now_ms() {
  auto now = std::chrono::system_clock::now();
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  std::time_t seconds = std::time_t(millis / 1000);
  std::tm utc{};
  gmtime_r(&seconds, &utc);
  char buffer[40];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", &utc);
  char result[48];
  std::snprintf(result, sizeof(result), "%s.%03lldZ", buffer, static_cast<long long>(millis % 1000));
  return result;
}

RoundStats compute_stats(std::vector<double> values) {
  RoundStats stats;
  if (values.empty()) return stats;
  std::sort(values.begin(), values.end());
  std::size_t const n = values.size();
  stats.min_us = values.front();
  stats.max_us = values.back();
  stats.median_us = (n % 2) ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
  stats.p10_us = values[std::size_t(0.1 * double(n - 1))];
  stats.p90_us = values[std::size_t(std::ceil(0.9 * double(n - 1)))];
  double sum = 0.0;
  for (double v : values) sum += v;
  stats.mean_us = sum / double(n);
  return stats;
}

TelemetrySample combine(TelemetrySample const& a, TelemetrySample const& b) {
  TelemetrySample result;
  result.valid = a.valid || b.valid;
  if (!a.valid && !b.valid) return result;
  auto average = [&](double x, double y) {
    bool xv = std::isfinite(x), yv = std::isfinite(y);
    if (xv && yv) return 0.5 * (x + y);
    return xv ? x : yv ? y : std::numeric_limits<double>::quiet_NaN();
  };
  result.sm_clock_mhz = average(a.sm_clock_mhz, b.sm_clock_mhz);
  result.mem_clock_mhz = average(a.mem_clock_mhz, b.mem_clock_mhz);
  result.power_w = average(a.power_w, b.power_w);
  result.temperature_c = std::isfinite(a.temperature_c) && std::isfinite(b.temperature_c)
      ? std::max(a.temperature_c, b.temperature_c)
      : std::isfinite(a.temperature_c) ? a.temperature_c : b.temperature_c;
  result.throttle_reasons = a.throttle_reasons | b.throttle_reasons;
  return result;
}

}  // namespace

TimingResult benchmark_plan(GemmPlan& plan, Problem const& problem, Options const& options,
                            cudaStream_t stream, Buffer* flush_buffer,
                            TelemetryReader* telemetry) {
  TimingResult result;
  result.effective_mode = options.timing_mode;
  int inner = options.inner;
  if (inner < 1) inner = 1;
  int samples = std::max(1, options.samples);
  int rounds = std::max(1, options.rounds);

  if (options.cache_mode == "cold" && inner != 1) {
    inner = 1;
    result.note += "cold cache forces inner=1; ";
  }

  bool use_graph = options.timing_mode == "graph";
  Graph graph;
  if (use_graph) {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    bool failed = false;
    std::string failure;
    try {
      for (int j = 0; j < inner; ++j) plan.run();
    } catch (std::exception const& error) {
      failed = true;
      failure = error.what();
    }
    cudaError_t const end_status = cudaStreamEndCapture(stream, &graph.graph);
    if (failed || end_status != cudaSuccess || graph.graph == nullptr) {
      (void)cudaGetLastError();
      if (graph.graph) {
        cudaGraphDestroy(graph.graph);
        graph.graph = nullptr;
      }
      use_graph = false;
      result.effective_mode = "launch";
      result.note += std::string("CUDA graph capture failed (") +
                     (failure.empty() ? cudaGetErrorString(end_status) : failure) +
                     "); fell back to launch; ";
    } else {
      CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph.executable, graph.graph, 0));
    }
  }

  auto batch = [&]() {
    if (use_graph) {
      CUDA_CHECK(cudaGraphLaunch(graph.executable, stream));
    } else {
      for (int j = 0; j < inner; ++j) plan.run();
    }
  };

  // 预热：launch 模式按 kernel 次数，graph 模式按 replay 次数折算。
  int warmup_launches = std::max(0, options.warmup);
  int warmup_batches = use_graph ? (warmup_launches + inner - 1) / inner : warmup_launches;
  if (use_graph) {
    for (int w = 0; w < warmup_batches; ++w) batch();
  } else {
    for (int w = 0; w < warmup_launches; ++w) plan.run();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  result.rounds.reserve(rounds);
  Event start;
  Event stop;
  for (int round = 0; round < rounds; ++round) {
    std::string const started_utc = utc_now_ms();
    TelemetrySample const telemetry_before = telemetry ? telemetry->sample() : TelemetrySample{};
    std::vector<double> sample_us;
    sample_us.reserve(samples);
    for (int s = 0; s < samples; ++s) {
      if (options.cache_mode == "cold") {
        if (!flush_buffer) throw std::runtime_error("cold cache requested without flush buffer");
        flush_caches(*flush_buffer, stream);
      }
      double elapsed_us = 0.0;
      if (options.timing_mode == "wall") {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto const begin = std::chrono::steady_clock::now();
        batch();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto const end = std::chrono::steady_clock::now();
        elapsed_us =
            std::chrono::duration<double, std::micro>(end - begin).count() / double(inner);
      } else if (options.timing_mode == "graph" && !use_graph) {
        // 回退到 launch 模式后仍使用 event 计时，口径已在 note 中记录。
        CUDA_CHECK(cudaEventRecord(start.value, stream));
        batch();
        CUDA_CHECK(cudaEventRecord(stop.value, stream));
        CUDA_CHECK(cudaEventSynchronize(stop.value));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.value, stop.value));
        elapsed_us = double(ms) * 1000.0 / double(inner);
      } else {
        CUDA_CHECK(cudaEventRecord(start.value, stream));
        batch();
        CUDA_CHECK(cudaEventRecord(stop.value, stream));
        CUDA_CHECK(cudaEventSynchronize(stop.value));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.value, stop.value));
        elapsed_us = double(ms) * 1000.0 / double(inner);
      }
      if (!(elapsed_us > 0.0) || !std::isfinite(elapsed_us)) {
        throw std::runtime_error("Invalid timer duration");
      }
      sample_us.push_back(elapsed_us);
    }
    TelemetrySample const telemetry_after = telemetry ? telemetry->sample() : TelemetrySample{};
    RoundMeasurement measurement;
    measurement.round = round;
    measurement.started_utc = started_utc;
    measurement.finished_utc = utc_now_ms();
    measurement.stats = compute_stats(sample_us);
    measurement.telemetry = combine(telemetry_before, telemetry_after);
    measurement.tflops =
        problem.flops() / (measurement.stats.median_us * 1e-6) / 1e12;
    result.rounds.push_back(measurement);
  }
  result.note += "warmup_batches=" + std::to_string(warmup_batches) +
                 " inner=" + std::to_string(inner) + " samples=" + std::to_string(samples) +
                 " rounds=" + std::to_string(rounds);
  return result;
}

}  // namespace gemm
