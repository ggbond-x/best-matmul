#pragma once
#include "common.cuh"

#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

// 本文件只负责计时、预热和统计。launch 回调只提交 kernel：
// 使用 config.stream，不在回调里做同步、内存分配或 CPU 参数准备。
namespace gemm_bench {

// 每个样本都是“一批启动的总时间 / 本批启动次数”，单位 ms。
// 因而 p50/p99 描述批均耗时，不能解释成单次 kernel 延迟的百分位。
struct PerformanceStats {
  std::size_t sample_count = 0;
  double elapsed_ms = 0.0;             // 正式批次总时间，不含预热和校验。
  double p50_ms = 0.0;
  double p99_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double drift_pct = 0.0;              // 后 1/4 相对前 1/4 的中位数变化。
  double tflops_p50 = 0.0;
  double tflops_at_p99_latency = 0.0;  // p99 延迟对应的吞吐量，非吞吐量 p99。
  int warmup_launches = 0;
  bool warmup_stable = false;          // 经验判定，不能保证没有温度漂移。
};

inline void validate_measurement_config(MeasurementConfig const& config)
{
  if (config.min_warmup_launches <= 0 || config.launches_per_batch <= 0 || config.batches < 2) {
    std::cerr << "Require min_warmup > 0, per_batch > 0, batches >= 2\n";
    std::exit(EXIT_FAILURE);
  }
  // 最低预热次数向上补齐一批，再加额外批次；保证实际启动计数不溢出 int。
  long long upper_bound = static_cast<long long>(config.min_warmup_launches) +
      (benchmark_policy::kExtraWarmupBatches + 1LL) * config.launches_per_batch;
  if (upper_bound > std::numeric_limits<int>::max()) {
    std::cerr << "Warmup count exceeds supported integer range\n";
    std::exit(EXIT_FAILURE);
  }
}

// Event 在所有批次之间复用，创建和销毁不进入测量区间。
// 这个小类只拥有计时资源；析构释放资源，禁止拷贝以免重复销毁 Event。
class BatchTimer {
 public:
  explicit BatchTimer(MeasurementConfig const& config)
      : mode_(config.timer_mode), stream_(config.stream) {
    if (mode_ == TimerMode::Event) {
      CUTE_CHECK_ERROR(cudaEventCreate(&begin_));
      CUTE_CHECK_ERROR(cudaEventCreate(&end_));
    }
  }

  ~BatchTimer() {
    if (mode_ == TimerMode::Event) {
      CUTE_CHECK_ERROR(cudaEventDestroy(begin_));
      CUTE_CHECK_ERROR(cudaEventDestroy(end_));
    }
  }

  BatchTimer(BatchTimer const&) = delete;
  BatchTimer& operator=(BatchTimer const&) = delete;

  // 先开始计时，连续提交一批，再等待本批完成；返回平均单次耗时。
  // 两种模式都逐批等待，避免无限积压启动队列。
  template <class Launch>
  double measure(Launch const& launch, int per_batch) {
    start();
    for (int i = 0; i < per_batch; ++i) {
      launch();
    }
    CUTE_CHECK_ERROR(cudaGetLastError());
    double elapsed_ms = finish_ms();
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0) {
      std::cerr << "Timer resolution insufficient; increase per_batch\n";
      std::exit(EXIT_FAILURE);
    }
    return elapsed_ms / double(per_batch);
  }

 private:
  void start() {
    if (mode_ == TimerMode::Event) {
      CUTE_CHECK_ERROR(cudaEventRecord(begin_, stream_));
    } else {
      host_begin_ = std::chrono::steady_clock::now();
    }
  }

  double finish_ms() {
    if (mode_ == TimerMode::Event) {
      CUTE_CHECK_ERROR(cudaEventRecord(end_, stream_));
      CUTE_CHECK_ERROR(cudaEventSynchronize(end_));
      float elapsed_ms = 0.0f;
      CUTE_CHECK_ERROR(cudaEventElapsedTime(&elapsed_ms, begin_, end_));
      return double(elapsed_ms);
    }
    // steady_clock 是单调时钟；同步本流后才停止，包含 CPU 提交和等待成本。
    CUTE_CHECK_ERROR(cudaStreamSynchronize(stream_));
    auto host_end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(host_end - host_begin_).count();
  }

  TimerMode mode_;
  cudaStream_t stream_;
  cudaEvent_t begin_{};
  cudaEvent_t end_{};
  std::chrono::steady_clock::time_point host_begin_;
};

// 输入已升序。用 q*(n-1) 线性插值：偶数个样本的 p50 是中间两项的平均。
inline double percentile_ms(std::vector<double> const& sorted_samples, double q)
{
  if (sorted_samples.empty() || !(q >= 0.0 && q <= 1.0)) {
    std::cerr << "Invalid percentile input\n";
    std::exit(EXIT_FAILURE);
  }
  double position = q * double(sorted_samples.size() - 1);
  std::size_t lo = std::size_t(position);
  std::size_t hi = std::min(lo + 1, sorted_samples.size() - 1);
  return sorted_samples[lo] + (position - double(lo)) *
         (sorted_samples[hi] - sorted_samples[lo]);
}

inline double median_ms(std::vector<double> samples)
{
  std::sort(samples.begin(), samples.end());
  return percentile_ms(samples, 0.5);
}

// 只复制并排序指定窗口，不改变原样本的时间顺序；预热和漂移统计共用。
inline double median_window_ms(std::vector<double> const& samples,
                               std::size_t first, std::size_t count)
{
  return median_ms(std::vector<double>(samples.begin() + first,
                                       samples.begin() + first + count));
}

struct WarmupResult {
  int launches = 0;
  bool stable = false;
};

// 每 5 批比较最近两个窗口，连续 2 次变化不超过 2% 才确认平台期。
// 默认最早在第 15 批确认稳定，即 20 次/批时至少预热 300 次。
// 用户指定最低次数后最多再给 20 批余量；不稳定也会结束，并在结果中标记。
template <class Launch>
WarmupResult warmup_until_stable(BatchTimer& timer, Launch const& launch,
                                MeasurementConfig const& config)
{
  constexpr std::size_t window = benchmark_policy::kWarmupWindowBatches;
  int required_batches = int((static_cast<long long>(config.min_warmup_launches) +
                             config.launches_per_batch - 1) / config.launches_per_batch);
  int max_batches = required_batches + benchmark_policy::kExtraWarmupBatches;
  std::vector<double> samples;
  WarmupResult result;
  int stable_windows = 0;

  for (int batch = 0; batch < max_batches; ++batch) {
    samples.push_back(timer.measure(launch, config.launches_per_batch));
    result.launches += config.launches_per_batch;
    std::size_t count = samples.size();
    if (count < 2 * window || count % window != 0) {
      continue;
    }
    double previous = median_window_ms(samples, count - 2 * window, window);
    double recent = median_window_ms(samples, count - window, window);
    bool stable = std::abs(recent / previous - 1.0) <=
                  benchmark_policy::kPlateauRelativeTolerance;
    stable_windows = stable ? stable_windows + 1 : 0;
    if (int(count) >= required_batches &&
        stable_windows >= benchmark_policy::kStableWindowCount) {
      result.stable = true;
      break;
    }
  }
  return result;
}

// GEMM 乘加按 2 FLOPs 计；先转 double 再乘，避免整数溢出。
inline double gemm_tflops(GemmProblem const& problem, double latency_ms)
{
  if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0 ||
      !std::isfinite(latency_ms) || latency_ms <= 0.0) {
    std::cerr << "Invalid GEMM dimensions or latency\n";
    std::exit(EXIT_FAILURE);
  }
  return 2.0 * double(problem.m) * double(problem.n) * double(problem.k) /
         (latency_ms * 1.0e9);
}

inline PerformanceStats summarize_performance(std::vector<double> const& samples,
                                              int per_batch, GemmProblem const& problem)
{
  if (samples.size() < 2 || per_batch <= 0) {
    std::cerr << "Need at least two batches and positive launches per batch\n";
    std::exit(EXIT_FAILURE);
  }
  for (double sample : samples) {
    if (!std::isfinite(sample) || sample <= 0.0) {
      std::cerr << "Invalid elapsed time; increase launches per batch\n";
      std::exit(EXIT_FAILURE);
    }
  }
  std::vector<double> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  PerformanceStats stats;
  stats.sample_count = samples.size();
  stats.elapsed_ms = std::accumulate(samples.begin(), samples.end(), 0.0) * per_batch;
  stats.p50_ms = percentile_ms(sorted, 0.5);
  stats.p99_ms = percentile_ms(sorted, 0.99);
  stats.min_ms = sorted.front();
  stats.max_ms = sorted.back();
  stats.tflops_p50 = gemm_tflops(problem, stats.p50_ms);
  stats.tflops_at_p99_latency = gemm_tflops(problem, stats.p99_ms);

  // 正值表示后段变慢。漂移使用原始顺序，不能使用上方排序后的样本。
  std::size_t window = std::max(std::size_t(1), samples.size() / 4);
  double first = median_window_ms(samples, 0, window);
  double last = median_window_ms(samples, samples.size() - window, window);
  stats.drift_pct = 100.0 * (last / first - 1.0);
  return stats;
}

// 可选状态记录在计时之外运行。用 PCI 地址选卡，避免 CUDA_VISIBLE_DEVICES
// 重排编号后查错 GPU；nvidia-smi 不可用时保留测量，但缺少状态佐证。
inline void print_gpu_state(char const* phase)
{
  int device = 0;
  char pci_bus_id[32] = {};
  CUTE_CHECK_ERROR(cudaGetDevice(&device));
  CUTE_CHECK_ERROR(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device));
  std::cout << "GPU state [" << phase << "] PCI=" << pci_bus_id << std::endl;
  char command[768] = {};
  std::snprintf(command, sizeof(command),
      "nvidia-smi -i %s --query-gpu=timestamp,uuid,driver_version,pstate,"
      "temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used "
      "--format=csv", pci_bus_id);
  std::fflush(stdout);
  if (std::system(command) != 0) {
    std::cout << "GPU telemetry unavailable (nvidia-smi); measurement continues\n";
    return;
  }
  std::snprintf(command, sizeof(command),
      "nvidia-smi -i %s --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory "
      "--format=csv", pci_bus_id);
  std::fflush(stdout);
  if (std::system(command) != 0) {
    std::cout << "GPU process list unavailable\n";
  }
}

// 测量主流程：预检查 -> 预热 -> 正式采样 -> 后检查 -> 汇总。
// 校验回调接收阶段名，只能在计时已结束、GPU 已完成时运行。
template <class Launch, class Validate>
PerformanceStats measure_gemm_performance(
  Launch const& launch, 
  Validate const& validate,
  GemmProblem const& problem,
  MeasurementConfig const& config)
{
  validate_measurement_config(config);
  BatchTimer timer(config);
  print_gpu_state("start");

  // 先校验再预热，避免 CPU 参考计算让已预热的 GPU 再次降频。
  launch();
  CUTE_CHECK_ERROR(cudaGetLastError());
  CUTE_CHECK_ERROR(cudaStreamSynchronize(config.stream));
  validate("precheck");

  WarmupResult warmup = warmup_until_stable(timer, launch, config);
  if (!warmup.stable) {
    std::cerr << "Warning: warmup did not reach a stable plateau; interpret results cautiously\n";
  }
  std::cout << "Warmup launches=" << warmup.launches
            << ", plateau=" << (warmup.stable ? "stable" : "not confirmed") 
            << std::endl;

  std::vector<double> samples;
  samples.reserve(config.batches);
  for (int batch = 0; batch < config.batches; ++batch) {
    samples.push_back(timer.measure(launch, config.launches_per_batch));
  }

  // 状态查询、显存回读和 CPU 校验均不进入正式采样时间。
  print_gpu_state("end");
  validate("postcheck");
  PerformanceStats stats = summarize_performance(samples, config.launches_per_batch, problem);
  stats.warmup_launches = warmup.launches;
  stats.warmup_stable = warmup.stable;
  return stats;
}

}  // namespace gemm_bench
