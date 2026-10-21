#pragma once
#include "measurement.cuh"
#include "validation.cuh"

#include <cerrno>
#include <cstring>
#include <iomanip>
#include <limits>

// 所有 kernel 共用的命令行、通用参数检查和结果输出。
namespace gemm_bench {

struct BenchResult {
  PerformanceStats performance;
  LaunchConfig launch;
};

inline void print_usage(char const* program)
{
  std::cout << "Usage: " << program
            << " [M N K transA transB input_mode event|wall per_batch min_warmup batches]\n"
            << "Defaults: 4096 4096 4096 N T 2 event 20 20 200\n"
            << "Modes: N T or T N; input_mode: 0=zero, 1=one, 2=random +/-1, 3=pattern\n"
            << "per_batch = launches per batch; min_warmup is a minimum, not a fixed count\n"
            << "p50/p99 describe batch-average per-launch latency\n";
}

// strtol 完整消费字符串，拒绝 20x、空串以及超出 int 范围的数值。
inline int parse_integer(char const* text, char const* name)
{
  char* end = nullptr;
  errno = 0;
  long value = std::strtol(text, &end, 10);
  if (errno == ERANGE || end == text || *end != '\0' ||
      value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
    std::cerr << "Invalid integer for " << name << ": " << text << '\n';
    std::exit(EXIT_FAILURE);
  }
  return int(value);
}

inline char parse_transpose(char const* text, char const* name)
{
  if (std::strlen(text) != 1) {
    std::cerr << name << " must be one character\n";
    std::exit(EXIT_FAILURE);
  }
  // 此处只解析字符；NT/TN 的组合合法性由 validate_options 统一检查。
  return text[0];
}

inline InputMode parse_input_mode(char const* text)
{
  int mode = parse_integer(text, "input_mode");
  if (mode < 0 || mode > 3) {
    std::cerr << "INPUT_MODE must be one of {0, 1, 2, 3}\n";
    std::exit(EXIT_FAILURE);
  }
  return static_cast<InputMode>(mode);
}

inline TimerMode parse_timer_mode(char const* text)
{
  if (std::strcmp(text, "event") == 0) return TimerMode::Event;
  if (std::strcmp(text, "wall") == 0) return TimerMode::Wall;
  std::cerr << "Timer mode must be wall or event\n";
  std::exit(EXIT_FAILURE);
}

// 先保留结构体默认值，再按命令行位置覆盖；各参数的格式检查留在小函数中。
inline BenchmarkOptions parse_options(int argc, char** argv)
{
  if (argc > 11) {
    print_usage(argv[0]);
    std::exit(EXIT_FAILURE);
  }
  BenchmarkOptions options;
  GemmProblem& problem = options.problem;
  MeasurementConfig& measurement = options.measurement;
  if (argc >= 2) problem.m = parse_integer(argv[1], "M");
  if (argc >= 3) problem.n = parse_integer(argv[2], "N");
  if (argc >= 4) problem.k = parse_integer(argv[3], "K");
  if (argc >= 5) problem.transA = parse_transpose(argv[4], "transA");
  if (argc >= 6) problem.transB = parse_transpose(argv[5], "transB");
  if (argc >= 7) options.input_mode = parse_input_mode(argv[6]);
  if (argc >= 8) measurement.timer_mode = parse_timer_mode(argv[7]);
  if (argc >= 9) measurement.launches_per_batch = parse_integer(argv[8], "per_batch");
  if (argc >= 10) measurement.min_warmup_launches = parse_integer(argv[9], "min_warmup");
  if (argc >= 11) measurement.batches = parse_integer(argv[10], "batches");
  return options;
}

inline void validate_options(BenchmarkOptions const& options)
{
  GemmProblem const& problem = options.problem;
  validate_measurement_config(options.measurement);
  if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0) {
    std::cerr << "Require M,N,K > 0\n";
    std::exit(EXIT_FAILURE);
  }
  bool is_tn = problem.transA == 'T' && problem.transB == 'N';
  if (!problem.is_nt() && !is_tn) {
    std::cerr << "Only NT and TN modes are implemented. Use N T or T N\n";
    std::exit(EXIT_FAILURE);
  }
  // 维持原基准支持的元素数量范围；实际分配和索引使用 size_t。
  // tile 对齐、共享内存和 pipeline 的约束由所选 kernel 适配器检查。
  std::size_t max_elements = std::size_t(std::numeric_limits<int>::max());
  if (problem.a_elements() > max_elements || problem.b_elements() > max_elements ||
      problem.c_elements() > max_elements) {
    std::cerr << "Matrix element count exceeds supported range\n";
    std::exit(EXIT_FAILURE);
  }
}

inline void print_configuration(cudaDeviceProp const& props, BenchmarkOptions const& options)
{
  GemmProblem const& problem = options.problem;
  MeasurementConfig const& measurement = options.measurement;
  std::printf("Device: %s, SM count=%d, compute capability=%d.%d\n",
              props.name, props.multiProcessorCount, props.major, props.minor);
  std::printf("Input mode: INPUT_MODE=%d\n", int(options.input_mode));
  std::printf("MIN_WARMUP=%d, LAUNCHES_PER_BATCH=%d, BATCHES=%d\n",
              measurement.min_warmup_launches, measurement.launches_per_batch, measurement.batches);
  std::printf("Timer=%s, scope=%s\n", timer_mode_name(measurement.timer_mode),
              measurement.timer_mode == TimerMode::Event ? "GPU event interval" : "host submission + stream completion");
  std::printf("Problem: M=%d, N=%d, K=%d, mode=%c%c\n",
              problem.m, problem.n, problem.k, problem.transA, problem.transB);
}

inline void print_result(BenchmarkOptions const& options, BenchResult const& result)
{
  GemmProblem const& problem = options.problem;
  LaunchConfig const& launch = result.launch;
  PerformanceStats const& stats = result.performance;
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Launch config: grid=(" << launch.grid.x << ',' << launch.grid.y << ',' << launch.grid.z << ')'
            << ", block=" << launch.block.x
            << ", cluster=(" << launch.cluster.x << ',' << launch.cluster.y << ',' << launch.cluster.z << ')'
            << ", smem=" << launch.smem_bytes << " bytes\n"
            << "M=" << problem.m << ", N=" << problem.n << ", K=" << problem.k << "\n"
            << "sample_scope=batch-average per-launch latency\n"
            << "n_batches=" << stats.sample_count
            << ", launches_per_batch=" << options.measurement.launches_per_batch << "\n"
            << "warmup_launches=" << stats.warmup_launches
            << ", warmup_stable=" << (stats.warmup_stable ? "yes" : "no") << "\n"
            << "elapsed_ms_measured_batches=" << stats.elapsed_ms << " ms\n"
            << "p50_ms=" << stats.p50_ms << " ms\n"
            << "p99_ms=" << stats.p99_ms << " ms\n"
            << "min_ms=" << stats.min_ms << " ms\n"
            << "max_ms=" << stats.max_ms << " ms\n"
            << "TFLOPS_p50=" << stats.tflops_p50 << "\n"
            << "TFLOPS_at_p99_latency=" << stats.tflops_at_p99_latency << "\n"
            << "batch_drift_pct=" << stats.drift_pct << " % (last/first quarter medians)\n";
  if (std::abs(stats.drift_pct) > benchmark_policy::kDriftWarningPct) {
    std::cerr << "Warning: latency drift exceeds 5%; inspect GPU state and repeat the run\n";
  }
}

}  // namespace gemm_bench
