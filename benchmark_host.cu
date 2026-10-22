// 阅读入口：解析参数 → 检查问题和设备 → 准备内核 → 校验、测量、输出。
// 编译脚本用 GEMM_KERNEL_HEADER 选择 teach 或 fast。描述符在 prepare() 里建好，
// 计时循环只调用 prepared()。
#include "host/driver.cuh"

#ifndef GEMM_KERNEL_HEADER
#define GEMM_KERNEL_HEADER "gemm/gemm_fast.cuh"
#endif
#include GEMM_KERNEL_HEADER

static_assert(gemm_backend::kInterfaceVersion == 1, "Unsupported GEMM backend interface");

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
gemm_bench::BenchResult run_benchmark(gemm_bench::BenchmarkOptions const& options)
{
  using namespace gemm_bench;
  GemmBuffers buffers(options);  // 初始化和 Host→Device 拷贝都在计时之前完成。
  auto prepared = gemm_backend::prepare(options, buffers);

  auto validate = [&](char const* phase) {
    validate_gemm_output(phase, options.problem, buffers.h_A, buffers.h_B,
                         buffers.h_C, buffers.d_C.data().get());
  };
  PerformanceStats stats = measure_gemm_performance(
      prepared, validate, options.problem, options.measurement);
  return BenchResult{stats, prepared.config};
}
#endif

int main(int argc, char** argv)
{
  using namespace gemm_bench;

  if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) {
    std::cout << "Kernel: " << gemm_backend::kName
              << "\nKernel header: " << GEMM_KERNEL_HEADER << '\n';
    gemm_backend::print_configuration();
    print_usage(argv[0]);
    return EXIT_SUCCESS;
  }

  BenchmarkOptions options = parse_options(argc, argv);
  validate_options(options);
  gemm_backend::validate_problem(options.problem);

  cudaDeviceProp props{};
  int device = 0;
  CUTE_CHECK_ERROR(cudaGetDevice(&device));
  CUTE_CHECK_ERROR(cudaGetDeviceProperties(&props, device));
  if (props.major != 9) {
    std::cerr << "This benchmark requires Hopper / compute capability 9.x\n";
    return EXIT_FAILURE;
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  std::cout << "Kernel: " << gemm_backend::kName
            << "\nKernel header: " << GEMM_KERNEL_HEADER << '\n';
  print_configuration(props, options);
  gemm_backend::print_configuration();
  BenchResult result = run_benchmark(options);
  print_result(options, result);
  return EXIT_SUCCESS;
#else
  std::cerr << "CUTLASS_ARCH_MMA_SM90_SUPPORTED must be enabled\n";
  return EXIT_FAILURE;
#endif
}
