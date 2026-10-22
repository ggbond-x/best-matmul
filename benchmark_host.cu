// 阅读入口：解析参数 → 检查问题和设备 → 准备内核 → 校验、测量、输出。
// 编译脚本用 GEMM_KERNEL_HEADER 选择内核；主机流程不随 v1-v5 改变。
#include "host/driver.cuh"

#ifndef GEMM_KERNEL_HEADER
#define GEMM_KERNEL_HEADER "kernels/wgmma_tma_fp16_v1.cuh"
#endif
#include GEMM_KERNEL_HEADER

static_assert(gemm_backend::kInterfaceVersion == 1, "Unsupported GEMM backend interface");

// 将一次 benchmark 的数据和回调放在同一个作用域中，保证它们始终有效。
// 这里的两个回调分别说明“如何校验”和“准备完成后做什么”。
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
gemm_bench::BenchResult run_benchmark(gemm_bench::BenchmarkOptions const& options)
{
  using namespace gemm_bench;
  GemmBuffers buffers(options); // 初始化和 Host→Device 拷贝都在计时之前完成。
  
  // validate 回调：如何校验输出
  auto validate = [&](char const* phase) {
    validate_gemm_output(
      phase, 
      options.problem,
      buffers.h_A, 
      buffers.h_B,
      buffers.h_C, 
      buffers.d_C.data().get());
  };

  // measure 回调：如何组织测量
  auto measure = [&](auto const& launch, LaunchConfig const& config) 
  {
    PerformanceStats stats = measure_gemm_performance(
        launch, 
        validate, 
        options.problem, 
        options.measurement);
        
    return BenchResult{stats, config};
  };

  // 适配器先构造 TMA 描述符和启动参数，再同步调用 measure。
  // launch 引用了这些参数，必须在 with_prepared 返回前完成全部使用。
  return gemm_backend::with_prepared(options, buffers, measure);
}
#endif

int main(int argc, char** argv)
{
  using namespace gemm_bench;
  
  // 1. --help 只展示接口和编译配置，不要求机器上有 Hopper GPU。
  if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) 
  {
    std::cout << "Kernel: " << gemm_backend::kName
              << "\nKernel header: " << GEMM_KERNEL_HEADER << '\n';
    gemm_backend::print_configuration();
    print_usage(argv[0]);
    return EXIT_SUCCESS;
  }
  
  // 2. 先检查公共参数，再检查所选内核的 tile 等专属约束。
  BenchmarkOptions options = parse_options(argc, argv);
  validate_options(options);
  gemm_backend::validate_problem(options.problem);

  // 3. 此处只支持 Hopper。失败时返回非零，避免被脚本记成有效性能数据。
  cudaDeviceProp props{};
  int device = 0;
  CUTE_CHECK_ERROR(cudaGetDevice(&device));
  CUTE_CHECK_ERROR(cudaGetDeviceProperties(&props, device));
  if (props.major != 9 || props.minor != 0) {
    std::cerr << "This benchmark requires Hopper / compute capability 9.0; device "
              << props.name << " has compute capability " << props.major << '.' << props.minor << '\n';
    return EXIT_FAILURE;
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  std::cout << "Kernel: " << gemm_backend::kName
            << "\nKernel header: " << GEMM_KERNEL_HEADER << '\n';
  
  print_configuration(props, options);
  gemm_backend::print_configuration();
  
  // 4. 完整流程包含预检查、预热、正式采样和最终校验，最后统一输出结果。
  BenchResult result = run_benchmark(options);
  print_result(options, result);
  
  return EXIT_SUCCESS;
#else
  std::cerr << "CUTLASS_ARCH_MMA_SM90_SUPPORTED must be enabled\n";
  return EXIT_FAILURE;
#endif
}
