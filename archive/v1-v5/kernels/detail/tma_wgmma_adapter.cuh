#pragma once
#include "../../../../host/common.cuh"
#include <cutlass/cluster_launch.hpp>

// CPU 侧准备：形状/布局 -> TMA 描述符 -> 启动参数 -> launch 回调。
// Tensor、TMA 和动态共享内存都取决于设备实现，因此放在 kernel 目录下；
// 公共 host 只接收一个 launch()，负责预热、计时和校验。
namespace gemm_backend {
// Major::MN：M/N 维连续，stride=(1,ld)，对应 NT。
// Major::K ：K 维连续，stride=(ld,1)，对应 TN。
// 两种 stride 的类型不同，if constexpr 只实例化所选类型。
template <GMMA::Major Major>
auto make_operand_stride(int leading_dimension)
{
  static_assert(Major == GMMA::Major::MN || Major == GMMA::Major::K);
  if constexpr (Major == GMMA::Major::MN) {
    return make_stride(Int<1>{}, leading_dimension);
  } else {
    return make_stride(leading_dimension, Int<1>{});
  }
}

// 将 SW128 基础布局铺满 (M或N, K, PIPE)。布局、WGMMA Major 与 TMA
// 描述符必须一致，否则“同一坐标”会落到不同共享内存地址。
template <GMMA::Major Major, class T, class Shape>
auto make_operand_smem_layout(Shape const& shape)
{
  static_assert(Major == GMMA::Major::MN || Major == GMMA::Major::K);
  if constexpr (Major == GMMA::Major::MN) {
    constexpr int mn = decltype(size<0>(Shape{}))::value;
    if constexpr ((mn & (mn - 1)) != 0) {
      // M=192 等非 2 次幂 tile：先铺 K，再铺 M/N，避免将 factor=3/5/7
      // 插进 swizzle 的连续地址模式。TMA 可按完整的 64 x K slab 分次搬运。
      return tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, shape, Step<_2, _1, _3>{});
    } else {
      return tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, shape);
    }
  } else {
    return tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, shape);
  }
}

template <GMMA::Major Major, class Runner>
decltype(auto) with_prepared_impl(gemm_bench::GemmProblem const& problem,
                                  gemm_bench::GemmBuffers& buffers,
                                  cudaStream_t stream, Runner const& runner)
{
  using Element = gemm_bench::Element;

  // 1. 选择 CTA 形状、内存布局和 WGMMA 指令。
  // Int<N> 将尺寸保留为编译期常量，CUTE 据此生成展开后的设备代码。
  auto prob_shape = make_shape(problem.m, problem.n, problem.k);
  auto bM = Int<PEAK_CTA_M>{};
  auto bN = Int<PEAK_CTA_N>{};
  auto bK = Int<PEAK_CTA_K>{};
  auto bP = Int<PIPE>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto dA = make_operand_stride<Major>(problem.ldA());
  auto dB = make_operand_stride<Major>(problem.ldB());
  auto dC = make_stride(Int<1>{}, problem.ldC());
  auto sA = make_operand_smem_layout<Major, Element>(make_shape(bM, bK, bP));
  auto sB = make_operand_smem_layout<Major, Element>(make_shape(bN, bK, bP));
  using GmmaAtom = typename Sm90F16F16F16SsSelector<GMMA_INST_N, Major, Major>::type;
  // v1-v4 使用一个计算 warpgroup；v5 沿 M 方向将输出分给多个 warpgroup。
  TiledMMA tiled_mma = make_tiled_mma(
      GmmaAtom{}, Layout<Shape<Int<kComputeWarpgroups>, _1, _1>>{});
  CUTE_STATIC_ASSERT_V(size(tiled_mma) == Int<kComputeThreads>{});

  // 2. TMA 描述符编码全局地址、stride 和共享内存 swizzle。
  // 只描述一个 PIPE 槽位；设备主循环决定每次加载哪个 K tile、写入哪个槽。
  // 这部分只执行一次，不计入后面的逐批测量。
  auto const* A = buffers.d_A.data().get();
  auto const* B = buffers.d_B.data().get();
  auto* C = buffers.d_C.data().get();
  Tensor mA = make_tensor(A, make_shape(problem.m, problem.k), dA);
  Tensor mB = make_tensor(B, make_shape(problem.n, problem.k), dB);
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));
  Element alpha(1.0f);
  Element beta(0.0f); // 每次覆盖 C，避免批次之间递归累加。

  // 3. 决定启动规模和动态共享内存大小。
  using Storage = SharedStorage<Element, Element, decltype(sA), decltype(sB)>;
  gemm_bench::LaunchConfig config;
  int mainloop_bytes = int(sizeof(Storage));
  // v3 在计算完成后复用 A/B 内存暂存 C，二者不会同时存活，取较大值即可。
  int epilogue_bytes = kCoalescedEpilogue
      ? PEAK_CTA_M * PEAK_CTA_N * int(sizeof(Element)) : 0;
  config.smem_bytes = std::max(mainloop_bytes, epilogue_bytes);
  config.block = dim3(kProducerThreads + size(tiled_mma));
  config.grid = dim3(round_up(size(ceil_div(problem.m, bM)), config.cluster.x),
                     round_up(size(ceil_div(problem.n, bN)), config.cluster.y));
  auto* kernel = &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                             Element, decltype(sA), decltype(tmaA),
                             Element, decltype(sB), decltype(tmaB),
                             Element, decltype(dC), decltype(tiled_mma),
                             decltype(alpha), decltype(beta)>;
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      config.smem_bytes));
  cutlass::ClusterLaunchParams params = {config.grid, config.block, config.cluster, config.smem_bytes};
  params.cuda_stream = stream; // 与计时 Event 和同步使用同一条流。

  // 4. 把带有 CUTE 类型的启动参数封装成零参数 launch()。
  // runner 会在当前作用域内反复调用它，计时框架无需理解这些模板类型。
  auto launch = [&]() {
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        params, reinterpret_cast<void const*>(kernel),
        prob_shape, cta_tiler, A, tmaA, B, tmaB, C, dC, tiled_mma, alpha, beta);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Error: Failed at kernel launch\n";
      std::exit(EXIT_FAILURE);
    }
  };
  // runner 必须在返回前完成对 launch 的使用，不能将它保存到外部：
  // 回调按引用捕获的形状、描述符等参数会在本函数返回时销毁。
  return runner(launch, config);
}

// 设备写回不带边界掩码，因此 M/N/K 必须分别整除 CTA 的对应维度。
// 正数等通用参数合法性由公共 host 检查。
inline void validate_problem(gemm_bench::GemmProblem const& problem)
{
  if (problem.m % PEAK_CTA_M != 0 || problem.n % PEAK_CTA_N != 0 ||
      problem.k % PEAK_CTA_K != 0) {
    std::cerr << "Selected kernel requires M % CTA_M == 0, N % CTA_N == 0, K % CTA_K == 0\n";
    std::exit(EXIT_FAILURE);
  }
}

inline void print_configuration()
{
  std::printf("Selected GMMA instruction: SM90_64x%dx16_F16F16F16_SS\n", GMMA_INST_N);
  std::printf("CTA tile: PEAK_CTA_M=%d, PEAK_CTA_N=%d, PEAK_CTA_K=%d, PIPE=%d\n",
              PEAK_CTA_M, PEAK_CTA_N, PEAK_CTA_K, PIPE);
  std::printf("WGMMA_MAX_INFLIGHT=%d, EPILOGUE_COALESCED=%d\n",
              kMaxInflight, int(kCoalescedEpilogue));
  std::printf("Producer threads=%d, WGMMA compute threads=%d (%d warpgroups), CTA threads=%d\n",
              kProducerThreads, kComputeThreads, kComputeWarpgroups,
              kProducerThreads + kComputeThreads);
}

// backend 入口：先按 NT/TN 选择编译好的布局，再准备并调用 runner。
// 布局分支只执行一次，正式测量中的 launch() 没有这个分支。
// decltype(auto) 保留 runner 返回值的原始类型，包括可能的引用类型。
template <class Runner>
decltype(auto) with_prepared(gemm_bench::BenchmarkOptions const& options,
                             gemm_bench::GemmBuffers& buffers, Runner const& runner)
{
  if (options.problem.is_nt()) {
    return with_prepared_impl<GMMA::Major::MN>(
        options.problem, buffers, options.measurement.stream, runner);
  }
  return with_prepared_impl<GMMA::Major::K>(
      options.problem, buffers, options.measurement.stream, runner);
}
}  // namespace gemm_backend
