// Standalone regression: host-only MMA ownership, then every output on Hopper.
#include "../host/common.cuh"

#ifndef GEMM_KERNEL_HEADER
#define GEMM_KERNEL_HEADER "../kernels/wgmma_tma_fp16_multi_warpgroup_v5.cuh"
#endif
#include GEMM_KERNEL_HEADER

#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <vector>

namespace {

// Use the same MMA atom and warpgroup layout as the adapter, without a GPU.
// Counting ownership catches overlaps, omissions and incorrect thread offsets.
bool check_output_mapping()
{
  using namespace cute;
  using Atom = typename gemm_backend::Sm90F16F16F16SsSelector<
      GMMA_INST_N, GMMA::Major::MN, GMMA::Major::MN>::type;
  auto mma = make_tiled_mma(
      Atom{}, Layout<Shape<Int<gemm_backend::kComputeWarpgroups>, _1, _1>>{});
  auto coordinates = make_identity_tensor(make_shape(Int<PEAK_CTA_M>{}, Int<PEAK_CTA_N>{}));
  std::vector<int> owners(PEAK_CTA_M * PEAK_CTA_N, 0);
  for (int thread = 0; thread < gemm_backend::kComputeThreads; ++thread) {
    auto partition = mma.get_thread_slice(thread).partition_C(coordinates);
    for (int entry = 0; entry < size(partition); ++entry) {
      auto coordinate = partition(entry);
      int row = int(get<0>(coordinate));
      int col = int(get<1>(coordinate));
      if (row < 0 || row >= PEAK_CTA_M || col < 0 || col >= PEAK_CTA_N) {
        std::cerr << "MMA partition out of bounds at thread " << thread << '\n';
        return false;
      }
      ++owners[row + col * PEAK_CTA_M];
    }
  }
  for (std::size_t entry = 0; entry < owners.size(); ++entry) {
    if (owners[entry] != 1) {
      std::cerr << "MMA output ownership count=" << owners[entry]
                << " at linear coordinate " << entry << '\n';
      return false;
    }
  }
  std::cout << "Mapping PASS: " << gemm_backend::kComputeThreads << " compute threads cover "
            << owners.size() << " output coordinates exactly once\n";
  return true;
}

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
bool check_case(bool nt, int k_tiles, gemm_bench::InputMode input)
{
  using namespace gemm_bench;
  BenchmarkOptions options;
  // Multiple CTAs in both dimensions also check CTA offsets and global strides.
  options.problem.m = 2 * PEAK_CTA_M;
  options.problem.n = 2 * PEAK_CTA_N;
  options.problem.k = k_tiles * PEAK_CTA_K;
  options.problem.transA = nt ? 'N' : 'T';
  options.problem.transB = nt ? 'T' : 'N';
  options.input_mode = input;
  gemm_backend::validate_problem(options.problem);
  GemmBuffers buffers(options);

  // A missing output store must fail even for the zero-input case.
  std::fill(buffers.h_C.begin(), buffers.h_C.end(),
            Element(std::numeric_limits<float>::quiet_NaN()));
  buffers.d_C = buffers.h_C;

  bool passed = gemm_backend::with_prepared(options, buffers,
      [&](auto const& launch, LaunchConfig const&) {
        // Run twice to catch stale pipeline state and ensure beta=0 overwrites.
        launch();
        launch();
        CUTE_CHECK_ERROR(cudaGetLastError());
        CUTE_CHECK_ERROR(cudaStreamSynchronize(options.measurement.stream));
        buffers.h_C = buffers.d_C;
        auto const& problem = options.problem;
        for (int col = 0; col < problem.n; ++col) {
          for (int row = 0; row < problem.m; ++row) {
            int expected = 0;
            for (int p = 0; p < problem.k; ++p) {
              expected += int(float(buffers.h_A[problem.a_index(row, p)])) *
                          int(float(buffers.h_B[problem.b_index(col, p)]));
              // All test sums must remain exactly representable in FP16.
              if (std::abs(expected) > 2048) {
                std::cerr << "Reference exceeds exact FP16 integer range; reduce test K\n";
                return false;
              }
            }
            float actual = float(buffers.h_C[problem.c_index(row, col)]);
            if (!std::isfinite(actual) || actual != float(expected)) {
              std::cerr << "Full-output failure: layout=" << (nt ? "NT" : "TN")
                        << ", input=" << int(input) << ", K tiles=" << k_tiles
                        << ", C(" << row << ',' << col << ")=" << actual
                        << ", reference=" << expected << '\n';
              return false;
            }
          }
        }
        return true;
      });
  if (passed) {
    std::cout << "Full-output PASS: " << (nt ? "NT" : "TN")
              << ", input=" << int(input) << ", K tiles=" << k_tiles
              << ", checked=" << options.problem.c_elements() << '\n';
  }
  return passed;
}
#endif

} // namespace

int main(int argc, char** argv)
{
  bool mapping_only = argc == 2 && std::strcmp(argv[1], "--mapping-only") == 0;
  if (argc > 1 && !mapping_only) {
    std::cerr << "Usage: " << argv[0] << " [--mapping-only]\n";
    return EXIT_FAILURE;
  }
  std::cout << "Kernel: " << gemm_backend::kName << '\n';
  gemm_backend::print_configuration();
  if (!check_output_mapping()) return EXIT_FAILURE;
  if (mapping_only) return EXIT_SUCCESS;

  cudaDeviceProp properties{};
  int device = 0;
  CUTE_CHECK_ERROR(cudaGetDevice(&device));
  CUTE_CHECK_ERROR(cudaGetDeviceProperties(&properties, device));
  if (properties.major != 9) {
    std::cerr << "Full-output validation requires Hopper / compute capability 9.x. "
                 "Use --mapping-only for host validation.\n";
    return EXIT_FAILURE;
  }
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  // Deduplicate PIPE=1/2 cases; reject zero K tiles from PIPE-1.
  std::set<int> tile_counts{1, 2, PIPE - 1, PIPE, PIPE + 1, 3 * PIPE + 1};
  tile_counts.erase(0);
  for (bool nt : {true, false}) {
    for (auto input : {gemm_bench::InputMode::Zero, gemm_bench::InputMode::One,
                       gemm_bench::InputMode::RandomPm1, gemm_bench::InputMode::Pattern}) {
      for (int tiles : tile_counts) {
        if (!check_case(nt, tiles, input)) return EXIT_FAILURE;
      }
    }
  }
  std::cout << "All full-output regressions PASS\n";
  return EXIT_SUCCESS;
#else
  std::cerr << "Compile with SM90a support enabled\n";
  return EXIT_FAILURE;
#endif
}
