# Hopper FP16 GEMM

Standalone Hopper FP16 GEMM teaching and benchmark repo (CUDA 12.6, CUTLASS/CuTe, `sm_90a`).
Two kernels share one tile, one WGMMA shape, and one coalesced epilogue.

FP16 inputs, accumulation, and outputs. `alpha=1`, `beta=0`. Logical shapes are A(M,K), B(N,K), C(M,N). Layouts are NT or TN.

## Defaults

There is one default, in `gemm/config.cuh`. `build.sh` does not override it.

| | teach | fast |
|---|---|---|
| CTA tile | 128×128×64 | 128×128×64 |
| PIPE (TMA stages) | 3 | 3 |
| WGMMA | `64×128×16` SS, FP16 | same |
| In-flight WGMMA batches per compute warpgroup | 2 | 2 |
| Producer threads | 0 (elected lane of the compute warpgroup issues TMA) | 128 (only thread 0 issues TMA) |
| Compute warpgroups | 1 (128 threads) | 2 (256 threads, split along M) |
| CTA threads | 128 | 384 |
| Epilogue | coalesced 16-byte store | same |

`M`, `N`, and `K` must be multiples of 128, 128, and 64. The empty mbarrier arrive count is the compute thread count. On `fast`, the producer drains every stage it used, and the last WGMMA batch arrives on empty after `wait<0>`.

## Layout

```text
benchmark_host.cu          # parse → check → prepare → sample → print
build.sh                   # build teach, fast, and the full-output checkers
gemm/
  config.cuh               # the only defaults
  smem_tma.cuh             # SharedStorage and TMA/SW128 layout helpers
  epilogue.cuh             # one coalesced writeback
  gemm_teach.cuh           # single warpgroup pipeline
  gemm_fast.cuh            # warp-specialized producer + two compute warpgroups
  launch.cuh               # prepare() builds TMA; operator() only launches
host/
  common.cuh               # problem, NT/TN index, buffers
  driver.cuh               # argv, checks, result printing
  measurement.cuh          # warmup and timing, outside descriptor setup
  validation.cuh           # sampled FP64 check
tests/
  full_output_validation.cu
  run_validation.sh
archive/v1-v5/             # previous v1–v5 sources, not the default path
experiments/               # optional NCU / nsys / sanitizer suite for v1–v5
```

## Build and run

Verified toolchain for this code: **CUDA 12.6 (nvcc 12.6)**, host **g++** with C++17, **CUTLASS 4.x** headers. The compiler must be a CUDA 12 `nvcc` that accepts `sm_90a`. `/usr/bin/nvcc` on some machines is far older; point `CUDACXX` at the CUDA 12 binary.

CUTLASS is resolved in this order: `--cutlass DIR`, then `$CUTLASS_PATH`, then `<repo>/third_party/cutlass` (the checkout root, the directory that contains `include/cute/tensor.hpp`). `$CUTLASS_ROOT` is accepted only as a legacy alias of `CUTLASS_PATH`.

nvcc is resolved in this order: `--nvcc PATH`, then `$CUDACXX`, then `/usr/local/cuda/bin/nvcc`, then `nvcc` on `PATH`.

```bash
git clone --depth 1 https://github.com/NVIDIA/cutlass.git third_party/cutlass
# or: export CUTLASS_PATH=/path/to/cutlass
export CUDACXX=/usr/local/cuda/bin/nvcc

bash build.sh all
./build/gemm_teach --help
./build/gemm_fast  --help

# Sampling check is inside the benchmark (short run, not a performance result).
./build/gemm_teach 256 256 64 N T 3 event 1 1 2
./build/gemm_teach 256 256 832 T N 3 event 1 1 2
./build/gemm_fast  256 256 64 N T 3 event 1 1 2
./build/gemm_fast  256 256 832 T N 3 event 1 1 2

# Performance sample. Measurement does not include allocation or TMA setup.
./build/gemm_fast 4096 4096 4096 N T 2 event 20 20 200

# Host MMA-ownership check, then every C element on a Hopper GPU.
bash tests/run_validation.sh --compile-only
bash tests/run_validation.sh
```

`bash build.sh teach` or `bash build.sh fast` builds one benchmark. Add `--validate` to also build `build/validate_teach` or `build/validate_fast`. `all` always builds both benchmarks and both checkers.

Runtime arguments, tail-optional:

```text
M    N    K    transA transB input_mode timer per_batch min_warmup batches
4096 4096 4096 N      T      2          event 20        20         200
```

`input_mode`: `0` zeros, `1` ones, `2` random ±1, `3` repeating −2..2. Transpose pairs are `N T` or `T N`.

## What to read

1. `benchmark_host.cu` — `prepare()` returns an object whose call operator submits one GEMM.
2. `gemm/config.cuh` — tile, pipe, instruction.
3. `gemm/gemm_teach.cuh` — one warpgroup: prefetch, two in-flight WGMMA batches, coalesced store.
4. `gemm/gemm_fast.cuh` — producer leader, two compute warpgroups, tail drain, final empty arrive.
5. `gemm/epilogue.cuh` and `gemm/smem_tma.cuh` — shared by both.

CuTe stays on the TMA descriptors, the SW128 layouts, and the tiled WGMMA. The instruction N is fixed, so there is no N-selector.

## Checks

The benchmark copies C back, rejects NaN/Inf, and compares 16 scattered entries with a CPU FP64 dot product. Tolerance is `1 + 2% * abs(reference)`. That is a sample, not a full-matrix proof.

`tests/full_output_validation.cu` checks every output for both kernels: NT and TN, four input patterns, and K tile counts around the pipeline depth (including several phase wraps). `--mapping-only` checks that the compute threads cover the CTA tile exactly once and does not launch a kernel.

## Archived v1–v5

`archive/v1-v5/` keeps the old five entry points. `experiments/build.sh` and `experiments/run.sh` still compile and profile that suite (NCU, nsys, sanitizer). They are not required to build or validate `teach` and `fast`. From `experiments/`:

```bash
CUTLASS_PATH=/path/to/cutlass bash build.sh --suite core --with-validation
bash run.sh --mode validate --gpu 0
```

Details: [experiments/H100_BENCHMARK.md](experiments/H100_BENCHMARK.md).
