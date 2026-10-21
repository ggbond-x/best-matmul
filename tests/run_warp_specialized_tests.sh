#!/usr/bin/env bash
# Six build configurations, each checking real MMA ownership on the CPU.
# The default additionally checks every GEMM output on a Hopper GPU.
set -euo pipefail

tests_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
modular_dir="$(cd -- "$tests_dir/.." && pwd)"
cutlass_root="${CUTLASS_ROOT:-}"
nvcc="${CUDACXX:-nvcc}"
output_dir="$modular_dir/build/warp_specialized_tests"
compile_only=0

usage() {
  cat <<'USAGE'
Usage: bash tests/run_warp_specialized_tests.sh --cutlass DIR [options]
  --cutlass DIR      CUTLASS root (or set CUTLASS_ROOT)
  --nvcc PATH        CUDA compiler (default: CUDACXX or nvcc)
  --output-dir DIR   Directory for validation binaries
  --compile-only    Compile and check CPU MMA ownership; skip GPU execution
  --help            Show this help

Builds v4 PIPE=1/3 and v5 with 128/256/384/512 compute threads, exercising
PIPE=1/2/3, one/two in-flight groups and both epilogues. GPU cases cover NT/TN,
zero/one/random -1,+1/pattern inputs, and K tile counts around the pipeline
depth and multiple phase wraps. Every output is checked against a CPU result.
USAGE
}

need_value() {
  if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
    printf 'Missing value for %s\n' "$1" >&2
    exit 2
  fi
}
while (( $# )); do
  case "$1" in
    --cutlass) need_value "$@"; cutlass_root="$2"; shift 2 ;;
    --nvcc) need_value "$@"; nvcc="$2"; shift 2 ;;
    --output-dir) need_value "$@"; output_dir="$2"; shift 2 ;;
    --compile-only) compile_only=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -f "$cutlass_root/include/cute/tensor.hpp" ]] || {
  printf 'Pass --cutlass DIR or set CUTLASS_ROOT to a CUTLASS checkout\n' >&2
  exit 2
}
command -v "$nvcc" >/dev/null 2>&1 || { printf 'CUDA compiler not found: %s\n' "$nvcc" >&2; exit 2; }
mkdir -p -- "$output_dir"
include_flags=(-I "$cutlass_root/include")
if [[ -d "$cutlass_root/tools/util/include" ]]; then
  include_flags+=(-I "$cutlass_root/tools/util/include")
fi

run_config() {
  local name="$1" version="$2" threads="$3" cta_m="$4" pipe="$5" inflight="$6" coalesced="$7"
  local header binary
  if [[ "$version" == v4 ]]; then
    header="$modular_dir/kernels/wgmma_tma_fp16_warp_specialized_v4.cuh"
  else
    header="$modular_dir/kernels/wgmma_tma_fp16_multi_warpgroup_v5.cuh"
  fi
  binary="$output_dir/$name"
  printf '\nBuilding %s\n' "$name"
  "$nvcc" -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_90a \
    "${include_flags[@]}" "-DGEMM_KERNEL_HEADER=\"$header\"" \
    "-DWGMMA_COMPUTE_THREADS=$threads" "-DPEAK_CTA_M=$cta_m" \
    -DPEAK_CTA_N=64 -DPEAK_CTA_K=64 -DGMMA_INST_N=64 \
    "-DPIPE=$pipe" "-DWGMMA_MAX_INFLIGHT=$inflight" "-DEPILOGUE_COALESCED=$coalesced" \
    "$tests_dir/full_output_validation.cu" -o "$binary"
  "$binary" --mapping-only
  if (( ! compile_only )); then
    "$binary"
  fi
}

# Distinct CTA tilings include both single and repeated M atoms per warpgroup.
run_config v4_p1_direct v4 128 128 1 1 0
run_config v4_p3_coalesced v4 128 128 3 2 1
run_config v5_t128_p2_coalesced v5 128 128 2 2 1
run_config v5_t256_p1_direct v5 256 128 1 1 0
run_config v5_t384_p3_coalesced v5 384 192 3 2 1
run_config v5_t512_p2_direct v5 512 256 2 2 0
if (( compile_only )); then
  printf '\nAll compile and CPU mapping checks PASS; GPU execution skipped\n'
else
  printf '\nAll compile, CPU mapping and full-output GPU checks PASS\n'
fi
