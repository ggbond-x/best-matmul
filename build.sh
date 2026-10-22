#!/usr/bin/env bash
# 仓库根目录的唯一构建入口：编译 teach、fast，以及可选的全输出校验。
# CUTLASS：--cutlass DIR，否则 $CUTLASS_PATH，否则 <repo>/third_party/cutlass。
# $CUTLASS_ROOT 只作为 CUTLASS_PATH 的旧别名。tile / INST_N / PIPE 来自 gemm/config.cuh。
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$script_dir"
target=""
with_validate=0
dry_run=0
cutlass_root="${CUTLASS_PATH:-${CUTLASS_ROOT:-}}"
nvcc="${CUDACXX:-}"
output=""
extra_flags=()

usage() {
  cat <<'USAGE'
Usage: bash build.sh teach|fast|all [options] [-- extra nvcc flags]
  teach|fast|all     要编译的内核。all = 两个基准程序和两个全输出校验程序
  --validate         同时编译该内核的全输出校验程序（all 总是编译）
  --cutlass DIR      CUTLASS 根目录。未指定时用 $CUTLASS_PATH，
                     再否则用 <repo>/third_party/cutlass
  --nvcc PATH        CUDA 编译器。未指定时用 $CUDACXX，
                     再否则用 /usr/local/cuda/bin/nvcc，再否则用 PATH 里的 nvcc
  --output PATH      单个内核的基准程序路径（不能和 all 一起用）
                     默认 build/gemm_teach 或 build/gemm_fast
  --dry-run          只打印 nvcc 命令
  --help             显示帮助

示例（在仓库根目录执行）：
  export CUTLASS_PATH=/path/to/cutlass
  export CUDACXX=/usr/local/cuda/bin/nvcc
  bash build.sh all
  ./build/gemm_teach 4096 4096 4096 N T 2 event 1 1 2
  ./build/gemm_fast  4096 4096 4096 N T 2 event 20 20 200
  ./build/validate_teach
  ./build/validate_fast

默认 CTA=128x128x64，PIPE=3，GMMA 64x128x16，两批在途，合并写回。
teach = 1 个计算 warpgroup（128 线程）。
fast  = 128 个 producer 线程 + 2 个计算 warpgroup（共 384 线程）。
v1–v5 对照实验在 experiments/，不从这里编译。
USAGE
}

fail() { printf 'build.sh: %s\n' "$1" >&2; exit 2; }
need_value() {
  if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then fail "Missing value for $1"; fi
}
absolute_path() {
  local path="$1" part result=""
  local -a parts=()
  [[ "$path" == /* ]] || path="$PWD/$path"
  IFS=/ read -r -a parts <<< "$path"
  for part in "${parts[@]}"; do
    case "$part" in ''|.) ;; ..) result="${result%/*}" ;; *) result+="/$part" ;; esac
  done
  printf '%s\n' "${result:-/}"
}
print_command() { printf '%q ' "$@"; printf '\n'; }

while (( $# )); do
  case "$1" in
    teach|fast|all)
      [[ -z "$target" ]] || fail "Only one of teach, fast, all"
      target="$1"; shift ;;
    --validate) with_validate=1; shift ;;
    --cutlass) need_value "$@"; cutlass_root="$2"; shift 2 ;;
    --nvcc) need_value "$@"; nvcc="$2"; shift 2 ;;
    --output) need_value "$@"; output="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; extra_flags=("$@"); break ;;
    *) usage >&2; fail "Unknown option: $1" ;;
  esac
done
[[ -n "$target" ]] || { usage >&2; fail "Specify teach, fast, or all"; }
if [[ "$target" == all && -n "$output" ]]; then
  fail "--output applies to one kernel; omit it with all"
fi
if [[ "$target" == all ]]; then with_validate=1; fi

if [[ -z "$cutlass_root" ]]; then
  for candidate in "$repo_root/third_party/cutlass" "$repo_root/third_party/CUTLASS"; do
    if [[ -f "$candidate/include/cute/tensor.hpp" ]]; then cutlass_root="$candidate"; break; fi
  done
  cutlass_root="${cutlass_root:-$repo_root/third_party/cutlass}"
fi
if [[ -z "$nvcc" ]]; then
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then nvcc=/usr/local/cuda/bin/nvcc
  else nvcc="$(command -v nvcc 2>/dev/null || true)"; nvcc="${nvcc:-nvcc}"; fi
fi
cutlass_root="$(absolute_path "$cutlass_root")"
[[ "$nvcc" != */* ]] || nvcc="$(absolute_path "$nvcc")"

header_for() {
  case "$1" in
    teach) printf '%s\n' "$repo_root/gemm/gemm_teach.cuh" ;;
    fast) printf '%s\n' "$repo_root/gemm/gemm_fast.cuh" ;;
    *) fail "Unknown kernel $1" ;;
  esac
}

compile_one() {
  local kind="$1" kernel="$2" destination="$3"
  local header include_flags=()
  header="$(header_for "$kernel")"
  include_flags=(-I "$cutlass_root/include")
  if [[ -d "$cutlass_root/tools/util/include" ]]; then
    include_flags+=(-I "$cutlass_root/tools/util/include")
  fi
  local source
  if [[ "$kind" == bench ]]; then
    source="$repo_root/benchmark_host.cu"
  else
    source="$repo_root/tests/full_output_validation.cu"
  fi
  local cmd=("$nvcc" -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_90a
    "${include_flags[@]}"
    "-DGEMM_KERNEL_HEADER=\"$header\""
    "${extra_flags[@]}"
    "$source" -o "$destination")
  print_command "${cmd[@]}"
  if (( dry_run )); then return 0; fi
  "${cmd[@]}"
}

build_kernel() {
  local kernel="$1" bench_out validate_out
  if [[ -n "$output" ]]; then
    bench_out="$output"
  else
    bench_out="$repo_root/build/gemm_$kernel"
  fi
  validate_out="$repo_root/build/validate_$kernel"
  if (( ! dry_run )); then
    mkdir -p -- "$(dirname -- "$bench_out")" "$repo_root/build"
  fi
  printf 'Kernel header: %s\nOutput: %s\n' "$(header_for "$kernel")" "$bench_out"
  compile_one bench "$kernel" "$bench_out"
  if (( with_validate )); then
    printf 'Validation output: %s\n' "$validate_out"
    compile_one validate "$kernel" "$validate_out"
  fi
}

if (( ! dry_run )); then
  for header in cute/tensor.hpp cute/util/debug.hpp cutlass/half.h; do
    [[ -f "$cutlass_root/include/$header" ]] || fail "Missing $cutlass_root/include/$header; set CUTLASS_PATH or pass --cutlass DIR"
  done
  nvcc_resolved="$(command -v "$nvcc" 2>/dev/null || true)"
  [[ -n "$nvcc_resolved" ]] || fail "CUDA compiler not found: $nvcc (set CUDACXX or pass --nvcc)"
  nvcc="$(absolute_path "$nvcc_resolved")"
  [[ -f "$(header_for teach)" && -f "$(header_for fast)" ]] || fail "Missing gemm headers"
fi

case "$target" in
  teach|fast) build_kernel "$target" ;;
  all) build_kernel teach; build_kernel fast ;;
esac
if (( dry_run )); then
  printf 'dry-run only; CUTLASS=%s nvcc=%s\n' "$cutlass_root" "$nvcc"
fi
