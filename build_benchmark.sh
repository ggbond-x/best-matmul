#!/usr/bin/env bash
# 编译流程：读参数 → 选择内核 → 检查依赖 → 编译公共入口。
# 内核在编译时确定；更换内核或 -D 参数后需要重新编译。
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
kernel="v1"
cutlass_root="${CUTLASS_ROOT:-}"
nvcc="${CUDACXX:-nvcc}"
output=""
extra_flags=()

usage() {
  cat <<'USAGE'
Usage: bash build_benchmark.sh --cutlass DIR [options] [-- extra nvcc flags]
  --kernel v1|v2|v3|v4|v5|HEADER
                           Kernel alias or header path (default: v1)
                           Relative paths are resolved from the modular directory.
  --cutlass DIR            CUTLASS checkout root (or set CUTLASS_ROOT)
  --nvcc PATH              CUDA compiler (default: CUDACXX or nvcc)
  --output PATH            Output binary (default: modular/build/<kernel basename>)
                           Explicit output paths are relative to the current directory.
  --help                   Show this help

Example:
  bash build_benchmark.sh --kernel v2 --cutlass /home/cutlass \
    --nvcc /usr/local/cuda-12.6/bin/nvcc --output build/gemm_v2 \
    -- -DPIPE=3 -DWGMMA_MAX_INFLIGHT=2
USAGE
}

fail() {
  printf '%s\n' "$1" >&2
  exit 2
}

# 只检查选项是否带值；参数仍由下面的 case 直接赋值，便于阅读和修改。
need_value() {
  if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
    fail "Missing value for $1"
  fi
}

# 1. 读取脚本选项。-- 后的内容原样传给 nvcc，例如 -DPIPE=3。
while (( $# )); do
  case "$1" in
    --kernel)  need_value "$@"; kernel="$2"; shift 2 ;;
    --cutlass) need_value "$@"; cutlass_root="$2"; shift 2 ;;
    --nvcc)    need_value "$@"; nvcc="$2"; shift 2 ;;
    --output)  need_value "$@"; output="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; extra_flags=("$@"); break ;;
    *) usage >&2; fail "Unknown option: $1" ;;
  esac
done

# 2. 内核相对路径以本脚本所在目录为基准，不受调用位置影响。
case "$kernel" in
  v1) kernel="kernels/wgmma_tma_fp16_v1.cuh" ;;
  v2) kernel="kernels/wgmma_tma_fp16_pipeline_v2.cuh" ;;
  v3) kernel="kernels/wgmma_tma_fp16_epilogue_v3.cuh" ;;
  v4) kernel="kernels/wgmma_tma_fp16_warp_specialized_v4.cuh" ;;
  v5) kernel="kernels/wgmma_tma_fp16_multi_warpgroup_v5.cuh" ;;
esac
if [[ "$kernel" != /* ]]; then
  kernel="$script_dir/$kernel"
fi
[[ -f "$kernel" ]] || fail "Kernel header does not exist: $kernel"
kernel="$(cd -- "$(dirname -- "$kernel")" && pwd)/$(basename -- "$kernel")"
# 路径将成为 C++ 的 #include 字符串，因此不能含双引号或换行。
if [[ "$kernel" == *'"'* || "$kernel" == *$'\n'* ]]; then
  fail "Kernel path must not contain a double quote or newline"
fi

# 3. 检查 CUTLASS 根目录和编译器。tools/util 为可选依赖。
[[ -n "$cutlass_root" ]] || fail "Missing CUTLASS root: set --cutlass DIR or CUTLASS_ROOT"
[[ -d "$cutlass_root" ]] || fail "CUTLASS directory does not exist: $cutlass_root"
cutlass_root="$(cd -- "$cutlass_root" && pwd)"
for header in cute/tensor.hpp cute/util/debug.hpp cutlass/half.h; do
  [[ -f "$cutlass_root/include/$header" ]] ||
    fail "Missing CUTLASS header: $cutlass_root/include/$header (pass the root containing include/)"
done
include_flags=(-I "$cutlass_root/include")
if [[ -d "$cutlass_root/tools/util/include" ]]; then
  include_flags+=(-I "$cutlass_root/tools/util/include")
fi
command -v "$nvcc" >/dev/null 2>&1 || fail "CUDA compiler not found: $nvcc"

if [[ -z "$output" ]]; then
  output="$script_dir/build/$(basename -- "${kernel%.*}")"
fi
mkdir -p -- "$(dirname -- "$output")"

# 4. 只编译一个入口；宏值保留双引号，使 #include GEMM_KERNEL_HEADER 生效。
# 用数组保留每个参数的边界，含空格的路径也能正确传递。
printf 'Kernel header: %s\nOutput: %s\n' "$kernel" "$output"
"$nvcc" -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_90a \
  "${include_flags[@]}" \
  "-DGEMM_KERNEL_HEADER=\"$kernel\"" \
  "${extra_flags[@]}" \
  "$script_dir/benchmark_host.cu" -o "$output"
