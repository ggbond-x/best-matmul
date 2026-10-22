#!/usr/bin/env bash
# Hopper 对照实验的统一编译入口；在 Linux 中运行 bash build.sh。
# 本脚本只编译、读取 --help 和检查 CPU 线程映射，不查询 GPU，也不启动 CUDA 内核。
# run.sh 只执行已编译程序；二者通过 manifest.tsv 交换配置，避免文件名与宏值不一致。
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/../../../../.." && pwd -P)"
suite="core"
cutlass_root="${CUTLASS_ROOT:-}"
nvcc="${CUDACXX:-}"
output_dir="$script_dir/build/h100_pcie"
dry_run=0
with_validation=0

usage() {
  cat <<'USAGE'
Usage: bash build.sh [options]
  --suite core|sweep|all  core=5 个版本；sweep=10 个额外配置；all=15 个（默认 core）
  --cutlass DIR          默认 CUTLASS_ROOT 或项目 third_party/cutlass
  --nvcc PATH            默认 CUDACXX、/usr/local/cuda/bin/nvcc 或 PATH 中的 nvcc
  --output-dir DIR       默认 modular/build/h100_pcie；相对路径以当前目录为基准
  --with-validation     为每个配置另编译 .validate 全输出校验程序（默认关闭）
  --dry-run              只显示命令与 manifest 预览，不编译、不创建文件
  --help                 显示帮助

Examples:
  bash build.sh --cutlass /workspace/peak_flops/third_party/cutlass
  bash build.sh --suite all --with-validation --cutlass /home/cutlass
  bash build.sh --suite sweep --dry-run

core: v1/v2/v3/v4/v5，统一 CTA=128x128x64、PIPE=3、GMMA_INST_N=128。
sweep: v3/v4/v5 的 PIPE=2/4，以及统一 CTA=256x128x64 的四个线程对照。
每次成功后 manifest.tsv 仅记录本次所选 suite；需要全部配置时使用 --suite all。
编译诊断、精确命令、--help 配置和 SHA256 保存在输出目录。
USAGE
}
fail() { printf 'build.sh: %s\n' "$1" >&2; exit 2; }
need_value() {
  if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then fail "Missing value for $1"; fi
}
# 只归一化字符串，不要求路径已存在，保证 dry-run 无需 CUDA/CUTLASS。
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
# %q 保留空格、引号等参数边界，生成的命令可直接交给 bash 重新执行。
print_command() { printf '%q ' "$@"; printf '\n'; }
while (( $# )); do
  case "$1" in
    --suite) need_value "$@"; suite="$2"; shift 2 ;;
    --cutlass) need_value "$@"; cutlass_root="$2"; shift 2 ;;
    --nvcc) need_value "$@"; nvcc="$2"; shift 2 ;;
    --output-dir) need_value "$@"; output_dir="$2"; shift 2 ;;
    --with-validation) with_validation=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; fail "Unknown option: $1" ;;
  esac
done
case "$suite" in core|sweep|all) ;; *) fail "--suite must be core, sweep, or all" ;; esac

# 显式参数及环境变量优先；其次寻找项目中常见的第三方目录。
if [[ -z "$cutlass_root" ]]; then
  for candidate in "$repo_root/third_party/cutlass" "$repo_root/third_party/CUTLASS" "$repo_root/third_party"; do
    if [[ -f "$candidate/include/cute/tensor.hpp" ]]; then cutlass_root="$candidate"; break; fi
  done
  cutlass_root="${cutlass_root:-$repo_root/third_party/cutlass}"
fi
if [[ -z "$nvcc" ]]; then
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then nvcc=/usr/local/cuda/bin/nvcc
  else nvcc="$(command -v nvcc 2>/dev/null || true)"; nvcc="${nvcc:-nvcc}"; fi
fi
cutlass_root="$(absolute_path "$cutlass_root")"
output_dir="$(absolute_path "$output_dir")"
[[ "$nvcc" != */* ]] || nvcc="$(absolute_path "$nvcc")"
# manifest 以制表符分列，允许路径含空格，拒绝不能无损表示的控制字符。
for path in "$script_dir" "$cutlass_root" "$output_dir" "$nvcc"; do
  if [[ "$path" == *$'\t'* || "$path" == *$'\r'* || "$path" == *$'\n'* ]]; then
    fail "Paths must not contain a tab, carriage return, or newline"
  fi
done

# 顺序：id version M N K PIPE GMMA_N compute producer inflight epilogue。
# v1/v2 的直接写回、v1 的单批在途由版本固定；仍传显式宏以便审计。
# producer=0 指 v1–v3 未预留专用 warpgroup，同一组线程兼做搬运与计算。
core_variants=(
  'v1 v1 128 128 64 3 128 128 0 1 0'
  'v2 v2 128 128 64 3 128 128 0 2 0'
  'v3 v3 128 128 64 3 128 128 0 2 1'
  'v4 v4 128 128 64 3 128 128 128 2 1'
  'v5 v5 128 128 64 3 128 256 128 2 1'
)
sweep_variants=(
  'v3_p2 v3 128 128 64 2 128 128 0 2 1'
  'v3_p4 v3 128 128 64 4 128 128 0 2 1'
  'v4_p2 v4 128 128 64 2 128 128 128 2 1'
  'v4_p4 v4 128 128 64 4 128 128 128 2 1'
  'v5_p2 v5 128 128 64 2 128 256 128 2 1'
  'v5_p4 v5 128 128 64 4 128 256 128 2 1'
  'v4_m256_t128 v4 256 128 64 3 128 128 128 2 1'
  'v5_m256_t128 v5 256 128 64 3 128 128 128 2 1'
  'v5_m256_t256 v5 256 128 64 3 128 256 128 2 1'
  'v5_m256_t512 v5 256 128 64 3 128 512 128 2 1'
)
variants=()
[[ "$suite" == sweep ]] || variants+=("${core_variants[@]}")
[[ "$suite" == core ]] || variants+=("${sweep_variants[@]}")
manifest_header=$'id\tversion\tcta_m\tcta_n\tcta_k\tpipe\tgmma_n\tcompute_threads\tproducer_threads\tinflight\tepilogue\tbinary'

# 优化等级和架构由 build_benchmark.sh 统一为 -O3 / C++17 / sm_90a。
# lineinfo 服务于 NCU 源码关联；ptxas=-v 记录寄存器、共享内存和 spill。
# 不使用 -G、maxrregcount 或 fast-math，避免给版本对照混入其他优化因素。
make_commands() {
  local destination="$1" validation_destination="$2" header
  case "$version" in
    v1) header="$script_dir/kernels/wgmma_tma_fp16_v1.cuh" ;;
    v2) header="$script_dir/kernels/wgmma_tma_fp16_pipeline_v2.cuh" ;;
    v3) header="$script_dir/kernels/wgmma_tma_fp16_epilogue_v3.cuh" ;;
    v4) header="$script_dir/kernels/wgmma_tma_fp16_warp_specialized_v4.cuh" ;;
    v5) header="$script_dir/kernels/wgmma_tma_fp16_multi_warpgroup_v5.cuh" ;;
  esac
  flags=(-lineinfo -Xptxas=-v
    "-DPEAK_CTA_M=$cta_m" "-DPEAK_CTA_N=$cta_n" "-DPEAK_CTA_K=$cta_k"
    "-DPIPE=$pipe" "-DGMMA_INST_N=$gmma_n" "-DWGMMA_COMPUTE_THREADS=$compute_threads"
    "-DWGMMA_MAX_INFLIGHT=$inflight" "-DEPILOGUE_COALESCED=$epilogue")
  build_command=(bash -x "$script_dir/build_benchmark.sh"
    --kernel "$version" --cutlass "$cutlass_root" --nvcc "$nvcc"
    --output "$destination" -- "${flags[@]}")
  # 校验入口单独指定源码，但架构、优化、指令形状、线程数和所有宏与主程序一致。
  # tools/util/include 即使不存在也可作为 -I 传入；不引入额外源码或库依赖。
  validation_command=("$nvcc" -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_90a
    -I "$cutlass_root/include" -I "$cutlass_root/tools/util/include"
    "-DGEMM_KERNEL_HEADER=\"$header\"" "${flags[@]}"
    "$script_dir/tests/full_output_validation.cu" -o "$validation_destination")
}
manifest_row() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$id" "$version" "$cta_m" "$cta_n" "$cta_k" "$pipe" "$gmma_n" \
    "$compute_threads" "$producer_threads" "$inflight" "$epilogue" "$binary"
}
if (( dry_run )); then
  printf '# suite=%s, variants=%d, with_validation=%s, output=%s\n' "$suite" "${#variants[@]}" "$with_validation" "$output_dir"
  for variant in "${variants[@]}"; do
    read -r id version cta_m cta_n cta_k pipe gmma_n compute_threads producer_threads inflight epilogue <<< "$variant"
    binary="$output_dir/$id"
    make_commands "$binary" "$binary.validate"
    print_command "${build_command[@]}"
    if (( with_validation )); then print_command "${validation_command[@]}"; fi
  done
  printf '\n# manifest.tsv preview (only selected suite)\n%s\n' "$manifest_header"
  for variant in "${variants[@]}"; do
    read -r id version cta_m cta_n cta_k pipe gmma_n compute_threads producer_threads inflight epilogue <<< "$variant"
    binary="$output_dir/$id"; manifest_row
  done
  exit 0
fi

# 真正编译才要求工具链和头文件齐全；机器不需要安装 Hopper GPU。
[[ -f "$script_dir/build_benchmark.sh" ]] || fail "Missing build_benchmark.sh"
for header in cute/tensor.hpp cute/util/debug.hpp cutlass/half.h; do
  [[ -f "$cutlass_root/include/$header" ]] || fail "Missing $cutlass_root/include/$header; pass --cutlass DIR"
done
nvcc_resolved="$(command -v "$nvcc" 2>/dev/null || true)"
[[ -n "$nvcc_resolved" ]] || fail "CUDA compiler not found: $nvcc"
nvcc="$(absolute_path "$nvcc_resolved")"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is required"
if (( with_validation )); then
  [[ -f "$script_dir/tests/full_output_validation.cu" ]] || fail "Missing tests/full_output_validation.cu"
fi
mkdir -p -- "$output_dir"

# manifest 只在所有构建成功后原子替换；编译失败不留下半套配置列表。
# 仅清理当前进程的暂存文件，不删除输出目录内已有的其他版本。
manifest_temp="$output_dir/.manifest.tsv.$$"
temporary_outputs=()
cleanup() {
  rm -f -- "$manifest_temp"
  if (( ${#temporary_outputs[@]} )); then rm -f -- "${temporary_outputs[@]}"; fi
}
trap cleanup EXIT
printf '%s\n' "$manifest_header" > "$manifest_temp"

# 保存可复现环境。UTC 适合合并来自不同机器的结果；git 只读，不改变工作区。
{
  printf 'build_utc=%s\nsuite=%s\nwith_validation=%s\nscript_dir=%s\nrepo_root=%s\noutput_dir=%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$suite" "$with_validation" "$script_dir" "$repo_root" "$output_dir"
  printf 'cutlass_root=%s\nnvcc=%s\nbash=%s\n' "$cutlass_root" "$nvcc" "$BASH_VERSION"
  printf '\n[system]\n'; uname -a
  printf '\n[nvcc --version]\n'; "$nvcc" --version
  if command -v git >/dev/null 2>&1 && git -C "$cutlass_root" rev-parse --git-dir >/dev/null 2>&1; then
    printf '\n[CUTLASS git root]\n'; git -C "$cutlass_root" rev-parse --show-toplevel
    printf '\n[CUTLASS git HEAD]\n'; git -C "$cutlass_root" rev-parse HEAD
    printf '\n[CUTLASS git branch]\n'; git -C "$cutlass_root" symbolic-ref --short -q HEAD || true
    printf '\n[CUTLASS git status, tracked files]\n'; git -C "$cutlass_root" status --short --untracked-files=no -- .
  else printf '\n[CUTLASS git]\nunavailable\n'; fi
  printf '\n[source sha256]\n'
  sha256sum "$script_dir/build.sh" "$script_dir/build_benchmark.sh" "$script_dir/benchmark_host.cu" \
    "$script_dir"/host/*.cuh "$script_dir"/kernels/*.cuh "$script_dir"/kernels/detail/*.cuh
  if (( with_validation )); then sha256sum "$script_dir/tests/full_output_validation.cu"; fi
} > "$output_dir/build_metadata.txt"

# 统一记录命令和编译器诊断；管道启用 pipefail，tee 成功不能掩盖 nvcc 失败。
compile_logged() {
  local stem="$1"; shift
  { printf '#!/usr/bin/env bash\n'; print_command "$@"; } > "$output_dir/$stem.command.sh"
  if ! "$@" 2>&1 | tee "$output_dir/$stem.build.log"; then
    fail "Build failed for $stem; see $output_dir/$stem.build.log (manifest unchanged)"
  fi
}
write_sha() {
  local source="$1" destination="$2" record="$3" binary_sha
  read -r binary_sha _ < <(sha256sum "$source")
  printf '%s  %s\n' "$binary_sha" "$destination" > "$record"
}
for variant in "${variants[@]}"; do
  read -r id version cta_m cta_n cta_k pipe gmma_n compute_threads producer_threads inflight epilogue <<< "$variant"
  binary="$output_dir/$id"
  temporary_binary="$output_dir/.$id.tmp.$$"
  temporary_validation="$output_dir/.$id.validate.tmp.$$"
  [[ ! -e "$temporary_binary" && ! -e "$temporary_validation" ]] || fail "Temporary output already exists for $id"
  temporary_outputs+=("$temporary_binary" "$temporary_validation")
  make_commands "$temporary_binary" "$temporary_validation"
  printf '\nBuilding %s: CTA=%sx%sx%s PIPE=%s compute=%s producer=%s\n' \
    "$id" "$cta_m" "$cta_n" "$cta_k" "$pipe" "$compute_threads" "$producer_threads"
  compile_logged "$id" "${build_command[@]}"
  [[ -x "$temporary_binary" ]] || fail "Compiler did not produce executable: $temporary_binary"
  "$temporary_binary" --help > "$output_dir/$id.help.txt" 2>&1 || fail "--help failed for $id; see $output_dir/$id.help.txt"
  write_sha "$temporary_binary" "$binary" "$output_dir/$id.sha256"
  if (( with_validation )); then
    compile_logged "$id.validate" "${validation_command[@]}"
    [[ -x "$temporary_validation" ]] || fail "Compiler did not produce executable: $temporary_validation"
    # 只检查 CPU 上的真实 CuTe 输出归属；全矩阵 GPU 校验由 run.sh --mode validate 执行。
    "$temporary_validation" --mapping-only > "$output_dir/$id.validate.mapping.txt" 2>&1 || \
      fail "CPU mapping check failed for $id; see $output_dir/$id.validate.mapping.txt"
    write_sha "$temporary_validation" "$binary.validate" "$output_dir/$id.validate.sha256"
    mv -f -- "$temporary_validation" "$binary.validate"
  fi
  mv -f -- "$temporary_binary" "$binary"
  manifest_row >> "$manifest_temp"
done
# 同一目录内 rename，避免 run.sh 读取到只写了一半的 manifest。
mv -f -- "$manifest_temp" "$output_dir/manifest.tsv"
printf '\nBuilt %d variants. Manifest: %s/manifest.tsv\n' "${#variants[@]}" "$output_dir"
