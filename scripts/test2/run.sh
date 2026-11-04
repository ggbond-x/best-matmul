#!/usr/bin/env bash
# V2 单变量扫描；在 Hopper Linux 环境下执行。
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd -- "$script_dir/../.." && pwd -P)"
project="$(cd -- "$repo/../../../../.." && pwd -P)"
runner="$repo/scripts/test1/run.sh"
cutlass="${CUTLASS_ROOT:-$project/third_party/cutlass}"
nvcc="${CUDACXX:-nvcc}"
gpu="${GPU:-0}"
output="${OUTPUT_DIR:-$repo/reports/test2/$(date +%Y%m%d_%H%M%S)_$$}"
repeats="${REPEATS:-3}"
per_batch="${PER_BATCH:-20}"
min_warmup="${MIN_WARMUP:-300}"
batches="${BATCHES:-100}"
telemetry="${TELEMETRY_MS:-1000}"
dry_run=0
build_only=0

usage() {
  cat <<'EOF'
Usage: bash scripts/test2/run.sh [--dry-run | --build-only] [--help]
  无参数：编译、全输出验证、先测同机 V1/V2 基准，再执行各参数对照。
  --dry-run    显示全部命令与配置，不检查工具/GPU，不创建文件。
  --build-only 只编译 benchmark 和验证程序，执行 CPU mapping 检查，不需要 GPU。
环境变量：CUTLASS_ROOT、CUDACXX、GPU、OUTPUT_DIR、REPEATS、PER_BATCH、
MIN_WARMUP、BATCHES、TELEMETRY_MS。OUTPUT_DIR 必须不存在。
EOF
}
die() { printf 'test2: %s\n' "$*" >&2; exit 1; }
while (($#)); do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --build-only) build_only=1 ;;
    --help|-h) usage; exit 0 ;;
    *) die "未知参数: $1" ;;
  esac
  shift
done
(( !(dry_run && build_only) )) || die '--dry-run 和 --build-only 不能同时使用'
[[ "$output" == /* ]] || output="$PWD/$output"
[[ "$cutlass" == /* ]] || cutlass="$PWD/$cutlass"
for value in "$output" "$cutlass" "$nvcc"; do
  [[ "$value" != *$'\t'* && "$value" != *$'\n'* && "$value" != *$'\r'* && "$value" != *'"'* ]] || die '路径不得包含制表符、换行或双引号'
done
for value in "$repeats" "$per_batch" "$min_warmup" "$batches" "$telemetry"; do
  [[ "$value" =~ ^(0|[1-9][0-9]{0,8})$ ]] || die '计时参数须为不超过 9 位的非负整数'
done
(( repeats >= 2 && per_batch > 0 && min_warmup > 0 && batches >= 2 && (telemetry == 0 || telemetry >= 100) )) || die '计时参数超出范围'

# id M N K PIPE GMMA_N。V1 和 V2 基准除 inflight 外使用相同参数。
configs=(
  'v1_p3_n128_m128n128 128 128 64 3 128'
  'v2_p2_n128_m128n128 128 128 64 2 128'
  'v2_p3_n128_m128n128 128 128 64 3 128'
  'v2_p4_n128_m128n128 128 128 64 4 128'
  'v2_p3_n64_m128n128 128 128 64 3 64'
  'v2_p3_n64_m64n128 64 128 64 3 64'
  'v2_p3_n64_m128n64 128 64 64 3 64'
  'v2_p3_n64_m256n128 256 128 64 3 64'
  'v2_p3_n64_m128n256 128 256 64 3 64'
)
build="$output/build"
shapes=4096x4096x256,4096x4096x4096
manifest_header=$'id\tversion\tcta_m\tcta_n\tcta_k\tpipe\tgmma_n\tcompute_threads\tproducer_threads\tinflight\tepilogue\tbinary'
print_cmd() { printf '%q ' "$@"; printf '\n'; }
execute() {
  local label="$1"; shift
  printf '\n[%s] ' "$label"; print_cmd "$@"
  if (( ! dry_run )); then
    { printf '# %s\n' "$label"; print_cmd "$@"; } >> "$output/commands.sh"
    "$@"
  fi
}

if (( ! dry_run )); then
  [[ -f "$runner" && -f "$repo/build_benchmark.sh" ]] || die '缺少已有的运行或编译脚本'
  [[ -f "$cutlass/include/cute/tensor.hpp" && -f "$cutlass/include/cutlass/half.h" ]] || die "CUTLASS 路径不正确: $cutlass"
  command -v "$nvcc" >/dev/null 2>&1 || die "找不到 nvcc: $nvcc"
  for tool in python3 sha256sum tee timeout; do command -v "$tool" >/dev/null 2>&1 || die "缺少工具: $tool"; done
  if (( ! build_only )); then
    command -v nvidia-smi >/dev/null 2>&1 || die '找不到 nvidia-smi'
    cap="$(nvidia-smi -i "$gpu" --query-gpu=compute_cap --format=csv,noheader)" || die 'GPU 查询失败'
    cap="${cap//[[:space:]]/}"
    [[ "$cap" == 9.0 ]] || die "需要 Hopper compute capability 9.0，当前 GPU 为 $cap"
  fi
  [[ ! -e "$output" ]] || die "输出目录已存在: $output"
  mkdir -p -- "$build"
  output="$(cd -- "$output" && pwd -P)"
  build="$output/build"
  cutlass="$(cd -- "$cutlass" && pwd -P)"
  stage=build
  printf 'RUNNING\n' > "$output/status.txt"
  finish() {
    local status=$?
    if (( status == 0 )); then printf 'COMPLETE\n' > "$output/status.txt"
    else printf 'FAILED stage=%s exit=%s\n' "$stage" "$status" > "$output/status.txt"; fi
  }
  trap finish EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  exec > >(tee -a "$output/run.log") 2>&1
  printf '# 可复现命令；日志重定向由脚本管理。\n' > "$output/commands.sh"
  printf '%s\n' "$manifest_header" > "$build/manifest.tsv"
  {
    printf 'with_validation=1\nbuild_utc=%s\ncutlass=%s\nnvcc=%s\n' "$(date -u +%FT%TZ)" "$cutlass" "$nvcc"
    "$nvcc" --version
    printf '\n[source hashes]\n'
    sha256sum "$script_dir/run.sh" "$runner" "$repo/build_benchmark.sh" "$repo/benchmark_host.cu" "$repo/tests/full_output_validation.cu" "$repo"/host/*.cuh "$repo"/kernels/*.cuh "$repo"/kernels/detail/*.cuh
    if git -C "$project" rev-parse HEAD >/dev/null 2>&1; then
      printf '\n[project git HEAD and tracked changes]\n'
      git -C "$project" rev-parse HEAD
      git -C "$project" status --short --untracked-files=no
    fi
    if git -C "$cutlass" rev-parse HEAD >/dev/null 2>&1; then
      printf '\n[CUTLASS git HEAD and tracked changes]\n'
      git -C "$cutlass" rev-parse HEAD
      git -C "$cutlass" status --short --untracked-files=no
    fi
  } > "$build/build_metadata.txt"
  if (( ! build_only )); then
    nvidia-smi -L > "$output/gpu_list.txt"
    nvidia-smi -i "$gpu" -q > "$output/gpu_before.txt"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv > "$output/processes_before.csv"
  fi
fi

for config in "${configs[@]}"; do
  read -r id m n k pipe gmma <<< "$config"
  version=v2
  inflight=2
  header="$repo/kernels/wgmma_tma_fp16_pipeline_v2.cuh"
  if [[ "$id" == v1_* ]]; then
    version=v1
    inflight=1
    header="$repo/kernels/wgmma_tma_fp16_v1.cuh"
  fi
  binary="$build/$id"
  flags=(-lineinfo -Xptxas=-v "-DPEAK_CTA_M=$m" "-DPEAK_CTA_N=$n" "-DPEAK_CTA_K=$k" "-DPIPE=$pipe" "-DGMMA_INST_N=$gmma" "-DWGMMA_MAX_INFLIGHT=$inflight")
  compile=(bash -x "$repo/build_benchmark.sh" --kernel "$version" --cutlass "$cutlass" --nvcc "$nvcc" --output "$binary" -- "${flags[@]}")
  validate_compile=("$nvcc" -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_90a -I "$cutlass/include" -I "$cutlass/tools/util/include" "-DGEMM_KERNEL_HEADER=\"$header\"" "${flags[@]}" "$repo/tests/full_output_validation.cu" -o "$binary.validate")
  if (( dry_run )); then
    execute "build $id" "${compile[@]}"
    execute "build $id.validate" "${validate_compile[@]}"
    execute "mapping $id" "$binary.validate" --mapping-only
    printf '# manifest: %s %s %s %s %s %s %s 128 0 %s 0 %s\n' "$id" "$version" "$m" "$n" "$k" "$pipe" "$gmma" "$inflight" "$binary"
  else
    execute "build $id" "${compile[@]}" 2>&1 | tee "$build/$id.build.log"
    execute "build $id.validate" "${validate_compile[@]}" 2>&1 | tee "$build/$id.validate.build.log"
    execute "mapping $id" "$binary.validate" --mapping-only | tee "$build/$id.mapping.log"
    sha256sum "$binary" "$binary.validate" > "$build/$id.sha256"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t128\t0\t%s\t0\t%s\n' "$id" "$version" "$m" "$n" "$k" "$pipe" "$gmma" "$inflight" "$binary" >> "$build/manifest.tsv"
  fi
done

if (( ! build_only )); then
  stage=validate
  execute validate bash "$runner" --mode validate --build-dir "$build" --gpu "$gpu" --variants all --output-dir "$output/validation"
  common=(--mode bench --build-dir "$build" --gpu "$gpu" --layouts NT,TN --shapes "$shapes" --repeats "$repeats" --per-batch "$per_batch" --min-warmup "$min_warmup" --batches "$batches" --telemetry-ms "$telemetry")
  stage=baseline
  execute "$stage" bash "$runner" "${common[@]}" --variants v1_p3_n128_m128n128,v2_p3_n128_m128n128 --baseline v1_p3_n128_m128n128 --output-dir "$output/00_baseline"
  stage=pipe
  execute "$stage" bash "$runner" "${common[@]}" --variants v2_p2_n128_m128n128,v2_p3_n128_m128n128,v2_p4_n128_m128n128 --baseline v2_p3_n128_m128n128 --output-dir "$output/01_pipe"
  stage=gmma
  execute "$stage" bash "$runner" "${common[@]}" --variants v2_p3_n64_m128n128,v2_p3_n128_m128n128 --baseline v2_p3_n128_m128n128 --output-dir "$output/02_gmma"
  stage=cta
  execute "$stage" bash "$runner" "${common[@]}" --variants v2_p3_n64_m64n128,v2_p3_n64_m128n64,v2_p3_n64_m128n128 --baseline v2_p3_n64_m128n128 --output-dir "$output/03_cta"
  stage=large_cta
  execute "$stage" bash "$runner" "${common[@]}" --variants v2_p3_n64_m128n128,v2_p3_n64_m256n128,v2_p3_n64_m128n256 --baseline v2_p3_n64_m128n128 --output-dir "$output/04_large_cta"
  if (( ! dry_run )); then
    nvidia-smi -i "$gpu" -q > "$output/gpu_after.txt"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv > "$output/processes_after.csv"
  fi
fi
printf '\n%s: %s\n' "$([[ "$dry_run" == 1 ]] && printf '计划输出目录（未创建）' || printf '输出目录')" "$output"
