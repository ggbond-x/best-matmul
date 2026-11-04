#!/usr/bin/env bash
# Hopper 完整实验；--dry-run 只看计划，--with-nsys 追加小矩阵时间线。
# 默认采集 60 份 cold focused + 1 份 v5 warm full NCU 报告，完整运行可能较久。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# 可用同名环境变量覆盖默认配置；相对路径以脚本目录为基准。
GPU="${GPU:-0}"
BUILD_DIR="${BUILD_DIR:-$PWD/build/h100_pcie}"
EXP_DIR="${EXP_DIR:-$PWD/reports/h100_pcie/manual_$(date +%Y%m%d_%H%M%S)_$$}"
CUTLASS_PATH="${CUTLASS_PATH:-/workspace/peak_flops/third_party/cutlass}"
NVCC_PATH="${NVCC_PATH:-/usr/local/cuda/bin/nvcc}"
NCU="${NCU:-ncu}"
CORE=v1,v2,v3,v4,v5
SWEEP_SHAPES=4096x4096x4096,4096x4096x256
export PATH="$(dirname -- "$NVCC_PATH"):$PATH"
export NCU

dry_run=0
with_nsys=0
while (($#)); do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --with-nsys) with_nsys=1 ;;
    --help|-h)
      cat <<'USAGE'
Usage: bash test.sh [--dry-run] [--with-nsys]
  --dry-run    只打印命令，不检查工具/GPU，不编译，不创建目录
  --with-nsys  追加五个核心版本在 1024^3、NT/TN 下的 NSYS 时间线
  环境变量：GPU、CUTLASS_PATH、NVCC_PATH、NCU、BUILD_DIR、EXP_DIR。
  EXP_DIR 必须不存在，且不能位于 BUILD_DIR 内部。
  示例：CUTLASS_PATH=/opt/cutlass GPU=0 bash test.sh --with-nsys
  正式成绩只取 bench_*，不使用 profiler/Sanitizer 日志中的耗时。
USAGE
      exit 0 ;;
    *) printf '未知参数：%s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done
[[ "$BUILD_DIR" == /* ]] || BUILD_DIR="$PWD/$BUILD_DIR"
[[ "$EXP_DIR" == /* ]] || EXP_DIR="$PWD/$EXP_DIR"

stage=preflight
run_step() {
  stage="$1"; shift
  printf '\n[%s] ' "$stage"; printf '%q ' "$@"; printf '\n'
  if (( ! dry_run )); then "$@"; fi
}

if (( ! dry_run )); then
  # 提前发现缺失工具，避免长时间编译/计时后才发现无法采集报告。
  # NCU 不一定安装在 CUDA bin 目录，可用 NCU=/opt/.../ncu 显式指定。
  for tool in "$NVCC_PATH" "$NCU" nvidia-smi compute-sanitizer cuobjdump python3 timeout sha256sum tee tar cp; do
    command -v "$tool" >/dev/null 2>&1 || { printf '缺少工具：%s\n' "$tool" >&2; exit 1; }
  done
  if (( with_nsys )); then
    command -v nsys >/dev/null 2>&1 || { printf '缺少工具：nsys\n' >&2; exit 1; }
  fi
  for header in cute/tensor.hpp cute/util/debug.hpp cutlass/half.h; do
    [[ -f "$CUTLASS_PATH/include/$header" ]] || { printf '缺少 CUTLASS 头文件：%s\n' "$header" >&2; exit 1; }
  done
  [[ ! -e "$EXP_DIR" ]] || { printf '实验目录已存在：%s\n' "$EXP_DIR" >&2; exit 1; }
  # 归档要复制 BUILD_DIR；禁止递归复制自己的输出，解析 .. 和符号链接后再检查。
  python3 - "$BUILD_DIR" "$EXP_DIR" <<'PY'
import pathlib, sys
build, experiment = map(lambda p: pathlib.Path(p).resolve(), sys.argv[1:])
if experiment == build or build in experiment.parents:
    raise SystemExit('EXP_DIR 不能等于 BUILD_DIR，也不能位于 BUILD_DIR 内部')
PY
  mkdir -p -- "$EXP_DIR"
  exec > >(tee -a "$EXP_DIR/test.log") 2>&1
  printf 'RUNNING\n' > "$EXP_DIR/status.txt"
  finish() {
    local status=$?
    if (( status == 0 )); then
      printf 'COMPLETE\n' > "$EXP_DIR/status.txt"
    else
      printf 'FAILED stage=%s exit=%s\n' "$stage" "$status" > "$EXP_DIR/status.txt"
      printf '实验未完成：阶段=%s，退出码=%s；已有材料保留在 %s\n' "$stage" "$status" "$EXP_DIR"
    fi
  }
  trap finish EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
fi

# 计划和真正编译使用相同路径/参数。共 15 个 benchmark + 15 个校验程序。
build_args=(--cutlass "$CUTLASS_PATH" --nvcc "$NVCC_PATH"
  --suite all --with-validation --output-dir "$BUILD_DIR")
run_step build_plan bash build.sh "${build_args[@]}" --dry-run
run_step build bash build.sh "${build_args[@]}"

# 在耗时的 GPU 分析前归档；后续重编译不会覆盖本次材料。
# 只打包明确列出的源码，不把 reports/build 递归装入源码包。
run_step archive_build cp -a -- "$BUILD_DIR" "$EXP_DIR/build_artifacts"
run_step archive_source tar -czf "$EXP_DIR/source.tar.gz" \
  benchmark_host.cu host kernels tests build.sh build_benchmark.sh run.sh test.sh README.md H100_BENCHMARK.md

run_step environment bash run.sh --mode env --build-dir "$BUILD_DIR" --gpu "$GPU" \
  --output-dir "$EXP_DIR/env"

# 全部配置完整输出校验通过后才计时，错误立即终止。
run_step validate bash run.sh --mode validate --build-dir "$BUILD_DIR" --gpu "$GPU" \
  --variants all --output-dir "$EXP_DIR/validate"

# 保存全部 15 组资源和 SASS，避免后面的 profiler 失败导致材料尚未导出。
run_step disasm bash run.sh --mode disasm --build-dir "$BUILD_DIR" \
  --variants all --output-dir "$EXP_DIR/disasm"

# 五版本、六形状、NT/TN、五轮，完成顺序位置轮换。
# 预热次数不保证热稳态，还需检查 telemetry、warmup_stable 和 drift。
run_step bench_core bash run.sh --mode bench --build-dir "$BUILD_DIR" --gpu "$GPU" \
  --variants "$CORE" --baseline v1 --layouts NT,TN --repeats 5 \
  --per-batch 20 --min-warmup 300 --batches 100 --telemetry-ms 1000 \
  --output-dir "$EXP_DIR/bench_core"

# PIPE 三配置各自比较，三轮完成位置轮换。
for version in v3 v4 v5; do
  run_step "bench_${version}_pipe" bash run.sh --mode bench --build-dir "$BUILD_DIR" --gpu "$GPU" \
    --variants "${version}_p2,${version},${version}_p4" --baseline "$version" \
    --shapes "$SWEEP_SHAPES" --layouts NT,TN --repeats 3 \
    --output-dir "$EXP_DIR/bench_${version}_pipe"
done

# 固定 M tile=256 比较 128/256/512 计算线程，四配置对应四轮。
run_step bench_m256_threads bash run.sh --mode bench --build-dir "$BUILD_DIR" --gpu "$GPU" \
  --variants v4_m256_t128,v5_m256_t128,v5_m256_t256,v5_m256_t512 \
  --baseline v5_m256_t128 --shapes "$SWEEP_SHAPES" --layouts NT,TN --repeats 4 \
  --output-dir "$EXP_DIR/bench_m256_threads"

# 15 配置 × NT/TN × 长/短 K = 60 份 focused 报告，统一 cold/kernel replay 条件。
# 短 K 用于分析启动、排空和写回。每份均保存 ncu-rep/raw.csv/details.txt/log。
for shape in 4096x4096x4096 4096x4096x256; do
  run_step "ncu_cold_${shape}" bash run.sh --mode ncu --build-dir "$BUILD_DIR" --gpu "$GPU" \
    --variants all --layouts NT,TN --profile-shape "$shape" \
    --ncu-set focused --ncu-cache cold --ncu-clock none --launch-skip 50 --timeout 7200 \
    --output-dir "$EXP_DIR/ncu_all_cold_${shape}"
done

# v5 warm full 仅用于深入查看该配置，不与其他版本 cold 指标混作缓存对照。
# 跨版本热缓存比较时，应为相关版本另采相同 warm/形状/布局的报告。
run_step ncu_v5_warm_full bash run.sh --mode ncu --build-dir "$BUILD_DIR" --gpu "$GPU" \
  --variants v5 --layouts NT --profile-shape 4096x4096x4096 \
  --ncu-set full --ncu-cache warm --ncu-clock none --launch-skip 50 --timeout 7200 \
  --output-dir "$EXP_DIR/ncu_v5_warm_full"

# 可选 CPU 提交/GPU 执行时间线；小矩阵表现异常时加 --with-nsys。
if (( with_nsys )); then
  run_step nsys_small_gemm bash run.sh --mode nsys --build-dir "$BUILD_DIR" --gpu "$GPU" \
    --variants "$CORE" --layouts NT,TN --profile-shape 1024x1024x1024 \
    --output-dir "$EXP_DIR/nsys_small_gemm"
fi

# 同步/内存诊断不参与成绩；失败时已经完成的计时/分析材料仍然保留。
run_step sanitize bash run.sh --mode sanitize --build-dir "$BUILD_DIR" --gpu "$GPU" \
  --variants v4,v5,v5_m256_t512 --layouts NT,TN \
  --sanitizers memcheck,racecheck,synccheck --timeout 7200 --output-dir "$EXP_DIR/sanitize"

if (( dry_run )); then
  printf '\n计划检查完成，未执行测试或创建目录。计划实验目录：%s\n' "$EXP_DIR"
else
  printf '\n实验全部完成。分析时请保留整个目录：%s\n' "$EXP_DIR"
fi
