#!/usr/bin/env bash
#=============================================================================
# run_cublas_all.sh
#
# 一键运行 test_paln.md 定义的 24 组核心 GEMM Shape（A/B/C/D 组）与
# 8 组扩展 Shape（E/F 组）的 cuBLAS 基准测试，结果汇总输出为 CSV。
#
# 用法：
#   ./run_cublas_all.sh                          # 编译（如需要）并运行 32 Shape × 4 转置
#   ./run_cublas_all.sh --dry-run                # 只打印将执行的命令，不实际运行
#   ./run_cublas_all.sh --skip-build             # 跳过编译，直接使用已有可执行文件
#   DTYPES="fp16 bf16" ./run_cublas_all.sh       # 通过环境变量扩展测试的数据类型
#   TRANSPOSES="NN NT TN TT" ./run_cublas_all.sh # 可选择转置子集，默认 NN NT TN TT
#
# 输出目录：output/matmul_cublas_<时间戳>/
#   results.csv    完整基准数据（gemm-v1 格式，每个用例一行，含 TFLOPS/校验/版本）
#   summary.csv    按测试计划 ShapeID 汇总的简表（M/N/K、耗时、TFLOPS、状态）
#   index.csv      ShapeID 与命令行参数、进程退出码对照
#   run.log        全部命令与控制台输出
#   run_info.txt   环境/构建元数据（nvcc 版本、GPU、二进制哈希等）
#
# 每个用例使用独立进程运行（进程隔离），单个用例崩溃不影响后续用例。
#=============================================================================
set -uo pipefail

#==================== 环境相关配置（不同机器按需修改） ====================#
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"             # CUDA Toolkit 根目录
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"                   # nvcc 路径（注意 /usr/bin/nvcc 可能是旧版本）
CUBLAS_LIB_DIR="${CUBLAS_LIB_DIR:-$CUDA_HOME/lib64}"  # libcublas.so / libcublasLt.so 所在目录
CMAKE_BIN="${CMAKE_BIN:-cmake}"                       # cmake 可执行文件路径
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
GPU_ARCHS="${GPU_ARCHS:-86;90}"                       # 目标 SM 架构：H100 用 "90"，RTX30 系用 "86;90"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"        # 编译并行线程数

#==================== 构建与可执行文件 ====================#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"             # 更换 NVCC/GPU_ARCHS 后建议改用新目录，避免 CMake 缓存混用
BENCH_EXE="${BENCH_EXE:-}"                            # 手动指定已编译的 gemm_bench；留空则用 $BUILD_DIR/gemm_bench
SKIP_BUILD=0                                          # 1 = 跳过编译

#==================== 测试参数 ====================#
TRANSPOSES="${TRANSPOSES:-NN NT TN TT}"                # A/B 转置组合，空格分隔；可选 NN NT TN TT
DTYPES="${DTYPES:-fp16}"                              # 可选 fp16 bf16 fp32 tf32，空格分隔可多个
BACKEND="${BACKEND:-cublas}"                          # cublas | cublaslt
MODE="${MODE:-launch}"                                # launch | graph
WARMUP="${WARMUP:-20}"
REPEATS="${REPEATS:-50}"
INNER="${INNER:-10}"
DEVICE="${DEVICE:-0}"
SEED="${SEED:-1234}"

#==================== 输出位置 ====================#
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/output}"        # 每次运行创建 matmul_cublas_<时间戳> 子目录
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUTPUT_ROOT/matmul_cublas_${RUN_TAG}"

#==================== 测试 Shape（test_paln.md：24 组核心 + 8 组扩展） ============#
# 格式: "ID M N K"
SHAPES=(
  # ---- A 组：Square GEMM（6 组核心） ----
  "A1 256 256 256"
  "A2 512 512 512"
  "A3 1024 1024 1024"
  "A4 2048 2048 2048"
  "A5 4096 4096 4096"
  "A6 8192 8192 8192"
  # ---- B 组：K 维度变化（5 组核心，固定 M=N=4096） ----
  "B1 4096 4096 128"
  "B2 4096 4096 256"
  "B3 4096 4096 512"
  "B4 4096 4096 1024"
  "B5 4096 4096 8192"
  # ---- C 组：M 维度变化（6 组核心，固定 N=K=4096） ----
  "C1 1 4096 4096"
  "C2 8 4096 4096"
  "C3 32 4096 4096"
  "C4 128 4096 4096"
  "C5 512 4096 4096"
  "C6 2048 4096 4096"
  # ---- D 组：LLM 推理常见形状（7 组核心） ----
  "D1 128 12288 4096"
  "D2 2048 12288 4096"
  "D3 2048 11008 4096"
  "D4 2048 4096 11008"
  "D5 32 14336 4096"
  "D6 32 4096 14336"
  "D7 512 32000 4096"
  # ---- E 组：N 维度变化（4 组扩展，固定 M=K=4096） ----
  "E1 4096 512 4096"
  "E2 4096 1024 4096"
  "E3 4096 8192 4096"
  "E4 4096 16384 4096"
  # ---- F 组：非对齐 Shape（4 组扩展） ----
  "F1 127 4096 4096"
  "F2 129 4096 4096"
  "F3 1024 4095 4096"
  "F4 1024 4096 4095"
)

#==================== 以下为脚本逻辑，一般无需修改 ====================#
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    *) echo "未知参数: $arg（支持 --dry-run --skip-build）" >&2; exit 2 ;;
  esac
done

read -ra DTYPES_ARR <<< "$DTYPES"
for d in "${DTYPES_ARR[@]}"; do
  case "$d" in fp16|bf16|fp32|tf32) ;; *) echo "非法 dtype: $d（可选 fp16 bf16 fp32 tf32）" >&2; exit 2 ;; esac
done
read -ra TRANSPOSES_ARR <<< "$TRANSPOSES"
if (( ${#TRANSPOSES_ARR[@]} == 0 )); then
  echo "TRANSPOSES 不能为空（可选 NN NT TN TT）" >&2; exit 2
fi
for trans in "${TRANSPOSES_ARR[@]}"; do
  case "$trans" in NN|NT|TN|TT) ;; *) echo "非法转置模式: $trans（可选 NN NT TN TT）" >&2; exit 2 ;; esac
done
case "$BACKEND" in cublas|cublaslt) ;; *) echo "非法 BACKEND: $BACKEND（可选 cublas cublaslt）" >&2; exit 2 ;; esac
case "$MODE" in launch|graph) ;; *) echo "非法 MODE: $MODE（可选 launch graph）" >&2; exit 2 ;; esac

if [[ -z "$BENCH_EXE" ]]; then
  BENCH_EXE="$BUILD_DIR/gemm_bench"
fi

exit_meaning() {
  case "$1" in
    0) echo "OK" ;;
    3) echo "SKIP" ;;
    1) echo "FAIL" ;;
    2) echo "SETUP_ERROR" ;;
    130|137) echo "INTERRUPTED" ;;
    *) echo "ERROR_$1" ;;
  esac
}

build_bench() {
  if [[ ! -x "$NVCC" ]]; then
    echo "错误: 找不到 nvcc: $NVCC（请修改脚本顶部 NVCC 变量）" >&2
    return 1
  fi
  echo "== 配置 CMake（nvcc=$NVCC arch=$GPU_ARCHS）=="
  "$CMAKE_BIN" -S "$ROOT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    -DGEMM_CUDA_ARCHITECTURES="$GPU_ARCHS" || return 1
  echo "== 编译 gemm_bench（-j$JOBS）=="
  "$CMAKE_BIN" --build "$BUILD_DIR" -j "$JOBS"
}

generate_summary() {
  local index_csv="$RUN_DIR/index.csv"
  local results_csv="$RUN_DIR/results.csv"
  local summary_csv="$RUN_DIR/summary.csv"
  [[ -f "$index_csv" ]] || return 0
  [[ -f "$results_csv" ]] || return 0
  awk '
    function clean(s) { gsub(/"/, "", s); return s }
    FNR==NR {   # index.csv：普通 CSV（字段内无逗号）
      if (FNR==1) next
      n=split($0, b, ",")
      key=b[2]","b[3]","b[4]","toupper(b[5])","b[6]","b[7]","b[8]
      ord[++cnt]=key
      id[key]=b[1]; xm[key]=b[2]; xn[key]=b[3]; xk[key]=b[4]; xd[key]=b[5]
      xta[key]=b[7]; xtb[key]=b[8]
      xex[key]=b[9]" ("b[10]")"
      next
    }
    FNR==1 {    # results.csv 表头：带引号 CSV，定位各列下标
      n=split($0, a, /","/)
      for (i=1; i<=n; i++) { h=a[i]; gsub(/"/, "", h); col[h]=i }
      cm=col["M"]; cn=col["N"]; ck=col["K"]; cd=col["Dtype"]
      cb=col["Backend"]; ca=col["TransA"]; cbt=col["TransB"]
      ct=col["Time_ms"]; cf=col["TFLOPS"]; cs=col["Status"]; cr=col["Reason"]
      print "ShapeID,M,N,K,Dtype,TransA,TransB,Time_ms,TFLOPS,Status,Reason"
      next
    }
    {          # results.csv 数据行：区分 Backend 和 TransA/TransB，避免转置结果覆盖
      n=split($0, a, /","/)
      key=clean(a[cm])","clean(a[cn])","clean(a[ck])","toupper(clean(a[cd]))","clean(a[cb])","clean(a[ca])","clean(a[cbt])
      rtime[key]=clean(a[ct]); rtf[key]=clean(a[cf])
      rst[key]=clean(a[cs]); rrs[key]=clean(a[cr]); has[key]=1
    }
    END {
      for (j=1; j<=cnt; j++) {
        key=ord[j]
        if (key in has)
          printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", id[key], xm[key], xn[key], xk[key], xd[key], xta[key], xtb[key], rtime[key], rtf[key], rst[key], rrs[key]
        else
          printf "%s,%s,%s,%s,%s,%s,%s,,,,%s\n", id[key], xm[key], xn[key], xk[key], xd[key], xta[key], xtb[key], "NO_ROW(exit=" xex[key] ")"
      }
    }' "$index_csv" "$results_csv" > "$summary_csv.tmp" && mv "$summary_csv.tmp" "$summary_csv"
}

# 运行前把 cuBLAS 库目录放到最前，避免链接到系统里错误版本的 libcublas
export LD_LIBRARY_PATH="$CUBLAS_LIB_DIR:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "==================== cuBLAS 全量基准测试 ===================="
echo "nvcc        : $NVCC"
echo "cublas 库   : $CUBLAS_LIB_DIR"
echo "可执行文件  : $BENCH_EXE"
echo "GPU 架构    : $GPU_ARCHS"
echo "dtype/后端  : ${DTYPES_ARR[*]} / $BACKEND (mode=$MODE)"
echo "转置模式    : ${TRANSPOSES_ARR[*]}"
echo "用例总数    : $(( ${#SHAPES[@]} * ${#DTYPES_ARR[@]} * ${#TRANSPOSES_ARR[@]} ))（每 dtype 各 ${#SHAPES[@]} Shape × ${#TRANSPOSES_ARR[@]} 转置）"
echo "输出目录    : $RUN_DIR"

if (( DRY_RUN )); then
  echo "==================== dry-run：仅打印命令 ===================="
  for dtype in "${DTYPES_ARR[@]}"; do
    for trans in "${TRANSPOSES_ARR[@]}"; do
      ta="${trans:0:1}"; tb="${trans:1:1}"
      for entry in "${SHAPES[@]}"; do
        read -r sid m n k <<< "$entry"
        echo "$BENCH_EXE --m $m --n $n --k $k --backend $BACKEND --dtype $dtype --transa $ta --transb $tb" \
             "--mode $MODE --warmup $WARMUP --repeats $REPEATS --inner $INNER --seed $SEED --device $DEVICE" \
             "--csv $RUN_DIR/results.csv   # $sid $trans"
      done
    done
  done
  exit 0
fi

if (( ! SKIP_BUILD )); then
  build_bench || { echo "构建失败" >&2; exit 2; }
fi
if [[ ! -x "$BENCH_EXE" ]]; then
  echo "错误: 可执行文件不存在或不可执行: $BENCH_EXE" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"

trap 'echo "被中断，已完成的结果保留在: $RUN_DIR"' INT

# 记录本次运行的环境元数据
{
  echo "run_tag        : $RUN_TAG"
  echo "utc            : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host           : $(hostname)"
  echo "kernel         : $(uname -srmo)"
  echo "nvcc           : $NVCC"
  "$NVCC" --version 2>&1 | tail -2
  echo "cmake          : $("$CMAKE_BIN" --version 2>/dev/null | head -1)"
  echo "cublas_lib_dir : $CUBLAS_LIB_DIR"
  echo "gpu_archs      : $GPU_ARCHS"
  echo "bench_exe      : $BENCH_EXE"
  echo "bench_sha256   : $(sha256sum "$BENCH_EXE" 2>/dev/null | cut -d" " -f1)"
  echo "dtypes         : ${DTYPES_ARR[*]}"
  echo "backend        : $BACKEND"
  echo "mode           : $MODE"
  echo "warmup/repeats/inner : $WARMUP / $REPEATS / $INNER"
  echo "device         : $DEVICE   seed: $SEED"
  echo "layout         : row_major"
  echo "transposes     : ${TRANSPOSES_ARR[*]}"
  echo "--------------- nvidia-smi ---------------"
  nvidia-smi 2>&1
} > "$RUN_DIR/run_info.txt"

echo "ShapeID,M,N,K,Dtype,Backend,TransA,TransB,ExitCode,ExitMeaning" > "$RUN_DIR/index.csv"
: > "$RUN_DIR/run.log"

total=$(( ${#SHAPES[@]} * ${#DTYPES_ARR[@]} * ${#TRANSPOSES_ARR[@]} ))
idx=0; n_pass=0; n_fail=0; n_skip=0
for dtype in "${DTYPES_ARR[@]}"; do
  for trans in "${TRANSPOSES_ARR[@]}"; do
    ta="${trans:0:1}"; tb="${trans:1:1}"
    for entry in "${SHAPES[@]}"; do
      read -r sid m n k <<< "$entry"
      idx=$((idx + 1))
      printf '[%d/%d] %s %sx%sx%s dtype=%s trans=%s\n' "$idx" "$total" "$sid" "$m" "$n" "$k" "$dtype" "$trans"
      printf '[%d/%d] %s %sx%sx%s dtype=%s trans=%s\n' "$idx" "$total" "$sid" "$m" "$n" "$k" "$dtype" "$trans" >> "$RUN_DIR/run.log"
      printf 'CMD: %s\n' "$BENCH_EXE --m $m --n $n --k $k --backend $BACKEND --dtype $dtype --transa $ta --transb $tb --mode $MODE --warmup $WARMUP --repeats $REPEATS --inner $INNER --seed $SEED --device $DEVICE --csv $RUN_DIR/results.csv" >> "$RUN_DIR/run.log"
      "$BENCH_EXE" --m "$m" --n "$n" --k "$k" \
                   --backend "$BACKEND" --dtype "$dtype" \
                   --transa "$ta" --transb "$tb" --mode "$MODE" \
                   --warmup "$WARMUP" --repeats "$REPEATS" --inner "$INNER" \
                   --seed "$SEED" --device "$DEVICE" \
                   --csv "$RUN_DIR/results.csv" 2>&1 | tee -a "$RUN_DIR/run.log"
      rc=${PIPESTATUS[0]}
      meaning=$(exit_meaning "$rc")
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$sid" "$m" "$n" "$k" "$dtype" "$BACKEND" "$ta" "$tb" "$rc" "$meaning" >> "$RUN_DIR/index.csv"
      case "$meaning" in
        OK) n_pass=$((n_pass + 1)) ;;
        SKIP) n_skip=$((n_skip + 1)) ;;
        *) n_fail=$((n_fail + 1)) ;;
      esac
    done
  done
done

generate_summary

echo "==========================================================="
echo "运行结束: OK=$n_pass FAIL=$n_fail SKIP=$n_skip（结果目录: $RUN_DIR）"
[[ -f "$RUN_DIR/summary.csv" ]] && echo "汇总预览:"
[[ -f "$RUN_DIR/summary.csv" ]] && column -s, -t "$RUN_DIR/summary.csv" 2>/dev/null || true

if (( n_fail > 0 )); then exit 1; fi
if (( n_pass == 0 )); then exit 3; fi
exit 0
