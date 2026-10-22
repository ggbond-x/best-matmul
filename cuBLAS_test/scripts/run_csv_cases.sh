#!/usr/bin/env bash
# Build gemm_bench and run every row in cuBLAS测试结果.csv with cuBLASEx/cuBLASLt.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
PYTHON="${PYTHON:-python3}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build_csv}"
BENCH_EXE="${BENCH_EXE:-$BUILD_DIR/gemm_bench}"
GPU_ARCHS="${GPU_ARCHS:-90}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
CUBLAS_LIB_DIR="${CUBLAS_LIB_DIR:-$CUDA_HOME/lib64}"

INPUT_CSV="${INPUT_CSV:-$ROOT_DIR/cuBLAS测试结果.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outptu}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/cublas_csv_$(date +%Y%m%d_%H%M%S)_$$}"
DEVICE="${DEVICE:-0}"
SEED="${SEED:-1234}"
CHECK_SAMPLES="${CHECK_SAMPLES:-512}"
LT_SELECTION="${LT_SELECTION:-tune}"
LT_CANDIDATES="${LT_CANDIDATES:-32}"

DRY_RUN=0
SKIP_BUILD=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_csv_cases.sh [options]
  --dry-run             Validate the CSV and print all commands; do not build or run.
  --skip-build          Use BENCH_EXE without invoking CMake.
  --input FILE          Input CSV; default cuBLAS测试结果.csv.
  --output-dir DIR      Exact result directory; default outptu/cublas_csv_<timestamp>_<pid>.
  --device N            CUDA device index; default DEVICE or 0.
  --help                Show this help.

Environment overrides:
  CUDA_HOME NVCC CMAKE_BIN PYTHON BUILD_DIR BENCH_EXE GPU_ARCHS JOBS
  CUBLAS_LIB_DIR INPUT_CSV OUTPUT_ROOT RUN_DIR DEVICE SEED CHECK_SAMPLES
  LT_SELECTION(first|tune) LT_CANDIDATES
USAGE
}

need_value() { [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }; }
while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --input) need_value "$@"; INPUT_CSV="$2"; shift 2 ;;
    --output-dir) need_value "$@"; RUN_DIR="$2"; shift 2 ;;
    --device) need_value "$@"; DEVICE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$INPUT_CSV" ]] || { echo "Input CSV not found: $INPUT_CSV" >&2; exit 2; }
command -v "$PYTHON" >/dev/null 2>&1 || { echo "Python not found: $PYTHON" >&2; exit 2; }
case "$DEVICE" in ''|*[!0-9]*) echo "DEVICE must be a non-negative integer" >&2; exit 2 ;; esac
case "$LT_SELECTION" in first|tune) ;; *) echo "LT_SELECTION must be first or tune" >&2; exit 2 ;; esac

runner=("$PYTHON" "$SCRIPT_DIR/run_csv_cases.py" --exe "$BENCH_EXE" --input "$INPUT_CSV"
        --output-dir "$RUN_DIR" --device "$DEVICE" --seed "$SEED"
        --check-samples "$CHECK_SAMPLES" --lt-selection "$LT_SELECTION"
        --lt-candidates "$LT_CANDIDATES")

if ((DRY_RUN)); then
  "${runner[@]}" --dry-run
  exit $?
fi

[[ ! -e "$RUN_DIR" ]] || { echo "Output directory already exists: $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR"
printf 'BUILDING\n' > "$RUN_DIR/status.txt"
trap 'rc=$?; if ((rc != 0)); then printf "FAILED_WRAPPER\nexit_code=%s\n" "$rc" > "$RUN_DIR/status.txt"; fi' EXIT
trap 'printf "INTERRUPTED\n" > "$RUN_DIR/status.txt"; exit 130' INT TERM

if ((!SKIP_BUILD)); then
  [[ -x "$NVCC" ]] || { echo "nvcc not executable: $NVCC" >&2; exit 2; }
  command -v "$CMAKE_BIN" >/dev/null 2>&1 || { echo "CMake not found: $CMAKE_BIN" >&2; exit 2; }
  {
    echo "== Configure: nvcc=$NVCC arch=$GPU_ARCHS =="
    "$CMAKE_BIN" -S "$ROOT_DIR" -B "$BUILD_DIR" \
      -DCMAKE_CUDA_COMPILER="$NVCC" \
      -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
      -DGEMM_CUDA_ARCHITECTURES="$GPU_ARCHS"
    echo "== Build: $BENCH_EXE =="
    "$CMAKE_BIN" --build "$BUILD_DIR" -j "$JOBS"
  } 2>&1 | tee "$RUN_DIR/build.log"
fi

[[ -x "$BENCH_EXE" ]] || { echo "Benchmark executable not found: $BENCH_EXE" >&2; exit 2; }
export LD_LIBRARY_PATH="$CUBLAS_LIB_DIR:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
printf 'RUNNING\n' > "$RUN_DIR/status.txt"
set +e
"${runner[@]}"
rc=$?
set -e
trap - EXIT
exit "$rc"
