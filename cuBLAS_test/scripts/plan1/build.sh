#!/usr/bin/env bash
# plan1 一键构建：定位项目根与 CUTLASS，配置 CMake，构建到 build/plan1/。
# 可从任意当前工作目录调用。
#
# 用法：
#   bash scripts/plan1/build.sh
#   bash scripts/plan1/build.sh --clean
#   CUTLASS_ROOT=/path/to/cutlass bash scripts/plan1/build.sh
#   GEMM_CUDA_ARCH=90a bash scripts/plan1/build.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BUILD_DIR="$ROOT/build/plan1"
JOBS="$(nproc 2>/dev/null || echo 4)"
ARCH="${GEMM_CUDA_ARCH:-90a}"
CLEAN=0
SKIP_SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) CLEAN=1; shift ;;
    --jobs|-j) JOBS="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --help|-h) sed -n '1,16p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

find_cutlass() {
  if [[ -n "${CUTLASS_ROOT:-}" && -f "$CUTLASS_ROOT/include/cutlass/cutlass.h" ]]; then
    printf '%s\n' "$CUTLASS_ROOT"
    return 0
  fi
  local dir="$ROOT"
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if [[ -f "$dir/cutlass/include/cutlass/cutlass.h" ]]; then
      printf '%s\n' "$dir/cutlass"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}
if ! CUTLASS_PATH="$(find_cutlass)"; then
  echo "找不到 CUTLASS（需要 include/cutlass/cutlass.h）。" >&2
  echo "请设置 CUTLASS_ROOT=/path/to/cutlass 后重试。" >&2
  exit 2
fi

# 选择支持 SM90a 的新版 nvcc：默认 PATH 里的 nvcc 可能来自旧 Toolkit（例如 10.1），
# 因此优先 CUDACXX / CUDA_HOME，其次在 /usr/local/cuda* 中挑选版本最高的一个。
nvcc_version_code() {
  "$1" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1\2/p' | head -n 1
}
find_cuda_compiler() {
  if [[ -n "${CUDACXX:-}" && -x "$CUDACXX" ]]; then
    printf '%s\n' "$CUDACXX"
    return 0
  fi
  if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
    printf '%s\n' "$CUDA_HOME/bin/nvcc"
    return 0
  fi
  local best="" best_code=0 cand code
  for cand in /usr/local/cuda*/bin/nvcc; do
    [[ -x "$cand" ]] || continue
    code="$(nvcc_version_code "$cand")"
    [[ -n "$code" ]] || continue
    if (( code > best_code )); then
      best="$cand"
      best_code="$code"
    fi
  done
  if [[ -n "$best" ]]; then
    printf '%s\n' "$best"
  else
    command -v nvcc || true
  fi
}
CUDA_NVCC="$(find_cuda_compiler)"
if [[ -z "$CUDA_NVCC" ]]; then
  echo "找不到 nvcc。请设置 CUDACXX 或 CUDA_HOME。" >&2
  exit 2
fi
CUDA_CODE="$(nvcc_version_code "$CUDA_NVCC")"
if [[ -n "$CUDA_CODE" ]] && (( CUDA_CODE < 1200 )); then
  echo "警告: $CUDA_NVCC 版本低于 12.0，SM90a 无法编译；请设置 CUDACXX。" >&2
fi

mkdir -p "$BUILD_DIR"
{
  printf '[%s] build.sh ROOT=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ROOT"
  printf '  arch=%s cutlass=%s jobs=%s clean=%s nvcc=%s\n' \
    "$ARCH" "$CUTLASS_PATH" "$JOBS" "$CLEAN" "$CUDA_NVCC"
} | tee -a "$BUILD_DIR/commands.log"

if [[ "$CLEAN" -eq 1 ]]; then
  rm -rf "$BUILD_DIR"
  mkdir -p "$BUILD_DIR"
fi

CMAKE_BIN="${CMAKE:-cmake}"
"$CMAKE_BIN" -S "$ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_NVCC" \
  -DGEMM_CUDA_ARCH="$ARCH" \
  -DCUTLASS_ROOT="$CUTLASS_PATH" \
  2>&1 | tee -a "$BUILD_DIR/commands.log"

"$CMAKE_BIN" --build "$BUILD_DIR" --parallel "$JOBS" 2>&1 | tee -a "$BUILD_DIR/commands.log"

GIT_SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
GIT_DIRTY="$(git -C "$ROOT" status --porcelain 2>/dev/null | head -n 1 || true)"
CUTLASS_VERSION="$(awk '/#define CUTLASS_(MAJOR|MINOR|PATCH)/ {printf "%s%s", sep, $3; sep="."}' \
  "$CUTLASS_PATH/include/cutlass/version.h" 2>/dev/null || echo unknown)"
{
  printf 'built_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'project_root=%s\n' "$ROOT"
  printf 'build_dir=%s\n' "$BUILD_DIR"
  printf 'cuda_arch=%s\n' "$ARCH"
  printf 'cutlass_root=%s\n' "$CUTLASS_PATH"
  printf 'cutlass_version=%s\n' "$CUTLASS_VERSION"
  printf 'git_sha=%s\n' "$GIT_SHA"
  printf 'git_dirty=%s\n' "${GIT_DIRTY:+true}"
  printf 'nvcc=%s (%s)\n' "$("$CUDA_NVCC" --version 2>/dev/null | tail -n 1 || echo 'nvcc not found')" "$CUDA_NVCC"
  printf 'cmake=%s\n' "$(cmake --version 2>/dev/null | head -n 1 || echo 'cmake not found')"
} > "$BUILD_DIR/build_info.txt"

printf '\nBuild OK: %s\n' "$BUILD_DIR/gemm_bench"
printf 'Build info: %s\n' "$BUILD_DIR/build_info.txt"
printf 'CUTLASS: %s (version %s)\n' "$CUTLASS_PATH" "$CUTLASS_VERSION"

if [[ "$SKIP_SMOKE" -eq 0 ]]; then
  echo
  echo "=== build smoke checks ==="
  impl_count="$("$BUILD_DIR/gemm_bench" --list-impls | wc -l)"
  case_count="$("$BUILD_DIR/gemm_bench" --list-cases | wc -l)"
  printf 'implementations=%s cases=%s\n' "$impl_count" "$case_count"
  if [[ "$impl_count" -lt 1 || "$case_count" -ne 12 ]]; then
    echo "build smoke check failed: unexpected impl/case counts" >&2
    exit 3
  fi
  "$BUILD_DIR/gemm_bench" --list-impls | sed -n '1,20p'
fi
