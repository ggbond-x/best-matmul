#!/usr/bin/env bash
# 编译 teach 与 fast，先做主机侧 MMA 映射检查。默认在 Hopper 上再跑全输出校验。
set -euo pipefail

tests_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$tests_dir/.." && pwd)"
compile_only=0
forward=()

usage() {
  cat <<'USAGE'
Usage: bash tests/run_validation.sh [options]
  --compile-only     只编译并运行主机侧 MMA 映射检查
  --cutlass DIR      传给 build.sh
  --nvcc PATH        传给 build.sh
  --help

两个校验程序都会覆盖 NT/TN、四种输入，以及 PIPE 附近和多次 phase 翻转的 K。
USAGE
}

while (( $# )); do
  case "$1" in
    --compile-only) compile_only=1; shift ;;
    --cutlass|--nvcc)
      [[ $# -ge 2 ]] || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
      forward+=("$1" "$2"); shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

bash "$repo_root/build.sh" all "${forward[@]}"
for kernel in teach fast; do
  binary="$repo_root/build/validate_$kernel"
  "$binary" --mapping-only
  if (( ! compile_only )); then
    "$binary"
  fi
done
if (( compile_only )); then
  printf '\nCompile and CPU mapping checks PASS; GPU execution skipped\n'
else
  printf '\nCompile, CPU mapping and full-output GPU checks PASS\n'
fi
