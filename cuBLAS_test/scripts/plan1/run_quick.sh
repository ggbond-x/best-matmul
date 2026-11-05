#!/usr/bin/env bash
# plan1 quick：S01 + S02，TN/NT，全部可用实现的正确性与 warm/event 基础性能。
# 每次调用都会创建新的 output/plan1/<timestamped_run>/，不会覆盖历史结果。
#
# 用法：
#   bash scripts/plan1/run_quick.sh
#   bash scripts/plan1/run_quick.sh --warmup 5 --inner 20 --samples 3 --rounds 2
#   bash scripts/plan1/run_quick.sh --set warmup=5 --set rounds=2
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../common/bench_run.sh"

bench_run_main quick "$@"
