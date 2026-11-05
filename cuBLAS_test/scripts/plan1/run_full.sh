#!/usr/bin/env bash
# plan1 full：S01–S12，TN/NT，全部可用实现；S01/S02 额外采集 cold-cache 独立榜单。
# 每次调用都会创建新的 output/plan1/<timestamped_run>/，绝不修改 quick 的结果。
#
# 用法：
#   bash scripts/plan1/run_full.sh
#   bash scripts/plan1/run_full.sh --set rounds=3 --set samples=3
#   bash scripts/plan1/run_full.sh --timing wall   # 独立口径，不与 event 混排
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../common/bench_run.sh"

bench_run_main full "$@"
