#!/usr/bin/env bash
# quick/full 共用的运行编排：
#   新运行目录 -> 环境快照 -> GPU 监控 -> gemm_bench -> 汇总 -> manifest 收尾。
# 不修改 GPU 时钟，不杀其他进程；中断时保留已写出的原始 CSV。

set -euo pipefail

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$_COMMON_DIR/../.." && pwd)"
PLAN="$ROOT/scripts/plan1/plan.yaml"
BIN="$ROOT/build/plan1/gemm_bench"
PLAN_UTILS="$ROOT/scripts/common/plan_utils.py"
ENV_SNAPSHOT="$ROOT/scripts/common/env_snapshot.py"
GPU_MONITOR="$ROOT/scripts/common/gpu_monitor.py"
SUMMARIZE="$ROOT/scripts/plan1/summarize.py"

RUN_DIR=""
MONITOR_PID=""
FINALIZED=0
SCRIPT_STATUS="RUNNING"

_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

_log() {
  local line="[$(_utc)] $*"
  printf '%s\n' "$line"
  if [[ -n "$RUN_DIR" ]]; then printf '%s\n' "$line" >> "$RUN_DIR/run.log"; fi
}

_run_cmd() {
  local rc=0
  printf '[%s] cwd=%s $ %s\n' "$(_utc)" "$ROOT" "$(printf '%q ' "$@")" >> "$RUN_DIR/commands.log"
  set +e
  "$@" 2>&1 | tee -a "$RUN_DIR/run.log"
  rc=${PIPESTATUS[0]}
  set -e
  printf '    exit=%d\n' "$rc" >> "$RUN_DIR/commands.log"
  return "$rc"
}

_stop_monitor() {
  if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill -TERM "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  MONITOR_PID=""
}

_manifest_status() {
  if [[ -f "$RUN_DIR/run_manifest.json" ]]; then
    python3 - "$RUN_DIR/run_manifest.json" <<'PY' 2>/dev/null || echo "PARTIAL"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.load(handle).get("status", "PARTIAL"))
PY
  else
    echo "PARTIAL"
  fi
}

_finalize_run() {
  local status="$1"
  if [[ "$FINALIZED" == "1" || -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
    return 0
  fi
  FINALIZED=1
  _stop_monitor

  if [[ -f "$RUN_DIR/environment.json" ]]; then
    _run_cmd python3 "$ENV_SNAPSHOT" --output "$RUN_DIR/environment.json" --merge \
      --phase final --gpu-index "${DEVICE:-0}" --before-file "$RUN_DIR/.env_before.json" \
      || { _log "WARNING: environment merge failed"; status="PARTIAL"; }
  else
    _run_cmd python3 "$ENV_SNAPSHOT" --output "$RUN_DIR/environment.json" --phase final \
      --gpu-index "${DEVICE:-0}" || { _log "WARNING: environment snapshot failed"; status="PARTIAL"; }
  fi

  if [[ -f "$RUN_DIR/csvs/raw_runs.csv" ]]; then
    _run_cmd python3 "$SUMMARIZE" --run-dir "$RUN_DIR" \
      || { _log "WARNING: summarize failed; raw data is preserved"; status="PARTIAL"; }
  else
    _log "WARNING: no csvs/raw_runs.csv; skipping summary"
    status="FAILED"
  fi

  _run_cmd python3 "$PLAN_UTILS" finalize --run-dir "$RUN_DIR" --status "$status" \
    || _log "WARNING: manifest finalize failed"

  rm -f "$RUN_DIR/.env_before.json"
  _log "finished status=$status"
}

_on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  _finalize_run "${SCRIPT_STATUS}"
  if [[ -n "$RUN_DIR" ]]; then
    printf '\nRUN_DIR=%s\n' "$RUN_DIR"
    printf 'SUMMARY=%s/csvs/summary.csv\n' "$RUN_DIR"
    printf 'ANALYSIS=%s/analysis.md\n' "$RUN_DIR"
  fi
  exit "$rc"
}

bench_run_main() {
  local mode="$1"
  shift || true
  local sets=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --set)
        [[ $# -ge 2 ]] || { echo "--set needs KEY=VALUE" >&2; return 2; }
        sets+=(--set "$2")
        shift 2
        ;;
      --warmup|--inner|--samples|--rounds|--seed|--device|--flush-mb|--lt-candidates|--lt-workspace-mb)
        [[ $# -ge 2 ]] || { echo "$1 needs a value" >&2; return 2; }
        key="${1#--}"; key="${key//-/_}"
        sets+=(--set "$key=$2")
        shift 2
        ;;
      --timing|--cache|--lt-selection)
        [[ $# -ge 2 ]] || { echo "$1 needs a value" >&2; return 2; }
        key="${1#--}"; key="${key//-/_}"
        sets+=(--set "$key=$2")
        shift 2
        ;;
      --cases|--cold-cases|--layouts)
        [[ $# -ge 2 ]] || { echo "$1 needs a value" >&2; return 2; }
        key="${1#--}"; key="${key//-/_}"
        sets+=(--set "$key=$2")
        shift 2
        ;;
      --impls|--implementations)
        [[ $# -ge 2 ]] || { echo "$1 needs a value" >&2; return 2; }
        sets+=(--set "implementations=$2")
        shift 2
        ;;
      *)
        echo "unknown option for run script: $1" >&2
        return 2
        ;;
    esac
  done

  if [[ ! -x "$BIN" ]]; then
    echo "gemm_bench 不存在: $BIN" >&2
    echo "请先运行: bash scripts/plan1/build.sh" >&2
    return 2
  fi

  local init_output
  if ! init_output="$(python3 "$PLAN_UTILS" init --plan "$PLAN" --mode "$mode" "${sets[@]}")"; then
    echo "无法解析测试计划" >&2
    return 2
  fi
  eval "$init_output"
  trap 'SCRIPT_STATUS=INTERRUPTED; exit 130' INT TERM
  trap _on_exit EXIT

  local git_sha=""
  git_sha="$(git -C "$ROOT" rev-parse --short=12 HEAD 2>/dev/null || true)"
  local clock_locked="${GEMM_CLOCK_LOCKED:-false}"
  local clock_target="${GEMM_CLOCK_TARGET_MHZ:-0}"

  : > "$RUN_DIR/run.log"
  _log "plan=$PLAN_ID mode=$MODE run_id=$RUN_ID"
  _log "config_hash=$CONFIG_HASH cases=$CASES cold_cases=${COLD_CASES:-none} layouts=$LAYOUTS"
  _log "timing=$TIMING_MODE cache=$CACHE_MODE warmup=$WARMUP inner=$INNER samples=$SAMPLES rounds=$ROUNDS"
  _log "implementations=$IMPLS"
  _log "clock_policy: no automatic clock change; clock_locked=$clock_locked target=$clock_target"
  printf '%s\n' "$init_output" >> "$RUN_DIR/run.log"

  local gpu_list=""
  gpu_list="$(nvidia-smi -L 2>&1 || true)"
  printf '%s\n' "$gpu_list" >> "$RUN_DIR/run.log"
  local apps=""
  apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
          --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "$apps" ]]; then
    _log "WARNING: 运行前检测到其他 GPU 计算进程（不会自动终止）:"
    printf '%s\n' "$apps" >> "$RUN_DIR/run.log"
    printf '%s\n' "$apps" > "$RUN_DIR/interfering_processes_before.txt"
  else
    _log "no pre-existing GPU compute processes reported by nvidia-smi"
  fi

  _run_cmd python3 "$ENV_SNAPSHOT" --output "$RUN_DIR/.env_before.json" --phase before \
    --gpu-index "$DEVICE" || _log "WARNING: before-environment snapshot failed"

  python3 "$GPU_MONITOR" --output "$RUN_DIR/clock_power.csv" \
    --interval "$MONITOR_INTERVAL" --gpu "$DEVICE" >> "$RUN_DIR/run.log" 2>&1 &
  MONITOR_PID=$!
  _log "gpu monitor started pid=$MONITOR_PID interval=${MONITOR_INTERVAL}s"

  local -a bench_args=(
    --mode "$MODE" --stage bench
    --cases "$CASES"
    --case-defs "$CASE_DEFS"
    --layouts "$LAYOUTS"
    --impl "$IMPLS"
    --timing "$TIMING_MODE"
    --cache "$CACHE_MODE"
    --warmup "$WARMUP"
    --inner "$INNER"
    --samples "$SAMPLES"
    --rounds "$ROUNDS"
    --flush-mb "$FLUSH_MB"
    --seed "$SEED"
    --lt-workspace-mb "$LT_WORKSPACE_MB"
    --lt-candidates "$LT_CANDIDATES"
    --lt-selection "$LT_SELECTION"
    --device "$DEVICE"
    --csv "$RUN_DIR/csvs/raw_runs.csv"
    --manifest "$RUN_DIR/run_manifest.json"
    --correctness-log "$RUN_DIR/correctness.log"
    --environment-json "$RUN_DIR/environment.json"
    --plan-id "$PLAN_ID"
    --run-id "$RUN_ID"
    --config-hash "$CONFIG_HASH"
    --clock-locked "$clock_locked"
    --clock-target "$clock_target"
    --notes "plan1 $MODE run; output=$RUN_DIR"
  )
  if [[ -n "$git_sha" ]]; then bench_args+=(--git-sha "$git_sha"); fi
  if [[ -n "$COLD_CASES" ]]; then bench_args+=(--cold-cases "$COLD_CASES"); fi

  local bench_rc=0
  _run_cmd "$BIN" "${bench_args[@]}" || bench_rc=$?
  _stop_monitor

  if [[ "$bench_rc" -eq 0 ]]; then
    SCRIPT_STATUS="$(_manifest_status)"
  else
    SCRIPT_STATUS="FAILED"
    _log "gemm_bench exit=$bench_rc; 已完成的原始记录会保留"
  fi
  exit "$bench_rc"
}
