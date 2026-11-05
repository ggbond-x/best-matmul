#!/usr/bin/env bash
# plan1 NCU：对重点 Case/实现做一次性 launch 的 Nsight Compute 诊断。
# 不替代 benchmark；NCU 的 Duration 不进入主排行榜。
#
# 用法：
#   bash scripts/plan1/run_ncu.sh
#   bash scripts/plan1/run_ncu.sh --run-dir output/plan1/20261105_113000_a1b2c3
#   bash scripts/plan1/run_ncu.sh --cases S01 --impls CUSTOM_V1_F16ACC,CUBLAS_EX_F32ACC \
#        --layouts TN,NT
#
# 不提供 --run-dir 时会创建新的 output/plan1/<timestamped_run>/；提供时只向该目录
# 追加 reports/ 与 csvs/ncu_summary.csv，不修改其 benchmark 原始数据。

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PLAN="$ROOT/scripts/plan1/plan.yaml"
BIN="$ROOT/build/plan1/gemm_bench"
PLAN_UTILS="$ROOT/scripts/common/plan_utils.py"
ENV_SNAPSHOT="$ROOT/scripts/common/env_snapshot.py"
SUMMARIZE="$ROOT/scripts/plan1/summarize.py"

RUN_DIR=""
REQUESTED_CASES=""
REQUESTED_LAYOUTS=""
REQUESTED_IMPLS=""
REQUESTED_SECTIONS=""
CLEAN_RUN=0

_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
_log() {
  local line="[$(_utc)] $*"
  printf '%s\n' "$line"
  [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]] && printf '%s\n' "$line" >> "$RUN_DIR/run.log"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --cases) REQUESTED_CASES="$2"; shift 2 ;;
    --layouts) REQUESTED_LAYOUTS="$2"; shift 2 ;;
    --impls|--implementations) REQUESTED_IMPLS="$2"; shift 2 ;;
    --sections) REQUESTED_SECTIONS="$2"; shift 2 ;;
    --help|-h)
      sed -n '1,25p' "$0"; exit 0 ;;
    *)
      echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -x "$BIN" ]] || { echo "gemm_bench 不存在，请先运行 build.sh" >&2; exit 2; }

NEW_RUN=0
if [[ -z "$RUN_DIR" ]]; then
  init_output="$(python3 "$PLAN_UTILS" init --plan "$PLAN" --mode ncu)"
  eval "$init_output"
  NEW_RUN=1
  : > "$RUN_DIR/run.log"
  : > "$RUN_DIR/commands.log"
  python3 - "$RUN_DIR/run_manifest.json" "$PLAN_ID" "$RUN_ID" <<'PY'
import json, sys
from datetime import datetime, timezone
path, plan_id, run_id = sys.argv[1:4]
with open(path, "w", encoding="utf-8") as handle:
    json.dump({
        "plan_id": plan_id,
        "run_id": run_id,
        "mode": "ncu",
        "stage": "ncu",
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cases": [],
        "note": "NCU-only run; benchmark 数据未采集",
    }, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
PY
else
  [[ -d "$RUN_DIR" ]] || { echo "run dir 不存在: $RUN_DIR" >&2; exit 2; }
  RUN_DIR="$(cd "$RUN_DIR" && pwd)"
  read_output="$(python3 "$PLAN_UTILS" read --run-dir "$RUN_DIR")"
  eval "$read_output"
  _log "attaching NCU results to existing run: $RUN_DIR"
fi

mkdir -p "$RUN_DIR/reports" "$RUN_DIR/csvs"
CAPTURE_DIR="$(mktemp -d "$RUN_DIR/reports/ncu_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
ATTEMPTS="$RUN_DIR/reports/ncu_attempts.csv"
if [[ ! -f "$ATTEMPTS" ]]; then
  printf 'case_id,layout,implementation,status,exit_code,ncu_report,ncu_csv,notes\n' > "$ATTEMPTS"
fi

CASES="${REQUESTED_CASES:-$NCU_CASES}"
LAYOUTS="${REQUESTED_LAYOUTS:-$NCU_LAYOUTS}"
IMPLS="${REQUESTED_IMPLS:-$NCU_IMPLS}"
SECTIONS="${REQUESTED_SECTIONS:-$NCU_SECTIONS}"
[[ -n "$CASES" ]] || CASES="S01"
[[ -n "$LAYOUTS" ]] || LAYOUTS="TN,NT"
[[ -n "$IMPLS" ]] || IMPLS="CUBLAS_EX_F32ACC"
[[ -n "$SECTIONS" ]] || SECTIONS="SpeedOfLight"

_log "NCU plan: cases=$CASES layouts=$LAYOUTS impls=$IMPLS sections=$SECTIONS"
_log "NCU 口径: cache_control=$NCU_CACHE_CONTROL replay_mode=$NCU_REPLAY_MODE clock_control=$NCU_CLOCK_CONTROL"

log_cmd() {
  printf '[%s] cwd=%s $ %s\n' "$(_utc)" "$ROOT" "$(printf '%q ' "$@")" >> "$RUN_DIR/commands.log"
}

append_attempt() {
  python3 - "$ATTEMPTS" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" <<'PY'
import csv, sys
path, case_id, layout, implementation, status, exit_code, report, csv_path, notes = sys.argv[1:10]
with open(path, "a", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerow([
        case_id, layout, implementation, status, exit_code, report, csv_path, notes,
    ])
PY
}

_finalize_ncu() {
  local status="$1"
  trap - EXIT INT TERM
  if [[ -f "$RUN_DIR/environment.json" && "$NEW_RUN" == "1" ]]; then
    python3 "$ENV_SNAPSHOT" --output "$RUN_DIR/environment.json" --merge --phase final \
      --gpu-index "${DEVICE:-0}" >> "$RUN_DIR/run.log" 2>&1 || \
      _log "WARNING: environment merge failed"
  elif [[ "$NEW_RUN" == "1" ]]; then
    python3 "$ENV_SNAPSHOT" --output "$RUN_DIR/environment.json" --phase final \
      --gpu-index "${DEVICE:-0}" >> "$RUN_DIR/run.log" 2>&1 || \
      _log "WARNING: environment snapshot failed"
  fi
  python3 "$SUMMARIZE" --run-dir "$RUN_DIR" >> "$RUN_DIR/run.log" 2>&1 || \
    _log "WARNING: summarize failed; NCU raw reports are preserved"
  if [[ "$NEW_RUN" == 1 ]]; then
    python3 "$PLAN_UTILS" finalize --run-dir "$RUN_DIR" --status "$status" \
      --note "NCU diagnostics; Duration 与 benchmark 排名分离" >> "$RUN_DIR/run.log" 2>&1 || \
      _log "WARNING: manifest finalize failed"
  else
    python3 - "$RUN_DIR/reports/ncu_sessions.jsonl" "$CAPTURE_DIR" "$status" <<'PY'
import datetime, json, os, sys
with open(sys.argv[1], 'a', encoding='utf-8') as handle:
    json.dump({'capture_dir': os.path.basename(sys.argv[2]), 'status': sys.argv[3],
               'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}, handle)
    handle.write('\n')
PY
  fi
  _log "NCU finished status=$status"
  printf '\nRUN_DIR=%s\n' "$RUN_DIR"
  printf 'NCU_SUMMARY=%s/csvs/ncu_summary.csv\n' "$RUN_DIR"
}

_on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  _finalize_ncu "${NCU_STATUS:-PARTIAL}"
  exit "$rc"
}
trap _on_exit EXIT
trap 'NCU_STATUS=INTERRUPTED; exit 130' INT TERM

NCU_STATUS="PARTIAL"

if ! command -v ncu >/dev/null 2>&1; then
  _log "WARNING: ncu 未安装或不在 PATH；不会伪造 NCU 数据"
  IFS=',' read -r -a _cases <<< "$CASES"
  IFS=',' read -r -a _layouts <<< "$LAYOUTS"
  IFS=',' read -r -a _impls <<< "$IMPLS"
  for case_id in "${_cases[@]}"; do
    for layout in "${_layouts[@]}"; do
      for impl in "${_impls[@]}"; do
        append_attempt "$case_id" "$layout" "$impl" "SKIPPED" "127" "" "" "ncu not found in PATH"
      done
    done
  done
  NCU_STATUS="SKIPPED"
  exit 0
fi

{
  printf '%s\n' "=== ncu --version ==="
  ncu --version 2>&1 || true
  printf '%s\n' "=== ncu --list-sections ==="
  ncu --list-sections 2>&1 || true
  printf '%s\n' "=== requested sections ==="
  printf '%s\n' "$SECTIONS"
} > "$CAPTURE_DIR/ncu_environment.txt"

IFS=',' read -r -a section_array <<< "$SECTIONS"
SAFE_SECTIONS=()
for section in "${section_array[@]}"; do
  section="$(printf '%s' "$section" | xargs)"
  [[ -n "$section" ]] && SAFE_SECTIONS+=("$section")
done
[[ ${#SAFE_SECTIONS[@]} -gt 0 ]] || SAFE_SECTIONS=("SpeedOfLight")

SAFE_NAME_IMPL=""
_overall_ok=1
_any=0

IFS=',' read -r -a case_array <<< "$CASES"
IFS=',' read -r -a layout_array <<< "$LAYOUTS"
IFS=',' read -r -a impl_array <<< "$IMPLS"

for case_id in "${case_array[@]}"; do
  case_id="$(printf '%s' "$case_id" | xargs)"
  for layout in "${layout_array[@]}"; do
    layout="$(printf '%s' "$layout" | xargs)"
    for impl in "${impl_array[@]}"; do
      impl="$(printf '%s' "$impl" | xargs)"
      [[ -n "$case_id" && -n "$layout" && -n "$impl" ]] || continue
      _any=1
      safe="$(printf '%s_%s_%s' "$case_id" "$layout" "$impl" | tr -c 'A-Za-z0-9_.-' '_')"
      report_base="$CAPTURE_DIR/${safe}"
      csv_out="$CAPTURE_DIR/${safe}.csv"
      err_out="$CAPTURE_DIR/${safe}.err"
      algo_args=()
      if [[ "$NEW_RUN" == 0 && "$impl" == CUBLAS_LT_* ]]; then
        if ! saved_algo="$(python3 "$ROOT/scripts/common/ncu_algorithm.py" "$RUN_DIR/csvs/raw_runs.csv" "$case_id" "$layout" "$impl" "${DEVICE:-0}" "$CASE_DEFS" "$CONFIG_HASH" 2>"$CAPTURE_DIR/${safe}.algo.err")"; then
          reason="$(tr '\n' ' ' < "$CAPTURE_DIR/${safe}.algo.err")"
          append_attempt "$case_id" "$layout" "$impl" "SKIPPED" "1" "" "" "$reason"
          _log "NCU $case_id $layout $impl -> SKIPPED: $reason"
          _overall_ok=0
          continue
        fi
        algo_args=(--lt-algo-hex "${saved_algo%%:*}" --lt-algo-version "${saved_algo#*:}")
      fi
      ncu_args=(
        ncu --csv --target-processes all --kernel-name-base demangled
        --launch-skip "$NCU_LAUNCH_SKIP" --launch-count "$NCU_LAUNCH_COUNT"
        --cache-control "$NCU_CACHE_CONTROL"
        --replay-mode "$NCU_REPLAY_MODE"
        --clock-control "$NCU_CLOCK_CONTROL"
      )
      for section in "${SAFE_SECTIONS[@]}"; do ncu_args+=(--section "$section"); done
      ncu_args+=(
        -o "$report_base"
        "$BIN" --mode ncu --stage ncu --cases "$case_id" --case-defs "$CASE_DEFS" --layouts "$layout" --impl "$impl"
        --seed "$SEED" --lt-workspace-mb "$LT_WORKSPACE_MB" --lt-candidates "$LT_CANDIDATES" "${algo_args[@]}"
        --device "${DEVICE:-0}" --plan-id "$PLAN_ID" --run-id "$RUN_ID"
        --config-hash "$CONFIG_HASH" --clock-locked "${GEMM_CLOCK_LOCKED:-false}"
        --clock-target "${GEMM_CLOCK_TARGET_MHZ:-0}"
      )
      log_cmd "${ncu_args[@]}"
      set +e
      "${ncu_args[@]}" > "$csv_out" 2> "$err_out"
      ncu_rc=$?
      set -e
      printf '    exit=%d\n' "$ncu_rc" >> "$RUN_DIR/commands.log"
      report_file="${report_base}.ncu-rep"
      if [[ ! -f "$report_file" ]]; then report_file=""; fi
      if [[ "$ncu_rc" -ne 0 ]]; then
        status="FAIL"
        _overall_ok=0
        notes="$(tail -n 3 "$err_out" | tr '\n' ' ' | cut -c1-500)"
      elif grep -q "NCU_LAUNCH_SKIPPED" "$csv_out" 2>/dev/null; then
        status="SKIPPED"
        notes="gemm_bench reported NCU_LAUNCH_SKIPPED (compute capability)"
      elif ! grep -qi "Metric Name" "$csv_out" 2>/dev/null; then
        if [[ -n "$report_file" ]]; then
          status="OK"
          notes="ncu-rep saved; CSV metric table unavailable (version-specific format)"
        else
          status="FAIL"
          _overall_ok=0
          notes="ncu produced neither a metric table nor an .ncu-rep; see ${safe}.err"
        fi
      else
        status="OK"
        notes=""
      fi
      if [[ "$NEW_RUN" == 1 && "$impl" == CUBLAS_LT_* ]]; then
        notes="${notes:+$notes; }standalone NCU uses first heuristic, not a benchmark-selected algorithm"
      fi
      printf '%s\n' "$status" >> "$CAPTURE_DIR/${safe}.status"
      append_attempt "$case_id" "$layout" "$impl" "$status" "$ncu_rc" \
        "$(realpath --relative-to="$RUN_DIR" "$report_file" 2>/dev/null || printf '%s' "$report_file")" \
        "$(realpath --relative-to="$RUN_DIR" "$csv_out")" "$notes"
      _log "NCU $case_id $layout $impl -> $status"
    done
  done
done

if [[ "$_any" -eq 0 ]]; then
  NCU_STATUS="FAILED"
elif [[ "$_overall_ok" -eq 1 ]]; then
  NCU_STATUS="COMPLETED"
else
  NCU_STATUS="PARTIAL"
fi
exit 0
