#!/usr/bin/env python3
"""把一个运行目录中的真实原始记录汇总为 summary.csv / ncu_summary.csv / analysis.md。

用法：
  python3 scripts/plan1/summarize.py --run-dir output/plan1/<timestamped_run>

约束：
  * 只读取指定运行目录，不扫描其他运行或计划。
  * 不创建新的 run_id，不插入推测/插值数据；缺失一律写 N/A 或 NOT_RUN。
  * NCU Duration 只进入 ncu_summary.csv，不参与 benchmark 排名。
"""

import argparse
import csv
import html
import json
import math
import os
import statistics
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import plan_utils  # noqa: E402

TRUTHY = ("true", "True", "TRUE", "1", "PASS")
THEORETICAL_H100_PCIE_FP16 = 756.0

SUMMARY_FIELDS = [
    "plan_id", "run_id", "case_id", "shape_id", "m", "n", "k", "layout",
    "a_dtype", "b_dtype", "accum_dtype", "accum_group", "out_dtype", "alpha", "beta",
    "a_storage", "b_storage", "lda", "ldb", "ldc",
    "implementation", "is_custom", "is_library", "kernel_name", "algorithm_id",
    "workspace_bytes", "timing_mode", "timer_scope", "cache_mode",
    "rounds_csv_rows", "rounds_ok", "samples", "inner",
    "duration_us_median", "duration_us_min", "duration_us_max",
    "duration_us_p10", "duration_us_p90", "variability_pct",
    "tflops_median", "tflops_min", "tflops_max",
    "best_library_tflops", "best_library_impl",
    "custom_pct_of_best_library", "gap_pct_vs_best_library",
    "pct_of_theoretical_peak",
    "correctness", "max_abs_error", "max_rel_error", "rel_l2_error",
    "mismatch_count", "nan_inf_count", "checked_elements",
    "sm_clock_mhz_avg", "mem_clock_mhz_avg", "power_w_avg", "power_w_max", "temp_c_max", "telemetry_scope",
    "clock_locked", "clock_target_mhz",
    "gpu_name", "compute_capability", "build_git_sha", "config_hash",
    "status", "reason", "notes",
]

NCU_FIELDS = [
    "run_id", "case_id", "layout", "implementation", "kernel_name", "launch_config",
    "metric", "value", "unit", "report_path", "report_csv", "status", "notes",
]

NCU_METRIC_PATTERNS = [
    ("duration_us", ("gpu__time_duration", "duration")),
    ("compute_throughput_pct", ("sm__throughput.avg.pct", "compute (sm) throughput")),
    ("memory_throughput_pct", ("gpu__compute_memory_throughput", "memory throughput")),
    ("dram_throughput_pct", ("dram__throughput", "dram throughput")),
    ("l2_hit_rate_pct", ("lts__t_sector_hit_rate", "l2 hit")),
    ("tensor_pipe_active_pct", ("pipe_tensor_cycles_active", "tensor pipe")),
    ("sm_issue_active_pct", ("inst_issued.avg.pct", "issue active", "sm issue")),
    ("eligible_warps_per_cycle", ("eligible_warps",)),
    ("achieved_occupancy_pct", ("achieved_occupancy", "achieved occupancy")),
    ("registers_per_thread", ("registers_per_thread", "registers per thread")),
    ("static_smem_bytes", ("static_shared_mem", "static shared")),
    ("dynamic_smem_bytes", ("dynamic_shared_mem", "dynamic shared")),
    ("stall_long_scoreboard_pct", ("long_scoreboard", "long scoreboard")),
    ("stall_barrier_pct", ("stall_barrier", "barrier")),
    ("stall_wait_pct", ("stall_wait", "wait")),
    ("stall_math_pipe_throttle_pct", ("math_pipe_throttle", "math pipe throttle")),
]


def to_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "N/A", "NaN", "nan", "None"):
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def fmt(value, digits=6):
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        return ("%." + str(digits) + "g") % value
    return str(value)


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def truthy(value):
    return str(value).strip() in TRUTHY


def read_rows(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "N/A") for key in fields})


# ---------------------------------------------------------------------------
# benchmark 汇总
# ---------------------------------------------------------------------------


def load_context(run_dir):
    context = {"resolved": {}, "manifest": {}, "environment": {}}
    resolved_path = os.path.join(run_dir, "resolved_plan.yaml")
    if os.path.isfile(resolved_path):
        try:
            context["resolved"] = plan_utils.load_plan(resolved_path)
        except Exception as error:  # noqa: BLE001 - 明确记录解析失败
            context["resolved_error"] = str(error)
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                context["manifest"] = json.load(handle)
        except ValueError as error:
            context["manifest_error"] = str(error)
    environment_path = os.path.join(run_dir, "environment.json")
    if os.path.isfile(environment_path):
        try:
            with open(environment_path, "r", encoding="utf-8") as handle:
                context["environment"] = json.load(handle)
        except ValueError as error:
            context["environment_error"] = str(error)
    return context


def group_key(row):
    return (
        row.get("case_id", ""), row.get("layout", ""), row.get("accum_dtype", ""),
        row.get("timing_mode", ""), row.get("timer_scope", ""), row.get("cache_mode", ""),
        row.get("implementation", ""),
    )


def _parse_utc(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def aggregate_group(rows, key, monitor_rows=()):
    first = rows[0]
    ok_rows = [row for row in rows if row.get("status") == "OK" and to_float(
        row.get("duration_us_median")) is not None]
    durations = [to_float(row.get("duration_us_median")) for row in ok_rows]
    tflops_values = [to_float(row.get("tflops_median")) for row in ok_rows]
    p10 = [to_float(row.get("duration_us_p10")) for row in ok_rows]
    p90 = [to_float(row.get("duration_us_p90")) for row in ok_rows]
    mins = [to_float(row.get("duration_us_min")) for row in ok_rows]
    maxs = [to_float(row.get("duration_us_max")) for row in ok_rows]
    duration_median = median(durations)
    duration_min = min([v for v in mins if v is not None], default=None)
    duration_max = max([v for v in maxs if v is not None], default=None)
    m = to_float(first.get("m")) or 0.0
    n = to_float(first.get("n")) or 0.0
    k = to_float(first.get("k")) or 0.0
    flops = 2.0 * m * n * k
    tflops_median = None
    tflops_min = None
    tflops_max = None
    if duration_median and duration_median > 0:
        tflops_median = flops / (duration_median * 1e-6) / 1e12
    if duration_max and duration_max > 0:
        tflops_min = flops / (duration_max * 1e-6) / 1e12
    if duration_min and duration_min > 0:
        tflops_max = flops / (duration_min * 1e-6) / 1e12
    variability = None
    if duration_median and duration_min is not None and duration_max is not None:
        variability = 100.0 * (duration_max - duration_min) / duration_median
    correctness_values = []
    correctness_reason = ""
    for row in ok_rows or rows:
        value = row.get("correctness_pass", "")
        if str(value).strip() != "":
            correctness_values.append(value)
        if row.get("status") != "OK" and row.get("notes"):
            correctness_reason = row["notes"]
    if not ok_rows:
        correctness = "FAIL" if any(row.get("status") == "FAIL" for row in rows) else "N/A"
    elif correctness_values and all(truthy(value) for value in correctness_values):
        correctness = "PASS"
    elif any(value.strip() in ("FAIL", "false", "False", "0") for value in correctness_values):
        correctness = "FAIL"
    else:
        correctness = "NOT_CHECKED"
    statuses = [row.get("status", "NOT_RUN") for row in rows]
    if "FAIL" in statuses:
        status = "FAIL"
    elif ok_rows and correctness == "PASS":
        status = "OK"
    elif ok_rows:
        status = "NOT_CHECKED"
    elif "UNSUPPORTED" in statuses:
        status = "UNSUPPORTED"
    else:
        status = statuses[0] if statuses else "NOT_RUN"
    notes = ""
    for row in rows:
        if row.get("notes"):
            notes = row["notes"]
            if status == "OK":
                break
    if correctness_reason and correctness_reason != notes:
        notes = (notes + " | " if notes else "") + correctness_reason
    algorithms = sorted({row.get("algorithm_id", "") for row in rows if row.get("algorithm_id")})
    windows = [(_parse_utc(r.get("round_started_utc")), _parse_utc(r.get("round_finished_utc")))
               for r in ok_rows]
    windows = [(start, end) for start, end in windows if start is not None and end is not None and start <= end]
    samples = []
    for sample in monitor_rows:
        at = _parse_utc(sample.get("timestamp_utc"))
        if at is not None and any(start <= at <= end for start, end in windows):
            samples.append(sample)
    def monitor_values(field):
        return [v for v in (to_float(sample.get(field)) for sample in samples) if v is not None]
    power_values = monitor_values("power_w")
    temperature_values = monitor_values("temperature_c")
    implementation = first.get("implementation", key[6])
    accum = first.get("accum_dtype", key[2])
    return {
        "plan_id": first.get("plan_id", ""),
        "run_id": first.get("run_id", ""),
        "case_id": key[0],
        "shape_id": first.get("shape_id", key[0]),
        "m": first.get("m", ""),
        "n": first.get("n", ""),
        "k": first.get("k", ""),
        "layout": key[1],
        "a_dtype": first.get("a_dtype", "FP16"),
        "b_dtype": first.get("b_dtype", "FP16"),
        "accum_dtype": accum,
        "accum_group": accum,
        "out_dtype": first.get("out_dtype", "FP16"),
        "alpha": first.get("alpha", ""),
        "beta": first.get("beta", ""),
        "a_storage": first.get("a_storage", ""),
        "b_storage": first.get("b_storage", ""),
        "lda": first.get("lda", ""),
        "ldb": first.get("ldb", ""),
        "ldc": first.get("ldc", ""),
        "implementation": implementation,
        "is_custom": "true" if implementation.startswith("CUSTOM") else "false",
        "is_library": "true" if implementation.startswith(("CUBLAS", "CUTLASS")) else "false",
        "kernel_name": first.get("kernel_name", ""),
        "algorithm_id": ",".join(algorithms) if algorithms else "N/A",
        "workspace_bytes": first.get("workspace_bytes", "N/A"),
        "timing_mode": key[3],
        "timer_scope": key[4],
        "cache_mode": key[5],
        "rounds_csv_rows": len(rows),
        "rounds_ok": len(ok_rows),
        "samples": first.get("samples", ""),
        "inner": first.get("inner", ""),
        "duration_us_median": fmt(duration_median),
        "duration_us_min": fmt(duration_min),
        "duration_us_max": fmt(duration_max),
        "duration_us_p10": fmt(median(p10)),
        "duration_us_p90": fmt(median(p90)),
        "variability_pct": fmt(variability, 4),
        "tflops_median": fmt(tflops_median, 5),
        "tflops_min": fmt(tflops_min, 5),
        "tflops_max": fmt(tflops_max, 5),
        "best_library_tflops": "N/A",
        "best_library_impl": "N/A",
        "custom_pct_of_best_library": "N/A",
        "gap_pct_vs_best_library": "N/A",
        "pct_of_theoretical_peak": "N/A",
        "correctness": correctness,
        "max_abs_error": fmt(mean([to_float(row.get("max_abs_error")) for row in rows]), 5),
        "max_rel_error": fmt(mean([to_float(row.get("max_rel_error")) for row in rows]), 5),
        "rel_l2_error": fmt(max([v for v in (to_float(row.get("rel_l2_error")) for row in rows)
                                 if v is not None], default=None), 5),
        "mismatch_count": first.get("mismatch_count", "N/A"),
        "nan_inf_count": first.get("nan_inf_count", "N/A"),
        "checked_elements": first.get("checked_elements", "N/A"),
        "sm_clock_mhz_avg": fmt(mean(monitor_values("sm_clock_mhz")), 5),
        "mem_clock_mhz_avg": fmt(mean(monitor_values("mem_clock_mhz")), 5),
        "power_w_avg": fmt(mean(power_values), 5),
        "power_w_max": fmt(max(power_values, default=None), 5),
        "temp_c_max": fmt(max(temperature_values, default=None), 5),
        "telemetry_scope": "monitor_samples_in_round_windows:%d" % len(samples) if samples else "N/A",
        "clock_locked": first.get("clock_locked", "N/A"),
        "clock_target_mhz": first.get("clock_target_mhz", "N/A"),
        "gpu_name": first.get("gpu_name", ""),
        "compute_capability": first.get("compute_capability", ""),
        "build_git_sha": first.get("build_git_sha", ""),
        "config_hash": first.get("config_hash", ""),
        "status": status,
        "reason": correctness_reason if status != "OK" else "",
        "notes": notes,
        "_tflops": tflops_median,
        "_duration_median": duration_median,
        "_m": m, "_n": n, "_k": k,
    }


def apply_baselines(rows, gpu_name):
    # 严格同 case / layout / 累加精度 / 计时口径 / cache 模式。
    groups = {}
    for row in rows:
        key = (row["case_id"], row["layout"], row["accum_dtype"], row["timing_mode"],
               row["timer_scope"], row["cache_mode"])
        groups.setdefault(key, []).append(row)
    for key, items in groups.items():
        libraries = [item for item in items
                     if item["status"] == "OK" and item["correctness"] == "PASS" and item["is_library"] == "true"
                     and item["_tflops"] is not None]
        best = max(libraries, key=lambda item: item["_tflops"]) if libraries else None
        for item in items:
            if item["status"] != "OK" or item["correctness"] != "PASS" or item["_tflops"] is None:
                continue
            if best is not None:
                item["best_library_tflops"] = fmt(best["_tflops"], 5)
                item["best_library_impl"] = best["implementation"]
                if item["is_custom"] == "true" and best["_tflops"] > 0:
                    pct = 100.0 * item["_tflops"] / best["_tflops"]
                    item["custom_pct_of_best_library"] = fmt(pct, 4)
                    item["gap_pct_vs_best_library"] = fmt(100.0 - pct, 4)
            if "h100 pcie" in (gpu_name or "").lower():
                item["pct_of_theoretical_peak"] = fmt(
                    100.0 * item["_tflops"] / THEORETICAL_H100_PCIE_FP16, 4
                )


def expected_not_run_rows(context, existing_keys):
    """根据 resolved plan 与 manifest 补充 NOT_RUN 行（不猜测性能）。"""
    resolved = context.get("resolved") or {}
    selected = resolved.get("selected") or {}
    shapes = {shape.get("id"): shape for shape in resolved.get("shapes", [])}
    implementations = selected.get("implementations") or []
    layouts = selected.get("layouts") or []
    cold_cases = set(selected.get("cold_cases") or [])
    cases = selected.get("cases") or list(shapes)
    timing_mode = (resolved.get("timing") or {}).get("mode", "event")
    rows = []
    for case_id in cases:
        shape = shapes.get(case_id)
        if not shape:
            continue
        cache_modes = ["warm"] + (["cold"] if case_id in cold_cases else [])
        for cache_mode in cache_modes:
            for layout in layouts:
                for implementation in implementations:
                    key = (case_id, layout, "", timing_mode, "", cache_mode, implementation)
                    matched = any(existing[0] == case_id and existing[1] == layout
                                  and existing[5] == cache_mode
                                  and existing[6] == implementation
                                  for existing in existing_keys)
                    if matched:
                        continue
                    rows.append({
                        "plan_id": resolved.get("plan_id", "plan1"),
                        "run_id": (resolved.get("run") or {}).get("run_id", ""),
                        "case_id": case_id,
                        "shape_id": shape.get("id", case_id),
                        "m": shape.get("m", ""), "n": shape.get("n", ""), "k": shape.get("k", ""),
                        "layout": layout,
                        "a_dtype": "FP16", "b_dtype": "FP16",
                        "accum_dtype": "N/A", "accum_group": "N/A", "out_dtype": "FP16",
                        "alpha": 1.0, "beta": 0.0,
                        "a_storage": "", "b_storage": "", "lda": "", "ldb": "", "ldc": "",
                        "implementation": implementation,
                        "is_custom": "true" if implementation.startswith("CUSTOM") else "false",
                        "is_library": "true" if implementation.startswith(("CUBLAS", "CUTLASS"))
                        else "false",
                        "kernel_name": "", "algorithm_id": "N/A", "workspace_bytes": "N/A",
                        "timing_mode": timing_mode, "timer_scope": "", "cache_mode": cache_mode,
                        "rounds_csv_rows": 0, "rounds_ok": 0, "samples": "", "inner": "",
                        "correctness": "N/A", "status": "NOT_RUN",
                        "reason": "no raw_runs.csv row for this combination",
                        "notes": "NOT_RUN: 计划中存在但本次没有产生原始记录",
                        "_tflops": None, "_duration_median": None,
                        "_m": shape.get("m", 0), "_n": shape.get("n", 0),
                        "_k": shape.get("k", 0),
                    })
    return rows


def build_summary(run_dir, context):
    raw_path = os.path.join(run_dir, "csvs", "raw_runs.csv")
    raw_rows = read_rows(raw_path)
    grouped = {}
    for row in raw_rows:
        grouped.setdefault(group_key(row), []).append(row)
    monitor_rows = read_rows(os.path.join(run_dir, "clock_power.csv"))
    summary = [aggregate_group(rows, key, monitor_rows) for key, rows in grouped.items()]
    existing = set(grouped.keys())
    summary.extend(expected_not_run_rows(context, existing))
    gpu_name = ""
    for row in raw_rows:
        if row.get("gpu_name"):
            gpu_name = row["gpu_name"]
            break
    if not gpu_name:
        gpu_name = ((context.get("environment") or {}).get("gpu") or {}).get("name", "")
    apply_baselines(summary, gpu_name)
    summary.sort(key=lambda row: (
        str(row.get("case_id")), str(row.get("layout")), str(row.get("accum_dtype")),
        str(row.get("cache_mode")), str(row.get("timing_mode")),
        str(row.get("implementation")),
    ))
    write_csv(os.path.join(run_dir, "csvs", "summary.csv"), SUMMARY_FIELDS, summary)
    return summary, raw_rows, gpu_name


# ---------------------------------------------------------------------------
# NCU 汇总：只做诊断，不参与排名
# ---------------------------------------------------------------------------


def parse_ncu_long_csv(path):
    records = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            header = None
            for row in reader:
                if not row:
                    continue
                lowered = [cell.strip().lower() for cell in row]
                if "metric name" in lowered:
                    header = {name: index for index, name in enumerate(lowered)}
                    break
            if header is None:
                return records
            for row in reader:
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                metric = row[header["metric name"]].strip()
                value = row[header["metric value"]].strip() if "metric value" in header else ""
                if not metric or not value:
                    continue
                kernel = row[header["kernel name"]].strip() if "kernel name" in header else ""
                unit = row[header["metric unit"]].strip() if "metric unit" in header else ""
                launch = ""
                if "grid size" in header and "block size" in header:
                    launch = "grid=%s block=%s" % (
                        row[header["grid size"]].strip(), row[header["block size"]].strip()
                    )
                records.append({
                    "kernel_name": kernel, "metric": metric, "value": value,
                    "unit": unit, "launch_config": launch,
                })
    except OSError:
        return records
    return records


def classify_metric(metric_name):
    lowered = metric_name.lower()
    for label, patterns in NCU_METRIC_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return label
    return None


def build_ncu_summary(run_dir, context):
    reports_dir = os.path.join(run_dir, "reports")
    attempts_path = os.path.join(reports_dir, "ncu_attempts.csv")
    attempts = read_rows(attempts_path)
    run_id = ""
    resolved = context.get("resolved") or {}
    run_id = (resolved.get("run") or {}).get("run_id", "")
    if not run_id:
        run_id = (context.get("manifest") or {}).get("run_id", "")
    rows = []
    for attempt in attempts:
        case_id = attempt.get("case_id", "")
        layout = attempt.get("layout", "")
        implementation = attempt.get("implementation", "")
        status = attempt.get("status", "FAIL")
        csv_path = attempt.get("ncu_csv", "")
        report_path = attempt.get("ncu_report", "")
        notes = attempt.get("notes", "")
        kernel = ""
        metrics = []
        if csv_path:
            full = csv_path if os.path.isabs(csv_path) else os.path.join(run_dir, csv_path)
            parsed = parse_ncu_long_csv(full)
            if parsed:
                kernel = parsed[0].get("kernel_name", "")
                seen = set()
                for record in parsed:
                    label = classify_metric(record["metric"])
                    if not label or label in seen:
                        continue
                    seen.add(label)
                    metrics.append((label, record))
        rows.append({
            "run_id": run_id, "case_id": case_id, "layout": layout,
            "implementation": implementation,
            "kernel_name": kernel or attempt.get("kernel_name", ""),
            "launch_config": metrics[0][1].get("launch_config", "") if metrics else "",
            "metric": "report_status", "value": status, "unit": "",
            "report_path": report_path, "report_csv": csv_path,
            "status": status, "notes": notes,
        })
        for label, record in metrics:
            rows.append({
                "run_id": run_id, "case_id": case_id, "layout": layout,
                "implementation": implementation,
                "kernel_name": record.get("kernel_name", kernel),
                "launch_config": record.get("launch_config", ""),
                "metric": label, "value": record.get("value", ""),
                "unit": record.get("unit", ""),
                "report_path": report_path, "report_csv": csv_path,
                "status": status, "notes": "",
            })
    write_csv(os.path.join(run_dir, "csvs", "ncu_summary.csv"), NCU_FIELDS, rows)
    return rows


# ---------------------------------------------------------------------------
# figures（可选，纯 SVG，数据全部来自 summary）
# ---------------------------------------------------------------------------


def svg_bar_chart(path, title, categories, series, y_label):
    # series: list of (name, values)
    width = max(720, 90 * max(1, len(categories)))
    height = 420
    margin_left, margin_bottom, margin_top = 80, 110, 50
    plot_w = width - margin_left - 30
    plot_h = height - margin_top - margin_bottom
    all_values = [value for _, values in series for value in values if value is not None]
    if not all_values:
        return
    maximum = max(all_values) or 1.0
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' % (width, height)]
    parts.append('<rect width="100%%" height="100%%" fill="#ffffff"/>')
    parts.append('<text x="%d" y="28" text-anchor="middle" font-size="16" font-family="sans-serif">%s</text>'
                 % (width // 2, html.escape(title)))
    group_width = plot_w / max(1, len(categories))
    bar_width = group_width * 0.8 / max(1, len(series))
    for index, value in enumerate([0, 0.25, 0.5, 0.75, 1.0]):
        y = margin_top + plot_h - value * plot_h
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e5e7eb"/>'
                     % (margin_left, y, width - 30, y))
        parts.append('<text x="%d" y="%.1f" text-anchor="end" font-size="11" '
                     'font-family="sans-serif">%.0f</text>'
                     % (margin_left - 8, y + 4, maximum * value))
    for group_index, category in enumerate(categories):
        base_x = margin_left + group_index * group_width + group_width * 0.1
        for series_index, (name, values) in enumerate(series):
            value = values[group_index]
            if value is None:
                continue
            bar_h = plot_h * value / maximum
            x = base_x + series_index * bar_width
            y = margin_top + plot_h - bar_h
            color = colors[series_index % len(colors)]
            parts.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s"/>'
                         % (x, y, bar_width * 0.9, bar_h, color))
        label = html.escape(str(category))
        parts.append('<text x="%.1f" y="%d" text-anchor="end" font-size="11" '
                     'font-family="sans-serif" transform="rotate(-35 %.1f %d)">%s</text>'
                     % (margin_left + (group_index + 0.5) * group_width,
                        height - margin_bottom + 18,
                        margin_left + (group_index + 0.5) * group_width,
                        height - margin_bottom + 18, label))
    for series_index, (name, _) in enumerate(series):
        x = margin_left + series_index * 160
        parts.append('<rect x="%d" y="%d" width="12" height="12" fill="%s"/>'
                     % (x, height - 24, colors[series_index % len(colors)]))
        parts.append('<text x="%d" y="%d" font-size="12" font-family="sans-serif">%s</text>'
                     % (x + 18, height - 14, html.escape(name)))
    parts.append('<text x="20" y="%d" font-size="12" font-family="sans-serif" '
                 'transform="rotate(-90 20 %d)">%s</text>'
                 % (margin_top + plot_h // 2, margin_top + plot_h // 2, html.escape(y_label)))
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))


def svg_line_chart(path, title, x_values, series, x_label, y_label):
    width, height = 820, 440
    margin_left, margin_bottom, margin_top = 80, 80, 50
    plot_w = width - margin_left - 40
    plot_h = height - margin_top - margin_bottom
    all_y = [value for _, values in series for value in values if value is not None]
    if not all_y or not x_values:
        return
    max_y = max(all_y)
    min_x, max_x = min(x_values), max(x_values)
    if max_x == min_x:
        max_x = min_x + 1
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]

    def px(value):
        return margin_left + (value - min_x) / (max_x - min_x) * plot_w

    def py(value):
        return margin_top + plot_h - value / max_y * plot_h

    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' % (width, height)]
    parts.append('<rect width="100%%" height="100%%" fill="#ffffff"/>')
    parts.append('<text x="%d" y="28" text-anchor="middle" font-size="16" '
                 'font-family="sans-serif">%s</text>' % (width // 2, html.escape(title)))
    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        y = margin_top + plot_h - fraction * plot_h
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e5e7eb"/>'
                     % (margin_left, y, width - 40, y))
        parts.append('<text x="%d" y="%.1f" text-anchor="end" font-size="11" '
                     'font-family="sans-serif">%.0f</text>'
                     % (margin_left - 8, y + 4, max_y * fraction))
    for value in sorted(set(x_values)):
        x = px(value)
        parts.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#f3f4f6"/>'
                     % (x, margin_top, x, margin_top + plot_h))
        parts.append('<text x="%.1f" y="%d" text-anchor="middle" font-size="11" '
                     'font-family="sans-serif">%s</text>'
                     % (x, height - margin_bottom + 20, html.escape(str(value))))
    for series_index, (name, values) in enumerate(series):
        points = [(px(x), py(y)) for x, y in zip(x_values, values) if y is not None]
        if len(points) >= 2:
            parts.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>'
                         % (colors[series_index % len(colors)],
                            " ".join("%.1f,%.1f" % point for point in points)))
        for x, y in points:
            parts.append('<circle cx="%.1f" cy="%.1f" r="3" fill="%s"/>'
                         % (x, y, colors[series_index % len(colors)]))
        x = margin_left + series_index * 170
        parts.append('<rect x="%d" y="%d" width="12" height="12" fill="%s"/>'
                     % (x, height - 24, colors[series_index % len(colors)]))
        parts.append('<text x="%d" y="%d" font-size="12" font-family="sans-serif">%s</text>'
                     % (x + 18, height - 14, html.escape(name)))
    parts.append('<text x="%d" y="%d" text-anchor="middle" font-size="12" '
                 'font-family="sans-serif">%s</text>'
                 % (width // 2, height - 8, html.escape(x_label)))
    parts.append('<text x="20" y="%d" font-size="12" font-family="sans-serif" '
                 'transform="rotate(-90 20 %d)">%s</text>'
                 % (margin_top + plot_h // 2, margin_top + plot_h // 2, html.escape(y_label)))
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))


def build_figures(run_dir, summary):
    figures = os.path.join(run_dir, "figures")
    os.makedirs(figures, exist_ok=True)
    event_rows = [row for row in summary if row["timing_mode"] == "event"
                  and row["cache_mode"] == "warm" and row["status"] == "OK"
                  and row["_tflops"] is not None]
    s01 = [row for row in event_rows if row["case_id"] == "S01"]
    if s01:
        for layout in ("TN", "NT"):
            categories = []
            values = []
            for row in sorted([r for r in s01 if r["layout"] == layout],
                              key=lambda r: (r["accum_dtype"], r["implementation"])):
                categories.append(row["implementation"] + " (" + row["accum_dtype"] + ")")
                values.append(row["_tflops"])
            svg_bar_chart(
                os.path.join(figures, "s01_%s_tflops.svg" % layout.lower()),
                "S01 4096^3 %s warm event TFLOPS" % layout,
                categories,
                [("TFLOPS", values)],
                "TFLOPS",
            )
    # K 扫描：同一实现、layout、accum 的 warm/event S01/S02/S03/S04/S08。
    k_cases = ["S02", "S08", "S03", "S04", "S01"]
    k_by_case = {}
    for case_id in k_cases:
        candidates = [row for row in event_rows if row["case_id"] == case_id
                      and row["layout"] == "TN" and row["accum_dtype"] == "FP16"]
        if candidates:
            k_by_case[case_id] = candidates[0]["_k"]
    if len(k_by_case) >= 2:
        x_values = [k_by_case[case_id] for case_id in k_cases if case_id in k_by_case]
        series = []
        for implementation in sorted({row["implementation"] for row in event_rows}):
            values = []
            for case_id in k_cases:
                if case_id not in k_by_case:
                    continue
                match = [row for row in event_rows
                         if row["case_id"] == case_id and row["layout"] == "TN"
                         and row["accum_dtype"] == "FP16"
                         and row["implementation"] == implementation]
                values.append(match[0]["_tflops"] if match else None)
            if any(value is not None for value in values):
                series.append((implementation, values))
        if series:
            svg_line_chart(
                os.path.join(figures, "k_scaling.svg"),
                "K scaling, TN, FP16 accumulation, warm event",
                x_values, series, "K", "TFLOPS",
            )


# ---------------------------------------------------------------------------
# analysis.md
# ---------------------------------------------------------------------------


def find_row(summary, case_id, layout, implementation, accum=None, timing="event", cache="warm"):
    for row in summary:
        if (row["case_id"] == case_id and row["layout"] == layout
                and row["implementation"] == implementation
                and row["status"] == "OK" and row["correctness"] == "PASS"
                and row["timing_mode"] == timing and row["cache_mode"] == cache
                and (accum is None or row["accum_dtype"] == accum)):
            return row
    return None


def table_fragment(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def best_row(summary, case_id, layout, accum, timing="event", cache="warm"):
    candidates = [row for row in summary
                  if row["case_id"] == case_id and row["layout"] == layout
                  and row["accum_dtype"] == accum and row["is_library"] == "true"
                  and row["status"] == "OK" and row["timing_mode"] == timing
                  and row["cache_mode"] == cache and row["_tflops"] is not None]
    return max(candidates, key=lambda row: row["_tflops"]) if candidates else None


def gap_bucket(gap):
    if gap is None:
        return "无有效对比（数据缺失或未在目标 GPU 成功运行）"
    if gap < 5:
        return "稳定在 5% 以内"
    if gap < 15:
        return "差距约 5%–15%"
    if gap < 30:
        return "差距约 15%–30%"
    return "差距超过 30%"


def k_saturation(summary, implementation, layout, accum):
    points = []
    for case_id in ("S02", "S08", "S03", "S04", "S01"):
        row = find_row(summary, case_id, layout, implementation, accum)
        if row and row["_tflops"] is not None and row["_k"]:
            points.append((row["_k"], row["_tflops"]))
    points.sort()
    if len(points) < 2:
        return None
    peak = max(value for _, value in points)
    for k, value in points:
        if value >= 0.9 * peak:
            return k, peak
    return points[-1][0], peak


def telemetry_stats(raw_rows, case_id="S01", cache="warm", timing="event"):
    rows = [row for row in raw_rows
            if row.get("case_id") == case_id and row.get("cache_mode") == cache
            and row.get("timing_mode") == timing and row.get("status") == "OK"]
    fallback = False
    if not rows:
        rows = [row for row in raw_rows if row.get("status") == "OK"]
        fallback = True
    sm = [to_float(row.get("sm_clock_mhz_avg")) for row in rows]
    mem = [to_float(row.get("mem_clock_mhz_avg")) for row in rows]
    power = [to_float(row.get("power_w_avg")) for row in rows]
    pmax = [to_float(row.get("power_w_max")) for row in rows]
    temp = [to_float(row.get("temp_c_max")) for row in rows]
    return {
        "sm": mean(sm), "mem": mean(mem), "power": mean(power),
        "power_max": max([v for v in pmax if v is not None], default=None),
        "temp": max([v for v in temp if v is not None], default=None),
        "rows": len(rows),
        "fallback": fallback,
    }


def monitor_stats(clock_power_path):
    rows = read_rows(clock_power_path)
    if not rows:
        return None
    sm = [to_float(row.get("sm_clock_mhz")) for row in rows]
    mem = [to_float(row.get("mem_clock_mhz")) for row in rows]
    power = [to_float(row.get("power_w")) for row in rows]
    temp = [to_float(row.get("temperature_c")) for row in rows]
    return {
        "samples": len(rows),
        "sm_avg": mean(sm), "sm_min": min([v for v in sm if v is not None], default=None),
        "sm_max": max([v for v in sm if v is not None], default=None),
        "mem_avg": mean(mem),
        "power_avg": mean(power),
        "power_min": min([v for v in power if v is not None], default=None),
        "power_max": max([v for v in power if v is not None], default=None),
        "temp_max": max([v for v in temp if v is not None], default=None),
    }


def analysis_markdown(run_dir, context, summary, raw_rows, ncu_rows, gpu_name):
    resolved = context.get("resolved") or {}
    manifest = context.get("manifest") or {}
    environment = context.get("environment") or {}
    run_id = (resolved.get("run") or {}).get("run_id", manifest.get("run_id", ""))
    target_gpu = ((resolved.get("target") or {}).get("gpu") or "")
    is_target = "h100 pcie" in (gpu_name or "").lower()
    lines = []
    lines.append("# plan1 运行分析：%s" % run_id)
    lines.append("")
    lines.append("- 运行目录：`%s`" % os.path.basename(run_dir))
    lines.append("- GPU：%s（compute capability %s）" % (
        gpu_name or "N/A",
        next((row.get("compute_capability") for row in raw_rows
              if row.get("compute_capability")), "N/A"),
    ))
    lines.append("- 目标 GPU：%s" % (target_gpu or "N/A"))
    lines.append("- 配置哈希：`%s`" % ((resolved.get("run") or {}).get("config_hash", "N/A")))
    lines.append("- 数据来源：`csvs/raw_runs.csv`（%d 行）、`csvs/summary.csv`、`csvs/ncu_summary.csv`" %
                 len(raw_rows))
    if not is_target:
        lines.append("")
        lines.append("> **未在目标 GPU 实测。** 本目录的数据来自 `%s`，不能用于回答 H100 PCIe 的"
                     "最终性能结论；它只验证了构建、正确性、计时与汇总流水线。若自研 Kernel 的"
                     "最低 compute capability 不满足，会在 manifest 中记录为 UNSUPPORTED。" % (gpu_name or "当前 GPU"))
    lines.append("")
    lines.append("## 数据口径")
    lines.append("")
    lines.append("- 主排行榜：`timing_mode=event` 且 `cache_mode=warm`；cold、wall、graph 各自独立，"
                 "不混合排名。")
    lines.append("- 正确性先于性能：`status=OK` 的行必须 `correctness_pass=true`；"
                 "UNSUPPORTED/FAIL/NOT_RUN 不参与基线选优。")
    lines.append("- 排名严格按 `accum_dtype` 分组；自研 Kernel 为 FP16 累加，未与 FP32 累加直接比较误差。")
    lines.append("- NCU 仅用于诊断；`ncu_summary.csv` 中的 Duration 不进入本文件的主排行榜。")
    lines.append("")

    # 问题 1
    lines.append("## 问题 1：S01 上自研 V1/V2 与各库基线分别达到多少 TFLOPS？")
    lines.append("")
    s01_rows = [row for row in summary if row["case_id"] == "S01"
                and row["timing_mode"] == "event" and row["cache_mode"] == "warm"
                and row["status"] in ("OK", "FAIL", "UNSUPPORTED")]
    if s01_rows:
        table_rows = []
        for layout in ("TN", "NT"):
            for accum in ("FP16", "FP32"):
                for row in sorted([r for r in s01_rows if r["layout"] == layout
                                   and r["accum_dtype"] == accum],
                                  key=lambda r: (r["status"] != "OK", -(
                                      r["_tflops"] or 0.0))):
                    table_rows.append([
                        layout, accum, row["implementation"], row["status"],
                        fmt(row["_tflops"], 5), fmt(row["variability_pct"], 3),
                        row["best_library_impl"],
                        fmt(row["custom_pct_of_best_library"], 4),
                        row["correctness"],
                    ])
        lines.append(table_fragment(
            ["layout", "accum", "implementation", "status", "TFLOPS", "波动%",
             "最佳同精度库", "自研/最佳库 %", "正确性"], table_rows))
    else:
        lines.append("本次运行没有 S01 的 warm/event 有效记录，无法给出吞吐值。"
                     "请检查 `run_manifest.json` 中的状态与原因。")
    lines.append("")

    # 问题 2
    lines.append("## 问题 2：差距是 5%、15% 还是 30% 以上？重复测试是否稳定？")
    lines.append("")
    gap_rows = []
    for layout in ("TN", "NT"):
        for implementation in ("CUSTOM_V1_F16ACC", "CUSTOM_V2_PIPE3_F16ACC",
                               "CUSTOM_V2_PIPE2_F16ACC", "CUSTOM_V2_PIPE4_F16ACC"):
            row = find_row(summary, "S01", layout, implementation, "FP16")
            if row and row["custom_pct_of_best_library"] != "N/A":
                pct = to_float(row["custom_pct_of_best_library"])
                gap_rows.append([
                    layout, implementation, row["best_library_impl"],
                    fmt(row["_tflops"], 5), fmt(row["best_library_tflops"], 5),
                    fmt(100.0 - pct if pct is not None else None, 4),
                    fmt(row["variability_pct"], 3),
                    gap_bucket(100.0 - pct if pct is not None else None),
                ])
    if gap_rows:
        lines.append(table_fragment(
            ["layout", "自研实现", "最佳同精度库", "自研 TFLOPS", "库 TFLOPS",
             "差距%", "轮间波动%", "差距分级"], gap_rows))
        lines.append("")
        lines.append("判断依据：差距来自同一 Shape、布局、累加精度、计时与缓存模式下的"
                     "`custom_tflops / best_library_tflops`；波动来自各轮中位数的 min–max "
                     "相对中位数。若波动接近或超过差距，应视为不稳定。")
    else:
        lines.append("没有可用的自研 vs 同精度库对比（自研实现可能为 UNSUPPORTED，"
                     "或本 GPU 上未产生 S01 数据）。不能据此声称 5%/15%/30% 的差距。")
    lines.append("")

    # 问题 3
    lines.append("## 问题 3：TN/NT 差异是自研独有，还是基线也存在？")
    lines.append("")
    tn_nt_rows = []
    implementations = sorted({row["implementation"] for row in s01_rows})
    for implementation in implementations:
        tn = find_row(summary, "S01", "TN", implementation)
        nt = find_row(summary, "S01", "NT", implementation)
        if not tn or not nt or tn["accum_dtype"] != nt["accum_dtype"]:
            continue
        if tn["_tflops"] is None or nt["_tflops"] is None:
            continue
        delta = 100.0 * (tn["_tflops"] - nt["_tflops"]) / nt["_tflops"]
        tn_nt_rows.append([
            implementation, tn["accum_dtype"], fmt(tn["_tflops"], 5),
            fmt(nt["_tflops"], 5), fmt(delta, 4),
        ])
    if tn_nt_rows:
        lines.append(table_fragment(
            ["implementation", "accum", "TN TFLOPS", "NT TFLOPS", "TN 相对 NT %"],
            tn_nt_rows))
        lines.append("")
        lines.append("若同样幅度的 TN/NT 差异同时出现在 cuBLAS/cuBLASLt/CUTLASS 上，"
                     "更可能是输入布局/缓存行为；若仅在自研实现上出现，应优先检查 TMA "
                     "swizzle、A/B major 选择与 Epilogue 写出路径。")
    else:
        lines.append("没有同实现、同精度的 TN 与 NT 双布局有效数据，无法判断。")
    lines.append("")

    # 问题 4
    lines.append("## 问题 4：K 从 256 增大到 4096 时，哪个实现最早进入高吞吐区？")
    lines.append("")
    k_table = []
    saturation = []
    k_implementations = sorted({row["implementation"] for row in summary
                                if row["timing_mode"] == "event"
                                and row["cache_mode"] == "warm"})
    for implementation in k_implementations:
        for layout in ("TN", "NT"):
            row = find_row(summary, "S01", layout, implementation)
            accum = row["accum_dtype"] if row else None
            if accum is None:
                continue
            values = []
            for case_id in ("S02", "S08", "S03", "S04", "S01"):
                current = find_row(summary, case_id, layout, implementation, accum)
                values.append(fmt(current["_tflops"], 4) if current else "N/A")
            if any(value != "N/A" for value in values):
                k_table.append([layout, accum, implementation] + values)
            result = k_saturation(summary, implementation, layout, accum)
            if result:
                saturation.append([implementation, layout, accum, result[0], fmt(result[1], 4)])
    if k_table:
        lines.append(table_fragment(
            ["layout", "accum", "implementation", "K=256", "K=512", "K=1024",
             "K=2048", "K=4096"], k_table))
        lines.append("")
        lines.append("“进入高吞吐区”的粗定义为：同一实现/布局/精度首次达到其 K 扫描最大值"
                     "90% 的 K。仅使用 warm/event 数据，缺失点写 N/A。")
        if saturation:
            lines.append("")
            lines.append(table_fragment(
                ["implementation", "layout", "accum", "首次 >=90% 峰值的 K", "该实现峰值 TFLOPS"],
                saturation))
    else:
        lines.append("本次运行没有足够的 K 扫描数据（至少需要两个 K 点）。")
    lines.append("")

    # 问题 5
    lines.append("## 问题 5：实际 GPU 时钟和功耗是否解释了一部分理论峰值差距？")
    lines.append("")
    stats = telemetry_stats(raw_rows)
    monitor = monitor_stats(os.path.join(run_dir, "clock_power.csv"))
    if stats["rows"] and (stats["sm"] or stats["power"]):
        scope = "全部有效 OK 行" if stats.get("fallback") else "S01 warm/event"
        lines.append("- %s的 NVML 轮次前后端点采样（不是稳态连续平均）：SM 时钟 %s MHz，"
                     "显存时钟 %s MHz，端点平均功耗 %s W，端点最大功耗 %s W，端点最高温度 %s °C。"
                     % (scope, fmt(stats["sm"], 5), fmt(stats["mem"], 5), fmt(stats["power"], 5),
                        fmt(stats["power_max"], 5), fmt(stats["temp"], 5)))
    else:
        lines.append("- raw_runs.csv 中没有可用的 NVML 行级采样（可能 NVML 不可用）。")
    if monitor:
        lines.append("- `clock_power.csv` 全程 nvidia-smi 采样 %d 次：SM 时钟 %s–%s MHz"
                     "（均值 %s），平均功耗 %s W（%s–%s W），最高温度 %s °C。"
                     % (monitor["samples"], fmt(monitor["sm_min"], 5),
                        fmt(monitor["sm_max"], 5), fmt(monitor["sm_avg"], 5),
                        fmt(monitor["power_avg"], 5), fmt(monitor["power_min"], 5),
                        fmt(monitor["power_max"], 5), fmt(monitor["temp_max"], 5)))
    else:
        lines.append("- 没有 `clock_power.csv` 采样。")
    if is_target:
        best_s01 = max([row["_tflops"] for row in s01_rows
                        if row["status"] == "OK" and row["_tflops"]], default=None)
        if best_s01:
            lines.append("- 运行中观测到的最佳 S01 吞吐 %s TFLOPS，对应 756 TFLOPS 标称峰值的 %s%%。"
                         "该分母是 H100 PCIe 官方稠密 FP16 标称值，不是实测硬件上限。"
                         % (fmt(best_s01, 5), fmt(100.0 * best_s01 / 756.0, 3)))
    else:
        lines.append("- 由于当前 GPU 不是 H100 PCIe，不使用 756 TFLOPS 作为分母，也不宣称"
                     "“理论峰值利用率”。理论差距需在目标 GPU 上重测。")
    lines.append("")
    lines.append("需要强调：几百微秒的单次 GEMM 无法与 nvidia-smi 瞬时值一一对应；"
                 "功耗/温度结论应理解为整个稳态批次区间的状态，而不是单个 Kernel 的功耗。")
    lines.append("")

    # 问题 6
    lines.append("## 问题 6：如果只能选 3 个下一步优化实验，选什么？")
    lines.append("")
    has_custom = any(row["is_custom"] == "true" and row["status"] == "OK" for row in summary)
    suggestions = []
    # 数据驱动的判断（仅在目标 GPU 有真实数据时给出强度更高的排序）。
    tn_nt_custom = []
    for implementation in ("CUSTOM_V1_F16ACC", "CUSTOM_V2_PIPE2_F16ACC",
                           "CUSTOM_V2_PIPE3_F16ACC", "CUSTOM_V2_PIPE4_F16ACC"):
        tn = find_row(summary, "S01", "TN", implementation)
        nt = find_row(summary, "S01", "NT", implementation)
        if tn and nt and tn["_tflops"] and nt["_tflops"]:
            tn_nt_custom.append(abs(tn["_tflops"] - nt["_tflops"]) / nt["_tflops"] * 100.0)
    short_k = [find_row(summary, "S02", layout, "CUSTOM_V2_PIPE3_F16ACC", "FP16")
               for layout in ("TN", "NT")]
    long_k = [find_row(summary, "S01", layout, "CUSTOM_V2_PIPE3_F16ACC", "FP16")
              for layout in ("TN", "NT")]
    short_gap = next((to_float(row["gap_pct_vs_best_library"]) for row in short_k if row), None)
    long_gap = next((to_float(row["gap_pct_vs_best_library"]) for row in long_k if row), None)
    if short_gap is not None and long_gap is not None and short_gap > long_gap + 5:
        suggestions.append(
            "**短 K 启动/Epilogue 专项**：S02 的差距（%s%%）明显大于 S01（%s%%）。"
            "优先测量 CTA 启动、TMA prologue、Epilogue 写回占比，并尝试减小 tile 或"
            "增加 CTA 并发；指标证据：S02 vs S01 的 `tflops_median` 与 `gap_pct_vs_best_library`。"
            % (fmt(short_gap, 3), fmt(long_gap, 3))
        )
    if tn_nt_custom and max(tn_nt_custom) > 5:
        suggestions.append(
            "**TN/NT 非对称性专项**：自研 S01 的 TN/NT 最大差异约 %.1f%%，而库基线差异较小。"
            "优先检查 TMA swizzle、A/B major 选择与 Epilogue 写回；指标证据：问题 3 的 TN/NT 表。"
            % max(tn_nt_custom)
        )
    if has_custom:
        suggestions.append(
            "**PIPE/CTA 驻留单变量实验**：固定 tile 与输入，只改 PIPE=2/3/4 与 CTA 并发，"
            "记录 SMEM/CTA、Registers/Thread、理论/实际驻留 CTA 与 TFLOPS；"
            "用 NCU 的 Tensor Active、Eligible Warps、Long Scoreboard/Barrier stall 判断"
            "瓶颈在 TMA 供数、WGMMA 发射/等待还是 CTA 并行度。"
        )
        suggestions.append(
            "**Epilogue 写回实验**：对比现有 Epilogue 与向量化/异步写回变体，保持数学语义不变；"
            "指标证据：S02 与 S01 的吞吐差、NCU MemoryWorkloadAnalysis 与 DRAM/L2 吞吐。"
        )
    else:
        suggestions.append(
            "**先在 H100 PCIe 上完成自研 Kernel 的正确性与基线测量**：当前运行未产生自研有效数据，"
            "无法判断差距来源。第一步应是 S01/S02 的 TN/NT warm/event 基线与 NCU 采集。"
        )
        suggestions.append(
            "**PIPE 深度与 CTA 驻留单变量实验**：在 H100 上固定 tile/输入，只改 PIPE=2/3/4 与"
            "CTA 并发，记录 SMEM、Registers、实际驻留 CTA 与 TFLOPS，并用 NCU 的"
            "Tensor Active / Eligible Warps / Long Scoreboard 分类瓶颈。"
        )
        suggestions.append(
            "**TMA 供数 vs Epilogue 写回拆分实验**：用 NCU 的 MemoryWorkloadAnalysis、"
            "TMA 统计与 S02/S01 对比判断短 K 的启动/写回占比，再决定优先优化 TMA 还是 Epilogue。"
        )
    for index, text in enumerate(suggestions[:3], start=1):
        lines.append("%d. %s" % (index, text))
    lines.append("")

    lines.append("## 限制与可复现性")
    lines.append("")
    lines.append("- 本报告只基于 `%s` 中的实际 CSV/JSON；缺失指标写 N/A，未插值、未补齐。"
                 % os.path.basename(run_dir))
    lines.append("- NCU 报告与普通 benchmark 分开；NCU 的 replay/cache-control 会改变运行条件，"
                 "其 Duration 不与 warm/event 排名混用。")
    lines.append("- 若 `run_manifest.json` 中存在 UNSUPPORTED/FAIL/NOT_RUN，代表该组合没有进入排名；"
                 "原因见 manifest 与 `correctness.log`。")
    lines.append("- 时钟策略：默认不主动修改 GPU 时钟；`clock_locked` 与目标频率逐行记录在 CSV 中。")
    lines.append("")
    lines.append("### 相关文件")
    lines.append("")
    for name in ("resolved_plan.yaml", "run_manifest.json", "environment.json", "commands.log",
                 "run.log", "correctness.log", "clock_power.csv", "csvs/raw_runs.csv",
                 "csvs/summary.csv", "csvs/ncu_summary.csv", "reports/"):
        lines.append("- `%s`" % name)
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="明确指定 output/plan1/<timestamped_run>")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise SystemExit("run dir 不存在: %s" % run_dir)
    context = load_context(run_dir)
    summary, raw_rows, gpu_name = build_summary(run_dir, context)
    ncu_rows = build_ncu_summary(run_dir, context)
    if not args.no_figures:
        try:
            build_figures(run_dir, summary)
        except Exception as error:  # noqa: BLE001 - 图是可选产物，失败不掩盖数据
            with open(os.path.join(run_dir, "figures", "figures_error.txt"), "w",
                      encoding="utf-8") as handle:
                handle.write(str(error))
    markdown = analysis_markdown(run_dir, context, summary, raw_rows, ncu_rows, gpu_name)
    with open(os.path.join(run_dir, "analysis.md"), "w", encoding="utf-8") as handle:
        handle.write(markdown + "\n")
    ok = sum(1 for row in summary if row["status"] == "OK")
    print("summary: %d groups (%d OK) -> %s" % (len(summary), ok,
                                                os.path.join(run_dir, "csvs", "summary.csv")))
    print("ncu: %d rows -> %s" % (len(ncu_rows), os.path.join(run_dir, "csvs", "ncu_summary.csv")))
    print("analysis: %s" % os.path.join(run_dir, "analysis.md"))


if __name__ == "__main__":
    main()
