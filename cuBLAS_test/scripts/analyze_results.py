#!/usr/bin/env python3
"""Analyze measured windows; never interpolate or invent missing sensors."""
import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

BASE = ["run_id", "case_id", "phase", "M", "N", "K", "transA", "transB", "dtype", "compute_type", "api",
        "algorithm_id", "workspace_bytes", "elapsed_ms_median", "elapsed_ms_mean", "elapsed_ms_min",
        "tflops", "tflops_mean", "peak_efficiency_pct", "power_avg_w", "power_max_w", "power_limit_w",
        "sm_clock_avg_mhz", "sm_clock_min_mhz", "sm_clock_max_mhz", "temperature_max_c",
        "power_throttle_observed", "thermal_throttle_observed", "correctness_pass", "status"]
EXTRA = ["algorithm", "gpu_uuid", "reason", "tensor_core_evidence", "instruction_path", "repeats",
         "telemetry_samples", "telemetry_coverage_ratio", "telemetry_quality", "power_samples", "clock_samples",
         "power_reason_samples", "thermal_reason_samples", "power_throttle_sample_ratio", "thermal_throttle_sample_ratio",
         "idle_sample_ratio", "application_clock_sample_ratio", "hw_slowdown_sample_ratio", "power_brake_sample_ratio",
         "near_power_limit_sample_ratio", "power_limit_min_w", "power_limit_max_w", "gpu_utilization_avg_pct",
         "memory_clock_avg_mhz", "other_compute_pids", "process_visibility", "gpu_event_duty_ratio",
         "start_monotonic_ns", "end_monotonic_ns", "sample_interval_ms_median",
         "clock_scaled_peak_tflops", "clock_scaled_efficiency_pct"]


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def ns(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return 0


def values(rows, field):
    return [v for r in rows if (v := num(r.get(field))) is not None]


def avg(v):
    return st.mean(v) if v else ""


def fmt(x, digits=2):
    v = num(x)
    return f"{v:.{digits}f}" if v is not None else "N/A"


def contained(sample, start, end):
    return start <= ns(sample.get("query_start_monotonic_ns")) <= ns(sample.get("query_end_monotonic_ns")) <= end


def telemetry_stats(samples, windows, known_pid=None):
    intervals = [(ns(w.get("start_monotonic_ns")), ns(w.get("end_monotonic_ns"))) for w in windows]
    intervals = [(s, e) for s, e in intervals if e > s]
    uuid = windows[0].get("gpu_uuid", "") if windows else ""
    selected = [s for s in samples if s.get("gpu_uuid", "").lower() == uuid.lower()
                and any(contained(s, start, end) for start, end in intervals)]
    timestamps = sorted(ns(s["timestamp_monotonic_ns"]) for s in selected)
    span = 0
    for start, end in intervals:
        stamps = [t for t in timestamps if start <= t <= end]
        if len(stamps) >= 2:
            span += stamps[-1] - stamps[0]
    duration = sum(e - s for s, e in intervals)
    coverage = span / duration if duration else 0
    quality = "OK" if len(selected) >= 3 and coverage >= .5 else "INSUFFICIENT"
    if not selected or not any(values(selected, field) for field in (
            "power_draw_w", "power_limit_w", "sm_clock_mhz", "temperature_c",
            "gpu_utilization_pct", "memory_clock_mhz", "power_throttle", "thermal_throttle")):
        quality = "MISSING"
    result = {"telemetry_samples": len(selected), "telemetry_coverage_ratio": coverage,
              "telemetry_quality": quality, "other_compute_pids": "", "process_visibility": "unavailable"}
    mapping = {"power_draw_w": ("power_avg_w", "power_max_w", None),
               "power_limit_w": ("power_limit_w", "power_limit_max_w", "power_limit_min_w"),
               "sm_clock_mhz": ("sm_clock_avg_mhz", "sm_clock_max_mhz", "sm_clock_min_mhz"),
               "temperature_c": (None, "temperature_max_c", None),
               "gpu_utilization_pct": ("gpu_utilization_avg_pct", None, None),
               "memory_clock_mhz": ("memory_clock_avg_mhz", None, None)}
    for field, (mean, maximum, minimum) in mapping.items():
        v = values(selected, field)
        if mean:
            result[mean] = avg(v)
        if maximum:
            result[maximum] = max(v) if v else ""
        if minimum:
            result[minimum] = min(v) if v else ""
    result["power_samples"] = len(values(selected, "power_draw_w"))
    result["clock_samples"] = len(values(selected, "sm_clock_mhz"))
    for src, prefix in [("power_throttle", "power_throttle"), ("thermal_throttle", "thermal_throttle"),
                        ("idle_reason", "idle"), ("application_clock_reason", "application_clock"),
                        ("hw_slowdown", "hw_slowdown"), ("power_brake", "power_brake")]:
        v = values(selected, src)
        result[prefix + "_sample_ratio"] = avg(v)
        if src in ("power_throttle", "thermal_throttle"):
            result[prefix + "_observed"] = ("true" if any(v) else "false" if len(v) >= 3 and quality == "OK" else "unknown")
            result[("power" if src == "power_throttle" else "thermal") + "_reason_samples"] = len(v)
    near, pids = [], set()
    for s in selected:
        p, limit = num(s.get("power_draw_w")), num(s.get("power_limit_w"))
        if p is not None and limit is not None and limit > 0:
            near.append(int(p >= .95 * limit))
        if s.get("compute_processes"):
            try:
                procs = json.loads(s["compute_processes"])
                result["process_visibility"] = "available"
                pids.update(str(p["pid"]) for p in procs if str(p["pid"]) != str(known_pid))
            except (ValueError, KeyError, TypeError):
                pass
    result["near_power_limit_sample_ratio"] = avg(near)
    result["other_compute_pids"] = ";".join(sorted(pids))
    gaps = [(b - a) / 1e6 for a, b in zip(timestamps, timestamps[1:])]
    result["sample_interval_ms_median"] = st.median(gaps) if gaps else ""
    return result


def aggregate(group, samples, index, peak):
    row = {k: group[0].get(k, "") for k in BASE + EXTRA}
    entry = index.get(row["run_id"])
    fail = next((r for r in group if r.get("status") != "PASS"), None)
    if fail:
        row.update(status=fail.get("status", "FAIL"), reason=fail.get("reason", ""), correctness_pass="false" if fail.get("status") == "FAIL" else "")
    elif entry is not None and entry.get("status") != "PASS":
        row.update(status=entry.get("status", "INCOMPLETE"), reason=entry.get("reason", ""), correctness_pass="")
    else:
        row["status"] = "PASS"
    measured = [r for r in group if ns(r.get("end_monotonic_ns")) > ns(r.get("start_monotonic_ns")) > 0]
    times = [float(v) for r in measured for v in r.get("samples_ms", "").split(";") if num(v) is not None and float(v) > 0]
    row.update(telemetry_stats(samples, measured, (entry or {}).get("pid")))
    if measured:
        row["start_monotonic_ns"] = min(ns(r["start_monotonic_ns"]) for r in measured)
        row["end_monotonic_ns"] = max(ns(r["end_monotonic_ns"]) for r in measured)
        wall = sum(num(r.get("wall_ms")) or 0 for r in measured)
        row["gpu_event_duty_ratio"] = sum(num(r.get("gpu_total_ms")) or 0 for r in measured) / wall if wall else ""
    if times:
        median, mean = st.median(times), st.mean(times)
        flops = 2 * int(row["M"]) * int(row["N"]) * int(row["K"])
        row.update(elapsed_ms_median=median, elapsed_ms_mean=mean, elapsed_ms_min=min(times),
                   tflops=flops / (median * 1e9), tflops_mean=flops / (mean * 1e9), repeats=len(times))
    if row["status"] != "PASS":
        row["tflops"] = row["tflops_mean"] = row["peak_efficiency_pct"] = ""
    elif num(row.get("tflops")) is not None and peak.get("verified_for_device"):
        row["peak_efficiency_pct"] = row["tflops"] / peak["dense_fp16_tflops"] * 100
        clock = num(row.get("sm_clock_avg_mhz"))
        if clock and peak.get("reference_clock_mhz") and row["telemetry_quality"] == "OK" and row["clock_samples"] >= 3:
            scaled = peak["dense_fp16_tflops"] * clock / peak["reference_clock_mhz"]
            row["clock_scaled_peak_tflops"] = scaled
            row["clock_scaled_efficiency_pct"] = row["tflops"] / scaled * 100
    return row


def correlation(rows, x, y):
    required_counts = {"sm_clock_avg_mhz": "clock_samples", "power_avg_w": "power_samples"}
    eligible = [r for r in rows if r.get("telemetry_quality") == "OK"
                and all((num(r.get(required_counts[f])) or 0) >= 3 for f in (x, y) if f in required_counts)]
    pairs = [(num(r.get(x)), num(r.get(y))) for r in eligible]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(pairs) < 8:
        return None, "有效同条件窗口少于 8 个"
    xs, ys = zip(*pairs)
    vx, vy = st.pstdev(xs), st.pstdev(ys)
    if vx < max(abs(st.mean(xs)) * .005, 1e-9) or vy < max(abs(st.mean(ys)) * .005, 1e-9):
        return None, "变化范围过小，相关性不稳定"
    r = sum((a - st.mean(xs)) * (b - st.mean(ys)) for a, b in pairs) / (len(pairs) * vx * vy)
    return r, f"{len(pairs)} 个有效窗口；相关不证明因果"


def plots(out, summary, windows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return [], "未安装 matplotlib；已保存 window_summary.csv，可安装后重新运行分析脚本生成图。"
    folder = out / "plots"
    folder.mkdir(exist_ok=True)
    links = []
    passed = [r for r in summary if r["status"] == "PASS" and r["telemetry_quality"] == "OK"]
    for name, x, y, xl, yl in [
        ("tflops_vs_sm_clock", "sm_clock_avg_mhz", "tflops", "SM clock (MHz)", "TFLOPS"),
        ("power_vs_sm_clock", "sm_clock_avg_mhz", "power_avg_w", "SM clock (MHz)", "Power (W)"),
        ("power_vs_tflops", "tflops", "power_avg_w", "TFLOPS", "Power (W)")]:
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        plotted = False
        for api in ("cublas", "cublaslt"):
            pairs = [(num(r.get(x)), num(r.get(y))) for r in passed if r["api"] == api]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            if pairs:
                ax.scatter(*zip(*pairs), label=api, alpha=.8)
                plotted = True
        if plotted:
            ax.legend()
        else:
            ax.text(.5, .5, "Insufficient telemetry data", transform=ax.transAxes, ha="center")
        ax.set(xlabel=xl, ylabel=yl, title="Measured runs (mixed shapes; descriptive only)")
        ax.grid(alpha=.2)
        fig.savefig(folder / (name + ".png"), dpi=150)
        plt.close(fig)
        links.append(f"plots/{name}.png")
    runs = sorted({r["run_id"] for r in windows if r["phase"] == "sustained" and r["status"] == "PASS"})
    for i, run in enumerate(runs):
        rows = sorted([r for r in windows if r["run_id"] == run], key=lambda r: ns(r["start_monotonic_ns"]))
        first = ns(rows[0]["start_monotonic_ns"])
        xs = [(ns(r["start_monotonic_ns"]) + ns(r["end_monotonic_ns"]) - 2 * first) / 2e9 for r in rows]
        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
        for ax, field, label in zip(axes, ("tflops", "sm_clock_avg_mhz", "power_avg_w"), ("TFLOPS", "SM clock (MHz)", "Power (W)")):
            ys = [num(r.get(field)) for r in rows]
            ax.plot(xs, [v if v is not None else math.nan for v in ys], marker=".", linewidth=1)
            if not any(v is not None for v in ys):
                ax.text(.5, .5, "Unavailable", transform=ax.transAxes, ha="center")
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
        axes[0].set_title(f'{rows[0]["case_id"]} / {rows[0]["api"]} (window averages)')
        axes[-1].set_xlabel("Elapsed formal-load time (s)")
        name = f"sustained_{i:02d}.png"
        fig.savefig(folder / name, dpi=150)
        plt.close(fig)
        links.append("plots/" + name)
    return links, ""


def report(out, env, summary, windows, chart_links, chart_note):
    peak = env.get("theoretical_peak", {})
    good = sorted([r for r in summary if r["status"] == "PASS" and num(r.get("tflops")) is not None],
                  key=lambda r: r["tflops"], reverse=True)
    lines = ["# FP16 Dense GEMM 性能、功耗与时钟分析", "",
             f'- GPU：{env.get("gpu_name", "未知")}；UUID：{env.get("gpu_uuid", "未知")}。',
             f'- 形态：{peak.get("form_factor", "待验证")}；SM 数：{env.get("sm_count", "未知")}；CUDA Runtime：{env.get("cuda_runtime", "未知")}；cuBLAS：{env.get("cublas_version", "未知")}。',
             "- FP16 A/B/C、FP32 累加，alpha=1、beta=0；column-major；Dense；固定随机输入并复用操作数。",
             "- TFLOPS 使用各 CUDA Event 批次的单次 GEMM 耗时中位数；同时保存 mean/min。初始化、校验、调优、预热不计入正式耗时。",
             f'- 有效 {len(good)} 组；其他状态 {len(summary)-len(good)} 组。运行状态：{env.get("run_status", "unknown")}。', ""]
    if peak.get("verified_for_device"):
        lines += [f'理论 Dense FP16 峰值 **{peak["dense_fp16_tflops"]} TFLOPS**；型号、SM 数及显存配置匹配。[规格来源]({peak["source"]})。', ""]
    else:
        lines += ["**未匹配 H800 PCIe 80GB 完整配置，峰值效率留空；当前结果不能作为 H800 性能结论。**", ""]
    if good:
        b = good[0]
        lines += ["## 本次最高有效成绩", "",
                  f'**{fmt(b["tflops"])} TFLOPS**；{b["M"]}×{b["N"]}×{b["K"]}，{b["transA"]}{b["transB"]}，{b["api"]}，阶段 {b["phase"]}。',
                  f'- 算法：{b.get("algorithm", "N/A")}；workspace：{b.get("workspace_bytes") or "N/A"} bytes。',
                  f'- 平均功耗 {fmt(b.get("power_avg_w"))} W；Power Limit {fmt(b.get("power_limit_w"))} W；平均 SM Clock {fmt(b.get("sm_clock_avg_mhz"))} MHz。',
                  f'- 功耗限频：{b.get("power_throttle_observed", "unknown")}；温度限频：{b.get("thermal_throttle_observed", "unknown")}；采样质量 {b["telemetry_quality"]}。',
                  f'- 标称峰值效率 {fmt(b.get("peak_efficiency_pct"))}%；Tensor Core 证据：{b.get("tensor_core_evidence")}；MMA/WGMMA 路径：{b.get("instruction_path")}。',
                  "- cuBLASLt 数值实现标志可确认 Tensor Core 家族，不能区分 HMMA/HGMMA；GemmEx 无指令证据时明确标记未验证，需单独 profiler/SASS 检查。", ""]
    else:
        lines += ["没有正确性通过且完整结束的有效测量；请查看 logs 和 summary 的 reason。", ""]
    lines += ["## API 与 Shape 对照", "", "| case | 阶段 | API | TFLOPS | 功耗 W | SM MHz | 样本/质量 |", "|---|---|---|---:|---:|---:|---|"]
    for r in sorted(good, key=lambda r: (r["phase"], r["case_id"], r["api"])):
        lines.append(f'| {r["case_id"]} | {r["phase"]} | {r["api"]} | {fmt(r["tflops"])} | {fmt(r.get("power_avg_w"))} | {fmt(r.get("sm_clock_avg_mhz"))} | {r["telemetry_samples"]}/{r["telemetry_quality"]} |')
    pair = defaultdict(dict)
    for r in good:
        if r["phase"] == "scan":
            pair[r["case_id"]][r["api"]] = r
    pairs = [x for x in pair.values() if len(x) == 2]
    if pairs:
        wins = sum(x["cublaslt"]["tflops"] > x["cublas"]["tflops"] for x in pairs)
        lines += ["", f"同 case 扫描有效配对 {len(pairs)} 组；Lt 数值更高 {wins} 组，Ex 更高或相等 {len(pairs)-wins} 组。微小差异未做显著性检验。"]
    lines += ["", "## 功耗、温度与频率证据", "",
              "- 功耗限频仅由 SW Power Cap/HW Power Brake 位判定；温度限频仅由 SW/HW Thermal 位判定。通用 HW Slowdown 单独记录，不能直接当作温度限频。",
              "- 达到 Power Limit 的 95% 仅为接近上限，不替代 reason。采样比例不是精确事件持续时长。",
              "- reason 缺失为 unknown；采样不足时未采到事件不等于没有事件。轮询周期不等于传感器刷新周期。",
              "- 只使用完整查询区间落在正式窗口内且 UUID 一致的点，排除跨阶段/跨窗口查询，不插值。",
              "- 少于 3 点或观测跨度覆盖不足 50% 标记 INSUFFICIENT，各指标另有有效样本数。", ""]
    for r in good:
        if r["power_throttle_observed"] == "true" or r["thermal_throttle_observed"] == "true" or r.get("other_compute_pids"):
            lines.append(f'- {r["case_id"]}/{r["api"]}/{r["phase"]}：power={r["power_throttle_observed"]}，thermal={r["thermal_throttle_observed"]}，其他 PID={r.get("other_compute_pids") or "未观察到"}。')
    if not any(r.get("process_visibility") == "available" for r in summary):
        lines.append("- 运行中进程可见性不足，不能排除其他工作负载；结合 environment.json 的运行前检查。")
    lines += ["", "## 持续负载窗口与趋势", ""]
    sustained = sorted({r["run_id"] for r in windows if r["phase"] == "sustained" and r["status"] == "PASS"})
    if not sustained:
        lines.append("没有有效持续负载窗口，不能判断衰减。")
    for run in sustained:
        rows = sorted([r for r in windows if r["run_id"] == run], key=lambda r: ns(r["start_monotonic_ns"]))
        lines.append(f'### {rows[0]["case_id"]} / {rows[0]["api"]}')
        if len(rows) < 8:
            lines += [f"只有 {len(rows)} 个窗口：不足以判断稳定趋势或相关性。", ""]
            continue
        third = max(1, len(rows) // 3)
        early, late = rows[:third], rows[-third:]
        a, b = st.median(values(early, "tflops")), st.median(values(late, "tflops"))
        lc = values(late, "sm_clock_avg_mhz")
        cv_t = st.pstdev(values(late, "tflops")) / b if b else math.inf
        cv_c = st.pstdev(lc) / st.mean(lc) if len(lc) >= 3 and st.mean(lc) else math.inf
        lines += [f"- 前/后三分之一 TFLOPS 中位数 {a:.2f} → {b:.2f}，变化 {(b/a-1)*100:.2f}%；是正式负载内的早晚对照，不是冷启动对照。",
                  f'- 后段稳定性筛选：{"通过" if cv_t <= .03 and cv_c <= .03 else "未通过或时钟数据不足"}（两项 CV ≤3%；通过不证明热稳态）。']
        for x, y in [("sm_clock_avg_mhz", "tflops"), ("sm_clock_avg_mhz", "power_avg_w"), ("tflops", "power_avg_w")]:
            if cv_t <= .03 and cv_c <= .03:
                corr, why = correlation(late, x, y)
            else:
                corr, why = None, "后段未通过稳定性筛选或时钟数据不足，不能判断稳态相关性"
            lines.append(f"- {y} vs {x}（仅后段窗口）：r={fmt(corr, 3)}；{why}。")
        matched = [r for r in rows if r["telemetry_quality"] == "OK" and r["power_throttle_observed"] == "true"]
        lines.append(f"- {len(matched)} 个窗口采到功耗限制位。与同窗频率和 TFLOPS 比较；同时下降仍不能独立证明因果。\n")
    lines += ["## 归因边界", "",
              "1. 小矩阵/短 K：同 API、转置下比较规模扩展，低吞吐可能涉及启动、调度及数据供给。",
              "2. 算法效率：固定 Shape、输入、计时协议比较 API。Tile/Stages/Cluster 是内部枚举，不换算成自定义 kernel 的 CTA 尺寸。",
              "3. 功耗/温度：先要求明确 reason 位，再检查同条件窗口的频率和吞吐变化；高功耗本身不是限频证据。",
              "4. 频率差：型号配置匹配且采样充分时，才计算 clock_scaled_peak。按平均频率线性折算只是参考模型，不是实测上限或因果分解。",
              "5. 其他负载：其他 PID 或进程信息不可见时须保留干扰可能。工具只读配置，不改 Power Limit、不锁频、不改持久化设置。",
              "6. 持续测试重新搜索合法算法，算法可能与扫描不同；必须按记录比较，不能假定内核不变。",
              "", "## 图表", ""]
    lines += [f"![{Path(p).stem}]({p})" for p in chart_links]
    if chart_note:
        lines.append(chart_note)
    evidence_files = sorted((out / "logs").glob("profile_*/tensor_core_evidence.json"))
    if evidence_files:
        lines += ["", "## 独立 Tensor Core 证据", "",
                  "以下为独立 profiler 重跑；其计时不参与上述排名，也不能直接证明原扫描执行了完全相同的内核。"]
        for path in evidence_files:
            try:
                evidence = json.loads(path.read_text(encoding="utf-8"))
                lines.append(f'- [{path.parent.name}]({path.relative_to(out).as_posix()})：'
                             f'API={evidence.get("api")}，动态计数器观察={evidence.get("tensor_core_observed")}，'
                             f'指令={evidence.get("instruction_path")}，状态={evidence.get("status")}。')
                if evidence.get("profiler_diagnostics") or evidence.get("error"):
                    lines.append("  profiler 存在诊断信息；请阅读证据 JSON 与 profile.log，驱动兼容性问题需在目标环境复核。")
            except (ValueError, OSError):
                lines.append(f"- {path.parent.name}：证据文件不可读，未用于结论。")
    lines += ["", "## 原始数据与复现", "",
              "- benchmark_raw.csv：逐窗口 Event 样本、单调时钟和 UTC 时间；phase_events.csv：初始化/调优/校验/预热/正式阶段。",
              "- gpu_telemetry.csv：原始值；telemetry_labeled.csv：严格时间关联标签；window_summary.csv：逐窗口关联指标。",
              "- benchmark_summary.csv：从全部样本重新计算统计；algorithm_trials.csv：实际执行候选的正确性及调优数据。",
              "- environment.json、run_index.csv、配置副本及 logs 保存设备、版本、命令和退出码。失败或中断的整个 run_id 排除排名。",
              "- 正确性为 CPU FP64 确定性抽样参考，加完整输出 NaN/Inf 检查；不是大型矩阵逐元素参考验证。", ""]
    (out / "performance_report.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(out):
    ep = out / "environment.json"
    env = json.loads(ep.read_text()) if ep.exists() else {}
    raw, telemetry = read_csv(out / "benchmark_raw.csv"), read_csv(out / "gpu_telemetry.csv")
    index_rows = read_csv(out / "run_index.csv")
    index = {r["run_id"]: r for r in index_rows}
    groups = defaultdict(list)
    for r in raw:
        groups[r["run_id"]].append(r)
    for r in index_rows:
        if r["run_id"] not in groups:
            groups[r["run_id"]].append({**r, "status": r.get("status", "INCOMPLETE")})
    if (out / "run_index.csv").exists():
        for run in groups:
            if run not in index:
                index[run] = {"status": "INCOMPLETE", "reason": "No completed process record"}
    summary, windows = [], []
    for run, group in groups.items():
        row = aggregate(group, telemetry, index, env.get("theoretical_peak", {}))
        summary.append(row)
        for w in group:
            if ns(w.get("end_monotonic_ns")) > ns(w.get("start_monotonic_ns")) > 0:
                win = aggregate([w], telemetry, index, env.get("theoretical_peak", {}))
                win["window"] = w.get("window", "")
                if row["status"] != "PASS":
                    win.update(status=row["status"], tflops="", tflops_mean="", peak_efficiency_pct="", correctness_pass="")
                windows.append(win)
    write_csv(out / "benchmark_summary.csv", summary, BASE + EXTRA)
    write_csv(out / "window_summary.csv", windows, BASE + EXTRA + ["window"])
    phases = read_csv(out / "phase_events.csv")
    labeled = []
    for sample in telemetry:
        r = dict(sample)
        matches = [e for e in phases if e.get("gpu_uuid", "").lower() == sample.get("gpu_uuid", "").lower()
                   and contained(sample, ns(e["start_monotonic_ns"]), ns(e["end_monotonic_ns"]))]
        if len(matches) == 1:
            r.update({k: matches[0][k] for k in ("run_id", "case_id", "phase")})
        else:
            t = ns(sample.get("timestamp_monotonic_ns"))
            inside = any(ns(e["start_monotonic_ns"]) <= t <= ns(e["end_monotonic_ns"]) for e in phases)
            r.update(run_id="", case_id="", phase="boundary" if inside else "idle_or_unclassified")
        labeled.append(r)
    if telemetry:
        write_csv(out / "telemetry_labeled.csv", labeled, list(telemetry[0]))
    links, note = plots(out, summary, windows)
    report(out, env, summary, windows, links, note)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    a = ap.parse_args()
    analyze(a.run_dir.resolve())
    print(a.run_dir.resolve() / "performance_report.md")
