#!/usr/bin/env python3
"""Optional separate Nsight Compute evidence run; timings never enter peak ranking."""
import argparse
import csv
import datetime as dt
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from analyze_results import read_csv


def tensor_counts(raw, metric):
    """Accept both NCU long metric rows and wide raw-page tables (including units rows)."""
    start = raw.find('"ID"')
    if start < 0:
        return []
    counts = []
    for row in csv.DictReader(io.StringIO(raw[start:])):
        value = row.get(metric) if metric in row else row.get("Metric Value") if row.get("Metric Name") == metric else None
        try:
            count = float(value.replace(",", ""))
            if math.isfinite(count) and count >= 0:
                counts.append(count)
        except (AttributeError, TypeError, ValueError):
            pass
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--api", choices=["cublas", "cublaslt"], default="cublas")
    ap.add_argument("--ncu", default=shutil.which("ncu") or "ncu")
    ap.add_argument("--metric", choices=["sm__inst_executed_pipe_tensor.sum", "smsp__inst_executed_pipe_tensor.sum"], default="sm__inst_executed_pipe_tensor.sum")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = a.run_dir.resolve()
    valid = [r for r in read_csv(out / "benchmark_summary.csv")
             if r["status"] == "PASS" and r["correctness_pass"] == "true" and r["api"] == a.api and r["tflops"]]
    if not valid:
        ap.error("No valid result for requested API")
    best = max(valid, key=lambda r: float(r["tflops"]))
    entry = next(r for r in read_csv(out / "run_index.csv") if r["run_id"] == best["run_id"])
    cmd = shlex.split(entry["command"])
    prof = out / "logs" / ("profile_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    # Force a short launch-mode evidence probe. The cuBLASLt heuristic is rerun;
    # its observed algorithm must be read from the separate profile raw CSV.
    changes = {"--raw": prof / "probe_raw.csv", "--events": prof / "probe_events.csv", "--trials": prof / "probe_trials.csv",
               "--seconds": ".1", "--window-seconds": ".1", "--warmup-seconds": ".1",
               "--warmup": "2", "--repeats": "1", "--timing": "launch", "--run-id": "profile_probe"}
    for key, val in changes.items():
        cmd[cmd.index(key) + 1] = str(val)
    cmd.append("--profile")
    full = [a.ncu, "--profile-from-start", "off", "--clock-control", "none",
            "--launch-count", "8", "--metrics", a.metric, "--export", str(prof / "tensor_core")] + cmd
    if a.dry_run:
        print(shlex.join(full))
        return 0
    prof.mkdir(parents=True, exist_ok=False)
    evidence = {"source_run_id": best["run_id"], "api": a.api, "gpu_uuid": best["gpu_uuid"],
                "status": "unverified", "tensor_core_observed": None, "instruction_path": "unverified",
                "command": full, "note": "Separate profiler run, launch mode; profiling timings excluded. Lt may select a different algorithm."}
    try:
        if not shutil.which(a.ncu):
            raise RuntimeError("Nsight Compute ncu unavailable")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=best["gpu_uuid"])
        with (prof / "profile.log").open("w") as log:
            p = subprocess.run(full, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        if p.returncode:
            raise RuntimeError(f"ncu exit {p.returncode}; check profile.log and counter permissions")
        rep = prof / "tensor_core.ncu-rep"
        raw = subprocess.check_output([a.ncu, "--import", str(rep), "--page", "raw", "--csv"], text=True)
        (prof / "metrics.csv").write_text(raw)
        # Read only a genuine dynamic Tensor instruction counter, not kernel-name guesses.
        counts = tensor_counts(raw, a.metric)
        if counts:
            evidence["tensor_core_observed"] = any(v > 0 for v in counts)
            evidence["status"] = "counter_observed"
            evidence["counter_values"] = counts
        else:
            evidence["counter_note"] = "Requested dynamic counter is missing or invalid; not evidence of zero execution."
        evidence["profiler_diagnostics"] = [
            line for line in (prof / "profile.log").read_text().splitlines()
            if "==ERROR==" in line or "==WARNING==" in line]
        sass = subprocess.check_output([a.ncu, "--import", str(rep), "--page", "source", "--print-source", "sass"], text=True)
        (prof / "sass.txt").write_text(sass)
        paths = [x for x in ("HGMMA", "HMMA") if re.search(r"\b" + x + r"(?:\.|\b)", sass)]
        evidence["instruction_path"] = "|".join(paths) or "unverified"
        evidence["sass_note"] = "SASS presence is static evidence; positive dynamic tensor counter confirms execution."
    except Exception as e:
        evidence["error"] = str(e)
    (prof / "tensor_core_evidence.json").write_text(json.dumps(evidence, indent=2))
    print(prof / "tensor_core_evidence.json")
    return 0 if evidence["tensor_core_observed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
