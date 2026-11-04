#!/usr/bin/env python3
"""Orchestrate FP16 peak scans and sustained load; no hardware settings are changed."""
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
SHANGHAI = dt.timezone(dt.timedelta(hours=8))
from analyze_results import analyze, read_csv, write_csv, BASE, aggregate
from monitor_gpu import snapshot, command, FIELDS

ROOT = Path(__file__).resolve().parents[1]
SHAPE_FIELDS = ["case_id", "M", "N", "K", "transA", "transB", "dtype", "compute_type", "alpha", "beta", "enabled", "category"]
DEFAULTS = {"scan_seconds": 3., "sustained_seconds": 45., "window_seconds": 1., "warmup_seconds": 2.,
            "warmup_batches": 20, "minimum_samples": 10, "inner": 20, "lt_candidates": 32,
            "workspace_mib": 128, "check_samples": 1024, "telemetry_interval_ms": 150.,
            "sustained_top": 3, "case_timeout_seconds": 600., "timing": "graph"}
INT_KEYS = {"warmup_batches", "minimum_samples", "inner", "lt_candidates", "workspace_mib", "check_samples", "sustained_top"}
INDEX_FIELDS = ["run_id", "case_id", "phase", "api", "M", "N", "K", "transA", "transB", "dtype", "compute_type",
                "gpu_uuid", "pid", "returncode", "status", "reason", "command", "start_monotonic_ns", "end_monotonic_ns"]


def load_config(path):
    cfg = dict(DEFAULTS)
    rows = read_csv(path)
    if not rows or not {"parameter", "value"}.issubset(rows[0]):
        raise ValueError("Config requires parameter,value CSV columns and rows")
    seen = set()
    for row in rows:
        key, value = row["parameter"].strip(), row["value"].strip()
        if key in seen or key not in cfg:
            raise ValueError(f"Duplicate/unknown config key: {key}")
        seen.add(key)
        if key == "timing":
            if value not in ("graph", "launch"):
                raise ValueError("timing must be graph or launch")
            cfg[key] = value
        else:
            x = float(value)
            if not math.isfinite(x) or x < 0 or (x == 0 and key not in ("workspace_mib", "warmup_batches", "warmup_seconds")):
                raise ValueError(f"Invalid config value {key}")
            if key in INT_KEYS:
                if x != int(x):
                    raise ValueError(f"{key} must be integral")
                x = int(x)
            cfg[key] = x
    if not 20 <= cfg["telemetry_interval_ms"] <= 10000 or not 1 <= cfg["lt_candidates"] <= 512:
        raise ValueError("Telemetry interval or candidate count outside supported range")
    if cfg["inner"] > 65536 or cfg["workspace_mib"] > 16384 or cfg["sustained_top"] > 10:
        raise ValueError("inner/workspace/top exceeds safety bound")
    if cfg["case_timeout_seconds"] < cfg["sustained_seconds"] + cfg["warmup_seconds"] + 5:
        raise ValueError("case_timeout_seconds too short for sustained load plus preparation")
    return cfg


def load_shapes(path):
    rows = read_csv(path)
    if not rows or not set(SHAPE_FIELDS).issubset(rows[0]):
        raise ValueError("Shape CSV missing required columns")
    seen, selected = set(), []
    for row in rows:
        row = {k: row[k].strip() for k in SHAPE_FIELDS}
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", row["case_id"]) or row["case_id"] in seen:
            raise ValueError(f"Invalid/duplicate case_id: {row['case_id']}")
        seen.add(row["case_id"])
        for key in ("M", "N", "K"):
            if not row[key].isdigit() or not 1 <= int(row[key]) <= 2147483647:
                raise ValueError(f"{row['case_id']}: invalid {key}")
            row[key] = int(row[key])
        if row["enabled"] not in ("0", "1"):
            raise ValueError("enabled must be 0 or 1")
        if row["transA"] not in ("N", "T") or row["transB"] not in ("N", "T"):
            raise ValueError("transA/transB must be N or T")
        if row["dtype"] != "FP16" or row["compute_type"] != "FP32" or row["alpha"] not in ("1", "1.0") or row["beta"] not in ("0", "0.0"):
            raise ValueError("This protocol requires FP16 A/B/C, FP32 compute, alpha=1, beta=0")
        if row["enabled"] == "1":
            selected.append(row)
    if not selected:
        raise ValueError("No enabled cases")
    return selected


def quick_shapes():
    result = []
    for ta, tb in (("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")):
        result.append(dict(zip(SHAPE_FIELDS, [f"quick_rect_{ta}{tb}", 128, 192, 64, ta, tb, "FP16", "FP32", "1", "0", "1", "quick"])))
    result.append(dict(zip(SHAPE_FIELDS, ["quick_square_NN", 1024, 1024, 1024, "N", "N", "FP16", "FP32", "1", "0", "1", "quick"])))
    return result


def choose_top(rows, shapes, count):
    allowed = {s["case_id"]: s for s in shapes}
    valid = [r for r in rows if r.get("status") == "PASS" and r.get("correctness_pass") == "true"
             and r.get("case_id") in allowed and r.get("tflops") not in ("", None)]
    chosen, sizes = [], set()
    for r in sorted(valid, key=lambda r: float(r["tflops"]), reverse=True):
        s = allowed[r["case_id"]]
        key = (s["M"], s["N"], s["K"])
        if key not in sizes:
            sizes.add(key)
            chosen.append(s)
            if len(chosen) == count:
                break
    return chosen


def peak_for_device(env):
    name = env.get("gpu_name", "")
    match = (name.strip() in ("NVIDIA H800 PCIe", "NVIDIA H800 PCIe 80GB") and env.get("sm_count") == 114
             and env.get("compute_capability") == "9.0" and 70 * 2**30 <= env.get("vram_bytes", 0) <= 85 * 2**30
             and str(env.get("gpu_uuid", "")).startswith("GPU-"))
    return {"verified_for_device": match, "form_factor": "PCIe" if match else "待验证",
            "dense_fp16_tflops": 756.5 if match else None, "reference_clock_mhz": 1620 if match else None,
            "source": "https://lenovopress.lenovo.com/lp1814-thinksystem-nvidia-h800-pcie-gen5-gpu",
            "clock_source": "https://arxiv.org/html/2501.12084v1",
            "matching_rule": "Exact H800 PCIe name, 114 SM, sm90, 70-85 GiB visible memory, physical GPU UUID",
            "note": "1513 sparse / 2 = 756.5 dense; reference clock scaling is illustrative, not a measured ceiling"}


def file_hash(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def terminate(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0, help="physical nvidia-smi GPU index; CUDA is bound by UUID")
    ap.add_argument("--mode", choices=["scan", "sustained", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true", help="validate and print plan; no GPU/build/output required")
    ap.add_argument("--quick", action="store_true", help="short functional test, not a peak claim")
    ap.add_argument("--shapes", type=Path, default=ROOT / "csvs/h800_pcie_fp16_shapes.csv")
    ap.add_argument("--config", type=Path, default=ROOT / "csvs/benchmark_config.csv")
    ap.add_argument("--scan-results", type=Path, help="existing summary for sustained-only selection")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--exe", type=Path)
    ap.add_argument("--monitor-backend", choices=["auto", "nvml", "nvidia-smi"], default="auto")
    a = ap.parse_args()
    if a.gpu < 0:
        ap.error("gpu must be nonnegative")
    cfg, shapes = load_config(a.config.resolve()), load_shapes(a.shapes.resolve())
    if a.quick:
        shapes = quick_shapes()
        cfg.update(scan_seconds=1., sustained_seconds=3., window_seconds=.6, warmup_seconds=.15,
                   warmup_batches=2, minimum_samples=4, lt_candidates=4, check_samples=256,
                   sustained_top=2, workspace_mib=32)
    if a.dry_run:
        print(json.dumps({"gpu_physical_index": a.gpu, "mode": a.mode, "config": cfg, "shapes": shapes,
                          "apis": ["cublas", "cublaslt"], "sustained_selection": "top distinct M/N/K from scan; both APIs",
                          "hardware_settings": "read-only", "output_root": str(ROOT / "output")}, ensure_ascii=False, indent=2))
        return 0
    stamp = dt.datetime.now(SHANGHAI).strftime("%Y%m%d_%H%M%S_%f")
    out = ROOT / "output" / f"h800_peak_{stamp}_{os.getpid()}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    shutil.copy2(a.config, out / "benchmark_config.csv")
    write_csv(out / "h800_pcie_fp16_shapes.csv", shapes, SHAPE_FIELDS)
    print(f"Output: {out}", flush=True)
    env = {"created_at": dt.datetime.now(SHANGHAI).isoformat(), "quick": a.quick,
           "mode": a.mode, "config": cfg, "python": sys.version, "run_status": "PREPARING",
           "warnings": [], "original_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "source_hashes": {str(p.relative_to(ROOT)): file_hash(p) for p in
                [Path(__file__), ROOT / "scripts/monitor_gpu.py", ROOT / "scripts/analyze_results.py",
                 ROOT / "src/h800_peak.cu", ROOT / "src/cublaslt_gemm.cpp", ROOT / "src/cublas_gemm.cpp"]}}
    for name in ("benchmark_raw.csv", "gpu_telemetry.csv", "phase_events.csv", "algorithm_trials.csv", "run_index.csv"):
        # Empty real-data artifacts even on early failure. No synthetic measurements.
        (out / name).touch()
    write_csv(out / "gpu_telemetry.csv", [], FIELDS)
    write_csv(out / "run_index.csv", [], INDEX_FIELDS)
    monitor = child = None
    monitor_log = None
    def save_env():
        (out / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    index_rows = []
    save_env()
    code = 0
    try:
        if not shutil.which("nvidia-smi"):
            raise RuntimeError("nvidia-smi missing")
        gpu_info = list(csv.reader(command(["nvidia-smi", "-i", str(a.gpu), "--query-gpu=uuid,name,driver_version",
                                           "--format=csv,noheader"]).splitlines()))
        if len(gpu_info) != 1:
            raise RuntimeError("Expected exactly one physical GPU")
        gpu_uuid, gpu_name, driver = [x.strip() for x in gpu_info[0]]
        child_env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
        env.update(gpu_uuid=gpu_uuid, gpu_name=gpu_name, driver_version=driver, physical_gpu_index=a.gpu)
        nvcc = os.environ.get("NVCC") or str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc")
        if not Path(nvcc).exists():
            nvcc = shutil.which("nvcc")
        build = Path(os.environ.get("H800_BUILD_DIR", str(ROOT / "build_h800_peak"))).resolve()
        exe = a.exe.resolve() if a.exe else build / "h800_peak_bench"
        if not a.skip_build:
            if not nvcc or not shutil.which("cmake"):
                raise RuntimeError("CUDA nvcc and cmake required (or use --skip-build --exe)")
            env["nvcc_version"] = command([nvcc, "--version"])
            with (out / "logs/build.log").open("w") as log:
                for cmd in [["cmake", "-S", str(ROOT), "-B", str(build), f"-DCMAKE_CUDA_COMPILER={nvcc}",
                             "-DCMAKE_BUILD_TYPE=Release", "-DGEMM_CUDA_ARCHITECTURES=" + os.environ.get("GPU_ARCHS", "80;90")],
                            ["cmake", "--build", str(build), "--target", "h800_peak_bench", "-j", os.environ.get("JOBS", "4")]]:
                    log.write(shlex.join(cmd) + "\n")
                    log.flush()
                    child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                    rc = child.wait()
                    child = None
                    if rc:
                        raise RuntimeError(f"Build failed ({rc}); see logs/build.log")
        if not exe.is_file():
            raise RuntimeError(f"Benchmark executable missing: {exe}")
        env["executable_sha256"] = file_hash(exe)
        info = json.loads(subprocess.check_output([str(exe), "--info", "--device", "0"], env=child_env, text=True))
        if info["gpu_uuid"].lower() != gpu_uuid.lower():
            raise RuntimeError("CUDA and nvidia-smi UUID differ; refusing wrong-GPU telemetry")
        env.update(info)
        env["theoretical_peak"] = peak_for_device(env)
        env["telemetry_before_run"] = snapshot(gpu_uuid)
        for module in ("pynvml", "matplotlib"):
            env[module + "_available"] = importlib.util.find_spec(module) is not None
        if not env["matplotlib_available"]:
            env["warnings"].append("matplotlib unavailable; plot source CSV will still be generated")
        try:
            processes = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"])
            env["compute_processes_before_run"] = processes
            if any(gpu_uuid.lower() in line.lower() for line in processes.splitlines()):
                env["warnings"].append("Other compute processes observed on target GPU before run")
        except Exception as e:
            env["warnings"].append(f"Process inventory unavailable: {e}")
        utilization = env["telemetry_before_run"].get("sample", {}).get("gpu_utilization_pct")
        if utilization is not None and float(utilization) > 10:
            env["warnings"].append(f"Target GPU initial utilization {utilization}%: possible other workload")
        for warning in env["warnings"]:
            print("WARNING:", warning, flush=True)
        monitor_log = (out / "logs/monitor.log").open("w")
        monitor = subprocess.Popen([sys.executable, str(ROOT / "scripts/monitor_gpu.py"), "--uuid", gpu_uuid,
                                   "--out", str(out / "gpu_telemetry.csv"), "--ready", str(out / "monitor_ready.json"),
                                   "--stop-file", str(out / "monitor.stop"), "--interval-ms", str(cfg["telemetry_interval_ms"]),
                                   "--backend", a.monitor_backend], stdout=monitor_log, stderr=subprocess.STDOUT)
        env["monitor_pid"] = monitor.pid
        deadline = time.monotonic() + 60
        while not (out / "monitor_ready.json").exists() and monitor.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        if (out / "monitor_ready.json").exists():
            env["monitor"] = json.loads((out / "monitor_ready.json").read_text())
        else:
            env["warnings"].append("Monitor did not become ready; measurements will be marked missing")
        env["run_status"] = "RUNNING"
        save_env()
        counter = 0

        def execute(shape, api, phase):
            nonlocal child, counter
            counter += 1
            rid = f"{counter:04d}_{phase}_{shape['case_id']}_{api}"
            cmd = [str(exe), "--device", "0", "--m", str(shape["M"]), "--n", str(shape["N"]), "--k", str(shape["K"]),
                   "--transA", shape["transA"], "--transB", shape["transB"], "--api", api,
                   "--case-id", shape["case_id"], "--run-id", rid, "--phase", phase,
                   "--seconds", str(cfg["scan_seconds"] if phase == "scan" else cfg["sustained_seconds"]),
                   "--window-seconds", str(cfg["window_seconds"]), "--warmup-seconds", str(cfg["warmup_seconds"]),
                   "--warmup", str(cfg["warmup_batches"]), "--inner", str(cfg["inner"]),
                   "--repeats", str(cfg["minimum_samples"]), "--samples", str(cfg["check_samples"]),
                   "--workspace-mib", str(cfg["workspace_mib"]), "--lt-candidates", str(cfg["lt_candidates"]),
                   "--timing", cfg["timing"], "--raw", str(out / "benchmark_raw.csv"),
                   "--events", str(out / "phase_events.csv"), "--trials", str(out / "algorithm_trials.csv")]
            row = {**shape, "run_id": rid, "api": api, "phase": phase, "gpu_uuid": gpu_uuid,
                   "command": shlex.join(cmd), "start_monotonic_ns": time.monotonic_ns(), "status": "INTERRUPTED"}
            print(f"[{counter}] {phase} {shape['case_id']} {api}", flush=True)
            try:
                with (out / "logs" / (rid + ".log")).open("w") as log:
                    child = subprocess.Popen(cmd, env=child_env, stdout=log, stderr=subprocess.STDOUT)
                    row["pid"] = child.pid
                    try:
                        rc = child.wait(timeout=cfg["case_timeout_seconds"])
                        row.update(returncode=rc, status="PASS" if rc == 0 else "SKIP" if rc == 3 else "FAIL",
                                   reason="" if rc == 0 else f"Process exit {rc}; see per-run log")
                    except subprocess.TimeoutExpired:
                        row.update(status="TIMEOUT", reason="Case timeout")
                        terminate(child)
            finally:
                terminate(child)
                child = None
                row["end_monotonic_ns"] = time.monotonic_ns()
                index_rows.append(row)
                write_csv(out / "run_index.csv", index_rows, INDEX_FIELDS)

        if a.mode in ("scan", "all"):
            for shape in shapes:
                for api in ("cublas", "cublaslt"):
                    execute(shape, api, "scan")
        if a.mode in ("sustained", "all"):
            if a.mode == "all":
                groups = {}
                for r in read_csv(out / "benchmark_raw.csv"):
                    groups.setdefault(r["run_id"], []).append(r)
                index = {r["run_id"]: r for r in index_rows}
                ranked = [aggregate(g, [], index, {}) for g in groups.values()]
                top = choose_top(ranked, shapes, cfg["sustained_top"])
            elif a.scan_results:
                top = choose_top(read_csv(a.scan_results.resolve()), shapes, cfg["sustained_top"])
            else:
                top = [s for s in shapes if s["transA"] == "N" and s["transB"] == "N"]
                top.sort(key=lambda s: (s["M"] != s["N"], -s["M"] * s["N"] * s["K"]))
                top = top[:cfg["sustained_top"]]
                env["warnings"].append("Sustained-only without prior scan: preset large shapes, not measured best")
            env["sustained_cases"] = [s["case_id"] for s in top]
            save_env()
            for shape in top:
                for api in ("cublas", "cublaslt"):
                    execute(shape, api, "sustained")
        env["run_status"] = "COMPLETE" if all(r["status"] == "PASS" for r in index_rows) else "COMPLETE_WITH_SKIPS_OR_FAILURES"
        if not any(r["status"] == "PASS" for r in index_rows) or any(r["status"] in ("FAIL", "TIMEOUT") for r in index_rows):
            code = 1
    except KeyboardInterrupt:
        env["run_status"] = "INTERRUPTED"
        code = 130
    except Exception as e:
        env["run_status"] = "FAILED"
        env["error"] = str(e)
        print("ERROR:", e, file=sys.stderr, flush=True)
        code = 1
    finally:
        terminate(child)
        (out / "monitor.stop").touch()
        if monitor:
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                terminate(monitor)
            env["monitor_returncode"] = monitor.returncode
        if monitor_log:
            monitor_log.close()
        env["finished_at"] = dt.datetime.now(SHANGHAI).isoformat()
        save_env()
        try:
            analyze(out)
        except Exception as e:
            env["analysis_error"] = str(e)
            save_env()
            (out / "performance_report.md").write_text(f"# Analysis failed\n\n{e}\n\nRaw data retained in this directory.\n")
            code = code or 1
        print(f"Report: {out / 'performance_report.md'}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
