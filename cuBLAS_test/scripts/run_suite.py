#!/usr/bin/env python3
"""Sequential, process-isolated GEMM batch runner; standard library only."""
import argparse
import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", type=Path, default=ROOT / "build/gemm_bench")
    p.add_argument("--cases", type=Path, default=ROOT / "scripts/cases.json")
    p.add_argument("--groups", nargs="+", default=["all"])
    p.add_argument("--backends", nargs="+", choices=["cublas", "cublaslt"], default=["cublas", "cublaslt"])
    p.add_argument("--dtypes", nargs="+", choices=["fp16", "bf16", "fp32", "tf32"], default=["fp16", "bf16", "fp32", "tf32"])
    p.add_argument("--transposes", nargs="+", choices=["NN", "NT", "TN", "TT"], default=["NN"])
    p.add_argument("--modes", nargs="+", choices=["launch", "graph"], default=["launch"])
    p.add_argument("--csv", type=Path, default=ROOT / "results/baseline.csv")
    p.add_argument("--append", action="store_true", help="Explicitly append another run to an existing result")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Trailing gemm_bench options")
    args = p.parse_args()
    config = json.loads(args.cases.read_text())
    groups = list(config) if args.groups == ["all"] else args.groups
    shapes = []
    for group in groups:
        if group not in config:
            p.error("Unknown group: " + group)
        for shape in config[group]:
            if len(shape) != 3 or any(type(x) is not int or not 0 < x <= 2147483647 for x in shape):
                p.error("Each shape must contain three positive int32 dimensions")
            if tuple(shape) not in shapes:
                shapes.append(tuple(shape))
    if not shapes:
        p.error("No cases selected")
    reserved = {"--m", "--n", "--k", "--backend", "--dtype", "--mode", "--transa", "--transb", "--csv", "--suite", "--help", "--device-info"}
    if any(x in reserved for x in args.extra):
        p.error("--extra must not override the case selection or CSV path")
    exe = args.exe.resolve()
    csv_path = args.csv.resolve()
    commands = []
    for (m, n, k), backend, dtype, trans, mode in itertools.product(shapes, args.backends, args.dtypes, args.transposes, args.modes):
        commands.append([str(exe), "--m", str(m), "--n", str(n), "--k", str(k),
                         "--backend", backend, "--dtype", dtype, "--transa", trans[0],
                         "--transb", trans[1], "--mode", mode, "--csv", str(csv_path)] + args.extra)
    if args.dry_run:
        for cmd in commands:
            print(shlex.join(cmd))
        print("Cases:", len(commands))
        return 0
    if not exe.is_file():
        p.error("Build the executable first: " + str(exe))
    if csv_path.exists() and csv_path.stat().st_size and not args.append:
        p.error("Output exists; use a new --csv or explicitly --append")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    manifest = csv_path.with_name(csv_path.stem + "." + stamp + ".manifest.json")
    meta = {
        "utc": stamp, "platform": platform.platform(), "commands": commands,
        "cases": config, "binary_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for directory in ("src", "include", "benchmark")
                          for path in sorted((ROOT / directory).rglob("*")) if path.is_file()},
        "environment": {k: os.environ.get(k, "") for k in (
            "CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG", "CUDA_MODULE_LOADING")},
        "return_codes": [],
    }
    try:
        probe = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        meta["nvidia_smi"] = probe.stdout + probe.stderr
    except (OSError, subprocess.TimeoutExpired) as error:
        meta["nvidia_smi"] = str(error)
    manifest.write_text(json.dumps(meta, indent=2) + "\n")
    failed = passed = skipped = 0
    with csv_path.with_suffix(".run.log").open("a") as log:
        for index, cmd in enumerate(commands, 1):
            print("[{}/{}] {}".format(index, len(commands), shlex.join(cmd)), flush=True)
            run = subprocess.run(cmd, capture_output=True, text=True)
            text = run.stdout + run.stderr
            print(text, end="", flush=True)
            log.write(shlex.join(cmd) + "\n" + text + "\n")
            log.flush()
            meta["return_codes"].append(run.returncode)
            manifest.write_text(json.dumps(meta, indent=2) + "\n")
            if run.returncode == 0:
                passed += 1
            elif run.returncode == 3:
                skipped += 1
            else:
                failed += 1
    print("Processes: PASS={} FAIL={} SKIP={}".format(passed, failed, skipped))
    print("Manifest:", manifest)
    return 1 if failed else 0 if passed else 3

if __name__ == "__main__":
    sys.exit(main())

