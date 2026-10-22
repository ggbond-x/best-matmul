#!/usr/bin/env python3
"""Real GPU correctness regression; small cases, not a performance report."""
import argparse
import csv
import itertools
from pathlib import Path
import subprocess
import sys
import tempfile

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    executable = str(args.exe.resolve())
    scenarios = []
    for backend, dtype, trans, beta in itertools.product(
            ("cublas", "cublaslt"), ("fp16", "bf16", "fp32", "tf32"), ("NN", "NT", "TN", "TT"), ("0", "0.5")):
        scenarios.append(["--backend", backend, "--dtype", dtype, "--transa", trans[0], "--transb", trans[1],
                          "--m", "17", "--n", "23", "--k", "31", "--alpha", "-0.75", "--beta", beta, "--check-all"])
    for backend, dtype, beta in itertools.product(("cublas", "cublaslt"), ("fp16", "bf16", "fp32", "tf32"), ("0", "2")):
        scenarios.append(["--backend", backend, "--dtype", dtype, "--mode", "graph", "--beta", beta,
                          "--m", "33", "--n", "47", "--k", "19", "--check-all"])
    for backend, dtype in itertools.product(("cublas", "cublaslt"), ("fp16", "bf16")):
        scenarios.append(["--backend", backend, "--dtype", dtype, "--output-dtype", "fp32",
                          "--m", "19", "--n", "37", "--k", "41", "--transa", "T", "--transb", "T",
                          "--alpha", "0.625", "--beta", "-0.25", "--check-all"])
    for mode, selection in itertools.product(("launch", "graph"), ("index", "tune")):
        scenarios.append(["--backend", "cublaslt", "--dtype", "bf16", "--mode", mode, "--lt-selection", selection,
                          "--lt-candidates", "8", "--workspace-mb", "8", "--m", "127", "--n", "63", "--k", "65"])
    scenarios += [
        ["--m", "1", "--n", "1", "--k", "1", "--alpha", "0", "--beta", "1"],
        ["--m", "3", "--n", "5", "--k", "7", "--alpha", "0", "--beta", "-0.5"],
        ["--m", "7", "--n", "11", "--k", "16385", "--dtype", "tf32", "--check-all"],
        ["--m", "1024", "--n", "1024", "--k", "1024", "--backend", "cublaslt", "--dtype", "fp16", "--lt-selection", "tune", "--lt-candidates", "8"],
    ]
    passed = skipped = failed = 0
    with tempfile.TemporaryDirectory(prefix="gemm-smoke-") as directory:
        result = args.csv or Path(directory) / "gpu.csv"
        if result.exists():
            parser.error("Use a fresh --csv file for regression")
        for index, flags in enumerate(scenarios, 1):
            cmd = [executable, "--warmup", "2", "--repeats", "4", "--inner", "3", "--csv", str(result)] + flags
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                passed += 1
            elif proc.returncode == 3:
                skipped += 1
            else:
                failed += 1
                print("FAILED", " ".join(flags), proc.stdout, proc.stderr, flush=True)
            if index % 16 == 0:
                print("GPU regression {}/{}: PASS={} SKIP={} FAIL={}".format(index, len(scenarios), passed, skipped, failed), flush=True)
        if result.exists():
            with result.open(newline="") as file:
                rows = list(csv.DictReader(file))
            for row in rows:
                if row["Status"] == "PASS":
                    assert int(row["Checked"]) > 0 and int(row["Bad"]) == 0
                    if float(row["Beta"]) != 0:
                        assert row["Inner"] == "1" and row["BetaPolicy"] == "reset_outside_timing"
                    assert len(row["Samples_ms"].split(";")) == int(row["Repeats"])
    print("GPU regression total: PASS={} SKIP={} FAIL={}".format(passed, skipped, failed))
    return 1 if failed else 0 if passed else 3

if __name__ == "__main__":
    sys.exit(main())

