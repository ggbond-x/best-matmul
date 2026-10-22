#!/usr/bin/env python3
"""CPU-side CLI validation plus optional-device failure paths."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--exe", type=Path, required=True)
args = parser.parse_args()
exe = str(args.exe.resolve())
for flags in (["--m", "0"], ["--k", "-1"], ["--alpha", "nan"], ["--alpha", "1e100"],
              ["--dtype", "fp8"], ["--backend", "unknown"], ["--transa", "X"],
              ["--layout", "diagonal"],
              ["--repeats", "0"], ["--inner", "0"], ["--m", "10junk"],
              ["--lt-selection", "index", "--lt-index", "32"], ["--atol", "-0.1"], ["--m"]):
    p = subprocess.run([exe] + flags, capture_output=True, text=True)
    assert p.returncode == 2, (flags, p.returncode, p.stdout, p.stderr)
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "no_gpu.csv"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([exe, "--m", "3", "--n", "5", "--k", "7", "--csv", str(path)],
                       env=environment, capture_output=True, text=True)
    assert p.returncode == 3, (p.returncode, p.stdout, p.stderr)
    with path.open(newline="") as file:
        result = list(csv.DictReader(file))
    assert len(result) == 1 and result[0]["Status"] == "SKIP"
    assert result[0]["Time_ms"] == "" and result[0]["TFLOPS"] == ""
print("CLI invalid-input and no-GPU checks passed")

