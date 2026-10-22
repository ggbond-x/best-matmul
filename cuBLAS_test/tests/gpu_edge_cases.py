#!/usr/bin/env python3
"""Verify OOM/overflow guards without actually exhausting device memory."""
import argparse
import csv
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--exe", type=Path, required=True)
args = parser.parse_args()
exe = str(args.exe.resolve())
with tempfile.TemporaryDirectory() as directory:
    for index, flags in enumerate((
        ["--m", "1000000", "--n", "1000000", "--k", "1000000"],
        ["--m", "2147483647", "--n", "2147483647", "--k", "2147483647", "--dtype", "fp32"],
        ["--backend", "cublaslt", "--workspace-mb", "1048576", "--m", "3", "--n", "5", "--k", "7"],
    )):
        path = Path(directory) / ("skip{}.csv".format(index))
        proc = subprocess.run([exe, "--csv", str(path)] + flags, capture_output=True, text=True)
        assert proc.returncode == 3, (proc.stdout, proc.stderr)
        with path.open(newline="") as file:
            result = list(csv.DictReader(file))[0]
        assert result["Status"] == "SKIP"
        assert result["Time_ms"] == "" and result["TFLOPS"] == ""
        assert "memory" in result["Reason"].lower() or "overflow" in result["Reason"].lower(), result
    path = Path(directory) / "after.csv"
    proc = subprocess.run([exe, "--m", "17", "--n", "19", "--k", "23", "--warmup", "1",
                           "--repeats", "2", "--csv", str(path)], capture_output=True, text=True)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
print("GPU OOM/overflow/workspace guards and subsequent valid case passed")

