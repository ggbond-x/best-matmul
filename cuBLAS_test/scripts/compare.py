#!/usr/bin/env python3
"""Compare only validated GEMMs with identical semantics and timing protocols."""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import math
from pathlib import Path
import statistics
import struct
import sys

# Never infer missing precision/layout fields from the datatype or backend.
KEY_FIELDS = [
    "GPU", "GPU_UUID", "SM", "Dtype", "InputDtype", "OutputDtype", "Compute", "Reduction",
    "Layout", "M", "N", "K", "TransA", "TransB", "LDA", "LDB", "LDC",
    "Alpha", "Beta", "Mode", "TimingProtocol", "CachePolicy", "BetaPolicy",
    "Warmup", "Repeats", "Inner", "Seed",
]
INT_FIELDS = {"SM", "M", "N", "K", "LDA", "LDB", "LDC", "Warmup", "Repeats", "Inner", "Seed"}
# Repeated identical experiments may be aggregated, but different algorithms,
# workspace limits or library versions must not be silently blended.
EXPERIMENT_FIELDS = ["Algorithm", "WorkspaceBudget_bytes", "WorkspaceUsed_bytes",
                     "CUDA_Runtime", "CUDA_Driver", "cuBLAS_Version",
                     "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG"]

def canonical(name, value):
    value = value.strip()
    if not value:
        raise ValueError("Missing value for " + name)
    if name in INT_FIELDS:
        number = Decimal(value)
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError("Invalid integer in " + name)
        return str(int(number))
    if name in ("Alpha", "Beta"):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Non-finite " + name)
        # alpha/beta are FP32 API scalars, irrespective of CSV display precision.
        number = struct.unpack("f", struct.pack("f", number))[0]
        if not math.isfinite(number):
            raise ValueError("FP32 overflow in " + name)
        return str(number)
    if name in ("Dtype", "InputDtype", "OutputDtype", "Compute", "Reduction", "TransA", "TransB"):
        return value.upper()
    return value

def key(row):
    return tuple(canonical(name, row.get(name, "")) for name in KEY_FIELDS)

def positive(row, name):
    number = float(row.get(name, ""))
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Invalid positive " + name)
    return number

def load(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = set(KEY_FIELDS + ["Backend", "Time_ms", "TFLOPS", "Status"])
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("{}: required fields missing: {}".format(path, ", ".join(sorted(missing))))
        rows = []
        for line, row in enumerate(reader, 2):
            if row["Status"].strip().upper() != "PASS":
                continue
            try:
                key(row)
                elapsed = positive(row, "Time_ms")
                tflops = positive(row, "TFLOPS")
                flops = 2 * int(row["M"]) * int(row["N"]) * int(row["K"])
                expected = flops / (elapsed * 1e9)
                if not math.isclose(tflops, expected, rel_tol=0.005):
                    raise ValueError("TFLOPS disagrees with 2*M*N*K / (Time_ms*1e9)")
                if not row["Backend"].strip():
                    raise ValueError("Backend is empty")
            except (ValueError, InvalidOperation, OverflowError) as error:
                raise ValueError("{}:{}: {}".format(path, line, error)) from error
            rows.append(row)
    return rows

def group(rows, duplicates):
    groups = {}
    for row in rows:
        groups.setdefault((key(row), row["Backend"]), []).append(row)
    result = {}
    for group_key, items in groups.items():
        if len(items) > 1:
            if duplicates == "error":
                raise ValueError("Duplicate matching rows for backend {}; use distinct files or --duplicates median".format(group_key[1]))
            fingerprints = {tuple(row.get(f, "") for f in EXPERIMENT_FIELDS) for row in items}
            if len(fingerprints) != 1:
                raise ValueError("Duplicates contain different algorithms/workspaces/versions; split the experiments")
        row = dict(items[0])
        row["Time_ms"] = str(statistics.median(float(item["Time_ms"]) for item in items))
        row["TFLOPS"] = str(2 * int(row["M"]) * int(row["N"]) * int(row["K"]) / (float(row["Time_ms"]) * 1e9))
        row["Runs"] = str(len(items))
        result[group_key] = row
    return result

def compare(baseline, custom, duplicates="error"):
    base = group(baseline, duplicates)
    test = group(custom, duplicates)
    output, unmatched = [], []
    for (identity, custom_backend), custom_row in test.items():
        matches = [(backend, row) for (k, backend), row in base.items() if k == identity]
        if not matches:
            unmatched.append(custom_row)
            continue
        for backend, baseline_row in matches:
            row = {name: custom_row[name] for name in KEY_FIELDS}
            row.update({
                "BaselineBackend": backend, "CustomBackend": custom_backend,
                "BaselineTime_ms": baseline_row["Time_ms"], "CustomTime_ms": custom_row["Time_ms"],
                "BaselineTFLOPS": baseline_row["TFLOPS"], "CustomTFLOPS": custom_row["TFLOPS"],
                "PerformanceRatio": float(custom_row["TFLOPS"]) / float(baseline_row["TFLOPS"]),
                "Speedup": float(baseline_row["Time_ms"]) / float(custom_row["Time_ms"]),
                "BaselineRuns": baseline_row["Runs"], "CustomRuns": custom_row["Runs"],
                "BaselineAlgorithm": baseline_row.get("Algorithm", ""),
                "BaselineWorkspace_bytes": baseline_row.get("WorkspaceBudget_bytes", ""),
            })
            output.append(row)
    return output, unmatched

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--custom", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-backend", choices=["all", "cublas", "cublaslt"], default="all")
    parser.add_argument("--duplicates", choices=["error", "median"], default="error")
    parser.add_argument("--strict", action="store_true", help="Nonzero exit if any custom row has no match")
    args = parser.parse_args()
    try:
        if args.output.resolve() in (args.baseline.resolve(), args.custom.resolve()):
            raise ValueError("Output must not overwrite either input CSV")
        baseline = [row for row in load(args.baseline)
                    if row["Backend"] in ("cublas", "cublaslt")
                    and (args.baseline_backend == "all" or row["Backend"] == args.baseline_backend)]
        custom = load(args.custom)
        if not baseline or not custom:
            raise ValueError("Both files must contain eligible PASS rows")
        output, unmatched = compare(baseline, custom, args.duplicates)
        if not output:
            raise ValueError("No comparable rows: check GPU, dtype, compute, output, layout, scalars and timing metadata")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
        print("Matched comparisons: {}; unmatched custom rows: {}".format(len(output), len(unmatched)))
        for row in unmatched:
            print("UNMATCHED {} {}x{}x{} {} {}{}".format(row["Backend"], row["M"], row["N"], row["K"], row["Compute"], row["TransA"], row["TransB"]))
        return 4 if unmatched and args.strict else 0
    except (OSError, ValueError, InvalidOperation, OverflowError) as error:
        parser.error(str(error))

if __name__ == "__main__":
    sys.exit(main())

