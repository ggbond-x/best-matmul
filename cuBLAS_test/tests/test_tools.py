import csv
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare", ROOT / "scripts/compare.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)

def row(backend="cublas", time=2.0):
    record = {
        "GPU": "test gpu", "GPU_UUID": "abc", "SM": "90", "Dtype": "FP16",
        "InputDtype": "FP16", "OutputDtype": "FP16", "Compute": "FP32_ACC",
        "Reduction": "FP32", "Layout": "row_major", "M": "64", "N": "128", "K": "256",
        "TransA": "N", "TransB": "T", "LDA": "256", "LDB": "256", "LDC": "128",
        "Alpha": "1", "Beta": "0", "Mode": "launch", "TimingProtocol": "cuda_event_batch_v1",
        "CachePolicy": "reuse_operands", "BetaPolicy": "overwrite", "Warmup": "10",
        "Repeats": "30", "Inner": "10", "Seed": "1234", "Status": "PASS", "Backend": backend,
        "Algorithm": "test", "Time_ms": str(time), "TFLOPS": str(2 * 64 * 128 * 256 / (time * 1e9))
    }
    return record

class CompareTests(unittest.TestCase):
    def test_separate_baselines_and_ratios(self):
        output, missing = compare.compare([row(), row("cublaslt", 1)], [row("my_kernel", 0.5)])
        self.assertFalse(missing)
        self.assertEqual(len(output), 2)
        self.assertEqual([r["PerformanceRatio"] for r in output], [4, 2])
        self.assertEqual([r["Speedup"] for r in output], [4, 2])

    def test_no_cross_precision_layout_or_protocol_match(self):
        for field, value in {"Compute": "TF32_FP32_ACC", "OutputDtype": "FP32",
                             "Mode": "graph", "TransB": "N", "GPU_UUID": "other",
                             "Inner": "1", "Beta": "0.5", "Layout": "column_major",
                             "Reduction": "FP16", "LDA": "264"}.items():
            custom = row("custom")
            custom[field] = value
            output, missing = compare.compare([row()], [custom])
            self.assertFalse(output, field)
            self.assertEqual(len(missing), 1, field)

    def test_missing_semantics_rejected(self):
        record = row()
        del record["Compute"]
        with self.assertRaises(ValueError):
            compare.key(record)

    def test_duplicates_require_explicit_policy(self):
        with self.assertRaises(ValueError):
            compare.compare([row(), row(time=4)], [row("custom")])
        output, _ = compare.compare([row(), row(time=4)], [row("custom")], "median")
        self.assertEqual(float(output[0]["BaselineTime_ms"]), 3)
        self.assertEqual(output[0]["Speedup"], 1.5)

    def test_algorithm_variants_cannot_be_blended(self):
        variant = row()
        variant["Algorithm"] = "different"
        with self.assertRaises(ValueError):
            compare.compare([row(), variant], [row("custom")], "median")

    def test_scalar_formatting(self):
        a, b = row(), row()
        b["Alpha"] = "1.0000000"
        self.assertEqual(compare.key(a), compare.key(b))
        b["Alpha"] = "NaN"
        with self.assertRaises(ValueError):
            compare.key(b)

    def test_reader_excludes_fail_and_checks_flops(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            def write(records):
                with path.open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
            failed = row()
            failed["Status"] = "FAIL"
            write([row(), failed])
            self.assertEqual(len(compare.load(path)), 1)
            malformed = row()
            malformed["TFLOPS"] = "1000"
            write([malformed])
            with self.assertRaises(ValueError):
                compare.load(path)

    def test_compare_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name, record in (("baseline", row()), ("custom", row("my_kernel", 1))):
                with (folder / (name + ".csv")).open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=list(record))
                    writer.writeheader()
                    writer.writerow(record)
            result = subprocess.run([sys.executable, str(ROOT / "scripts/compare.py"),
                                     "--baseline", str(folder / "baseline.csv"),
                                     "--custom", str(folder / "custom.csv"),
                                     "--output", str(folder / "comparison.csv"), "--strict"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (folder / "comparison.csv").open(newline="") as file:
                records = list(csv.DictReader(file))
            self.assertEqual(float(records[0]["Speedup"]), 2)
            self.assertEqual(float(records[0]["PerformanceRatio"]), 2)

class SuiteTests(unittest.TestCase):
    def test_custom_configuration_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "cases.json"
            config.write_text('{"mine": [[17, 23, 31], [17, 23, 31]]}')
            result = subprocess.run([sys.executable, str(ROOT / "scripts/run_suite.py"),
                                     "--cases", str(config), "--groups", "mine",
                                     "--backends", "cublas", "--dtypes", "fp16",
                                     "--transposes", "NN", "TT", "--dry-run"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Cases: 2", result.stdout)

if __name__ == "__main__":
    unittest.main()


