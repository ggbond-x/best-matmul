#!/usr/bin/env python3
"""Unit fixtures are artificial and temporary; never written to benchmark output."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_results import telemetry_stats, aggregate, correlation
from profile_h800_peak import tensor_counts
from monitor_gpu import decode_reasons
from run_h800_peak import load_shapes, load_config, choose_top, peak_for_device


class AnalysisTests(unittest.TestCase):
    def sample(self, t, power="", thermal="", uuid="GPU-test", start=None, end=None):
        return dict(timestamp_monotonic_ns=str(t), query_start_monotonic_ns=str(start if start is not None else t),
                    query_end_monotonic_ns=str(end if end is not None else t), gpu_uuid=uuid,
                    power_throttle=power, thermal_throttle=thermal, sm_clock_mhz="1000", power_draw_w="90", power_limit_w="100")

    def windows(self):
        return [dict(start_monotonic_ns="100", end_monotonic_ns="1000", gpu_uuid="GPU-test")]

    def test_missing_is_not_false(self):
        r = telemetry_stats([self.sample(t) for t in (200, 500, 800)], self.windows())
        self.assertEqual(r["power_throttle_observed"], "unknown")
        self.assertEqual(r["power_throttle_sample_ratio"], "")
        self.assertEqual(r["telemetry_quality"], "OK")

    def test_cross_phase_and_wrong_uuid_excluded(self):
        s = [self.sample(100, start=90, end=110), self.sample(300, uuid="GPU-other"), self.sample(500)]
        r = telemetry_stats(s, self.windows())
        self.assertEqual(r["telemetry_samples"], 1)
        self.assertEqual(r["telemetry_quality"], "INSUFFICIENT")

    def test_observed_event_even_sparse(self):
        r = telemetry_stats([self.sample(500, power="1")], self.windows())
        self.assertEqual(r["power_throttle_observed"], "true")
        self.assertEqual(r["thermal_throttle_observed"], "unknown")

    def test_zero_with_sufficient_samples(self):
        r = telemetry_stats([self.sample(t, "0", "0") for t in (200, 500, 800)], self.windows())
        self.assertEqual(r["power_throttle_observed"], "false")

    def test_reasons_do_not_conflate_slowdown(self):
        r = decode_reasons(8)
        self.assertEqual(r["hw_slowdown"], 1)
        self.assertEqual(r["thermal_throttle"], 0)
        self.assertEqual(decode_reasons(128)["power_throttle"], 1)
        self.assertEqual(decode_reasons(None), {})

    def test_post_failure_invalidates_ranking(self):
        measured = dict(run_id="r", M="4", N="4", K="4", status="PASS", correctness_pass="true",
                        start_monotonic_ns="100", end_monotonic_ns="1000", samples_ms="1;2;3", gpu_uuid="GPU-test")
        r = aggregate([measured, dict(run_id="r", status="FAIL", reason="post-check")], [], {}, {})
        self.assertEqual(r["status"], "FAIL")
        self.assertEqual(r["tflops"], "")

    def test_process_interrupt_invalidates_ranking(self):
        measured = dict(run_id="r", M="4", N="4", K="4", status="PASS", correctness_pass="true",
                        start_monotonic_ns="100", end_monotonic_ns="1000", samples_ms="1;2;3", gpu_uuid="GPU-test")
        r = aggregate([measured], [], {"r": {"status": "INTERRUPTED"}}, {})
        self.assertEqual(r["tflops"], "")

    def test_aggregate_raw_samples_not_medians_of_windows(self):
        a = dict(run_id="r", M="4", N="4", K="4", status="PASS", correctness_pass="true", samples_ms="1;2;3",
                 start_monotonic_ns="100", end_monotonic_ns="1000", gpu_uuid="GPU-test")
        b = dict(a, samples_ms="100", start_monotonic_ns="1100", end_monotonic_ns="2000")
        r = aggregate([a, b], [], {}, {})
        self.assertEqual(r["elapsed_ms_median"], 2.5)
        self.assertEqual(r["elapsed_ms_mean"], 26.5)

    def test_no_spurious_correlation(self):
        rows = [dict(telemetry_quality="OK", x=1, y=2) for _ in range(20)]
        self.assertIsNone(correlation(rows, "x", "y")[0])

    def test_empty_sensors_are_missing(self):
        samples = [dict(timestamp_monotonic_ns=str(t), query_start_monotonic_ns=str(t),
                        query_end_monotonic_ns=str(t), gpu_uuid="GPU-test") for t in (200, 500, 800)]
        r = telemetry_stats(samples, self.windows())
        self.assertEqual(r["telemetry_quality"], "MISSING")
        self.assertEqual(r["power_throttle_observed"], "unknown")

    def test_correlation_requires_per_metric_sample_count(self):
        rows = [dict(telemetry_quality="OK", sm_clock_avg_mhz=1000+i*20, tflops=10+i,
                     clock_samples=1) for i in range(10)]
        self.assertIsNone(correlation(rows, "sm_clock_avg_mhz", "tflops")[0])
        for row in rows:
            row["clock_samples"] = 4
        self.assertAlmostEqual(correlation(rows, "sm_clock_avg_mhz", "tflops")[0], 1.)

    def test_ncu_long_and_wide_csv(self):
        metric = "sm__inst_executed_pipe_tensor.sum"
        wide = '"ID","' + metric + '"\n"","inst"\n"0","589,824"\n"1","N/A"\n"2","nan"\n'
        long = '"ID","Metric Name","Metric Value"\n"0","' + metric + '","589,824"\n'
        self.assertEqual(tensor_counts(wide, metric), [589824.])
        self.assertEqual(tensor_counts(long, metric), [589824.])
        self.assertEqual(tensor_counts(wide, "missing"), [])

    def test_other_gpu_has_no_h800_peak(self):
        self.assertFalse(peak_for_device({"gpu_name": "NVIDIA RTX 3050", "sm_count": 20})["verified_for_device"])

    def test_nominal_peak_requires_full_configuration(self):
        e = dict(gpu_name="NVIDIA H800 PCIe", gpu_uuid="GPU-test", sm_count=114, compute_capability="9.0", vram_bytes=80*2**30)
        self.assertTrue(peak_for_device(e)["verified_for_device"])
        e["sm_count"] = 14
        self.assertFalse(peak_for_device(e)["verified_for_device"])

    def test_default_csv_and_transposes(self):
        shapes = load_shapes(ROOT / "csvs/h800_pcie_fp16_shapes.csv")
        self.assertEqual(len(shapes), 40)
        self.assertEqual({s["transA"] + s["transB"] for s in shapes}, {"NN", "NT", "TN", "TT"})
        self.assertEqual(load_config(ROOT / "csvs/benchmark_config.csv")["sustained_seconds"], 45)

    def test_duplicate_and_invalid_shapes(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "cases.csv"
            content = (ROOT / "csvs/h800_pcie_fp16_shapes.csv").read_text()
            p.write_text(content + content.splitlines()[1] + "\n")
            with self.assertRaises(ValueError):
                load_shapes(p)
            p.write_text(content.replace("FP16", "FP32", 1))
            with self.assertRaises(ValueError):
                load_shapes(p)

    def test_dry_run_from_arbitrary_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            p = subprocess.run([sys.executable, str(ROOT / "scripts/run_h800_peak.py"), "--dry-run", "--quick"],
                               cwd=folder, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(len(json.loads(p.stdout)["shapes"]), 5)

    def test_top_distinct_shape_and_failure_filter(self):
        shapes = load_shapes(ROOT / "csvs/h800_pcie_fp16_shapes.csv")
        rows = [dict(case_id="S1_NN", status="FAIL", correctness_pass="false", tflops="999"),
                dict(case_id="S2_NN", status="PASS", correctness_pass="true", tflops="20"),
                dict(case_id="S2_TT", status="PASS", correctness_pass="true", tflops="19"),
                dict(case_id="S3_NN", status="PASS", correctness_pass="true", tflops="18")]
        self.assertEqual([r["case_id"] for r in choose_top(rows, shapes, 3)], ["S2_NN", "S3_NN"])


if __name__ == "__main__":
    unittest.main()
