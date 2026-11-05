#!/usr/bin/env python3
"""CPU-only regression tests for plan1 result integrity and reproducibility."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/common'))
import plan_utils

spec = importlib.util.spec_from_file_location('summarize', ROOT / 'scripts/plan1/summarize.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def raw(impl='CUBLAS_EX_F16ACC', correctness='PASS', status='OK', duration='10'):
    return dict(case_id='S01', shape_id='S01', m='128', n='128', k='64', layout='TN',
                accum_dtype='FP16', timing_mode='event', timer_scope='gpu_kernel_batch',
                cache_mode='warm', implementation=impl, status=status, correctness_pass=correctness,
                duration_us_median=duration, duration_us_min=duration, duration_us_max=duration,
                round_started_utc='2026-11-05T01:00:00.000Z',
                round_finished_utc='2026-11-05T01:00:01.000Z')


class PlanFixes(unittest.TestCase):
    def test_plan_dimensions_and_thresholds(self):
        plan = plan_utils.load_plan(str(ROOT / 'scripts/plan1/plan.yaml'))
        plan['shapes'][0]['m'] = 128
        plan['correctness']['thresholds']['FP16']['atol'] = '1 + K * 0.25'
        config = plan_utils.resolve_config(plan, 'quick', [])
        entry = plan_utils.case_definitions(config).split(',')[0].split(':')
        self.assertEqual(entry[0:5], ['S01', '128', '4096', '4096', 'P0'])
        self.assertEqual(float(entry[5]), 1025.0)
        plan['correctness']['thresholds']['FP16']['atol'] = '__import__("os").system("x")'
        with self.assertRaises(SystemExit):
            plan_utils.resolve_config(plan, 'quick', [])

    def test_unchecked_and_failed_never_rank(self):
        candidate = raw(correctness='NOT_CHECKED')
        other = raw('CUSTOM_V1_F16ACC')
        a = summary.aggregate_group([candidate], summary.group_key(candidate))
        b = summary.aggregate_group([other], summary.group_key(other))
        self.assertEqual(a['status'], 'NOT_CHECKED')
        summary.apply_baselines([a, b], 'NVIDIA H100 PCIe')
        self.assertEqual(b['best_library_impl'], 'N/A')
        passed = raw()
        failed = raw(status='FAIL', duration='N/A')
        a = summary.aggregate_group([passed, failed], summary.group_key(passed))
        self.assertEqual(a['status'], 'FAIL')
        summary.apply_baselines([a, b], 'NVIDIA H100 PCIe')
        self.assertEqual(b['best_library_impl'], 'N/A')

    def test_windowed_monitor_samples(self):
        row = raw()
        samples = [dict(timestamp_utc='2026-11-05T01:00:00.500Z', sm_clock_mhz='1000',
                        mem_clock_mhz='500', power_w='200', temperature_c='70'),
                   dict(timestamp_utc='2026-11-05T01:00:02.000Z', sm_clock_mhz='3000',
                        mem_clock_mhz='900', power_w='900', temperature_c='90')]
        result = summary.aggregate_group([row], summary.group_key(row), samples)
        self.assertEqual(result['sm_clock_mhz_avg'], '1000')
        self.assertEqual(result['power_w_avg'], '200')
        self.assertEqual(result['telemetry_scope'], 'monitor_samples_in_round_windows:1')
        self.assertEqual(summary.aggregate_group([row], summary.group_key(row))['power_w_avg'], 'N/A')

    def test_ncu_algorithm_requires_valid_same_gpu_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            nvidia = root / 'nvidia-smi'
            nvidia.write_text('#!/bin/sh\necho GPU-test\n')
            nvidia.chmod(0o755)
            data = root / 'raw.csv'
            row = dict(case_id='S01', layout='NT', implementation='CUBLAS_LT_F16ACC',
                       cache_mode='warm', timing_mode='event', status='OK', correctness_pass='PASS',
                       gpu_uuid='GPU-test', m='64', n='64', k='64', config_hash='config-1',
                       notes=json.dumps({'algo_blob': 'abcd', 'cublaslt_version': '120600'}))
            with data.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'])
            cmd = [sys.executable, str(ROOT / 'scripts/common/ncu_algorithm.py'), str(data),
                   'S01', 'NT', 'CUBLAS_LT_F16ACC', '0',
                   'S01:64:64:64:P0:0.5:0.1:0.1:0:0.1:0.1:0.1:0', 'config-1']
            result = subprocess.run(cmd, env=env, text=True, capture_output=True)
            self.assertEqual((result.returncode, result.stdout.strip()), (0, 'abcd:120600'))
            nvidia.write_text('#!/bin/sh\necho GPU-other\n')
            result = subprocess.run(cmd, env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('GPU UUID', result.stderr)

    def test_ncu_repeated_attachment_preserves_reports_and_manifest(self):
        if not (ROOT / 'build/plan1/gemm_bench').is_file():
            self.skipTest('requires built executable for script preflight; fake NCU does not run it')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run([sys.executable, str(ROOT / 'scripts/common/plan_utils.py'), 'init',
                            '--plan', str(ROOT / 'scripts/plan1/plan.yaml'), '--mode', 'quick',
                            '--output-root', str(root), '--run-id', 'unit'], check=True, capture_output=True)
            run = root / 'unit'
            manifest = run / 'run_manifest.json'
            manifest.write_text('{"status":"COMPLETED","script_status":"COMPLETED","cases":[]}\n')
            previous = manifest.read_text()
            (run / 'commands.log').write_text('')
            ncu = root / 'ncu'
            ncu.write_text('''#!/usr/bin/env bash
set -eu
while (($#)); do
  if [[ $1 == -o ]]; then printf fake > "$2.ncu-rep"; break; fi
  shift
done
printf '"Metric Name","Metric Value"\\n"duration","1"\\n'
''')
            ncu.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'])
            cmd = ['bash', str(ROOT / 'scripts/plan1/run_ncu.sh'), '--run-dir', str(run),
                   '--cases', 'S01', '--layouts', 'NT', '--impls', 'CUSTOM_V1_F16ACC']
            for _ in range(2):
                subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, timeout=60)
            reports = list((run / 'reports').glob('ncu_*/*.ncu-rep'))
            self.assertEqual(len(reports), 2)
            self.assertEqual(manifest.read_text(), previous)
            self.assertEqual(len((run / 'reports/ncu_sessions.jsonl').read_text().splitlines()), 2)


if __name__ == '__main__':
    unittest.main()
