#!/usr/bin/env python3
"""CPU-only integration check of the experiment matrix and runner wiring.

The fake compiler and GPU commands do not test CUDA correctness or performance.
"""
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    script = Path(__file__).resolve().with_name('run.sh')
    with tempfile.TemporaryDirectory(prefix='v2-sweep-test-') as tmp:
        root = Path(tmp)
        tools = root / 'tools'
        tools.mkdir()
        cutlass = root / 'cutlass'
        for name in ('cute/tensor.hpp', 'cute/util/debug.hpp', 'cutlass/half.h'):
            header = cutlass / 'include' / name
            header.parent.mkdir(parents=True, exist_ok=True)
            header.touch()
        nvcc = tools / 'nvcc'
        nvcc.write_text('''#!/usr/bin/env bash
set -eu
if [[ ${1:-} == --version ]]; then echo "mock nvcc"; exit 0; fi
out=''
while (($#)); do
  if [[ $1 == -o ]]; then out=$2; shift 2; else shift; fi
done
[[ -n $out ]]
cat > "$out" <<'PROGRAM'
#!/usr/bin/env bash
echo 'p50_ms=1 p99_ms=1 TFLOPS_p50=1 warmup_launches=300 warmup_stable=yes batch_drift_pct=0'
PROGRAM
chmod +x "$out"
''')
        nvcc.chmod(0o755)
        smi = tools / 'nvidia-smi'
        smi.write_text('''#!/usr/bin/env bash
case " $* " in
  *--query-gpu=compute_cap*) echo 9.0 ;;
  *--query-gpu=uuid*) echo GPU-test-uuid ;;
  *--query-gpu=name*) echo 'Mock Hopper' ;;
  *) echo 'mock GPU data' ;;
esac
''')
        smi.chmod(0o755)
        output = root / 'result'
        build = root / 'build' / 'plan2' / 'result'
        env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ['PATH'],
                   CUDACXX=str(nvcc), CUTLASS_ROOT=str(cutlass), OUTPUT_DIR=str(output), BUILD_DIR=str(build),
                   REPEATS='2', PER_BATCH='1', MIN_WARMUP='1', BATCHES='2', TELEMETRY_MS='0')
        default_env = dict(env)
        default_env.pop('BUILD_DIR')
        plan = subprocess.run(['bash', str(script), '--dry-run'], env=default_env, check=True,
                              capture_output=True, text=True, timeout=30).stdout
        assert str(script.parents[2] / 'build' / 'plan2' / 'result') in plan
        assert 'scripts/plan1/run.sh' not in plan
        assert not output.exists()
        subprocess.run(['bash', str(script)], env=env, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, text=True, timeout=120)
        assert (output / 'status.txt').read_text().strip() == 'COMPLETE'
        assert (output / 'build_dir.txt').read_text().strip() == str(build)
        assert (build / 'output_dir.txt').read_text().strip() == str(output)
        assert not (output / 'build').exists()
        with (build / 'manifest.tsv').open() as f:
            configs = list(csv.DictReader(f, delimiter='\t'))
        assert all(Path(row['binary']).parent == build for row in configs)
        assert len(configs) == 9 and len({r['id'] for r in configs}) == 9
        actual = {(r['cta_m'], r['cta_n'], r['cta_k'], r['pipe'], r['gmma_n']) for r in configs}
        assert actual == {
            ('128', '128', '64', '2', '128'), ('128', '128', '64', '3', '128'),
            ('128', '128', '64', '4', '128'), ('128', '128', '64', '3', '64'),
            ('64', '128', '64', '3', '64'), ('128', '64', '64', '3', '64'),
            ('256', '128', '64', '3', '64'), ('128', '256', '64', '3', '64'),
        }
        assert all((r['compute_threads'], r['producer_threads'], r['epilogue']) == ('128', '0', '0') for r in configs)
        v1 = [r for r in configs if r['version'] == 'v1']
        assert len(v1) == 1 and v1[0]['id'] == 'v1_p3_n128_m128n128' and v1[0]['inflight'] == '1'
        assert all(r['version'] == 'v2' and r['inflight'] == '2' for r in configs if r not in v1)
        commands = (output / 'commands.sh').read_text()
        assert 'scripts/plan1/run.sh' not in commands
        assert '--kernel v1 ' in commands and 'wgmma_tma_fp16_v1.cuh' in commands
        for group, count in [('00_baseline', 2), ('01_pipe', 3), ('02_gmma', 2), ('03_cta', 3), ('04_large_cta', 3)]:
            with (output / group / 'summary.csv').open() as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == count * 2 * 2, (group, len(rows))
            assert all(row['speedup_vs_baseline'] == '1.0' for row in rows)
        assert len(list((output / 'validation').glob('validate_*.log'))) == 9
        status_files = sorted((output / 'validation').glob('validate_*.status.tsv'))
        assert len(status_files) == 9
        with status_files[0].open(newline='') as f:
            status_row = next(csv.DictReader(f, delimiter='\t'))
        assert status_row['exit_code'] == '0' and status_row['status'] == 'PASS'
        assert len(status_row['manifest_sha256']) == 64
        with (output / '00_baseline/summary.csv').open() as f:
            baseline_rows = list(csv.DictReader(f))
        assert {r['id'] for r in baseline_rows} == {'v1_p3_n128_m128n128', 'v2_p3_n128_m128n128'}
        assert all(r['baseline'] == 'v1_p3_n128_m128n128' for r in baseline_rows)
        build_only_output = root / 'build-only-result'
        build_only_dir = root / 'build' / 'plan2' / 'build-only-result'
        build_only_env = dict(env, OUTPUT_DIR=str(build_only_output), BUILD_DIR=str(build_only_dir))
        subprocess.run(['bash', str(script), '--build-only'], env=build_only_env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=120)
        assert (build_only_output / 'status.txt').read_text().strip() == 'COMPLETE'
        assert (build_only_output / 'build_dir.txt').read_text().strip() == str(build_only_dir)
        assert (build_only_dir / 'manifest.tsv').is_file() and not (build_only_output / 'validation').exists()
        runner = Path(__file__).resolve().with_name('run_variants.sh')
        missing_build = subprocess.run(['bash', str(runner), '--dry-run'], env=env,
                                       capture_output=True, text=True, timeout=30)
        assert missing_build.returncode != 0 and '缺少 --build-dir' in missing_build.stderr
        dry_validate = subprocess.run(
            ['bash', str(runner), '--mode', 'validate', '--build-dir', str(build_only_dir),
             '--dry-run'], env=env, capture_output=True, text=True, timeout=30)
        assert dry_validate.returncode == 0, dry_validate.stderr
        assert str(build_only_dir) in dry_validate.stdout
        assert str(build / 'v1_p3_n128_m128n128') not in dry_validate.stdout
        assert 'v1_p3_n128_m128n128.validate' in dry_validate.stdout
        nonexistent = subprocess.run(
            ['bash', str(runner), '--mode', 'validate',
             '--build-dir', str(root / 'no-such-build'), '--dry-run'],
            env=env, capture_output=True, text=True, timeout=30)
        assert nonexistent.returncode != 0 and '未找到 manifest.tsv' in nonexistent.stderr
        collision_env = dict(env, OUTPUT_DIR=str(root / 'new-result'))
        collision = subprocess.run(['bash', str(script)], env=collision_env, capture_output=True, text=True, timeout=30)
        assert collision.returncode != 0 and '构建目录已存在' in collision.stderr
        assert not (root / 'new-result').exists()
        overlap_env = dict(env, OUTPUT_DIR=str(root / 'overlap'), BUILD_DIR=str(root / 'overlap' / 'build'))
        overlap = subprocess.run(['bash', str(script)], env=overlap_env, capture_output=True, text=True, timeout=30)
        assert overlap.returncode != 0 and '不能相同或互相嵌套' in overlap.stderr
        assert not (root / 'overlap').exists()
        collector = Path(__file__).resolve().with_name('collect_results.py')
        csv_dir = root / 'collected'
        subprocess.run([sys.executable, str(collector), str(output), '--output-dir', str(csv_dir)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=60)
        with (csv_dir / 'summary_all.csv').open(newline='') as f:
            summary_all = list(csv.DictReader(f))
        with (csv_dir / 'runs_all.csv').open(newline='') as f:
            runs_all = list(csv.DictReader(f))
        with (csv_dir / 'manifests_all.csv').open(newline='') as f:
            manifest_all = list(csv.DictReader(f))
        with (csv_dir / 'validation.csv').open(newline='') as f:
            validation_all = list(csv.DictReader(f))
        with (csv_dir / 'dataset.csv').open(newline='') as f:
            dataset = next(csv.DictReader(f))
        assert len(summary_all) == 52 and len(runs_all) == 104
        assert {r['group'] for r in summary_all} == {'00_baseline', '01_pipe', '02_gmma', '03_cta', '04_large_cta'}
        assert len(manifest_all) == 54 and len(validation_all) == 13
        assert all(r['validation_status'] == 'UNKNOWN' for r in summary_all)
        assert all(r['detail'] == 'missing_final_success_marker'
                   for r in validation_all)
        assert all(r['manifest_source'] == 'group' for r in summary_all)
        assert all(len(r['manifest_sha256']) == 64 for r in summary_all)
        assert {(r['version'], r['inflight']) for r in summary_all} == {('v1', '1'), ('v2', '2')}
        assert dataset['run_state'] == 'COMPLETE' and dataset['partial'] == 'no'
        assert dataset['missing_groups'] == ''
        print('PASS: mock build, validation, baseline, four sweeps and CSV collection')


if __name__ == '__main__':
    main()
