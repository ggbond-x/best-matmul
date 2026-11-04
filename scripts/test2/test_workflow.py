#!/usr/bin/env python3
"""CPU-only integration check of the experiment matrix and runner wiring.

The fake compiler and GPU commands do not test CUDA correctness or performance.
"""
import csv
import os
from pathlib import Path
import subprocess
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
        env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ['PATH'],
                   CUDACXX=str(nvcc), CUTLASS_ROOT=str(cutlass), OUTPUT_DIR=str(output),
                   REPEATS='2', PER_BATCH='1', MIN_WARMUP='1', BATCHES='2', TELEMETRY_MS='0')
        subprocess.run(['bash', str(script)], env=env, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, text=True, timeout=120)
        assert (output / 'status.txt').read_text().strip() == 'COMPLETE'
        with (output / 'build/manifest.tsv').open() as f:
            configs = list(csv.DictReader(f, delimiter='\t'))
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
        assert '--kernel v1 ' in commands and 'wgmma_tma_fp16_v1.cuh' in commands
        for group, count in [('00_baseline', 2), ('01_pipe', 3), ('02_gmma', 2), ('03_cta', 3), ('04_large_cta', 3)]:
            with (output / group / 'summary.csv').open() as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == count * 2 * 2, (group, len(rows))
            assert all(row['speedup_vs_baseline'] == '1.0' for row in rows)
        assert len(list((output / 'validation').glob('validate_*.log'))) == 9
        with (output / '00_baseline/summary.csv').open() as f:
            baseline_rows = list(csv.DictReader(f))
        assert {r['id'] for r in baseline_rows} == {'v1_p3_n128_m128n128', 'v2_p3_n128_m128n128'}
        assert all(r['baseline'] == 'v1_p3_n128_m128n128' for r in baseline_rows)
        print('PASS: mock build, validation, baseline, four sweeps and summaries')


if __name__ == '__main__':
    main()
