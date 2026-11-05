#!/usr/bin/env python3
"""Offline regression tests for collect_results.py.

Covers validation evidence, per-group manifest snapshots and result selection.
No GPU, NCU or real experiment data is used.
"""
import csv
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

COLLECTOR = Path(__file__).resolve().with_name('collect_results.py')
MANIFEST_FIELDS = ['id', 'version', 'cta_m', 'cta_n', 'cta_k', 'pipe', 'gmma_n',
                   'compute_threads', 'producer_threads', 'inflight', 'epilogue', 'binary']
SUMMARY_FIELDS = ['id', 'shape', 'layout', 'median_p50_ms']
RUNS_FIELDS = ['id', 'shape', 'layout', 'p50_ms']
STATUS_FIELDS = ['id', 'exit_code', 'status', 'start_utc', 'end_utc', 'log',
                 'validate_binary', 'manifest_sha256']


def write_delimited(path, fields, rows, delimiter):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n',
                                delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def write_csv(path, fields, rows):
    write_delimited(path, fields, rows, ',')


def write_tsv(path, rows, fields=MANIFEST_FIELDS):
    write_delimited(path, fields, rows, '\t')


def manifest_row(ident, gmma='128', pipe='3', build=None):
    return {
        'id': ident, 'version': 'v2', 'cta_m': '128', 'cta_n': '128', 'cta_k': '64',
        'pipe': pipe, 'gmma_n': gmma, 'compute_threads': '128', 'producer_threads': '0',
        'inflight': '2', 'epilogue': '0',
        'binary': str((build or Path('/tmp/build')) / ident),
    }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_group(result, group, ids, build):
    write_csv(result / group / 'summary.csv', SUMMARY_FIELDS,
              [{'id': ident, 'shape': '4096x4096x4096', 'layout': 'NT',
                'median_p50_ms': '1.0'} for ident in ids])
    write_csv(result / group / 'runs.csv', RUNS_FIELDS,
              [{'id': ident, 'shape': '4096x4096x4096', 'layout': 'NT',
                'p50_ms': '1.0'} for ident in ids])
    write_tsv(result / group / 'manifest.tsv', [manifest_row(ident, build=build) for ident in ids])


def add_validation(result, ident, manifest_hash, *, log=None, exit_code='0',
                   process_status='PASS'):
    validation = result / 'validation'
    validation.mkdir(parents=True, exist_ok=True)
    if log is not None:
        (validation / f'validate_{ident}.log').write_text(log)
    row = {
        'id': ident, 'exit_code': exit_code, 'status': process_status,
        'start_utc': '2026-01-01T00:00:00Z', 'end_utc': '2026-01-01T00:00:01Z',
        'log': f'validate_{ident}.log', 'validate_binary': f'/build/{ident}.validate',
        'manifest_sha256': manifest_hash,
    }
    write_tsv(validation / f'validate_{ident}.status.tsv', [row], STATUS_FIELDS)


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def run_collector(result, *extra, root=None):
    command = [sys.executable, str(COLLECTOR)]
    if result is not None:
        command.append(str(result))
    if root is not None:
        command += ['--output-root', str(root)]
    command += list(extra)
    return subprocess.run(command, capture_output=True, text=True, timeout=60)


def validation_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    build = root / 'build'
    build.mkdir()
    result = root / 'result'
    result.mkdir()
    ids = ['ok', 'lower_fail', 'timeout', 'interrupted', 'nonzero_pass',
           'nonzero_pass_status_pass', 'mapping_only', 'missing_log', 'missing',
           'running', 'legacy', 'build_mismatch']
    add_group(result, '00_baseline', ids, build)
    manifest_hash = digest(result / '00_baseline' / 'manifest.tsv')
    (result / 'status.txt').write_text('COMPLETE\n')

    add_validation(result, 'ok', manifest_hash,
                   log='All full-output regressions PASS\n')
    add_validation(result, 'lower_fail', manifest_hash,
                   log='Mapping PASS: 128 compute threads\nFull-output failure: layout=NT, input=2\n')
    add_validation(result, 'timeout', manifest_hash, log='Mapping PASS: 128 compute threads\n',
                   exit_code='124', process_status='TIMEOUT')
    add_validation(result, 'interrupted', manifest_hash, log='Mapping PASS\n',
                   exit_code='130', process_status='INTERRUPTED')
    add_validation(result, 'nonzero_pass', manifest_hash,
                   log='All full-output regressions PASS\n', exit_code='1', process_status='FAIL')
    # 侧车状态异常地写着 PASS，但退出码非零：仍不得判为通过。
    add_validation(result, 'nonzero_pass_status_pass', manifest_hash,
                   log='All full-output regressions PASS\n', exit_code='1', process_status='PASS')
    add_validation(result, 'mapping_only', manifest_hash,
                   log='Mapping PASS: 128 compute threads cover the output\n')
    add_validation(result, 'missing_log', manifest_hash)
    add_validation(result, 'running', manifest_hash, log='Mapping PASS\n',
                   exit_code='', process_status='RUNNING')
    (result / 'validation' / 'validate_legacy.log').write_text(
        'All full-output regressions PASS\n')
    add_validation(result, 'build_mismatch', '0' * 64,
                   log='All full-output regressions PASS\n')
    return result


def test_validation_evidence(root):
    result = validation_fixture(root)
    before = {path.name: digest(path) for path in (result / '00_baseline').iterdir()}
    completed = run_collector(result, '--output-dir', str(root / 'csv'))
    assert completed.returncode == 0, completed.stderr
    statuses = {row['id']: row for row in read_csv(root / 'csv' / 'validation.csv')}
    expected = {
        'ok': ('PASS', 'ok'),
        'lower_fail': ('FAIL', 'log_failure'),
        'timeout': ('FAIL', 'timeout'),
        'interrupted': ('FAIL', 'interrupted'),
        'nonzero_pass': ('FAIL', 'process_failed'),
        'nonzero_pass_status_pass': ('FAIL', 'exit_1'),
        'mapping_only': ('UNKNOWN', 'missing_final_success_marker'),
        'missing_log': ('UNKNOWN', 'log_missing'),
        'missing': ('UNKNOWN', 'missing_validation_record'),
        'running': ('UNKNOWN', 'validation_running'),
        'legacy': ('UNKNOWN', 'no_exit_status'),
        'build_mismatch': ('UNKNOWN', 'validation_build_mismatch'),
    }
    assert set(statuses) == set(expected), statuses
    for ident, (status, detail) in expected.items():
        assert (statuses[ident]['status'], statuses[ident]['detail']) == (status, detail), ident
    summaries = read_csv(root / 'csv' / 'summary_all.csv')
    runs = read_csv(root / 'csv' / 'runs_all.csv')
    for row in summaries + runs:
        assert (row['validation_status'], row['validation_detail']) == expected[row['id']]
    after = {path.name: digest(path) for path in (result / '00_baseline').iterdir()}
    assert before == after


def snapshot_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    build = root / 'build'
    build.mkdir()
    result = root / 'result'
    result.mkdir()
    add_group(result, '00_baseline', ['cfg'], build)
    write_tsv(build / 'manifest.tsv', [manifest_row('cfg', gmma='64', build=build)])
    (result / 'build_dir.txt').write_text(str(build) + '\n')
    (result / 'status.txt').write_text('COMPLETE\n')
    return result, build


def test_manifest_snapshots(root):
    result, build = snapshot_fixture(root)
    completed = run_collector(result, '--output-dir', str(root / 'csv'))
    assert completed.returncode == 0, completed.stderr
    row = read_csv(root / 'csv' / 'summary_all.csv')[0]
    assert row['gmma_n'] == '128' and row['manifest_source'] == 'group'
    assert 'build_manifest_differs' in row['manifest_note']
    manifests = read_csv(root / 'csv' / 'manifests_all.csv')
    assert {(r['group'], r['gmma_n']) for r in manifests} == {
        ('00_baseline', '128'), ('(build)', '64')}
    # 原构建路径不可访问时仍应仅依赖组内快照。
    (result / 'build_dir.txt').write_text('/nonexistent/build/plan2/old\n')
    relocated = run_collector(result, '--output-dir', str(root / 'csv_relocated'))
    assert relocated.returncode == 0, relocated.stderr
    assert read_csv(root / 'csv_relocated' / 'summary_all.csv')[0]['gmma_n'] == '128'
    # 显式 manifest 与组内快照冲突必须报错。
    explicit = root / 'explicit.tsv'
    write_tsv(explicit, [manifest_row('cfg', gmma='64', build=build)])
    conflict = run_collector(result, '--manifest', str(explicit),
                             '--output-dir', str(root / 'csv_explicit'))
    assert conflict.returncode != 0 and '与 --manifest 不一致' in conflict.stderr
    # 组间快照冲突必须报错。
    add_group(result, '01_pipe', ['cfg'], build)
    (result / '01_pipe' / 'manifest.tsv').write_text(
        (result / '01_pipe' / 'manifest.tsv').read_text().replace('\t3\t128\t', '\t4\t128\t'))
    group_conflict = run_collector(result, '--output-dir', str(root / 'csv_conflict'))
    assert group_conflict.returncode != 0 and '不同实验组快照中不一致' in group_conflict.stderr


def test_missing_group_manifest(root):
    root.mkdir(parents=True, exist_ok=True)
    build = root / 'build'
    build.mkdir()
    result = root / 'result'
    result.mkdir()
    add_group(result, '00_baseline', ['cfg'], build)
    (result / '00_baseline' / 'manifest.tsv').unlink()
    (result / 'status.txt').write_text('COMPLETE\n')
    completed = run_collector(result, '--output-dir', str(root / 'csv'))
    assert completed.returncode == 0, completed.stderr
    row = read_csv(root / 'csv' / 'summary_all.csv')[0]
    assert row['gmma_n'] == '' and row['manifest_source'] == 'missing'
    assert 'group_manifest_missing' in row['manifest_note']
    assert row['validation_status'] == 'UNKNOWN'


def make_run(root, name, state, groups=('00_baseline',), complete_groups=False):
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    build = root / 'builds' / name
    build.mkdir(parents=True)
    actual_groups = ['00_baseline', '01_pipe', '02_gmma', '03_cta', '04_large_cta'] \
        if complete_groups else list(groups)
    for group in actual_groups:
        add_group(run, group, ['cfg'], build)
    if state is not None:
        (run / 'status.txt').write_text(state + '\n')
    return run


def test_result_selection(root):
    output_root = root / 'output'
    old = make_run(output_root, 'run_old', 'COMPLETE', complete_groups=True)
    new = make_run(output_root, 'run_new', 'COMPLETE', complete_groups=True)
    make_run(output_root, 'run_failed', 'FAILED stage=bench')
    make_run(output_root, 'run_running', 'RUNNING')
    make_run(output_root, 'run_unknown', None)
    make_run(output_root, 'build_only', 'COMPLETE', groups=())
    make_run(output_root, 'run_partial', 'COMPLETE')

    now = time.time()
    os.utime(old / 'status.txt', (now - 3600, now - 3600))
    os.utime(new / 'status.txt', (now, now))
    # 旧运行的 summary 修改时间更晚，也不能让它被选中。
    newest_summary = max(old.glob('*/summary.csv'), key=lambda path: path.name)
    os.utime(newest_summary, (now + 3600, now + 3600))

    selected = run_collector(None, root=output_root)
    assert selected.returncode == 0, selected.stderr
    dataset = read_csv(new / 'csv' / 'dataset.csv')[0]
    assert dataset['run_state'] == 'COMPLETE' and dataset['partial'] == 'no'
    assert dataset['result_dir'] == str(new.resolve())

    # 没有完整运行时不自动选择部分/失败结果。
    partial_root = root / 'only_partial'
    make_run(partial_root, 'run_failed', 'FAILED stage=bench')
    make_run(partial_root, 'run_running', 'RUNNING')
    make_run(partial_root, 'build_only', 'COMPLETE', groups=())
    make_run(partial_root, 'run_partial', 'COMPLETE')
    default_fail = run_collector(None, root=partial_root)
    assert default_fail.returncode == 2
    assert '不会按 summary 修改时间猜测' in default_fail.stderr

    explicit = run_collector(partial_root / 'run_partial', '--output-dir', str(root / 'csv_partial'))
    assert explicit.returncode == 0, explicit.stderr
    dataset = read_csv(root / 'csv_partial' / 'dataset.csv')[0]
    assert dataset['run_state'] == 'PARTIAL' and dataset['partial'] == 'yes'
    assert dataset['missing_groups'] == '01_pipe,02_gmma,03_cta,04_large_cta'

    failed = run_collector(partial_root / 'run_failed', '--output-dir', str(root / 'csv_failed'))
    assert failed.returncode == 0, failed.stderr
    assert read_csv(root / 'csv_failed' / 'dataset.csv')[0]['run_state'] == 'FAILED'

    build_only = run_collector(partial_root / 'build_only', '--output-dir', str(root / 'csv_build'))
    assert build_only.returncode == 2 and '没有包含 summary.csv' in build_only.stderr


def main():
    with tempfile.TemporaryDirectory(prefix='collect-results-test-') as tmp:
        root = Path(tmp)
        test_validation_evidence(root / 'validation')
        test_manifest_snapshots(root / 'manifests')
        test_missing_group_manifest(root / 'missing-manifest')
        test_result_selection(root / 'selection')
        print('PASS: validation evidence, manifest snapshots, conflicts and result selection')


if __name__ == '__main__':
    main()
