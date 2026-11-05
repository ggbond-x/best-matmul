#!/usr/bin/env python3
"""把 plan2 结果目录整理成扁平 CSV；不启动 GPU，也不重新计算耗时或 TFLOPS。

结果可信度约定：
- 每组配置优先使用该组运行时保存的 manifest.tsv 快照；
- 验证只有“进程正常退出 + 完整成功标记 + 构建快照匹配”才记为 PASS；
- 默认只自动选择 status.txt 为 COMPLETE 且包含全部预期实验组的运行；
- 原始 CSV、manifest 和日志只读，不修改。
"""
import argparse
import csv
import datetime
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

EXPECTED_GROUPS = ('00_baseline', '01_pipe', '02_gmma', '03_cta', '04_large_cta')
META_FIELDS = ('version', 'gmma_n', 'inflight')
CONFIG_FIELDS = ('version', 'cta_m', 'cta_n', 'cta_k', 'pipe', 'gmma_n',
                 'compute_threads', 'producer_threads', 'inflight', 'epilogue')
FINAL_SUCCESS = 'All full-output regressions PASS'
FAILURE_PATTERN = re.compile(
    r'(?i)\b(?:fail(?:ed|ure)?|error|mismatch|abort(?:ed)?|exception|incorrect)\b')


def parse_args():
    parser = argparse.ArgumentParser(
        description='合并 plan2 输出目录中的 summary/runs/验证日志为 CSV。')
    parser.add_argument('result_dir', nargs='?',
                        help='结果目录；省略时从 --output-root 自动选择已完成运行')
    parser.add_argument('-o', '--output-dir',
                        help='CSV 输出目录；默认 RESULT_DIR/csv')
    parser.add_argument('--manifest',
                        help='显式参考 manifest.tsv；仅作配置核对和缺失快照的补充，不覆盖组内快照')
    parser.add_argument('--output-root',
                        help='自动选择时使用的输出根目录；默认 best-matmul/output/plan2')
    return parser.parse_args()


def fail(message):
    print(f'collect_results: {message}', file=sys.stderr)
    raise SystemExit(2)


def warn(message):
    print(f'collect_results: 警告: {message}', file=sys.stderr)


def read_rows(path, delimiter=','):
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            return [], []
        return list(reader.fieldnames), [dict(row) for row in reader]


def merge_fields(base, rows):
    fields = list(base)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_run(directory):
    """返回结果目录的运行状态；不会把 --build-only 的 COMPLETE 当作 benchmark 完成。"""
    present = sorted(
        path.name for path in directory.iterdir()
        if path.is_dir() and (path / 'summary.csv').is_file())
    status_path = directory / 'status.txt'
    status_text = ''
    if status_path.is_file():
        lines = status_path.read_text(encoding='utf-8', errors='replace').splitlines()
        status_text = next((line.strip() for line in lines if line.strip()), '')
    if status_text.startswith('FAILED'):
        state = 'FAILED'
    elif status_text == 'RUNNING':
        state = 'RUNNING'
    elif status_text == 'COMPLETE':
        if not present:
            state = 'BUILD_ONLY'
        elif all(group in present for group in EXPECTED_GROUPS):
            state = 'COMPLETE'
        else:
            state = 'PARTIAL'
    else:
        state = 'UNKNOWN'
    missing = [group for group in EXPECTED_GROUPS if group not in present]
    return {
        'path': directory,
        'state': state,
        'status_text': status_text,
        'present': present,
        'missing': missing,
        'status_mtime': status_path.stat().st_mtime if status_path.is_file() else 0.0,
    }


def describe_run(info):
    text = info['state']
    if info['missing']:
        text += f"，缺失组: {','.join(info['missing'])}"
    return text


def find_result_dir(root):
    if not root.is_dir():
        fail(f'未找到 {root}；请显式传入结果目录')
    candidates = []
    for directory in root.iterdir():
        if not directory.is_dir() or directory.name.startswith('.') or directory.name == 'csv':
            continue
        if not (directory / 'status.txt').is_file() and \
                not any(directory.glob('*/summary.csv')):
            continue
        candidates.append(inspect_run(directory))
    if not candidates:
        fail(f'{root} 下没有可识别的 plan2 结果目录；请显式传入')
    complete = [info for info in candidates if info['state'] == 'COMPLETE']
    if not complete:
        detail = '\n'.join(f'  {info["path"].name}: {describe_run(info)}' for info in candidates)
        fail('没有可确认完成的运行；不会按 summary 修改时间猜测。'
             f'请显式传入结果目录，或先完成一次运行。\n候选:\n{detail}')
    # status.txt 在运行结束时写入；用它的时间而不是 summary.csv 的修改时间排序。
    return max(complete, key=lambda info: (info['status_mtime'], info['path'].name))


def load_manifest(path, label, required=True):
    if path is None:
        return None
    if not path.is_file():
        if required:
            fail(f'{label} manifest 不存在: {path}')
        return None
    fields, rows = read_rows(path, delimiter='\t')
    missing = [field for field in CONFIG_FIELDS if field not in fields]
    if 'id' not in fields or missing:
        problem = f'{label} manifest 缺少字段: {",".join(missing or ["id"])}'
        if required:
            fail(problem)
        warn(problem)
        return None
    mapping = {}
    for row in rows:
        ident = (row.get('id') or '').strip()
        if not ident:
            continue
        if ident in mapping:
            problem = f'{label} manifest 中 ID 重复: {ident}'
            if required:
                fail(problem)
            warn(problem)
            return None
        mapping[ident] = row
    return {
        'path': path,
        'hash': sha256_file(path),
        'rows': mapping,
        'fields': fields,
        'label': label,
    }


def find_build_manifest(result):
    marker = result / 'build_dir.txt'
    if not marker.is_file():
        return None
    lines = marker.read_text(encoding='utf-8', errors='replace').splitlines()
    if not lines:
        return None
    candidate = Path(lines[0].strip()) / 'manifest.tsv'
    return candidate if candidate.is_file() else None


def config_tuple(row):
    return tuple((row.get(field) or '').strip() for field in CONFIG_FIELDS)


def detect_manifest_conflicts(group_manifests, explicit, build):
    conflict_ids = set()
    by_id = defaultdict(list)
    for group, info in group_manifests.items():
        if info is None:
            continue
        for ident, row in info['rows'].items():
            by_id[ident].append((f'组 {group}', row))
    for ident, sources in by_id.items():
        values = {config_tuple(row) for _, row in sources}
        if len(values) > 1:
            detail = '; '.join(f'{name}={config_tuple(row)}' for name, row in sources)
            fail(f'配置 {ident} 在不同实验组快照中不一致: {detail}')
    if explicit is not None:
        for ident, sources in by_id.items():
            if ident not in explicit['rows']:
                continue
            values = {config_tuple(row) for _, row in sources}
            values.add(config_tuple(explicit['rows'][ident]))
            if len(values) > 1:
                detail = '; '.join(f'{name}={config_tuple(row)}' for name, row in sources)
                fail(f'配置 {ident} 与 --manifest 不一致: {detail}; explicit='
                     f'{config_tuple(explicit["rows"][ident])}')
    if build is not None:
        affected = set()
        for group, info in group_manifests.items():
            if info is None:
                continue
            for ident, row in info['rows'].items():
                if ident in build['rows'] and \
                        config_tuple(row) != config_tuple(build['rows'][ident]):
                    affected.add(ident)
        if explicit is not None:
            for ident in set(explicit['rows']) & set(build['rows']):
                if config_tuple(explicit['rows'][ident]) != config_tuple(build['rows'][ident]):
                    affected.add(ident)
        if affected:
            conflict_ids.update(affected)
            warn('当前构建 manifest 与组内快照不一致；历史行保留组内快照并标记 '
                 f'build_manifest_differs: {",".join(sorted(affected))}')
    return conflict_ids


def validation_id(name, suffix):
    match = re.match(r'^validate_(.+)' + re.escape(suffix) + r'$', name)
    return match.group(1) if match else None


def load_validation_entries(result):
    entries = {}
    validation = result / 'validation'
    for log in sorted(validation.glob('validate_*.log')):
        ident = validation_id(log.name, '.log')
        if ident:
            entries.setdefault(ident, {})['log_path'] = log
    for sidecar in sorted(validation.glob('validate_*.status.tsv')):
        ident = validation_id(sidecar.name, '.status.tsv')
        if not ident:
            continue
        entry = entries.setdefault(ident, {})
        entry['status_path'] = sidecar
        fields, rows = read_rows(sidecar, delimiter='\t')
        row = rows[0] if rows else {}
        entry['process_status'] = (row.get('status') or '').strip()
        entry['exit_code'] = (row.get('exit_code') or '').strip()
        entry['manifest_sha256'] = (row.get('manifest_sha256') or '').strip()
        entry['validate_binary'] = (row.get('validate_binary') or '').strip()
        entry['recorded_log'] = (row.get('log') or '').strip()
    return entries


def relative(path, result):
    if path is None:
        return ''
    try:
        return str(path.relative_to(result))
    except ValueError:
        return str(path)


def evaluate_validation(entry, reference, result):
    """返回 (status, detail, exit_code, process_status, log, status_file, binary, hash)。"""
    status_file = relative(entry.get('status_path'), result) if entry else ''
    log_path = entry.get('log_path') if entry else None
    log = relative(log_path, result) if log_path else ''
    binary = (entry.get('validate_binary') or '') if entry else ''
    manifest_sha = (entry.get('manifest_sha256') or '') if entry else ''
    exit_code = (entry.get('exit_code') or '') if entry else ''
    process_status = (entry.get('process_status') or '') if entry else ''
    if entry is None:
        return 'UNKNOWN', 'missing_validation_record', '', '', '', '', '', ''
    text = None
    if log_path and log_path.is_file():
        text = log_path.read_text(encoding='utf-8', errors='replace')
    failure = bool(text is not None and FAILURE_PATTERN.search(text))
    final_success = bool(text is not None and FINAL_SUCCESS in text)

    def result_tuple(status, detail):
        return status, detail, exit_code, process_status, log, status_file, binary, manifest_sha

    if failure:
        return result_tuple('FAIL', 'log_failure')
    if process_status == 'FAIL':
        return result_tuple('FAIL', 'process_failed')
    if process_status == 'TIMEOUT':
        return result_tuple('FAIL', 'timeout')
    if process_status == 'INTERRUPTED':
        return result_tuple('FAIL', 'interrupted')
    if process_status in ('RUNNING', ''):
        if process_status == 'RUNNING':
            return result_tuple('UNKNOWN', 'validation_running')
        if exit_code == '':
            return result_tuple('UNKNOWN', 'no_exit_status')
    if reference is None:
        return result_tuple('UNKNOWN', 'group_manifest_missing')
    if not manifest_sha:
        return result_tuple('UNKNOWN', 'validation_build_unknown')
    if manifest_sha != reference['hash']:
        return result_tuple('UNKNOWN', 'validation_build_mismatch')
    if exit_code == '':
        return result_tuple('UNKNOWN', 'validation_exit_unknown')
    try:
        code = int(exit_code)
    except ValueError:
        return result_tuple('UNKNOWN', 'invalid_exit_code')
    if code != 0:
        return result_tuple('FAIL', f'exit_{code}')
    if text is None:
        return result_tuple('UNKNOWN', 'log_missing')
    if not final_success:
        return result_tuple('UNKNOWN', 'missing_final_success_marker')
    return result_tuple('PASS', 'ok')


def decorate(fields, rows, run_state, group_manifests, explicit, conflict_ids, statuses):
    """为 summary/runs 行加入运行状态、manifest 来源和验证状态。"""
    base = ['group', 'run_state'] + fields
    insert_at = base.index('id') + 1 if 'id' in base else len(base)
    for field in META_FIELDS:
        if field not in base:
            base.insert(insert_at, field)
            insert_at += 1
    for field in ('manifest_source', 'manifest_note', 'manifest_sha256',
                  'validation_status', 'validation_detail'):
        if field not in base:
            base.append(field)
    output = []
    for group, row in rows:
        ident = (row.get('id') or '').strip()
        info = group_manifests.get(group)
        if info is not None and ident in info['rows']:
            source = 'group'
            metadata = info['rows'][ident]
            manifest_sha = info['hash']
        elif explicit is not None and ident in explicit['rows']:
            source = 'explicit'
            metadata = explicit['rows'][ident]
            manifest_sha = explicit['hash']
        else:
            source = 'missing'
            metadata = {}
            manifest_sha = ''
        merged = {'group': group, 'run_state': run_state}
        merged.update(row)
        for field in META_FIELDS:
            if not merged.get(field):
                merged[field] = metadata.get(field, '')
        notes = []
        if source == 'explicit':
            notes.append('group_manifest_missing_id' if info is not None else 'group_manifest_missing')
        elif source == 'missing':
            notes.append('group_manifest_missing_id' if info is not None else 'group_manifest_missing')
        if ident in conflict_ids:
            notes.append('build_manifest_differs')
        merged['manifest_source'] = source
        merged['manifest_note'] = ';'.join(notes)
        merged['manifest_sha256'] = manifest_sha
        status, detail = statuses.get((group, ident), ('UNKNOWN', 'not_evaluated'))
        merged['validation_status'] = status
        merged['validation_detail'] = detail
        output.append(merged)
    return merge_fields(base, output), output


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = Path(args.output_root) if args.output_root else repo / 'output' / 'plan2'
    if args.result_dir:
        result = Path(args.result_dir)
        if not result.is_dir():
            fail(f'结果目录不存在: {result}')
        result = result.resolve()
        run = inspect_run(result)
    else:
        run = find_result_dir(root.resolve())
        result = run['path']
    groups = sorted(
        (path for path in result.iterdir()
         if path.is_dir() and (path / 'summary.csv').is_file()),
        key=lambda path: path.name)
    if not groups:
        fail(f'{result} 下没有包含 summary.csv 的子目录')

    if run['state'] != 'COMPLETE':
        warn(f'显式结果不是可确认完成的运行: {describe_run(run)}；将按部分结果汇总并写入 run_state')

    group_manifests = {
        group.name: load_manifest(group / 'manifest.tsv', f'组 {group.name}', required=False)
        for group in groups
    }
    missing_manifests = [name for name, info in group_manifests.items() if info is None]
    if missing_manifests:
        warn('以下实验组缺少 manifest.tsv 快照，配置字段将留空: '
             + ','.join(missing_manifests))
    explicit = load_manifest(Path(args.manifest), '显式', required=True) if args.manifest else None
    build_manifest_path = find_build_manifest(result)
    build = load_manifest(build_manifest_path, '当前构建', required=False) \
        if build_manifest_path else None
    conflict_ids = detect_manifest_conflicts(group_manifests, explicit, build)

    validation_entries = load_validation_entries(result)
    statuses = {}

    def status_for(group, ident):
        key = (group, ident)
        if key not in statuses:
            reference = group_manifests.get(group) if group else explicit
            evaluated = evaluate_validation(validation_entries.get(ident), reference, result)
            statuses[key] = (evaluated[0], evaluated[1])
        return statuses[key]

    summary_fields = []
    summary_rows = []
    runs_fields = []
    runs_rows = []
    group_ids = {}
    for group in groups:
        fields, rows = read_rows(group / 'summary.csv')
        for field in fields:
            if field not in summary_fields:
                summary_fields.append(field)
        summary_rows.extend((group.name, row) for row in rows)
        ids = {(row.get('id') or '').strip() for row in rows}
        ids.discard('')
        group_ids[group.name] = ids
        for ident in ids:
            status_for(group.name, ident)
        if (group / 'runs.csv').is_file():
            fields, rows = read_rows(group / 'runs.csv')
            for field in fields:
                if field not in runs_fields:
                    runs_fields.append(field)
            runs_rows.extend((group.name, row) for row in rows)
            for row in rows:
                ident = (row.get('id') or '').strip()
                if ident:
                    group_ids[group.name].add(ident)
                status_for(group.name, ident)

    output = Path(args.output_dir) if args.output_dir else result / 'csv'
    output.mkdir(parents=True, exist_ok=True)

    fields, rows = decorate(summary_fields, summary_rows, run['state'], group_manifests,
                            explicit, conflict_ids, statuses)
    write_csv(output / 'summary_all.csv', fields, rows)
    fields, rows = decorate(runs_fields, runs_rows, run['state'], group_manifests,
                            explicit, conflict_ids, statuses)
    write_csv(output / 'runs_all.csv', fields, rows)

    validation_rows = []
    seen = set()
    for group in groups:
        reference = group_manifests.get(group.name)
        for ident in sorted(group_ids[group.name]):
            evaluated = evaluate_validation(validation_entries.get(ident), reference, result)
            validation_rows.append({
                'group': group.name,
                'id': ident,
                'status': evaluated[0],
                'process_status': evaluated[3],
                'exit_code': evaluated[2],
                'detail': evaluated[1],
                'log': evaluated[4],
                'status_file': evaluated[5],
                'validate_binary': evaluated[6],
                'manifest_sha256': evaluated[7],
            })
            seen.add(ident)
    unassigned = sorted(set(validation_entries) - seen)
    for ident in unassigned:
        evaluated = evaluate_validation(validation_entries[ident], explicit, result)
        validation_rows.append({
            'group': '',
            'id': ident,
            'status': evaluated[0],
            'process_status': evaluated[3],
            'exit_code': evaluated[2],
            'detail': evaluated[1],
            'log': evaluated[4],
            'status_file': evaluated[5],
            'validate_binary': evaluated[6],
            'manifest_sha256': evaluated[7],
        })
    write_csv(output / 'validation.csv',
              ('group', 'id', 'status', 'process_status', 'exit_code', 'detail',
               'log', 'status_file', 'validate_binary', 'manifest_sha256'),
              validation_rows)

    manifest_rows = []
    manifest_fields = ['group', 'manifest_source', 'manifest_sha256']
    for group in groups:
        info = group_manifests.get(group.name)
        if info is None:
            continue
        for row in info['rows'].values():
            merged = dict(row)
            merged.update({'group': group.name, 'manifest_source': 'group',
                           'manifest_sha256': info['hash']})
            manifest_rows.append(merged)
            for field in info['fields']:
                if field not in manifest_fields:
                    manifest_fields.append(field)
    for label, info, source in (('(explicit)', explicit, 'explicit'),
                                ('(build)', build, 'build')):
        if info is None:
            continue
        for row in info['rows'].values():
            merged = dict(row)
            merged.update({'group': label, 'manifest_source': source,
                           'manifest_sha256': info['hash']})
            manifest_rows.append(merged)
            for field in info['fields']:
                if field not in manifest_fields:
                    manifest_fields.append(field)
    if manifest_rows:
        write_csv(output / 'manifests_all.csv',
                  merge_fields(manifest_fields, manifest_rows), manifest_rows)

    write_csv(output / 'dataset.csv',
              ('result_dir', 'run_state', 'status_text', 'expected_groups',
               'present_groups', 'missing_groups', 'partial', 'collected_utc'),
              [{
                  'result_dir': str(result),
                  'run_state': run['state'],
                  'status_text': run['status_text'],
                  'expected_groups': ','.join(EXPECTED_GROUPS),
                  'present_groups': ','.join(run['present']),
                  'missing_groups': ','.join(run['missing']),
                  'partial': 'no' if run['state'] == 'COMPLETE' else 'yes',
                  'collected_utc': datetime.datetime.now(datetime.timezone.utc)
                  .strftime('%Y-%m-%dT%H:%M:%SZ'),
              }])

    passed = sum(1 for row in validation_rows if row['status'] == 'PASS')
    failed = sum(1 for row in validation_rows if row['status'] == 'FAIL')
    unknown = sum(1 for row in validation_rows if row['status'] == 'UNKNOWN')
    print(f'结果目录: {result}')
    print(f'运行状态: {run["state"]}'
          + (f'（缺失组: {",".join(run["missing"])}）' if run['missing'] else ''))
    print(f'CSV 输出: {output}')
    print(f'  summary_all.csv: {len(summary_rows)} 行，{len(groups)} 个实验组')
    print(f'  runs_all.csv:    {len(runs_rows)} 行')
    print(f'  validation.csv:  {len(validation_rows)} 条（PASS={passed}, FAIL={failed}, '
          f'UNKNOWN={unknown}）')
    if manifest_rows:
        print(f'  manifests_all.csv: {len(manifest_rows)} 条配置记录')
    else:
        print('  manifests_all.csv: 未找到 manifest.tsv，跳过', file=sys.stderr)
    if run['state'] != 'COMPLETE':
        print('注意: 这是部分/未确认完成的结果，dataset.csv 中 partial=yes；'
              '不要当作完整实验结论。', file=sys.stderr)


if __name__ == '__main__':
    main()
