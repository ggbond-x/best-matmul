#!/usr/bin/env python3
"""Recover the exact validated warm-event cuBLASLt algorithm for an NCU attachment."""
import csv
import json
import subprocess
import sys

path, case, layout, implementation, device, definitions, config_hash = sys.argv[1:]
try:
    matching = [entry.split(':') for entry in definitions.split(',') if entry.split(':')[0] == case]
    if len(matching) != 1 or len(matching[0]) != 13:
        raise ValueError('no unique case in resolved plan')
    dimensions = tuple(matching[0][1:4])
    gpu_uuid = subprocess.check_output(
        ['nvidia-smi', '-i', device, '--query-gpu=uuid', '--format=csv,noheader'],
        text=True, stderr=subprocess.DEVNULL).strip()
    selected = set()
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if (row.get('case_id'), row.get('layout'), row.get('implementation'),
                row.get('cache_mode'), row.get('timing_mode'), row.get('status'),
                row.get('correctness_pass')) != (case, layout, implementation, 'warm', 'event', 'OK', 'PASS'):
                continue
            if row.get('gpu_uuid') != gpu_uuid:
                raise ValueError('benchmark GPU UUID does not match the current GPU')
            if tuple(row.get(key) for key in ('m', 'n', 'k')) != dimensions or row.get('config_hash') != config_hash:
                raise ValueError('benchmark dimensions or config hash do not match the resolved plan')
            details, _ = json.JSONDecoder().raw_decode(row.get('notes', ''))
            selected.add((details['algo_blob'], str(details['cublaslt_version'])))
    if len(selected) != 1:
        raise ValueError('no unique validated saved algorithm; rerun the benchmark with this version')
    blob, version = selected.pop()
    if not blob or not all(ch in '0123456789abcdef' for ch in blob) or not version.isdigit():
        raise ValueError('invalid saved algorithm')
    print(blob + ':' + version)
except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
    print(str(exc), file=sys.stderr)
    sys.exit(1)
