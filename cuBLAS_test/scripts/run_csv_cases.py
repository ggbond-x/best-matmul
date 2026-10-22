#!/usr/bin/env python3
"""Run every row in cuBLAS测试结果.csv with cuBLASEx and cuBLASLt."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {
    "Shape (M×N×K)", "A/B转置", "A/B数据类型", "M", "N", "K", "C输出类型",
    "Compute Type", "A/B/C物理布局", "Alpha", "Beta", "lda", "ldb", "ldc",
    "Ex Algo", "Lt Algo ID", "Lt Tile ID", "Lt Stages ID", "Lt Split-K",
    "Lt Workspace (MiB)", "Warmup", "Repeat", "cuBLASEx 耗时(ms)",
    "cuBLASEx TFLOPS", "cuBLASLt 耗时(ms)", "cuBLASLt TFLOPS",
    "cuBLAS最佳(TFLOPS)", "cuBLAS/理论峰值", "正确性", "备注",
}


def positive_int(value, name, line):
    try:
        number = int(value)
    except ValueError as error:
        raise ValueError(f"第 {line} 行 {name} 不是整数: {value!r}") from error
    if number <= 0:
        raise ValueError(f"第 {line} 行 {name} 必须大于 0: {number}")
    return number


def finite_float(value, name, line):
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"第 {line} 行 {name} 不是浮点数: {value!r}") from error
    if not (-float("inf") < number < float("inf")):
        raise ValueError(f"第 {line} 行 {name} 必须为有限值")
    return number


def normalized_number(value):
    return format(float(value), ".12g")


def load_cases(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("输入 CSV 没有表头")
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError("输入 CSV 缺少列: " + ", ".join(missing))
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    if not rows:
        raise ValueError("输入 CSV 没有测试用例")

    parsed = []
    for index, row in enumerate(rows, 2):
        m = positive_int(row["M"], "M", index)
        n = positive_int(row["N"], "N", index)
        k = positive_int(row["K"], "K", index)
        trans = row["A/B转置"].strip().upper()
        if len(trans) != 2 or any(flag not in "NT" for flag in trans):
            raise ValueError(f"第 {index} 行 A/B转置 必须是 NN/NT/TN/TT: {trans!r}")
        dtype = row["A/B数据类型"].strip().lower()
        if dtype not in {"fp16", "bf16"}:
            raise ValueError(f"第 {index} 行只支持 fp16/bf16: {dtype!r}")
        if row["C输出类型"].strip().lower() != dtype:
            raise ValueError(f"第 {index} 行 C输出类型必须与 A/B数据类型一致")
        if row["Compute Type"].strip() != "CUBLAS_COMPUTE_32F":
            raise ValueError(f"第 {index} 行 Compute Type 仅支持 CUBLAS_COMPUTE_32F")
        if row["A/B/C物理布局"].strip().lower() not in {"colmajor", "column_major"}:
            raise ValueError(f"第 {index} 行物理布局必须是 ColMajor")
        if row["Ex Algo"].strip() != "CUBLAS_GEMM_DEFAULT":
            raise ValueError(f"第 {index} 行 Ex Algo 仅支持 CUBLAS_GEMM_DEFAULT")
        alpha = finite_float(row["Alpha"], "Alpha", index)
        beta = finite_float(row["Beta"], "Beta", index)
        warmup = positive_int(row["Warmup"], "Warmup", index)
        repeats = positive_int(row["Repeat"], "Repeat", index)
        workspace = positive_int(row["Lt Workspace (MiB)"], "Lt Workspace (MiB)", index)
        expected_lda = k if trans[0] == "T" else m
        expected_ldb = n if trans[1] == "T" else k
        expected_ldc = m
        actual_lds = tuple(positive_int(row[name], name, index) for name in ("lda", "ldb", "ldc"))
        expected_lds = (expected_lda, expected_ldb, expected_ldc)
        if actual_lds != expected_lds:
            raise ValueError(
                f"第 {index} 行 ColMajor leading dimensions 应为 {expected_lds}，实际为 {actual_lds}"
            )
        shape = row["Shape (M×N×K)"].strip().replace("x", "×")
        expected_shape = f"{m}×{n}×{k}"
        if shape != expected_shape:
            raise ValueError(f"第 {index} 行 Shape 应为 {expected_shape}，实际为 {shape}")
        parsed.append({
            "source_line": index,
            "row": dict(row),
            "m": m, "n": n, "k": k, "trans": trans, "dtype": dtype,
            "alpha": alpha, "beta": beta, "warmup": warmup,
            "repeats": repeats, "workspace": workspace,
        })
    return fieldnames, parsed


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_raw(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_algorithm(description):
    values = {}
    for item in description.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            values[key] = value
    return values


def build_command(args, case, backend, raw_csv):
    command = [
        str(args.exe),
        "--m", str(case["m"]), "--n", str(case["n"]), "--k", str(case["k"]),
        "--backend", backend, "--dtype", case["dtype"], "--output-dtype", "same",
        "--layout", "column_major", "--transa", case["trans"][0], "--transb", case["trans"][1],
        "--alpha", normalized_number(case["alpha"]), "--beta", normalized_number(case["beta"]),
        "--mode", "launch", "--warmup", str(case["warmup"]),
        "--repeats", str(case["repeats"]), "--inner", "1",
        "--workspace-mb", str(case["workspace"]), "--device", str(args.device),
        "--seed", str(args.seed), "--check-samples", str(args.check_samples),
        "--csv", str(raw_csv),
    ]
    if backend == "cublaslt":
        command += ["--lt-selection", args.lt_selection, "--lt-candidates", str(args.lt_candidates)]
    return command


def verify_record(record, case, backend):
    expected = {
        "Backend": backend,
        "Dtype": case["dtype"].upper(),
        "Layout": "column_major",
        "M": str(case["m"]), "N": str(case["n"]), "K": str(case["k"]),
        "TransA": case["trans"][0], "TransB": case["trans"][1],
        "LDA": case["row"]["lda"], "LDB": case["row"]["ldb"], "LDC": case["row"]["ldc"],
        "Warmup": str(case["warmup"]), "Repeats": str(case["repeats"]), "Inner": "1",
    }
    mismatches = [f"{key}={record.get(key)!r} (expected {value!r})"
                  for key, value in expected.items() if record.get(key) != value]
    if mismatches:
        raise RuntimeError("结果语义不匹配: " + "; ".join(mismatches))
    for key, expected_number in (("Alpha", case["alpha"]), ("Beta", case["beta"])):
        if abs(float(record[key]) - expected_number) > 1e-7:
            raise RuntimeError(f"结果语义不匹配: {key}={record[key]}")


def write_results(path, fieldnames, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def update_template(case, results):
    row = case["row"]
    ex = results.get("cublas")
    lt = results.get("cublaslt")
    if ex and ex.get("Status") == "PASS":
        row["cuBLASEx 耗时(ms)"] = ex.get("Time_ms", "")
        row["cuBLASEx TFLOPS"] = ex.get("TFLOPS", "")
    if lt and lt.get("Status") == "PASS":
        row["cuBLASLt 耗时(ms)"] = lt.get("Time_ms", "")
        row["cuBLASLt TFLOPS"] = lt.get("TFLOPS", "")
        algo = parse_algorithm(lt.get("Algorithm", ""))
        row["Lt Algo ID"] = algo.get("id", "")
        row["Lt Tile ID"] = algo.get("tile", "")
        row["Lt Stages ID"] = algo.get("stages", "")
        row["Lt Split-K"] = algo.get("splitK", "")
    valid_tflops = []
    for result in (ex, lt):
        if result and result.get("Status") == "PASS" and result.get("TFLOPS"):
            valid_tflops.append(result["TFLOPS"])
    row["cuBLAS最佳(TFLOPS)"] = max(valid_tflops, key=float) if valid_tflops else ""
    statuses = [result.get("Status", "ERROR") if result else "ERROR" for result in (ex, lt)]
    if statuses == ["PASS", "PASS"]:
        row["正确性"] = "PASS"
    elif "FAIL" in statuses or "ERROR" in statuses:
        row["正确性"] = "FAIL"
    elif statuses == ["SKIP", "SKIP"]:
        row["正确性"] = "SKIP"
    else:
        row["正确性"] = "PARTIAL"
    notes = [row.get("备注", "").strip()]
    for name, result in (("cuBLASEx", ex), ("cuBLASLt", lt)):
        if not result:
            notes.append(f"{name}: 未生成结果行")
        elif result.get("Status") != "PASS":
            notes.append(f"{name}: {result.get('Status', 'ERROR')} {result.get('Reason', '')}".strip())
    row["备注"] = "; ".join(note for note in notes if note)


def capture(command, timeout=20):
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return completed.stdout + completed.stderr
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=ROOT / "build_csv/gemm_bench")
    parser.add_argument("--input", type=Path, default=ROOT / "cuBLAS测试结果.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--check-samples", type=int, default=512)
    parser.add_argument("--lt-selection", choices=["first", "tune"], default="tune")
    parser.add_argument("--lt-candidates", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.device < 0 or args.seed < 0 or args.check_samples <= 0 or args.lt_candidates <= 0:
        parser.error("device/seed 必须非负，check-samples/lt-candidates 必须为正数")
    args.input = args.input.resolve()
    args.exe = args.exe.resolve()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = (args.output_dir or ROOT / "outptu" / f"cublas_csv_{stamp}").resolve()

    try:
        fieldnames, cases = load_cases(args.input)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    raw_csv = args.output_dir / "raw_results.csv"
    commands = [build_command(args, case, backend, raw_csv)
                for case in cases for backend in ("cublas", "cublaslt")]
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))
        print(f"CSV rows: {len(cases)}; processes: {len(commands)}")
        print(f"Result directory (not created): {args.output_dir}")
        return 0
    if not args.exe.is_file():
        parser.error(f"基准程序不存在，请先编译: {args.exe}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protected = ["raw_results.csv", "cuBLAS测试结果.csv", "index.csv", "run.log", "manifest.json"]
    existing = [name for name in protected if (args.output_dir / name).exists()]
    if existing:
        parser.error("输出目录已包含结果文件: " + ", ".join(existing))

    shutil.copy2(args.input, args.output_dir / "input_cases.csv")
    output_csv = args.output_dir / "cuBLAS测试结果.csv"
    index_csv = args.output_dir / "index.csv"
    run_log = args.output_dir / "run.log"
    manifest_path = args.output_dir / "manifest.json"
    status_path = args.output_dir / "status.txt"
    output_rows = [case["row"] for case in cases]
    write_results(output_csv, fieldnames, output_rows)
    status_path.write_text("RUNNING\n", encoding="utf-8")

    metadata = {
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": str(args.input), "input_sha256": sha256(args.input),
        "executable": str(args.exe), "executable_sha256": sha256(args.exe),
        "platform": platform.platform(), "python": sys.version,
        "lt_selection": args.lt_selection, "lt_candidates": args.lt_candidates,
        "commands": [command for command in commands], "return_codes": [],
    }
    manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    device_info = capture([str(args.exe), "--device-info", "--device", str(args.device)])
    nvidia_smi = capture(["nvidia-smi", "-i", str(args.device)])
    (args.output_dir / "run_info.txt").write_text(
        f"started_utc={metadata['started_utc']}\ninput_sha256={metadata['input_sha256']}\n"
        f"executable_sha256={metadata['executable_sha256']}\nlt_selection={args.lt_selection}\n"
        f"lt_candidates={args.lt_candidates}\n\n[gemm_bench --device-info]\n{device_info}"
        f"\n[nvidia-smi]\n{nvidia_smi}", encoding="utf-8")

    passed = failed = skipped = 0
    command_index = 0
    with index_csv.open("w", newline="", encoding="utf-8") as index_file, \
            run_log.open("w", encoding="utf-8", buffering=1) as log:
        index_writer = csv.writer(index_file)
        index_writer.writerow(["InputLine", "Shape", "Transpose", "Dtype", "Backend", "ExitCode", "Status", "Reason"])
        try:
            for case_number, case in enumerate(cases, 1):
                results = {}
                for backend in ("cublas", "cublaslt"):
                    command = commands[command_index]
                    command_index += 1
                    before = len(read_raw(raw_csv))
                    label = f"[{command_index}/{len(commands)}] CSV行={case['source_line']} {case['m']}x{case['n']}x{case['k']} {case['trans']} {case['dtype']} {backend}"
                    print(label, flush=True)
                    print("CMD:", shlex.join(command), flush=True)
                    log.write(label + "\nCMD: " + shlex.join(command) + "\n")
                    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True, bufsize=1)
                    assert process.stdout is not None
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                    return_code = process.wait()
                    raw_rows = read_raw(raw_csv)
                    record = raw_rows[-1] if len(raw_rows) == before + 1 else None
                    if record is not None:
                        try:
                            verify_record(record, case, backend)
                        except Exception as error:
                            record = dict(record)
                            record["Status"] = "FAIL"
                            record["Reason"] = str(error)
                        expected_return = 0 if record.get("Status") == "PASS" else 3 if record.get("Status") == "SKIP" else 1
                        if return_code != expected_return:
                            original_status = record.get("Status", "")
                            original_reason = record.get("Reason", "")
                            record = dict(record)
                            record["Status"] = "FAIL"
                            record["Reason"] = (
                                f"进程退出码 {return_code} 与结果状态不一致；"
                                f"原状态={original_status} 原因={original_reason}"
                            )
                    status = record.get("Status", "ERROR") if record else "ERROR"
                    reason = record.get("Reason", "未生成结果行") if record else "未生成结果行"
                    results[backend] = record
                    metadata["return_codes"].append(return_code)
                    manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    index_writer.writerow([case["source_line"], case["row"]["Shape (M×N×K)"], case["trans"],
                                           case["dtype"], backend, return_code, status, reason])
                    index_file.flush()
                    if status == "PASS" and return_code == 0:
                        passed += 1
                    elif status == "SKIP" and return_code == 3:
                        skipped += 1
                    else:
                        failed += 1
                update_template(case, results)
                output_rows[case_number - 1] = case["row"]
                write_results(output_csv, fieldnames, output_rows)
        except KeyboardInterrupt:
            status_path.write_text(f"INTERRUPTED\nPASS={passed}\nFAIL={failed}\nSKIP={skipped}\n", encoding="utf-8")
            print(f"\n已中断；已完成结果保留在 {args.output_dir}", file=sys.stderr)
            return 130

    final_status = "COMPLETE" if failed == 0 else "FAILED"
    status_path.write_text(f"{final_status}\nPASS={passed}\nFAIL={failed}\nSKIP={skipped}\n", encoding="utf-8")
    print(f"完成：PASS={passed} FAIL={failed} SKIP={skipped}")
    print(f"结果目录：{args.output_dir}")
    if failed:
        return 1
    return 0 if passed else 3


if __name__ == "__main__":
    sys.exit(main())
