#!/usr/bin/env python3
"""GPU/软件环境快照。

用法：
  env_snapshot.py --output .env_before.json --phase before
  env_snapshot.py --output environment.json --merge --phase final

--merge 会保留目标文件中由 gemm_bench 写入的 GPU/CUDA/cuBLAS/CUTLASS 字段，
再补充 nvidia-smi 细节、主机信息、Git 状态、进程与时钟策略快照。
所有查询失败都记录为错误文本，不填充推测值。
"""

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def run(cmd, timeout=30):
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout,
        )
        return {"command": cmd, "returncode": proc.returncode, "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip()}
    except FileNotFoundError:
        return {"command": cmd, "returncode": None, "stdout": "", "stderr": "not found"}
    except subprocess.TimeoutExpired:
        return {"command": cmd, "returncode": None, "stdout": "", "stderr": "timeout"}
    except OSError as error:
        return {"command": cmd, "returncode": None, "stdout": "", "stderr": str(error)}


def parse_gpu_query(text):
    fields = [
        "name", "uuid", "serial", "pci_bus_id", "pcie_link_gen_current",
        "pcie_link_gen_max", "pcie_link_width_current", "pcie_link_width_max",
        "mig_mode_current", "mig_mode_pending", "driver_version", "vbios_version",
        "memory_total", "clocks_sm", "clocks_mem", "clocks_max_sm", "clocks_max_mem",
        "power_draw", "power_limit", "power_max_limit", "temperature_gpu",
        "utilization_gpu", "persistence_mode", "compute_mode", "ecc_mode_current",
    ]
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        row = {}
        for index, name in enumerate(fields):
            if index < len(parts):
                row[name] = parts[index]
        rows.append(row)
    return rows


def nvidia_smi_snapshot(gpu_index, phase, out_dir):
    result = {
        "phase": phase,
        "sampled_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "errors": [],
    }
    base_fields = [
        "name", "uuid", "serial", "pci.bus_id", "pcie.link.gen.current",
        "pcie.link.gen.max", "pcie.link.width.current", "pcie.link.width.max",
        "mig.mode.current", "mig.mode.pending", "driver_version", "vbios_version",
        "memory.total", "clocks.sm", "clocks.mem", "clocks.max.sm", "clocks.max.mem",
        "power.draw", "power.limit", "power.max_limit", "temperature.gpu",
        "utilization.gpu", "persistence_mode", "compute_mode", "ecc.mode.current",
    ]
    query = run(
        ["nvidia-smi", "--query-gpu=" + ",".join(base_fields),
         "--format=csv,noheader,nounits", "--id=%d" % gpu_index]
    )
    if query["returncode"] == 0 and query["stdout"]:
        result["gpu"] = parse_gpu_query(query["stdout"])
    else:
        result["errors"].append(query)
        # 字段在旧驱动上可能不存在，回退到最基础的集合。
        fallback = run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,clocks.sm,"
             "clocks.mem,power.draw,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits", "--id=%d" % gpu_index]
        )
        result["gpu_fallback_raw"] = fallback
        if fallback["returncode"] == 0:
            result["gpu"] = parse_gpu_query(fallback["stdout"])

    detail = run(["nvidia-smi", "-q", "-i", str(gpu_index)])
    detail_path = os.path.join(out_dir, "nvidia_smi_q_%s.txt" % phase)
    try:
        with open(detail_path, "w", encoding="utf-8") as handle:
            handle.write(("command: %s\n\n" % " ".join(detail["command"])) + detail["stdout"])
        result["nvidia_smi_q_file"] = os.path.basename(detail_path)
    except OSError as error:
        result["errors"].append({"command": detail["command"], "error": str(error)})
    if detail["returncode"] != 0:
        result["errors"].append(detail)

    listing = run(["nvidia-smi", "-L"])
    result["gpu_list"] = listing["stdout"]
    if listing["returncode"] != 0:
        result["errors"].append(listing)

    apps = run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"]
    )
    result["compute_apps_raw"] = apps["stdout"] if apps["returncode"] == 0 else ""
    if apps["returncode"] != 0:
        result["errors"].append(apps)

    topo = run(["nvidia-smi", "topo", "-m"])
    result["topology"] = topo["stdout"] if topo["returncode"] == 0 else ""
    return result


def toolchain_snapshot():
    nvcc = run(["nvcc", "--version"])
    cutlass_root = os.environ.get("CUTLASS_ROOT", "")
    cutlass = {"root": cutlass_root, "version": "unknown"}
    candidates = []
    if cutlass_root:
        candidates.append(os.path.join(cutlass_root, "include", "cutlass", "version.h"))
    here = PROJECT_ROOT
    for _ in range(9):
        candidates.append(os.path.join(here, "cutlass", "include", "cutlass", "version.h"))
        here = os.path.dirname(here)
    for candidate in candidates:
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as handle:
                    text = handle.read()
                values = {
                    key: re.search(r"#define\s+%s\s+(\d+)" % key, text)
                    for key in ("CUTLASS_MAJOR", "CUTLASS_MINOR", "CUTLASS_PATCH")
                }
                if all(values.values()):
                    cutlass = {
                        "root": os.path.dirname(os.path.dirname(os.path.dirname(candidate))),
                        "version": "%s.%s.%s"
                        % tuple(values[key].group(1) for key in
                                ("CUTLASS_MAJOR", "CUTLASS_MINOR", "CUTLASS_PATCH")),
                        "version_header": candidate,
                    }
                break
            except OSError:
                pass
    return {
        "nvcc": nvcc,
        "cutlass": cutlass,
        "python": sys.version,
        "cmake": run(["cmake", "--version"]),
        "ncu": run(["ncu", "--version"]),
        "nvidia_smi_version": run(["nvidia-smi", "--version"]),
        "make": run(["make", "--version"]),
    }


def git_snapshot():
    head = run(["git", "-C", PROJECT_ROOT, "rev-parse", "HEAD"])
    status = run(["git", "-C", PROJECT_ROOT, "status", "--porcelain"])
    return {
        "head": head["stdout"] if head["returncode"] == 0 else "",
        "dirty": bool(status["stdout"]) if status["returncode"] == 0 else None,
        "status": status["stdout"] if status["returncode"] == 0 else "",
        "root": PROJECT_ROOT,
    }


def host_snapshot():
    os_release = {}
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as handle:
            for line in handle:
                if "=" in line:
                    key, value = line.rstrip("\n").split("=", 1)
                    os_release[key] = value.strip('"')
    except OSError:
        pass
    now = datetime.now().astimezone()
    return {
        "hostname": socket.gethostname(),
        "uname": run(["uname", "-a"])["stdout"],
        "os_release": os_release,
        "timezone": now.tzname(),
        "utc_offset": now.strftime("%z"),
        "local_time": now.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cpu_count": os.cpu_count(),
    }


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", default="final", choices=["before", "after", "final"])
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--before-file", default="")
    args = parser.parse_args()

    out_path = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    snapshot = {
        "phase": args.phase,
        "sampled_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": host_snapshot(),
        "git": git_snapshot(),
        "toolchain": toolchain_snapshot(),
        "nvidia_smi": nvidia_smi_snapshot(args.gpu_index, args.phase, out_dir),
    }
    if args.merge:
        base = load_json(out_path) or {}
        if not isinstance(base, dict):
            base = {}
        if args.phase == "after":
            base["snapshot_after"] = snapshot
        else:
            base["snapshot_final"] = snapshot
            base.setdefault("snapshot_after", snapshot)
        # 顶层补充便于快速检索的字段。
        base.setdefault("hostname", snapshot["host"]["hostname"])
        base.setdefault("git_head", snapshot["git"]["head"])
        base["git_dirty"] = snapshot["git"]["dirty"]
        base["nvcc_version_text"] = snapshot["toolchain"]["nvcc"]["stdout"]
        base["cutlass"] = snapshot["toolchain"]["cutlass"]
        base["nvidia_smi"] = snapshot["nvidia_smi"]
        if args.before_file:
            before = load_json(args.before_file)
            if before is not None:
                base["snapshot_before"] = before
        base.setdefault("clock_policy", {
            "change_clocks": False,
            "note": "plan1 默认不修改 GPU 时钟；clock_locked 以 benchmark CSV 为准。",
        })
        output = base
    else:
        output = snapshot
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("environment snapshot: %s" % out_path)


if __name__ == "__main__":
    main()
