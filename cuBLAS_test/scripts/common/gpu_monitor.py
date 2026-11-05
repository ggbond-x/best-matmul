#!/usr/bin/env python3
"""按固定间隔采样 GPU 时钟/功耗/温度/利用率，写入 clock_power.csv。

用法：
  gpu_monitor.py --output clock_power.csv --interval 0.5 --gpu 0

由 run 脚本后台启动，并在退出 trap 中发送 SIGTERM。采样失败不终止基准，
CSV 中对应字段写 N/A，并在 stderr 输出一次错误。
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

FIELDS = [
    "timestamp_utc",
    "elapsed_s",
    "gpu_index",
    "gpu_name",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "power_w",
    "temperature_c",
    "utilization_gpu_pct",
    "utilization_memory_pct",
    "pstate",
    "throttle_reasons",
]

QUERY = (
    "timestamp,index,name,clocks.sm,clocks.mem,power.draw,temperature.gpu,"
    "utilization.gpu,utilization.memory,pstate,clocks_throttle_reasons.active"
)


def clean(value):
    value = (value or "").strip()
    if value in ("[N/A]", "[Not Supported]", "N/A", ""):
        return "N/A"
    return value


def sample(gpu_index):
    """返回 (10 个数据列, 错误文本)。数据列为 index..throttle_reasons。"""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=" + QUERY,
                "--format=csv,noheader,nounits",
                "--id=%d" % gpu_index,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as error:
        return [timestamp] + ["N/A"] * 10, str(error)
    if proc.returncode != 0:
        return [timestamp] + ["N/A"] * 10, proc.stderr.strip() or "nvidia-smi failed"
    parts = [part.strip() for part in proc.stdout.strip().split(",")]
    # QUERY 的 11 个字段：timestamp,index,name,clocks.sm,clocks.mem,power.draw,
    # temperature.gpu,utilization.gpu,utilization.memory,pstate,throttle。
    # nvidia-smi 的时间戳形如 "2026/11/05 12:00:00.000"，不含逗号，因此可安全切分。
    values = parts[1:] if len(parts) >= 11 else parts
    while len(values) < 10:
        values.append("N/A")
    return [timestamp] + [clean(value) for value in values[:10]], ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    interval = max(0.1, args.interval)
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    running = {"value": True}

    def stop(_signum, _frame):
        running["value"] = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    started = time.monotonic()
    header = ["elapsed_s"] + [field for field in FIELDS if field != "elapsed_s"]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        handle.flush()
        errors = 0
        while running["value"]:
            tick = time.monotonic()
            values, error = sample(args.gpu)
            if error:
                errors += 1
                if errors == 1:
                    sys.stderr.write("gpu_monitor: %s\n" % error)
            writer.writerow(["%.3f" % (tick - started)] + values)
            handle.flush()
            elapsed = time.monotonic() - tick
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
    if errors:
        sys.stderr.write("gpu_monitor: %d sampling error(s)\n" % errors)


if __name__ == "__main__":
    main()
