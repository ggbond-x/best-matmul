#!/usr/bin/env python3
"""Read-only GPU telemetry. Missing values stay empty; sensor cadence is unknown."""
import argparse
import csv
import datetime as dt
import importlib
import json
import signal
import subprocess
import time
from pathlib import Path

REASONS = {1: "gpu_idle", 2: "applications_clocks_setting", 4: "sw_power_cap",
           8: "hw_slowdown", 16: "sync_boost", 32: "sw_thermal_slowdown",
           64: "hw_thermal_slowdown", 128: "hw_power_brake_slowdown",
           256: "display_clock_setting"}
FIELDS = ["timestamp_utc", "timestamp_unix_ns", "timestamp_monotonic_ns",
          "query_start_monotonic_ns", "query_end_monotonic_ns", "query_duration_ms",
          "gpu_uuid", "source", "gpu_utilization_pct", "power_draw_w", "power_limit_w",
          "sm_clock_mhz", "memory_clock_mhz", "temperature_c", "memory_used_mib",
          "clock_reasons_mask", "clock_reasons", "power_throttle", "thermal_throttle",
          "idle_reason", "application_clock_reason", "hw_slowdown", "power_brake",
          "compute_processes", "errors", "run_id", "case_id", "phase"]
SMI_FIELDS = {
    "gpu_uuid": "uuid", "gpu_utilization_pct": "utilization.gpu",
    "power_draw_w": "power.draw", "power_limit_w": "power.limit",
    "sm_clock_mhz": "clocks.sm", "memory_clock_mhz": "clocks.mem",
    "temperature_c": "temperature.gpu", "memory_used_mib": "memory.used",
    "clock_reasons_mask": "clocks_event_reasons.active"}


def command(argv, timeout=8):
    p = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if p.returncode:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip() or f"exit {p.returncode}")
    return p.stdout.strip()


def number(value):
    try:
        x = float(str(value).strip())
        return x if __import__("math").isfinite(x) else None
    except (ValueError, TypeError):
        return None


def decode_reasons(value):
    if value is None or str(value).strip() in ("", "N/A", "[N/A]", "Not Supported", "[Not Supported]"):
        return {}
    try:
        mask = int(str(value), 0) if isinstance(value, str) else int(value)
    except (ValueError, TypeError):
        return {}
    return {"clock_reasons_mask": mask,
            "clock_reasons": "|".join(v for k, v in REASONS.items() if mask & k) or "none",
            "power_throttle": int(bool(mask & (4 | 128))),
            "thermal_throttle": int(bool(mask & (32 | 64))),
            "idle_reason": int(bool(mask & 1)), "application_clock_reason": int(bool(mask & 2)),
            "hw_slowdown": int(bool(mask & 8)), "power_brake": int(bool(mask & 128))}


class NVML:
    source = "nvml"

    def __init__(self, uuid):
        self.n = importlib.import_module("pynvml")
        self.n.nvmlInit()
        try:
            self.h = self.n.nvmlDeviceGetHandleByUUID(uuid)
        except TypeError:
            self.h = self.n.nvmlDeviceGetHandleByUUID(uuid.encode())
        actual = self.n.nvmlDeviceGetUUID(self.h)
        if isinstance(actual, bytes):
            actual = actual.decode()
        if actual.lower() != uuid.lower():
            raise RuntimeError("NVML/CUDA UUID mismatch")
        self.uuid = actual

    def sample(self):
        n, h = self.n, self.h
        r, errors = {"gpu_uuid": self.uuid}, []

        def read(key, fn):
            try:
                r[key] = fn()
            except Exception as e:
                errors.append(f"{key}:{type(e).__name__}")

        read("gpu_utilization_pct", lambda: n.nvmlDeviceGetUtilizationRates(h).gpu)
        read("power_draw_w", lambda: n.nvmlDeviceGetPowerUsage(h) / 1000)
        read("power_limit_w", lambda: n.nvmlDeviceGetPowerManagementLimit(h) / 1000)
        read("sm_clock_mhz", lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM))
        read("memory_clock_mhz", lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM))
        read("temperature_c", lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU))
        read("memory_used_mib", lambda: n.nvmlDeviceGetMemoryInfo(h).used / 1048576)
        def reasons():
            fn = getattr(n, "nvmlDeviceGetCurrentClocksEventReasons", None)
            if fn is None:
                fn = n.nvmlDeviceGetCurrentClocksThrottleReasons
            return fn(h)
        read("clock_reasons_mask", reasons)
        read("compute_processes", lambda: json.dumps([{"pid": p.pid} for p in n.nvmlDeviceGetComputeRunningProcesses(h)]))
        r.update(decode_reasons(r.get("clock_reasons_mask")))
        r["errors"] = ";".join(errors)
        return r


class SMI:
    source = "nvidia-smi"

    def __init__(self, uuid):
        self.uuid = uuid
        help_text = command(["nvidia-smi", "--help-query-gpu"])
        self.fields = dict(SMI_FIELDS)
        if '"clocks_event_reasons.active"' not in help_text:
            self.fields["clock_reasons_mask"] = "clocks_throttle_reasons.active"
        try:
            self.query(self.fields)
        except Exception:
            # Some drivers reject an entire query for one unknown field.
            supported = {}
            for key, field in self.fields.items():
                try:
                    self.query({key: field})
                    supported[key] = field
                except Exception:
                    pass
            self.fields = supported
        if "gpu_uuid" not in self.fields:
            raise RuntimeError("nvidia-smi cannot resolve target UUID")

    def query(self, fields):
        out = command(["nvidia-smi", "-i", self.uuid, "--query-gpu=" + ",".join(fields.values()),
                       "--format=csv,noheader,nounits"])
        rows = list(csv.reader(out.splitlines()))
        if len(rows) != 1 or len(rows[0]) != len(fields):
            raise RuntimeError("Unexpected nvidia-smi target/CSV shape")
        return dict(zip(fields, [v.strip() for v in rows[0]]))

    def sample(self):
        raw = self.query(self.fields)
        if raw["gpu_uuid"].lower() != self.uuid.lower():
            raise RuntimeError("nvidia-smi/CUDA UUID mismatch")
        r = {"gpu_uuid": raw["gpu_uuid"]}
        for key, val in raw.items():
            if key not in ("gpu_uuid", "clock_reasons_mask"):
                r[key] = number(val)
        r.update(decode_reasons(raw.get("clock_reasons_mask")))
        r["errors"] = ";".join(f"{key}:unavailable" for key in SMI_FIELDS if r.get(key) is None)
        # Process ownership must not be guessed from GPU utilization.
        r["compute_processes"] = ""
        return r


def sampler(uuid, force="auto"):
    failures = []
    for cls in ([NVML, SMI] if force == "auto" else [NVML] if force == "nvml" else [SMI]):
        try:
            return cls(uuid), failures
        except Exception as e:
            failures.append(f"{cls.source}: {e}")
    raise RuntimeError("; ".join(failures))


def snapshot(uuid):
    try:
        s, warnings = sampler(uuid)
        r = s.sample()
        return {"source": s.source, "sample": r, "warnings": warnings}
    except Exception as e:
        return {"source": "unavailable", "sample": {}, "warnings": [str(e)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uuid", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ready", type=Path, required=True)
    ap.add_argument("--stop-file", type=Path, required=True)
    ap.add_argument("--interval-ms", type=float, default=150)
    ap.add_argument("--backend", choices=["auto", "nvml", "nvidia-smi"], default="auto")
    a = ap.parse_args()
    if not 20 <= a.interval_ms <= 10000:
        ap.error("interval-ms must be 20..10000")
    stop = False

    def stopped(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    s = None
    try:
        s, warnings = sampler(a.uuid, a.backend)
    except Exception as e:
        warnings = [str(e)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        f.flush()
        a.ready.write_text(json.dumps({"source": s.source if s else "unavailable", "warnings": warnings,
                                     "target_interval_ms": a.interval_ms, "sensor_refresh_period": "unknown",
                                     "gpu_uuid": a.uuid}, indent=2))
        deadline = time.monotonic()
        failures = 0
        while not stop and not a.stop_file.exists():
            start, unix_start = time.monotonic_ns(), time.time_ns()
            try:
                r = s.sample() if s else {"gpu_uuid": a.uuid, "errors": "Telemetry backend unavailable"}
                failures = 0
            except Exception as e:
                failures += 1
                r = {"gpu_uuid": a.uuid, "errors": str(e)}
                if failures >= 3 and s and s.source == "nvml" and a.backend == "auto":
                    try:
                        s = SMI(a.uuid)
                    except Exception:
                        pass
            end, unix_end = time.monotonic_ns(), time.time_ns()
            r.update(timestamp_utc=dt.datetime.fromtimestamp((unix_start + unix_end) / 2e9, dt.timezone.utc).isoformat(),
                     timestamp_unix_ns=(unix_start + unix_end) // 2,
                     timestamp_monotonic_ns=(start + end) // 2,
                     query_start_monotonic_ns=start, query_end_monotonic_ns=end,
                     query_duration_ms=(end - start) / 1e6, source=s.source if s else "unavailable",
                     run_id="", case_id="", phase="unassigned")
            writer.writerow({k: "" if r.get(k) is None else r.get(k, "") for k in FIELDS})
            f.flush()
            deadline += a.interval_ms / 1000
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                deadline = time.monotonic()  # Do not burst catch-up samples.
    return 0 if s else 2


if __name__ == "__main__":
    raise SystemExit(main())
