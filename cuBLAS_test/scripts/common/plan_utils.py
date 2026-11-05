#!/usr/bin/env python3
"""跨计划复用的计划解析与运行目录工具。

命令：
  init     解析 plan.yaml + 命令行覆盖，创建新的运行目录并打印 shell 变量
  read     读取已有运行目录的 resolved_plan.yaml 并打印同样格式的 shell 变量
  finalize 把脚本侧的运行状态、文件清单合并进 run_manifest.json
  hash     打印某份计划在指定覆盖下的 config_hash

设计约束：
  * 不扫描、不猜测运行目录；run_id 只由本工具创建。
  * 不覆盖已存在的运行目录。
  * 只依赖 Python 标准库；PyYAML 可用时使用，否则使用内置的受限 YAML 解析器。
"""

import argparse
import ast
import hashlib
import json
import math
import os
import re
import secrets
import shlex
import socket
import sys
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "plan1")


# ---------------------------------------------------------------------------
# YAML：优先 PyYAML；不可用时使用覆盖本计划文件语法的受限解析器。
# ---------------------------------------------------------------------------


def _strip_comment(text):
    out = []
    quote = None
    for ch in text:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            continue
        if ch == "#":
            break
        out.append(ch)
    return "".join(out).rstrip()


def _scalar(text):
    text = text.strip()
    if not text:
        return None
    if text in ("~", "null", "Null", "NULL"):
        return None
    if text in ("true", "True", "TRUE"):
        return True
    if text in ("false", "False", "FALSE"):
        return False
    if (text.startswith("'") and text.endswith("'")) or (
        text.startswith('"') and text.endswith('"')
    ):
        try:
            return json.loads(text) if text.startswith('"') else text[1:-1]
        except ValueError:
            return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_scalar(part) for part in _split_flow(inner)]
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        result = {}
        if inner:
            for part in _split_flow(inner):
                key, value = _split_key_value(part)
                result[key] = _scalar(value)
        return result
    try:
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        if re.fullmatch(r"[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?", text):
            return float(text)
    except ValueError:
        pass
    return text


def _split_flow(text):
    parts = []
    depth = 0
    current = []
    quote = None
    for ch in text:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            current.append(ch)
            continue
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _split_key_value(text):
    quote = None
    depth = 0
    for index, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        elif ch == ":" and depth == 0:
            return text[:index].strip(), text[index + 1 :].strip()
    raise ValueError("expected 'key: value' in %r" % text)


def _fallback_yaml(text):
    lines = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        content = _strip_comment(raw.strip())
        if content:
            lines.append((indent, content))

    def parse_block(index, indent):
        if index >= len(lines):
            return {}, index
        is_list = lines[index][1].startswith("- ")
        result = [] if is_list else {}
        while index < len(lines):
            line_indent, content = lines[index]
            if line_indent < indent:
                break
            if line_indent > indent:
                break
            if is_list:
                if not content.startswith("- "):
                    break
                rest = content[2:].strip()
                if not rest:
                    index += 1
                    value, index = parse_block(index, indent + 2)
                    result.append(value)
                    continue
                if rest[0] in "[{'\"":
                    result.append(_scalar(rest))
                    index += 1
                    continue
                key, sep, value_text = rest.partition(":")
                if sep and (value_text.startswith(" ") or not value_text):
                    item = {}
                    item[key.strip()] = _scalar(value_text)
                    index += 1
                    # 同一列表项的后续键位于更深的缩进。
                    while index < len(lines) and lines[index][0] > indent:
                        sub_indent, sub_content = lines[index]
                        if sub_content.startswith("- "):
                            break
                        skey, svalue = _split_key_value(sub_content)
                        if not svalue:
                            index += 1
                            nested, index = parse_block(index, sub_indent + 2)
                            item[skey] = nested
                        else:
                            item[skey] = _scalar(svalue)
                            index += 1
                    result.append(item)
                else:
                    result.append(_scalar(rest))
                    index += 1
            else:
                if content.startswith("- "):
                    break
                key, value_text = _split_key_value(content)
                if value_text in (">", ">-", "|", "|-"):
                    preserve = value_text.startswith("|")
                    index += 1
                    collected = []
                    while index < len(lines) and lines[index][0] > indent:
                        collected.append(lines[index][1])
                        index += 1
                    joined = ("\n" if preserve else " ").join(collected)
                    result[key] = joined + ("" if value_text.endswith("-") else "\n")
                elif value_text:
                    result[key] = _scalar(value_text)
                    index += 1
                else:
                    index += 1
                    if index < len(lines) and lines[index][0] > indent:
                        nested, index = parse_block(index, lines[index][0])
                        result[key] = nested
                    else:
                        result[key] = None
        return result, index

    value, _ = parse_block(0, 0)
    return value


def load_plan(path):
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    if os.environ.get("PLAN_FORCE_FALLBACK") != "1":
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text)
        except ImportError:
            pass
    return _fallback_yaml(text)


def dump_yaml(path, data):
    try:
        import yaml  # type: ignore

        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
            return
    except ImportError:
        pass
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")


# ---------------------------------------------------------------------------
# 配置解析
# ---------------------------------------------------------------------------


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)


def deep_get(data, path, default=None):
    node = data
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def deep_set(data, path, value):
    keys = path.split(".")
    node = data
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value)]


def canonical_hash(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _threshold_value(value, k):
    """Evaluate a numeric threshold or a small arithmetic expression in K, without eval."""
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "K":
            return float(k)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = visit(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            return left / right
        raise ValueError("threshold permits only numbers, K and + - * /")
    result = visit(ast.parse(str(value), mode="eval"))
    if not math.isfinite(result) or result < 0:
        raise ValueError("threshold must be finite and nonnegative")
    return result


def case_definitions(config):
    """Resolve dimensions and per-K thresholds to the exact binary CLI representation."""
    thresholds = config["correctness"]["thresholds"]
    entries, seen = [], set()
    for shape in config["shapes"]:
        ident, priority = str(shape["id"]), str(shape.get("priority", "P1"))
        if not re.fullmatch(r"[A-Za-z0-9_]+", ident) or ident in seen or not re.fullmatch(r"[A-Za-z0-9_]+", priority):
            raise ValueError("invalid or duplicate case ID/priority: " + ident)
        seen.add(ident)
        dims = [int(shape[x]) for x in ("m", "n", "k")]
        if any(v <= 0 or v > 1000000000 for v in dims):
            raise ValueError("invalid dimensions for " + ident)
        fields = [ident] + [str(v) for v in dims] + [priority]
        for precision in ("FP16", "FP32"):
            for key in ("atol", "rtol", "rel_l2_max", "mismatch_fraction_max"):
                number = _threshold_value(thresholds[precision][key], dims[2])
                if key == "mismatch_fraction_max" and number > 1:
                    raise ValueError("mismatch_fraction_max exceeds 1")
                fields.append(repr(number))
        entries.append(":".join(fields))
    if not entries:
        raise ValueError("plan must contain shapes")
    return ",".join(entries)


def apply_overrides(config, overrides):
    """把 --set 的便捷键写入生效配置；未知键直接报错，避免静默忽略。"""
    timing_warm_keys = {"warmup", "inner", "samples", "rounds"}
    apply_to = {
        "cases": ("selected", "cases"),
        "cold_cases": ("selected", "cold_cases"),
        "layouts": ("selected", "layouts"),
        "implementations": ("selected", "implementations"),
        "timing": ("timing", "mode"),
        "cache": ("selected", "cache_mode"),
        "seed": ("input", "seed"),
        "flush_mb": ("timing", "cold", "flush_mb"),
        "lt_workspace_mb": ("cublaslt", "workspace_mb"),
        "lt_candidates": ("cublaslt", "candidates"),
        "lt_selection": ("cublaslt", "selection"),
        "device": ("run", "device"),
        "notes": ("run", "notes"),
    }
    for item in overrides:
        if "=" not in item:
            raise SystemExit("--set 需要 KEY=VALUE，收到: %s" % item)
        key, raw = item.split("=", 1)
        key = key.strip()
        if key in timing_warm_keys:
            target = ("timing", "warm", key)
        elif key in apply_to:
            target = apply_to[key]
        else:
            raise SystemExit(
                "未知覆盖键 %r；可用: %s"
                % (key, ", ".join(sorted(list(apply_to) + list(timing_warm_keys))))
            )
        if key in ("seed", "flush_mb", "lt_workspace_mb", "lt_candidates", "device"):
            value = int(raw)
        elif key == "cases" or key == "cold_cases" or key == "layouts" or key == "implementations":
            value = as_list(raw)
        else:
            value = raw
        node = config
        for part in target[:-1]:
            node = node.setdefault(part, {})
        node[target[-1]] = value


def resolve_config(plan, mode, overrides):
    if "modes" not in plan or mode not in plan["modes"]:
        raise SystemExit("plan.yaml 缺少 modes.%s" % mode)
    selected = plan["modes"][mode]
    config = {
        "plan_id": plan.get("plan_id", "plan1"),
        "name": plan.get("name", ""),
        "purpose": plan.get("purpose", ""),
        "target": plan.get("target", {}),
        "input": plan.get("input", {}),
        "layouts": plan.get("layouts", ["TN", "NT"]),
        "layout_definitions": plan.get("layout_definitions", {}),
        "shapes": plan.get("shapes", []),
        "implementations": plan.get("implementations", {}),
        "correctness": plan.get("correctness", {}),
        "timing": plan.get("timing", {}),
        "cublaslt": plan.get("cublaslt", {}),
        "ncu": plan.get("ncu", {}),
        "clock_policy": plan.get("clock_policy", {}),
        "monitoring": plan.get("monitoring", {}),
        "failures": plan.get("failures", {}),
        "selected": {
            "mode": mode,
            "cases": as_list(selected.get("cases")),
            "cold_cases": as_list(selected.get("cold_cases")),
            "layouts": as_list(selected.get("layouts") or plan.get("layouts")),
            "implementations": as_list(selected.get("implementations") or "all"),
            "cache_mode": "warm",
        },
        "run": {
            "run_id": "",
            "started_utc": "",
            "output_dir": "",
            "device": 0,
            "notes": "",
            "overrides": list(overrides),
            "hostname": socket.gethostname(),
            "cwd": os.getcwd(),
        },
    }
    apply_overrides(config, overrides)

    input_config = config.get("input") or {}
    if (input_config.get("dtype") != "FP16" or input_config.get("output_dtype") != "FP16"
            or input_config.get("alpha") != 1.0 or input_config.get("beta") != 0.0):
        raise SystemExit("当前程序仅支持 FP16 输入/输出、alpha=1、beta=0；拒绝与计划不一致的配置")
    correctness_config = config.get("correctness") or {}
    if (correctness_config.get("reference") != "full_gpu_fp64"
            or correctness_config.get("sampled_fallback") != "sampled_cpu_fp64"):
        raise SystemExit("当前程序仅支持 full_gpu_fp64 参考和 sampled_cpu_fp64 回退")
    if (config.get("timing", {}).get("warm", {}).get("order", "rotate") != "rotate"):
        raise SystemExit("当前程序仅支持 rotate 轮次顺序")

    available = as_list(plan.get("implementations", {}).get("custom")) + as_list(
        plan.get("implementations", {}).get("libraries")
    )
    selected_impls = config["selected"]["implementations"]
    if selected_impls in (["all"], ["ALL"], ["All"]):
        selected_impls = available
    unknown = [name for name in selected_impls if name not in available]
    if unknown:
        raise SystemExit("未知实现: %s" % ", ".join(unknown))
    config["selected"]["implementations"] = selected_impls

    if config["selected"]["cache_mode"] not in ("warm", "cold"):
        raise SystemExit("cache_mode 必须是 warm/cold")
    if config["selected"]["mode"] != "ncu":
        if config["timing"].get("mode") not in ("event", "wall", "graph"):
            raise SystemExit("timing.mode 必须是 event/wall/graph")
    try:
        case_definitions(config)
    except (KeyError, ValueError, TypeError, ZeroDivisionError, SyntaxError) as error:
        raise SystemExit("无效的 Shape 或正确性阈值: %s" % error)
    shape_ids = [shape["id"] for shape in config["shapes"]]
    for case_id in config["selected"]["cases"] + config["selected"]["cold_cases"]:
        if case_id not in shape_ids:
            raise SystemExit("未知 case: %s" % case_id)
    for layout in config["selected"]["layouts"]:
        if layout not in ("TN", "NT"):
            raise SystemExit("未知 layout: %s" % layout)
    return config


def shell_vars(config):
    selected = config["selected"]
    warm = config["timing"].get("warm", {})
    cold = config["timing"].get("cold", {})
    monitoring = config.get("monitoring", {})
    values = {
        "PLAN_ID": config["plan_id"],
        "PLAN_NAME": config.get("name", ""),
        "RUN_ID": config["run"]["run_id"],
        "RUN_DIR": config["run"]["output_dir"],
        "CONFIG_HASH": config["run"].get("config_hash", ""),
        "MODE": selected["mode"],
        "CASES": ",".join(selected["cases"]),
        "CASE_DEFS": case_definitions(config),
        "COLD_CASES": ",".join(selected["cold_cases"]),
        "LAYOUTS": ",".join(selected["layouts"]),
        "IMPLS": ",".join(selected["implementations"]),
        "CACHE_MODE": selected["cache_mode"],
        "TIMING_MODE": config["timing"].get("mode", "event"),
        "WARMUP": str(warm.get("warmup", 20)),
        "INNER": str(warm.get("inner", 100)),
        "SAMPLES": str(warm.get("samples", 5)),
        "ROUNDS": str(warm.get("rounds", 5)),
        "SEED": str(config["input"].get("seed", 20261105)),
        "FLUSH_MB": str(cold.get("flush_mb", 256)),
        "LT_WORKSPACE_MB": str(config["cublaslt"].get("workspace_mb", 64)),
        "LT_CANDIDATES": str(config["cublaslt"].get("candidates", 32)),
        "LT_SELECTION": str(config["cublaslt"].get("selection", "tune")),
        "DEVICE": str(config["run"].get("device", 0)),
        "MONITOR_INTERVAL": str(monitoring.get("interval_s", 0.5)),
        "TARGET_GPU": str(config.get("target", {}).get("gpu", "")),
        "THEORETICAL_TFLOPS": str(config.get("target", {}).get("theoretical_dense_fp16_tflops", "")),
        "PLAN_STARTED_UTC": config["run"].get("started_utc", ""),
    }
    ncu = config.get("ncu", {}) or {}
    values.update({
        "NCU_CASES": ",".join(as_list(ncu.get("cases"))),
        "NCU_LAYOUTS": ",".join(as_list(ncu.get("layouts"))),
        "NCU_IMPLS": ",".join(as_list(ncu.get("implementations"))),
        "NCU_SECTIONS": ",".join(as_list(ncu.get("sections"))),
        "NCU_CACHE_CONTROL": str(ncu.get("cache_control", "all")),
        "NCU_REPLAY_MODE": str(ncu.get("replay_mode", "kernel")),
        "NCU_CLOCK_CONTROL": str(ncu.get("clock_control", "none")),
        "NCU_LAUNCH_SKIP": str(ncu.get("launch_skip", 0)),
        "NCU_LAUNCH_COUNT": str(ncu.get("launch_count", 5)),
    })
    for key, value in values.items():
        sys.stdout.write("%s=%s\n" % (key, shlex.quote(str(value))))


def cmd_init(args):
    plan = load_plan(args.plan)
    config = resolve_config(plan, args.mode, args.set or [])
    output_root = os.path.abspath(args.output_root or os.path.join(PROJECT_ROOT, "output", "plan1"))
    run_id = args.run_id or new_run_id()
    run_dir = os.path.join(output_root, run_id)
    if os.path.exists(run_dir):
        raise SystemExit("运行目录已存在，拒绝覆盖: %s" % run_dir)
    os.makedirs(os.path.join(run_dir, "csvs"))
    os.makedirs(os.path.join(run_dir, "reports"))
    os.makedirs(os.path.join(run_dir, "figures"))
    config["run"]["run_id"] = run_id
    config["run"]["started_utc"] = utc_now()
    config["run"]["output_dir"] = run_dir
    config["run"]["config_hash"] = canonical_hash(
        {key: value for key, value in config.items() if key != "run"}
    )
    dump_yaml(os.path.join(run_dir, "resolved_plan.yaml"), config)
    shell_vars(config)


def cmd_read(args):
    path = os.path.join(os.path.abspath(args.run_dir), "resolved_plan.yaml")
    if not os.path.isfile(path):
        raise SystemExit("找不到 resolved_plan.yaml: %s" % path)
    config = load_plan(path)
    config.setdefault("run", {})["output_dir"] = os.path.abspath(args.run_dir)
    shell_vars(config)


def _relative_files(run_dir, skip_hidden=True):
    result = []
    for root, dirs, files in os.walk(run_dir):
        dirs[:] = sorted(d for d in dirs if not (skip_hidden and d.startswith(".")))
        for name in sorted(files):
            if skip_hidden and name.startswith("."):
                continue
            full = os.path.join(root, name)
            result.append(os.path.relpath(full, run_dir))
    return result


def cmd_finalize(args):
    run_dir = os.path.abspath(args.run_dir)
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        manifest = {"cases": []}
    manifest["run_dir"] = run_dir
    manifest["script_status"] = args.status
    manifest["script_finished_utc"] = utc_now()
    manifest["resolved_plan"] = "resolved_plan.yaml"
    manifest["environment"] = "environment.json" if os.path.isfile(
        os.path.join(run_dir, "environment.json")
    ) else "N/A"
    manifest["clock_power_csv"] = "clock_power.csv" if os.path.isfile(
        os.path.join(run_dir, "clock_power.csv")
    ) else "N/A"
    manifest["summary_csv"] = "csvs/summary.csv" if os.path.isfile(
        os.path.join(run_dir, "csvs", "summary.csv")
    ) else "N/A"
    manifest["analysis"] = "analysis.md" if os.path.isfile(
        os.path.join(run_dir, "analysis.md")
    ) else "N/A"
    manifest["files"] = _relative_files(run_dir)
    if args.note:
        manifest["script_note"] = args.note
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def cmd_hash(args):
    plan = load_plan(args.plan)
    config = resolve_config(plan, args.mode, args.set or [])
    print(
        canonical_hash(
            {key: value for key, value in config.items() if key != "run"}
        )
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="创建新运行目录并输出 shell 变量")
    init.add_argument("--plan", required=True)
    init.add_argument("--mode", required=True, choices=["quick", "full", "ncu"])
    init.add_argument("--output-root", default="")
    init.add_argument("--run-id", default="")
    init.add_argument("--set", action="append", default=[])
    init.set_defaults(func=cmd_init)

    read = sub.add_parser("read", help="读取 resolved_plan.yaml 并输出 shell 变量")
    read.add_argument("--run-dir", required=True)
    read.set_defaults(func=cmd_read)

    finalize = sub.add_parser("finalize", help="合并脚本侧状态到 run_manifest.json")
    finalize.add_argument("--run-dir", required=True)
    finalize.add_argument("--status", required=True)
    finalize.add_argument("--note", default="")
    finalize.set_defaults(func=cmd_finalize)

    hash_cmd = sub.add_parser("hash", help="打印配置哈希")
    hash_cmd.add_argument("--plan", required=True)
    hash_cmd.add_argument("--mode", default="quick")
    hash_cmd.add_argument("--set", action="append", default=[])
    hash_cmd.set_defaults(func=cmd_hash)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
