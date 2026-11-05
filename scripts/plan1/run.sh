#!/usr/bin/env bash
# Hopper v1–v5 完整对照实验入口。在 Linux/Bash 中执行；编译命令全部在 build.sh。
# 具体执行逻辑在 scripts/common/runner_common.sh，本文件只提供 plan1 默认值。
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

runner_invocation="run.sh"
default_build_dir="build/plan1/h100_pcie"
default_output_root="output/h100_pcie"
default_shapes="1024x1024x1024,4096x4096x4096,8192x8192x8192,4096x4096x256,16384x1024x4096,1024x16384x4096"
shapes_help="见下方"
default_shapes_text="1024^3、4096^3、8192^3、4096x4096x256、16384x1024x4096、1024x16384x4096"
default_baseline="v1"
variants_example="v3,v4,v5"
build_dir_text="build.sh 的输出目录"
build_hint="bash build.sh"
validate_hint="bash build.sh --with-validation"
audit_runner="run.sh"
usage_footer="build/plan1/h100_pcie 和 output/h100_pcie 的目录名也可用于 H800 等 Hopper GPU。"

source "$script_dir/../common/runner_common.sh"
