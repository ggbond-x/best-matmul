#!/usr/bin/env bash
# plan2 已有配置的执行入口。只读取 scripts/plan2/run.sh --build-only 生成的 manifest。
# 具体执行逻辑在 scripts/common/runner_common.sh，本文件只提供 plan2 默认值。
# 必须用 --build-dir 指定某次构建，避免误选其他历史构建。
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

runner_invocation="scripts/plan2/run_variants.sh"
# --build-dir 必填，因此不提供默认构建目录。
default_output_root="output/plan2"
default_shapes="4096x4096x256,4096x4096x4096"
default_shapes_text="4096x4096x256、4096x4096x4096"
default_baseline="v2_p3_n128_m128n128"
variants_example="v2_p3_n128_m128n128"
build_dir_text="构建目录；必填，指向 run.sh --build-only 的输出"
build_dir_required=1
build_hint="bash scripts/plan2/run.sh --build-only"
validate_hint="bash scripts/plan2/run.sh --build-only"
audit_runner="scripts/plan2/run_variants.sh"

source "$script_dir/../common/runner_common.sh"
