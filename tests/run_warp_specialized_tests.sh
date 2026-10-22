#!/usr/bin/env bash
# 旧的多配置入口。主路径现在只校验 teach 与 fast 的固定默认值。
printf 'run_warp_specialized_tests.sh now builds teach and fast only.\n' >&2
printf 'Historical v1–v5 sweeps: bash experiments/build.sh --with-validation\n' >&2
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_validation.sh" "$@"
