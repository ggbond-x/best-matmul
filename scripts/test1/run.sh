#!/usr/bin/env bash
# Hopper 对照测试入口。在 Linux/Bash 中执行；编译命令全部在 build.sh。
# 计时、正确性检查、NCU replay、Compute Sanitizer 分别保存，禁止混用其耗时。
# 不自动修改应用时钟、功耗上限、MIG、持久化模式或驱动计数器权限。
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
build_dir="$script_dir/build/h100_pcie"
output_dir=""
mode=bench
gpu=0
variants=all
layouts=NT,TN
shapes=1024x1024x1024,4096x4096x4096,8192x8192x8192,4096x4096x256,16384x1024x4096,1024x16384x4096
profile_shape=4096x4096x4096
repeats=3
per_batch=20
min_warmup=300
batches=100
ncu_bin="${NCU:-ncu}"
ncu_set=focused
ncu_cache=cold
ncu_clock=none
launch_skip=50
metrics=""
sanitizers=memcheck,racecheck,synccheck
timeout_sec=1800
telemetry_ms=1000
dry_run=0
baseline=v1

usage() {
  cat <<'USAGE'
Usage: bash run.sh [options]
  --mode env|check|validate|bench|ncu|nsys|sanitize|disasm|all   默认 bench
  --build-dir DIR       build.sh 的输出目录，默认 build/h100_pcie
  --output-dir DIR      新建的结果目录；必须不存在，避免覆盖实验（默认 reports/时间戳）
  --gpu ID             nvidia-smi 的卡索引或 GPU UUID，默认 0
  --variants CSV       manifest 中的配置 ID，如 v3,v4,v5；默认 all
  --layouts CSV        NT、TN 或 NT,TN（默认两者）
  --shapes CSV         bench 的 MxNxK 列表，默认见下方
  --profile-shape MxNxK NCU / Nsight Systems 的单一问题形状，默认 4096x4096x4096
  --repeats N          独立进程重复次数，默认 3，至少 2
  --per-batch N        每个计时批次启动数，默认 20
  --min-warmup N       最少预热启动数，默认 300；实际还需满足自适应稳定判据
  --batches N          正式计时批次数，默认 100，至少 2
  --baseline ID        聚合表的参考配置，默认 v1；未测试时相对值留空
  --ncu PATH           NCU 可执行文件，默认 NCU 环境变量或 ncu
  --ncu-set focused|full 默认 focused（9 个性能分析 section）；full 更慢
  --ncu-cache cold|warm 默认 cold=kernel replay+flush；warm=application replay+不清缓存
  --ncu-clock none|base 默认 none；base 仅影响 NCU，不影响普通计时
  --launch-skip N       跳过匹配的 GEMM 启动数，默认 50，至少 1
  --metrics CSV        追加已由本机 --query-metrics 确认支持的指标
  --sanitizers CSV     默认 memcheck,racecheck,synccheck
  --timeout N          每个命令超时秒数，默认 1800；0 表示不限时
  --telemetry-ms N     bench 期间 GPU 状态采样间隔毫秒，默认 1000；0 关闭，非零至少 100
  --dry-run            显示将执行的命令，不启动 GPU、不创建结果目录；仍需 manifest
  --help               显示帮助

bench 默认问题：1024^3、4096^3、8192^3、4096x4096x256、16384x1024x4096、1024x16384x4096。
all = 环境记录 + 全输出 validate + bench + ncu + disasm；需要 build.sh --with-validation。
check 是主程序抽样检查；validate 才是小矩阵每个 C 元素的参考校验。
sanitize 单独运行，代价高；其耗时与 NCU 日志中的 TFLOPS 都不能当作性能结果。
nsys 单独运行，用 Nsight Systems 检查 CPU 提交空隙和 GPU 时间线，不参与成绩汇总。
env/disasm 不启动内核；env 不需要 manifest。其余模式需要 Hopper GPU（计算能力 9.0）。
build/h100_pcie 和 reports/h100_pcie 为沿用的目录名，也可用于 H800 等 Hopper GPU。
USAGE
}
die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --dry-run) dry_run=1; shift ;;
    --mode|--build-dir|--output-dir|--gpu|--variants|--layouts|--shapes|--profile-shape|--repeats|--per-batch|--min-warmup|--batches|--baseline|--ncu|--ncu-set|--ncu-cache|--ncu-clock|--launch-skip|--metrics|--sanitizers|--timeout|--telemetry-ms)
      (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die "$1 缺少值"
      opt="${1#--}"; opt="${opt//-/_}"
      [[ "$opt" != ncu ]] || opt=ncu_bin
      [[ "$opt" != timeout ]] || opt=timeout_sec
      printf -v "$opt" '%s' "$2"; shift 2 ;;
    *) die "未知选项 $1" ;;
  esac
done
case "$mode" in env|check|validate|bench|ncu|nsys|sanitize|disasm|all) ;; *) die "不支持的 mode: $mode" ;; esac
case "$ncu_set" in focused|full) ;; *) die 'ncu-set 必须为 focused/full' ;; esac
case "$ncu_cache" in cold|warm) ;; *) die 'ncu-cache 必须为 cold/warm' ;; esac
case "$ncu_clock" in none|base) ;; *) die 'ncu-clock 必须为 none/base' ;; esac
for opt in repeats per_batch min_warmup batches launch_skip timeout_sec telemetry_ms; do
  value="${!opt}"
  [[ "$value" =~ ^(0|[1-9][0-9]{0,8})$ ]] || die "$opt 必须为不超过 9 位的非负十进制整数"
done
(( repeats >= 2 && per_batch > 0 && min_warmup > 0 && batches >= 2 && launch_skip >= 1 )) || die '重复次数/计时/预热/skip 参数超出范围'
(( min_warmup + 21 * per_batch <= 2147483647 )) || die '预热启动总数可能超出 INT_MAX'
(( telemetry_ms == 0 || telemetry_ms >= 100 )) || die 'telemetry-ms 为 0 或至少 100'
IFS=, read -r -a layout_list <<< "$layouts"
IFS=, read -r -a shape_list <<< "$shapes"
IFS=, read -r -a sanitizer_list <<< "$sanitizers"
[[ -n "$layouts" && -n "$shapes" && -n "$sanitizers" ]] || die '列表不能为空'
[[ "$layouts$shapes$sanitizers" != *,,* && "$layouts" != *, && "$shapes" != *, && "$sanitizers" != *, ]] || die '列表包含空项'
reject_duplicates() {
  local label="$1" item; shift
  local -A seen=()
  for item in "$@"; do
    [[ -n "$item" ]] || die "$label 包含空项"
    [[ -z "${seen[$item]+exists}" ]] || die "$label 包含重复项: $item"
    seen[$item]=1
  done
}
# 文件名含 layout/shape/repeat，拒绝重复项，防止覆盖日志后在汇总表中重复计数。
reject_duplicates layouts "${layout_list[@]}"
reject_duplicates shapes "${shape_list[@]}"
reject_duplicates sanitizers "${sanitizer_list[@]}"
for layout in "${layout_list[@]}"; do [[ "$layout" == NT || "$layout" == TN ]] || die "不支持布局 $layout"; done
for sanitizer in "${sanitizer_list[@]}"; do
  case "$sanitizer" in memcheck|racecheck|synccheck) ;; *) die "不支持 sanitizer: $sanitizer" ;; esac
done

# 所有元数据从编译时生成的 manifest 读取，不根据文件名猜测宏值。
declare -a ids=()
declare -A version cta_m cta_n cta_k pipes gmma compute producer inflight epilogue binary
if [[ "$mode" != env ]]; then
  [[ -f "$build_dir/manifest.tsv" ]] || die '未找到 manifest.tsv；先执行 bash build.sh'
  expected_header=$'id\tversion\tcta_m\tcta_n\tcta_k\tpipe\tgmma_n\tcompute_threads\tproducer_threads\tinflight\tepilogue\tbinary'
  IFS= read -r header < "$build_dir/manifest.tsv"
  [[ "$header" == "$expected_header" ]] || die 'manifest 格式不匹配，请重新编译'
  while IFS=$'\t' read -r id ver cm cn ck pp gn ct pt wi ep bin; do
    [[ "$id" =~ ^[a-zA-Z0-9_]+$ && -z "${binary[$id]+exists}" ]] || die "manifest ID 非法或重复: $id"
    for value in "$cm" "$cn" "$ck" "$pp" "$gn" "$ct" "$wi"; do
      [[ "$value" =~ ^[1-9][0-9]{0,4}$ ]] || die "manifest 数值非法: $id"
    done
    [[ "$pt" =~ ^(0|128)$ && "$ep" =~ ^[01]$ && "$ver" =~ ^v[1-5]$ && "$bin" == /* ]] || die "manifest 配置非法: $id"
    version[$id]="$ver"; cta_m[$id]="$cm"; cta_n[$id]="$cn"; cta_k[$id]="$ck"
    pipes[$id]="$pp"; gmma[$id]="$gn"; compute[$id]="$ct"; producer[$id]="$pt"
    inflight[$id]="$wi"; epilogue[$id]="$ep"; binary[$id]="$bin"
    ids+=("$id")
  done < <(tail -n +2 "$build_dir/manifest.tsv")
  ((${#ids[@]})) || die 'manifest 没有配置'
  if [[ "$variants" != all ]]; then
    IFS=, read -r -a ids <<< "$variants"
    [[ -n "$variants" && "$variants" != *, && "$variants" != *,,* ]] || die 'variants 列表不能为空'
    declare -A selected=()
    for id in "${ids[@]}"; do
      [[ "$id" =~ ^[a-zA-Z0-9_]+$ && -n "${binary[$id]+exists}" ]] || die "manifest 中没有 $id"
      [[ -z "${selected[$id]+exists}" ]] || die "重复 ID: $id"
      selected[$id]=1
    done
  fi
fi

parse_shape() {
  local id
  [[ "$1" =~ ^([1-9][0-9]{0,8})x([1-9][0-9]{0,8})x([1-9][0-9]{0,8})$ ]] || die "形状应为 MxNxK: $1"
  m="${BASH_REMATCH[1]}"; n="${BASH_REMATCH[2]}"; k="${BASH_REMATCH[3]}"
  (( m*n <= 2147483647 && m*k <= 2147483647 && n*k <= 2147483647 )) || die "矩阵元素数超出当前 int 索引范围: $1"
  for id in "${ids[@]}"; do
    (( m % cta_m[$id] == 0 && n % cta_n[$id] == 0 && k % cta_k[$id] == 0 )) || die "$1 不能被 $id 的 CTA tile 整除"
  done
}
if [[ "$mode" == bench || "$mode" == all ]]; then
  for shape in "${shape_list[@]}"; do parse_shape "$shape"; done
fi
if [[ "$mode" == ncu || "$mode" == nsys || "$mode" == all ]]; then parse_shape "$profile_shape"; fi

[[ -n "$output_dir" ]] || output_dir="$script_dir/reports/h100_pcie/$(date +%Y%m%d_%H%M%S)_${mode}_$$"
[[ "$output_dir" == /* ]] || output_dir="$PWD/$output_dir"
need_tool() { command -v "$1" >/dev/null 2>&1 || die "未找到 $1"; }
if (( ! dry_run )); then
  [[ ! -e "$output_dir" ]] || die "结果目录已存在: $output_dir，请使用新目录"
  (( timeout_sec == 0 )) || need_tool timeout
  if [[ "$mode" == bench || "$mode" == all ]]; then need_tool python3; fi
  if [[ "$mode" == ncu || "$mode" == all ]]; then need_tool "$ncu_bin"; fi
  if [[ "$mode" == nsys ]]; then need_tool nsys; fi
  if [[ "$mode" == sanitize ]]; then need_tool compute-sanitizer; fi
  if [[ "$mode" == disasm || "$mode" == all ]]; then need_tool cuobjdump; fi
  for id in "${ids[@]}"; do
    [[ -x "${binary[$id]}" ]] || die "不可执行: ${binary[$id]}"
    if [[ "$mode" == validate || "$mode" == all ]]; then
      [[ -x "${binary[$id]}.validate" ]] || die '缺少全输出校验程序；请先执行 build.sh --with-validation'
    fi
  done
  if [[ "$mode" == validate || "$mode" == all ]]; then
    # 未带 --with-validation 的后续编译可能留下上一次的 .validate 文件，不能误用。
    [[ -f "$build_dir/build_metadata.txt" ]] && grep -qx 'with_validation=1' "$build_dir/build_metadata.txt" || \
      die '本次构建未声明启用全输出校验；请重新执行 build.sh --with-validation'
  fi
  mkdir -p -- "$output_dir"
  printf 'mode=%s\nvariants=%s\n' "$mode" "$variants" > "$output_dir/run_metadata.txt"
  printf '# 命令审计清单；请使用 run.sh 重跑实验，不要整体执行此文件。\n# 包含持续采样命令；日志重定向及后台进程回收由 run.sh 管理。\n' > "$output_dir/commands.sh"
  [[ "$mode" == env ]] || cp -- "$build_dir/manifest.tsv" "$output_dir/manifest.tsv"
  [[ ! -f "$build_dir/build_metadata.txt" ]] || cp -- "$build_dir/build_metadata.txt" "$output_dir/"
fi

# 每条命令均保存为可复现的 shell 引用格式；外部命令退出非零即停止。
# SIGTERM 超时后再等 30 秒强制退出，避免同步错误使整套实验永久挂起。
record_command() {
  local log="$1"; shift
  printf '[%s] ' "$log"; printf '%q ' "$@"; printf '\n'
  if (( ! dry_run )); then
    { printf '# %s\n' "$log"; printf '%q ' "$@"; printf '\n'; } >> "$output_dir/commands.sh"
  fi
}
run_logged() {
  local log="$1"; shift
  local -a cmd=("$@")
  (( timeout_sec == 0 )) || cmd=(timeout --kill-after=30s "${timeout_sec}s" "${cmd[@]}")
  record_command "$log" "${cmd[@]}"
  if (( ! dry_run )); then
    printf '%s\tstart\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$log" >> "$output_dir/timeline.tsv"
    if "${cmd[@]}" > "$output_dir/$log" 2>&1; then :; else
      local status=$?
      tail -n 30 -- "$output_dir/$log" >&2
      die "命令失败(exit=$status)，见 $output_dir/$log"
    fi
    printf '%s\tend\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$log" >> "$output_dir/timeline.tsv"
  fi
}
optional_log() {
  local log="$1"; shift
  local -a cmd=("$@")
  (( timeout_sec == 0 )) || cmd=(timeout --kill-after=30s "${timeout_sec}s" "${cmd[@]}")
  record_command "$log" "${cmd[@]}"
  if (( ! dry_run )); then
    "${cmd[@]}" > "$output_dir/$log" 2>&1 || printf 'Warning: 可选环境查询失败，见 %s\n' "$log" >&2
  fi
}

telemetry_pid=""
stop_telemetry() {
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
    telemetry_pid=""
  fi
}
trap stop_telemetry EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
start_telemetry() {
  (( telemetry_ms > 0 )) || return 0
  local -a cmd=(nvidia-smi -i "$gpu"
    --query-gpu=timestamp,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used
    --format=csv --loop-ms="$telemetry_ms")
  record_command telemetry.csv "${cmd[@]}"
  if (( ! dry_run )); then
    # 后台采样由 EXIT trap 回收；不会遗留监控进程。采样不是每次 kernel 的精细测量。
    "${cmd[@]}" > "$output_dir/telemetry.csv" 2> "$output_dir/telemetry.stderr.txt" &
    telemetry_pid=$!
  fi
}

# 物理 GPU 索引先解析为 UUID，再用 CUDA_VISIBLE_DEVICES 隔离。
# 进程内 device 0 就是指定的物理卡；不用 NCU --devices 猜测重编号后的设备。
if [[ "$mode" != disasm ]]; then
  if (( dry_run )); then
    printf '# 将 nvidia-smi -i %q 查询到的 UUID 设置为 CUDA_VISIBLE_DEVICES\n' "$gpu"
  else
    need_tool nvidia-smi
    gpu_query=(nvidia-smi -i "$gpu")
    (( timeout_sec == 0 )) || gpu_query=(timeout --kill-after=30s "${timeout_sec}s" "${gpu_query[@]}")
    gpu_uuid="$("${gpu_query[@]}" --query-gpu=uuid --format=csv,noheader)" || die 'GPU 选择失败'
    gpu_name="$("${gpu_query[@]}" --query-gpu=name --format=csv,noheader)" || die 'GPU 名称查询失败'
    [[ "$gpu_uuid" == GPU-* && "$gpu_uuid" != *$'\n'* ]] || die '请指定一张完整 GPU 的索引或 UUID'
    # sm_90a 内核按架构放行；H800/H200/GH200 等型号不应被名称过滤。
    # env 仅记录环境，允许在其他架构上使用；实际执行前 CUDA 程序仍会检查设备。
    if [[ "$mode" != env ]]; then
      gpu_compute_cap="$("${gpu_query[@]}" --query-gpu=compute_cap --format=csv,noheader)" || \
        die "GPU 计算能力查询失败，无法确认 $gpu_name 是否支持 Hopper / sm_90a"
      gpu_compute_cap="${gpu_compute_cap//[[:space:]]/}"
      [[ "$gpu_compute_cap" == 9.0 ]] || \
        die "本测试需要 Hopper / compute capability 9.0，当前设备为 $gpu_name（计算能力 $gpu_compute_cap）"
      printf 'compute_cap=%s\n' "$gpu_compute_cap" >> "$output_dir/run_metadata.txt"
    fi
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu_uuid"
    printf 'export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=%q\n' "$gpu_uuid" >> "$output_dir/commands.sh"
    printf 'gpu=%s\nuuid=%s\n' "$gpu_name" "$gpu_uuid" >> "$output_dir/run_metadata.txt"
  fi
fi

collect_environment() {
  optional_log env_date.txt date --iso-8601=seconds
  optional_log env_uname.txt uname -a
  optional_log env_cpu.txt lscpu
  optional_log env_gpu_list.txt nvidia-smi -L
  optional_log env_gpu_before.txt nvidia-smi -i "$gpu" -q
  optional_log env_topology.txt nvidia-smi topo -m
  optional_log env_processes.txt nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
  # -q 含型号、PCIe 链路、MIG/ECC、温度、时钟、功耗和降频原因，字段随驱动变化。
  # 计数器/section 名随 NCU 变化，保存本机目录，避免硬编码网上过时的指标名。
  if [[ "$mode" == env || "$mode" == ncu || "$mode" == all ]]; then
    optional_log env_ncu_version.txt "$ncu_bin" --version
    optional_log env_ncu_sections.txt "$ncu_bin" --list-sections
    optional_log env_ncu_sets.txt "$ncu_bin" --list-sets
    optional_log env_ncu_metrics_gh100.txt "$ncu_bin" --query-metrics --chips gh100 --query-metrics-mode all
  fi
  if [[ "$mode" == env || "$mode" == nsys ]]; then optional_log env_nsys_version.txt nsys --version; fi
  if [[ "$mode" == sanitize ]]; then optional_log env_sanitizer_version.txt compute-sanitizer --version; fi
  for id in "${ids[@]}"; do
    optional_log "${id}.help.txt" "${binary[$id]}" --help
    optional_log "${id}.sha256.txt" sha256sum "${binary[$id]}"
  done
}
[[ "$mode" == disasm ]] || collect_environment

run_check() {
  local -A checked=()
  # 快速抽样检查覆盖：K 短于 PIPE、恰好 PIPE、环形缓冲回绕和排空。
  # 主程序只抽样 16 个输出；完整校验另见 run_validate，不将这里当作全矩阵验证。
  for id in "${ids[@]}"; do
    for layout in "${layout_list[@]}"; do
      for tiles in 1 2 "${pipes[$id]}" "$((pipes[$id]+1))" "$((2*pipes[$id]+1))"; do
        [[ -z "${checked[${id}_${layout}_${tiles}]+exists}" ]] || continue
        checked[${id}_${layout}_${tiles}]=1
        run_logged "check_${id}_${layout}_k${tiles}.log" "${binary[$id]}" \
          "$((2*cta_m[$id]))" "$((2*cta_n[$id]))" "$((tiles*cta_k[$id]))" \
          "${layout:0:1}" "${layout:1:1}" 3 event 1 1 2
      done
    done
  done
}
run_validate() {
  # 独立校验程序遍历 NT/TN、四类输入、短 K/PIPE 边界/多次回绕，每个 C 元素比 CPU。
  # 输入为可精确累加的小整数，避免把 FP16 累加舍入误判为线程同步错误。
  for id in "${ids[@]}"; do
    run_logged "validate_${id}.log" "${binary[$id]}.validate"
  done
}
run_bench() {
  start_telemetry
  if (( ! dry_run )); then printf 'id\tshape\tlayout\trepeat\tlog\n' > "$output_dir/bench_index.tsv"; fi
  for shape in "${shape_list[@]}"; do
    parse_shape "$shape"
    for layout in "${layout_list[@]}"; do
      for ((rep=1; rep<=repeats; ++rep)); do
        # 每轮循环移位测试顺序，降低固定先后顺序与温度/boost 漂移的相关性。
        # 仍须检查环境与 drift；轮换不能消除其他进程和功耗上限的干扰。
        for ((j=0; j<${#ids[@]}; ++j)); do
          id="${ids[$(((j+rep-1)%${#ids[@]}))]}"
          log="bench_${id}_${shape}_${layout}_r${rep}.log"
          run_logged "$log" "${binary[$id]}" "$m" "$n" "$k" "${layout:0:1}" "${layout:1:1}" \
            2 event "$per_batch" "$min_warmup" "$batches"
          if (( ! dry_run )); then printf '%s\t%s\t%s\t%s\t%s\n' "$id" "$shape" "$layout" "$rep" "$log" >> "$output_dir/bench_index.tsv"; fi
        done
      done
    done
  done
  if (( ! dry_run )) && [[ -n "$telemetry_pid" ]] && ! kill -0 "$telemetry_pid" 2>/dev/null; then
    printf 'Warning: GPU 状态采样已停止，请查看 telemetry.stderr.txt\n' >&2
  fi
  stop_telemetry
  (( ! dry_run )) || return 0
  # 只解析 bench_index 指向的普通计时日志，绝不读取 NCU/Sanitizer 日志里的性能数字。
  # p50/p99 是批均延迟百分位；跨进程取 p50 的中位数，不把多个进程的 p99 再合成 p99。
  python3 - "$output_dir" "$baseline" <<'PY'
import csv, math, pathlib, re, statistics, sys
root, baseline = pathlib.Path(sys.argv[1]), sys.argv[2]
with (root / 'manifest.tsv').open() as f:
    configs = {row['id']: row for row in csv.DictReader(f, delimiter='\t')}
fields = ['p50_ms', 'p99_ms', 'TFLOPS_p50', 'warmup_launches', 'warmup_stable', 'batch_drift_pct']
rows, groups = [], {}
with (root / 'bench_index.tsv').open() as f:
    index = list(csv.DictReader(f, delimiter='\t'))
for row in index:
    content = (root / row['log']).read_text()
    for key in fields:
        match = re.search(r'\b' + key + r'=([^\s,]+)', content)
        if not match:
            raise SystemExit(f"Missing {key} in {row['log']}")
        row[key] = match[1]
    for key in ('p50_ms', 'p99_ms', 'TFLOPS_p50', 'batch_drift_pct'):
        number = float(row[key])
        if not math.isfinite(number) or (key != 'batch_drift_pct' and number <= 0):
            raise SystemExit(f"Invalid {key} in {row['log']}")
    row['warning'] = ';'.join(filter(None, [
        'warmup_not_stable' if row['warmup_stable'] != 'yes' else '',
        'drift_over_5pct' if abs(float(row['batch_drift_pct'])) > 5 else '']))
    rows.append(row)
    groups.setdefault((row['id'], row['shape'], row['layout']), []).append(row)
with (root / 'runs.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
summaries = []
for (ident, shape, layout), items in groups.items():
    times = [float(x['p50_ms']) for x in items]
    median = statistics.median(times)
    m, n, k = map(int, shape.split('x'))
    config = configs[ident]
    reference = groups.get((baseline, shape, layout))
    # 相对列只是同问题耗时比；不同 CTA/PIPE 的 sweep 不能解释为单因素的版本收益。
    ratio = statistics.median(float(x['p50_ms']) for x in reference) / median if reference else ''
    summaries.append(dict(id=ident, shape=shape, layout=layout, repeats=len(items),
        cta=f"{config['cta_m']}x{config['cta_n']}x{config['cta_k']}", pipe=config['pipe'],
        compute_threads=config['compute_threads'], producer_threads=config['producer_threads'],
        median_p50_ms=median, min_p50_ms=min(times), max_p50_ms=max(times),
        tflops_from_median_p50=2*m*n*k/(median*1e9), baseline=baseline,
        speedup_vs_baseline=ratio, flagged_runs=sum(bool(x['warning']) for x in items)))
with (root / 'summary.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
print(f"Saved {len(rows)} runs and {len(summaries)} groups to {root}")
PY
}
run_ncu() {
  parse_shape "$profile_shape"
  local -a opts=(--kernel-name-base function --kernel-name 'regex:gemm_device'
    --launch-skip "$launch_skip" --launch-count 1 --clock-control "$ncu_clock" --check-exit-code yes)
  if [[ "$ncu_cache" == cold ]]; then
    # 隔离单个 kernel 的可复现分析；每个 replay pass 前清 cache，区别于长期热缓存计时。
    opts+=(--replay-mode kernel --cache-control all)
  else
    # 热缓存对照重启整个应用，重建目标前的执行历史，代价显著高于 kernel replay。
    # kernel replay + cache-control none 不能保证每一 pass 都有相同的 cache 状态。
    opts+=(--replay-mode application --cache-control none)
  fi
  if [[ "$ncu_set" == full ]]; then opts+=(--set full); else
    for section in LaunchStats Occupancy SpeedOfLight ComputeWorkloadAnalysis MemoryWorkloadAnalysis SchedulerStats WarpStateStats InstructionStats SourceCounters; do
      opts+=(--section "$section")
    done
  fi
  [[ -z "$metrics" ]] || opts+=(--metrics "$metrics")
  for id in "${ids[@]}"; do
    for layout in "${layout_list[@]}"; do
      local stem="ncu_${id}_${profile_shape}_${layout}_${ncu_cache}"
      # 启动 1 是测量前校验；skip=50 时抓取第 51 个 GEMM，即第 50 次预热。
      # min_warmup=skip+100 保证目标位于自适应停止判定之前；Thrust 初始化不匹配过滤器。
      # --launch-skip 只计数匹配 kernel；不能替换为 --launch-skip-before-match。
      run_logged "$stem.log" "$ncu_bin" "${opts[@]}" --export "$output_dir/$stem" \
        "${binary[$id]}" "$m" "$n" "$k" "${layout:0:1}" "${layout:1:1}" \
        2 event 1 "$((launch_skip+100))" 2
      if (( ! dry_run )); then
        [[ -s "$output_dir/$stem.ncu-rep" ]] || die "NCU 未生成报告，请检查 kernel 过滤器/计数器权限: $stem.log"
      fi
      # CSV 单独从报告导出，避免应用输出与 ==PROF== 状态行污染 CSV。
      run_logged "$stem.raw.csv" "$ncu_bin" --import "$output_dir/$stem.ncu-rep" --page raw --csv --print-units base
      run_logged "$stem.details.txt" "$ncu_bin" --import "$output_dir/$stem.ncu-rep" --page details
    done
  done
}
run_sanitize() {
  for id in "${ids[@]}"; do
    for layout in "${layout_list[@]}"; do
      for sanitizer in "${sanitizer_list[@]}"; do
        # 小矩阵、多次槽位回绕；用足够新的工具识别 Hopper TMA/WGMMA 同步语义。
        run_logged "sanitize_${id}_${layout}_${sanitizer}.log" compute-sanitizer \
          --tool "$sanitizer" --error-exitcode 99 "${binary[$id]}" \
          "$((2*cta_m[$id]))" "$((2*cta_n[$id]))" "$(((2*pipes[$id]+1)*cta_k[$id]))" \
          "${layout:0:1}" "${layout:1:1}" 3 event 1 1 2
      done
    done
  done
}
run_nsys() {
  parse_shape "$profile_shape"
  for id in "${ids[@]}"; do
    for layout in "${layout_list[@]}"; do
      local stem="nsys_${id}_${profile_shape}_${layout}"
      # Event 批计时仍可能包含 GPU 等待 CPU 提交的空隙，小 GEMM 尤其明显。
      # 采集短运行的 CUDA API/GPU/OS runtime 时间线，用于定位空隙；不开 CPU 栈采样。
      # 包含初始化/校验/预热，因此 GUI 中须定位稳定的连续 GEMM 区间，不看全程序平均。
      run_logged "$stem.log" nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none \
        --stats=true --output "$output_dir/$stem" "${binary[$id]}" \
        "$m" "$n" "$k" "${layout:0:1}" "${layout:1:1}" 2 event 10 100 20
      if (( ! dry_run )); then
        [[ -s "$output_dir/$stem.nsys-rep" ]] || die "Nsight Systems 未生成报告: $stem.log"
      fi
    done
  done
}
run_disasm() {
  optional_log env_cuobjdump_version.txt cuobjdump --version
  for id in "${ids[@]}"; do
    # ptxas 的寄存器/spill 编译报告在 build 目录；这里保留最终二进制的资源和指令证据。
    run_logged "${id}.resources.txt" cuobjdump --dump-resource-usage "${binary[$id]}"
    run_logged "${id}.sass.txt" cuobjdump --dump-sass "${binary[$id]}"
    # 仅 sm_90a cubin 时可能没有嵌入 PTX；空 PTX 不代表没有生成 WGMMA。
    run_logged "${id}.ptx.txt" cuobjdump --dump-ptx "${binary[$id]}"
  done
}

case "$mode" in
  env) ;;
  check) run_check ;;
  validate) run_validate ;;
  bench) run_bench ;;
  ncu) run_ncu ;;
  nsys) run_nsys ;;
  sanitize) run_sanitize ;;
  disasm) run_disasm ;;
  all) run_validate; run_bench; run_ncu; run_disasm ;;
esac
if [[ "$mode" != disasm ]]; then optional_log env_gpu_after.txt nvidia-smi -i "$gpu" -q; fi
printf '完成 %s；%s：%s\n' "$mode" "$([[ "$dry_run" == 1 ]] && printf '计划结果目录' || printf '结果目录')" "$output_dir"
