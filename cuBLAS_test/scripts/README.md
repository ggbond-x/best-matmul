# 测试计划总览

本目录按**测试计划**组织：`src/` 与顶层 CMake 是跨计划复用的公共测试逻辑，`scripts/planN/` 只放该计划的配置和执行入口，结果统一写入 `output/planN/<timestamped_run>/`。

## 计划索引

| Plan ID | 名称 | 测试目标 | 主要改动 | 执行入口 | 输出 |
|---|---|---|---|---|---|
| `plan1` | H100 PCIe FP16 GEMM 公平基线 | S01–S12 × TN/NT 上比较 CUSTOM_V1/V2、cuBLAS、cuBLASLt、CUTLASS SM90；建立可复现的 warm/cold 基线与 NCU 诊断 | 首个完整基线；统一 FP64 参考、物理打包、累加精度分组 | `scripts/plan1/build.sh`、`run_quick.sh`、`run_full.sh`、`run_ncu.sh` | `output/plan1/<timestamped_run>/` |

新增计划时应复制的是**配置与入口**，不是 `src/`：在 `scripts/planN/` 下提供自己的 `plan.yaml`、运行脚本与汇总脚本，复用 `scripts/common/`。

## 共享工具（`scripts/common/`）

| 文件 | 作用 |
|---|---|
| `plan_utils.py` | 解析 `plan.yaml`（PyYAML 或内置受限解析器）、应用 `--set` 覆盖、计算 SHA-256 `config_hash`、创建唯一运行目录、写出 `resolved_plan.yaml`、生成 shell 变量、manifest 收尾 |
| `env_snapshot.py` | 采集 GPU 名称/UUID/SM 数/CC/驱动、MIG、`nvidia-smi` 详情、CUDA/NVCC/CMake/NCU 版本、cuBLAS/CUTLASS 版本、Git HEAD 与 dirty、主机与时区；支持 before/after 合并 |
| `gpu_monitor.py` | 以固定间隔把 SM/显存时钟、功耗、温度、利用率、P-state、throttle 原因写入 `clock_power.csv`；SIGTERM/SIGINT 时安全退出 |
| `bench_run.sh` | quick/full 共用编排：创建运行目录 → 环境快照 → 干扰进程检查 → GPU 监控 → `gemm_bench` → 汇总 → manifest 收尾；中断时保留已有数据，不锁频、不杀进程 |

## 运行与结果约定

1. **计划 ↔ 结果一一对应**：`scripts/planN/` 只写 `output/planN/`，不跨计划读写。
2. **不覆盖历史**：quick/full/NCU 默认生成新的 `<timestamped_run>`（`YYYYMMDD_HHMMSS_<6hex>`）；`run_ncu.sh --run-dir` 只向指定目录追加 `reports/` 与 `csvs/ncu_summary.csv`。
3. **配置可追溯**：`plan.yaml` 是源配置；每次运行保存 `resolved_plan.yaml`（含 CLI 覆盖），并计算 `config_hash` 写进每条原始记录。
4. **唯一原始数据**：`csvs/raw_runs.csv`；`summary.csv` 与 `analysis.md` 只能来自该文件的真实记录。`ncu_summary.csv` 与 benchmark 分开。
5. **状态明确**：运行级 `RUNNING/COMPLETED/PARTIAL/FAILED`；Case 级 `OK/FAIL/UNSUPPORTED/NOT_RUN`。部分失败保留原因与已有数据。
6. **环境安全**：默认不改变 GPU 时钟/功耗设置；如手工锁定，需通过 `GEMM_CLOCK_LOCKED`/`GEMM_CLOCK_TARGET_MHZ` 如实记录，并在测试后自行恢复。

## 当前验证状态

- `plan1` 已在 RTX 3050 Laptop（SM86，CUDA 12.6，CUTLASS 4.4.1）完成构建、cuBLAS/cuBLASLt 正确性、warm/cold 计时、遥测、汇总的端到端验证。
- 自研 Kernel 与 CUTLASS SM90 在 SM86 上按设计返回 `UNSUPPORTED`；**尚无 H100 实测数据**。
