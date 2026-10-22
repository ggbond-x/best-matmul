# 本机验证记录

验证日期：2026-10-22（Asia/Shanghai）。

用户最初说明本机可以编译、没有目标测试环境；检查后发现 WSL 中可访问 RTX 3050 Laptop，因此进行了本地功能验证。它不是目标 Hopper/H100 平台，本机回归耗时不能当成 H100 的性能基线。

## 环境

| 项目 | 检查结果 |
|---|---|
| 系统 | Windows 上的 Ubuntu WSL |
| 默认 nvcc | /usr/bin/nvcc，CUDA 10.1；未用于本项目构建 |
| 实际 nvcc | /usr/local/cuda-12.6/bin/nvcc，12.6.85 |
| CMake / g++ | 3.16.3 / 9.4.0 |
| Python | 3.8.10 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| Compute capability | 8.6 |
| SM 数 | 16 |
| 显存 | 4294443008 bytes，约 4 GiB |
| CUDA runtime / driver API | 12060 / 12060 |
| cuBLAS | cublasGetVersion = 120604 |
| 动态库 | CUDA 12.6 的 libcublas.so.12.6.4.1、libcublasLt.so.12.6.4.1 |

## 已完成的检查

1. **实际编译与链接成功**：主构建 build/（SM80）与独立 build_hopper/（SM86、SM90）。
   build_hopper 的 nvcc 标志中确认存在 compute_86/sm_86 和 compute_90/sm_90。
   主机 C++ 编译启用 -Wall -Wextra -Wpedantic；最终构建没有输出警告。
2. **阶段性验证**：FP16 首先完成全量参考校验；随后 BF16、严格 FP32、TF32 的 TT、非方阵、负 alpha/非零 beta 校验通过；再验证 cuBLASLt 调优与 Graph。
3. **GPU 回归：92 PASS，0 FAIL，0 SKIP**，完整记录在 results/gpu_regression.csv：
   - 2 后端 × 4 dtype × 4 转置 × 2 beta = 64 个 case；
   - 2 后端 × 4 dtype × 2 beta 的 Graph = 16 个 case；
   - 低精度输入、FP32 输出、非零 beta = 4 个 case；
   - Lt index/tune × launch/graph = 4 个 case；
   - alpha=0、1×1×1、较大 K TF32、1024³ Lt 调优 = 4 个 case。
4. **批量脚本端到端：8 PASS，0 FAIL，0 SKIP**，results/batch_smoke.csv，包含 2 个形状、2 个后端、FP16/BF16；同时生成 .run.log 与带 UTC 时间戳的 .manifest.json。
5. **CTest：3/3 通过**：
   - cli_help；
   - compare_and_suite_tools（内部 9 个 unittest，含比较 CLI 的完整读写链路）；
   - cli_and_no_gpu（13 组非法参数，CUDA_VISIBLE_DEVICES 为空时正常写 SKIP 且无虚假时间）。
6. **显存/边界保护通过**：tests/gpu_edge_cases.py 检查超大矩阵、int32 极限尺寸引发的字节数溢出、1 TiB Workspace 预算，均在实际大分配前输出 SKIP；后续合法小矩阵正常 PASS。
7. **CSV 对比规则验证**：精度、输出类型、转置、leading dimension、GPU UUID、模式、重复次数、归约方式不一致不匹配；缺少语义字段报错；不同算法的重复结果不能混合；合成测试数据的 Ratio/Speedup 计算正确。

这些脚本中的合成 CSV 只用于软件测试，不是自定义 kernel 的真实性能数据。

## 可复查命令

在项目目录、已有构建下：

~~~bash
cmake --build build -j 4
(cd build && ctest --output-on-failure)
python3 tests/test_tools.py
python3 tests/gpu_smoke.py --exe build/gemm_bench --csv results/gpu_regression_rerun.csv
python3 tests/gpu_edge_cases.py --exe build/gemm_bench
python3 scripts/run_suite.py --cases tests/cases_smoke.json \
  --dtypes fp16 bf16 --csv results/batch_smoke_rerun.csv \
  --extra --warmup 2 --repeats 4 --inner 3
~~~

GPU 回归和批量输出使用新文件名，避免与已有记录重复。

## 尚未验证

- H100/Hopper 实机性能，以及 WGMMA/TMA 指令选择；编译 SM90 不能替代在 SM90 上运行。
- 8192 等完整大型集合的正式性能测量、持续运行的温度/频率稳定性。
- Windows 原生 Visual Studio 构建、CUDA 11.x 或其他 Toolkit 的实际兼容性。
- 全部 cuBLASLt 算法和全部 Workspace 组合；当前是限定 heuristic 候选的实测选择。
- 任意用户输入分布、极端 alpha/beta 的溢出/次正规数数值行为。
- 用户自定义 MatMul 的真实性能对比，因为尚未提供该算子的 CSV。
- Nsight Compute 对实际 Tensor Core/TF32 指令的确认。

本机部分短 kernel 的 launch 样本有明显调度波动。结果保留原值，不筛掉慢样本；正式报告应在目标 GPU 上使用一致协议重新采集。Graph 和 launch 已分别标记，比较工具不会跨模式计算速度比。

