# 命令审计清单；请使用 run.sh 重跑实验，不要整体执行此文件。
# 包含持续采样命令；日志重定向及后台进程回收由 run.sh 管理。
# env_cuobjdump_version.txt
timeout --kill-after=30s 1800s cuobjdump --version 
# v1.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /workspace/peak_flops/src/sm90/nsight_compute/matmul/modular/build/h100_pcie/v1 
# v1.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /workspace/peak_flops/src/sm90/nsight_compute/matmul/modular/build/h100_pcie/v1 
# v1.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /workspace/peak_flops/src/sm90/nsight_compute/matmul/modular/build/h100_pcie/v1 
# v2.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /workspace/peak_flops/src/sm90/nsight_compute/matmul/modular/build/h100_pcie/v2 
# v2.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /workspace/peak_flops/src/sm90/nsight_compute/matmul/modular/build/h100_pcie/v2 
# v2.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /workspace/peak_flops/src/sm90/nsight_compute/matmul/modular/build/h100_pcie/v2 
# v3.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /workspace/peak_flops/src/sm90/nsight_compute/matmul/modular/build/h100_pcie/v3 
# v3.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /workspace/peak_flops/src/sm90/nsight_compute/matmul/modular/build/h100_pcie/v3 
# v3.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /workspace/peak_flops/src/sm90/nsight_compute/matmul/modular/build/h100_pcie/v3 
# v4.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /workspace/peak_flops/src/sm90/nsight_compute/matmul/modular/build/h100_pcie/v4 
# v4.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /workspace/peak_flops/src/sm90/nsight_compute/matmul/modular/build/h100_pcie/v4 
# v4.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /workspace/peak_flops/src/sm90/nsight_compute/matmul/modular/build/h100_pcie/v4 
# v5.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /workspace/peak_flops/src/sm90/nsight_compute/matmul/modular/build/h100_pcie/v5 
# v5.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /workspace/peak_flops/src/sm90/nsight_compute/matmul/modular/build/h100_pcie/v5 
# v5.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /workspace/peak_flops/src/sm90/nsight_compute/matmul/modular/build/h100_pcie/v5 
# v5_m256_t512.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /workspace/peak_flops/src/sm90/nsight_compute/matmul/modular/build/h100_pcie/v5_m256_t512 
# v5_m256_t512.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /workspace/peak_flops/src/sm90/nsight_compute/matmul/modular/build/h100_pcie/v5_m256_t512 
# v5_m256_t512.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /workspace/peak_flops/src/sm90/nsight_compute/matmul/modular/build/h100_pcie/v5_m256_t512 
