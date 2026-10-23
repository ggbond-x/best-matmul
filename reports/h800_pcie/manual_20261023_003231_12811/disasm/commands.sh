# 命令审计清单；请使用 run.sh 重跑实验，不要整体执行此文件。
# 包含持续采样命令；日志重定向及后台进程回收由 run.sh 管理。
# env_cuobjdump_version.txt
timeout --kill-after=30s 1800s cuobjdump --version 
# v1.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /root/scg/best-matmul/build/h100_pcie/v1 
# v1.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /root/scg/best-matmul/build/h100_pcie/v1 
# v1.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /root/scg/best-matmul/build/h100_pcie/v1 
# v2.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /root/scg/best-matmul/build/h100_pcie/v2 
# v2.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /root/scg/best-matmul/build/h100_pcie/v2 
# v2.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /root/scg/best-matmul/build/h100_pcie/v2 
# v3.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /root/scg/best-matmul/build/h100_pcie/v3 
# v3.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /root/scg/best-matmul/build/h100_pcie/v3 
# v3.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /root/scg/best-matmul/build/h100_pcie/v3 
# v4.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /root/scg/best-matmul/build/h100_pcie/v4 
# v4.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /root/scg/best-matmul/build/h100_pcie/v4 
# v4.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /root/scg/best-matmul/build/h100_pcie/v4 
# v5.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /root/scg/best-matmul/build/h100_pcie/v5 
# v5.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /root/scg/best-matmul/build/h100_pcie/v5 
# v5.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /root/scg/best-matmul/build/h100_pcie/v5 
# v3_p2.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /root/scg/best-matmul/build/h100_pcie/v3_p2 
# v3_p2.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /root/scg/best-matmul/build/h100_pcie/v3_p2 
# v3_p2.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /root/scg/best-matmul/build/h100_pcie/v3_p2 
# v3_p4.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /root/scg/best-matmul/build/h100_pcie/v3_p4 
# v3_p4.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /root/scg/best-matmul/build/h100_pcie/v3_p4 
# v3_p4.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /root/scg/best-matmul/build/h100_pcie/v3_p4 
# v4_p2.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /root/scg/best-matmul/build/h100_pcie/v4_p2 
# v4_p2.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /root/scg/best-matmul/build/h100_pcie/v4_p2 
# v4_p2.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /root/scg/best-matmul/build/h100_pcie/v4_p2 
# v4_p4.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /root/scg/best-matmul/build/h100_pcie/v4_p4 
# v4_p4.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /root/scg/best-matmul/build/h100_pcie/v4_p4 
# v4_p4.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /root/scg/best-matmul/build/h100_pcie/v4_p4 
# v5_p2.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /root/scg/best-matmul/build/h100_pcie/v5_p2 
# v5_p2.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /root/scg/best-matmul/build/h100_pcie/v5_p2 
# v5_p2.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /root/scg/best-matmul/build/h100_pcie/v5_p2 
# v5_p4.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /root/scg/best-matmul/build/h100_pcie/v5_p4 
# v5_p4.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /root/scg/best-matmul/build/h100_pcie/v5_p4 
# v5_p4.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /root/scg/best-matmul/build/h100_pcie/v5_p4 
# v4_m256_t128.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 
# v4_m256_t128.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 
# v4_m256_t128.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 
# v5_m256_t128.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 
# v5_m256_t128.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 
# v5_m256_t128.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 
# v5_m256_t256.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 
# v5_m256_t256.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 
# v5_m256_t256.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 
# v5_m256_t512.resources.txt
timeout --kill-after=30s 1800s cuobjdump --dump-resource-usage /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 
# v5_m256_t512.sass.txt
timeout --kill-after=30s 1800s cuobjdump --dump-sass /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 
# v5_m256_t512.ptx.txt
timeout --kill-after=30s 1800s cuobjdump --dump-ptx /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 
