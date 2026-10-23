# 命令审计清单；请使用 run.sh 重跑实验，不要整体执行此文件。
# 包含持续采样命令；日志重定向及后台进程回收由 run.sh 管理。
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-89384b53-c02b-0f66-c2ab-b966923d789f
# env_date.txt
timeout --kill-after=30s 1800s date --iso-8601=seconds 
# env_uname.txt
timeout --kill-after=30s 1800s uname -a 
# env_cpu.txt
timeout --kill-after=30s 1800s lscpu 
# env_gpu_list.txt
timeout --kill-after=30s 1800s nvidia-smi -L 
# env_gpu_before.txt
timeout --kill-after=30s 1800s nvidia-smi -i 0 -q 
# env_topology.txt
timeout --kill-after=30s 1800s nvidia-smi topo -m 
# env_processes.txt
timeout --kill-after=30s 1800s nvidia-smi --query-compute-apps=gpu_uuid\,pid\,process_name\,used_gpu_memory --format=csv 
# v1.help.txt
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 --help 
# v1.sha256.txt
timeout --kill-after=30s 1800s sha256sum /root/scg/best-matmul/build/h100_pcie/v1 
# v2.help.txt
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 --help 
# v2.sha256.txt
timeout --kill-after=30s 1800s sha256sum /root/scg/best-matmul/build/h100_pcie/v2 
# v3.help.txt
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 --help 
# v3.sha256.txt
timeout --kill-after=30s 1800s sha256sum /root/scg/best-matmul/build/h100_pcie/v3 
# v4.help.txt
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 --help 
# v4.sha256.txt
timeout --kill-after=30s 1800s sha256sum /root/scg/best-matmul/build/h100_pcie/v4 
# v5.help.txt
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 --help 
# v5.sha256.txt
timeout --kill-after=30s 1800s sha256sum /root/scg/best-matmul/build/h100_pcie/v5 
# v3_p2.help.txt
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3_p2 --help 
# v3_p2.sha256.txt
timeout --kill-after=30s 1800s sha256sum /root/scg/best-matmul/build/h100_pcie/v3_p2 
# v3_p4.help.txt
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3_p4 --help 
# v3_p4.sha256.txt
timeout --kill-after=30s 1800s sha256sum /root/scg/best-matmul/build/h100_pcie/v3_p4 
# v4_p2.help.txt
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_p2 --help 
# v4_p2.sha256.txt
timeout --kill-after=30s 1800s sha256sum /root/scg/best-matmul/build/h100_pcie/v4_p2 
# v4_p4.help.txt
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_p4 --help 
# v4_p4.sha256.txt
timeout --kill-after=30s 1800s sha256sum /root/scg/best-matmul/build/h100_pcie/v4_p4 
# v5_p2.help.txt
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_p2 --help 
# v5_p2.sha256.txt
timeout --kill-after=30s 1800s sha256sum /root/scg/best-matmul/build/h100_pcie/v5_p2 
# v5_p4.help.txt
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_p4 --help 
# v5_p4.sha256.txt
timeout --kill-after=30s 1800s sha256sum /root/scg/best-matmul/build/h100_pcie/v5_p4 
# v4_m256_t128.help.txt
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 --help 
# v4_m256_t128.sha256.txt
timeout --kill-after=30s 1800s sha256sum /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 
# v5_m256_t128.help.txt
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 --help 
# v5_m256_t128.sha256.txt
timeout --kill-after=30s 1800s sha256sum /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 
# v5_m256_t256.help.txt
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 --help 
# v5_m256_t256.sha256.txt
timeout --kill-after=30s 1800s sha256sum /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 
# v5_m256_t512.help.txt
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 --help 
# v5_m256_t512.sha256.txt
timeout --kill-after=30s 1800s sha256sum /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 
# validate_v1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1.validate 
# validate_v2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2.validate 
# validate_v3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3.validate 
# validate_v4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4.validate 
# validate_v5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5.validate 
# validate_v3_p2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3_p2.validate 
# validate_v3_p4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3_p4.validate 
# validate_v4_p2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_p2.validate 
# validate_v4_p4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_p4.validate 
# validate_v5_p2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_p2.validate 
# validate_v5_p4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_p4.validate 
# validate_v4_m256_t128.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_m256_t128.validate 
# validate_v5_m256_t128.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t128.validate 
# validate_v5_m256_t256.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t256.validate 
# validate_v5_m256_t512.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t512.validate 
# env_gpu_after.txt
timeout --kill-after=30s 1800s nvidia-smi -i 0 -q 
