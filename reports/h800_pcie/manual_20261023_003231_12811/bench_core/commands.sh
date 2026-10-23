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
# telemetry.csv
nvidia-smi -i 0 --query-gpu=timestamp\,pstate\,temperature.gpu\,power.draw\,power.limit\,clocks.sm\,clocks.mem\,utilization.gpu\,utilization.memory\,memory.used --format=csv --loop-ms=1000 
# bench_v1_1024x1024x1024_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 1024 1024 N T 2 event 20 300 100 
# bench_v2_1024x1024x1024_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 1024 1024 N T 2 event 20 300 100 
# bench_v3_1024x1024x1024_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 1024 1024 N T 2 event 20 300 100 
# bench_v4_1024x1024x1024_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 1024 1024 N T 2 event 20 300 100 
# bench_v5_1024x1024x1024_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 1024 1024 N T 2 event 20 300 100 
# bench_v2_1024x1024x1024_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 1024 1024 N T 2 event 20 300 100 
# bench_v3_1024x1024x1024_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 1024 1024 N T 2 event 20 300 100 
# bench_v4_1024x1024x1024_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 1024 1024 N T 2 event 20 300 100 
# bench_v5_1024x1024x1024_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 1024 1024 N T 2 event 20 300 100 
# bench_v1_1024x1024x1024_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 1024 1024 N T 2 event 20 300 100 
# bench_v3_1024x1024x1024_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 1024 1024 N T 2 event 20 300 100 
# bench_v4_1024x1024x1024_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 1024 1024 N T 2 event 20 300 100 
# bench_v5_1024x1024x1024_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 1024 1024 N T 2 event 20 300 100 
# bench_v1_1024x1024x1024_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 1024 1024 N T 2 event 20 300 100 
# bench_v2_1024x1024x1024_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 1024 1024 N T 2 event 20 300 100 
# bench_v4_1024x1024x1024_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 1024 1024 N T 2 event 20 300 100 
# bench_v5_1024x1024x1024_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 1024 1024 N T 2 event 20 300 100 
# bench_v1_1024x1024x1024_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 1024 1024 N T 2 event 20 300 100 
# bench_v2_1024x1024x1024_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 1024 1024 N T 2 event 20 300 100 
# bench_v3_1024x1024x1024_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 1024 1024 N T 2 event 20 300 100 
# bench_v5_1024x1024x1024_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 1024 1024 N T 2 event 20 300 100 
# bench_v1_1024x1024x1024_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 1024 1024 N T 2 event 20 300 100 
# bench_v2_1024x1024x1024_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 1024 1024 N T 2 event 20 300 100 
# bench_v3_1024x1024x1024_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 1024 1024 N T 2 event 20 300 100 
# bench_v4_1024x1024x1024_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 1024 1024 N T 2 event 20 300 100 
# bench_v1_1024x1024x1024_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 1024 1024 T N 2 event 20 300 100 
# bench_v2_1024x1024x1024_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 1024 1024 T N 2 event 20 300 100 
# bench_v3_1024x1024x1024_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 1024 1024 T N 2 event 20 300 100 
# bench_v4_1024x1024x1024_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 1024 1024 T N 2 event 20 300 100 
# bench_v5_1024x1024x1024_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 1024 1024 T N 2 event 20 300 100 
# bench_v2_1024x1024x1024_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 1024 1024 T N 2 event 20 300 100 
# bench_v3_1024x1024x1024_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 1024 1024 T N 2 event 20 300 100 
# bench_v4_1024x1024x1024_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 1024 1024 T N 2 event 20 300 100 
# bench_v5_1024x1024x1024_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 1024 1024 T N 2 event 20 300 100 
# bench_v1_1024x1024x1024_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 1024 1024 T N 2 event 20 300 100 
# bench_v3_1024x1024x1024_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 1024 1024 T N 2 event 20 300 100 
# bench_v4_1024x1024x1024_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 1024 1024 T N 2 event 20 300 100 
# bench_v5_1024x1024x1024_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 1024 1024 T N 2 event 20 300 100 
# bench_v1_1024x1024x1024_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 1024 1024 T N 2 event 20 300 100 
# bench_v2_1024x1024x1024_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 1024 1024 T N 2 event 20 300 100 
# bench_v4_1024x1024x1024_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 1024 1024 T N 2 event 20 300 100 
# bench_v5_1024x1024x1024_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 1024 1024 T N 2 event 20 300 100 
# bench_v1_1024x1024x1024_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 1024 1024 T N 2 event 20 300 100 
# bench_v2_1024x1024x1024_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 1024 1024 T N 2 event 20 300 100 
# bench_v3_1024x1024x1024_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 1024 1024 T N 2 event 20 300 100 
# bench_v5_1024x1024x1024_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 1024 1024 T N 2 event 20 300 100 
# bench_v1_1024x1024x1024_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 1024 1024 T N 2 event 20 300 100 
# bench_v2_1024x1024x1024_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 1024 1024 T N 2 event 20 300 100 
# bench_v3_1024x1024x1024_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 1024 1024 T N 2 event 20 300 100 
# bench_v4_1024x1024x1024_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 1024 1024 T N 2 event 20 300 100 
# bench_v1_4096x4096x4096_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 4096 N T 2 event 20 300 100 
# bench_v2_4096x4096x4096_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 4096 N T 2 event 20 300 100 
# bench_v3_4096x4096x4096_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 4096 N T 2 event 20 300 100 
# bench_v4_4096x4096x4096_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 4096 N T 2 event 20 300 100 
# bench_v5_4096x4096x4096_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 4096 N T 2 event 20 300 100 
# bench_v2_4096x4096x4096_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 4096 N T 2 event 20 300 100 
# bench_v3_4096x4096x4096_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 4096 N T 2 event 20 300 100 
# bench_v4_4096x4096x4096_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 4096 N T 2 event 20 300 100 
# bench_v5_4096x4096x4096_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 4096 N T 2 event 20 300 100 
# bench_v1_4096x4096x4096_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 4096 N T 2 event 20 300 100 
# bench_v3_4096x4096x4096_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 4096 N T 2 event 20 300 100 
# bench_v4_4096x4096x4096_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 4096 N T 2 event 20 300 100 
# bench_v5_4096x4096x4096_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 4096 N T 2 event 20 300 100 
# bench_v1_4096x4096x4096_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 4096 N T 2 event 20 300 100 
# bench_v2_4096x4096x4096_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 4096 N T 2 event 20 300 100 
# bench_v4_4096x4096x4096_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 4096 N T 2 event 20 300 100 
# bench_v5_4096x4096x4096_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 4096 N T 2 event 20 300 100 
# bench_v1_4096x4096x4096_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 4096 N T 2 event 20 300 100 
# bench_v2_4096x4096x4096_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 4096 N T 2 event 20 300 100 
# bench_v3_4096x4096x4096_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 4096 N T 2 event 20 300 100 
# bench_v5_4096x4096x4096_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 4096 N T 2 event 20 300 100 
# bench_v1_4096x4096x4096_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 4096 N T 2 event 20 300 100 
# bench_v2_4096x4096x4096_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 4096 N T 2 event 20 300 100 
# bench_v3_4096x4096x4096_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 4096 N T 2 event 20 300 100 
# bench_v4_4096x4096x4096_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 4096 N T 2 event 20 300 100 
# bench_v1_4096x4096x4096_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 4096 T N 2 event 20 300 100 
# bench_v2_4096x4096x4096_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 4096 T N 2 event 20 300 100 
# bench_v3_4096x4096x4096_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 4096 T N 2 event 20 300 100 
# bench_v4_4096x4096x4096_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 4096 T N 2 event 20 300 100 
# bench_v5_4096x4096x4096_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 4096 T N 2 event 20 300 100 
# bench_v2_4096x4096x4096_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 4096 T N 2 event 20 300 100 
# bench_v3_4096x4096x4096_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 4096 T N 2 event 20 300 100 
# bench_v4_4096x4096x4096_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 4096 T N 2 event 20 300 100 
# bench_v5_4096x4096x4096_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 4096 T N 2 event 20 300 100 
# bench_v1_4096x4096x4096_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 4096 T N 2 event 20 300 100 
# bench_v3_4096x4096x4096_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 4096 T N 2 event 20 300 100 
# bench_v4_4096x4096x4096_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 4096 T N 2 event 20 300 100 
# bench_v5_4096x4096x4096_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 4096 T N 2 event 20 300 100 
# bench_v1_4096x4096x4096_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 4096 T N 2 event 20 300 100 
# bench_v2_4096x4096x4096_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 4096 T N 2 event 20 300 100 
# bench_v4_4096x4096x4096_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 4096 T N 2 event 20 300 100 
# bench_v5_4096x4096x4096_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 4096 T N 2 event 20 300 100 
# bench_v1_4096x4096x4096_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 4096 T N 2 event 20 300 100 
# bench_v2_4096x4096x4096_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 4096 T N 2 event 20 300 100 
# bench_v3_4096x4096x4096_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 4096 T N 2 event 20 300 100 
# bench_v5_4096x4096x4096_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 4096 T N 2 event 20 300 100 
# bench_v1_4096x4096x4096_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 4096 T N 2 event 20 300 100 
# bench_v2_4096x4096x4096_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 4096 T N 2 event 20 300 100 
# bench_v3_4096x4096x4096_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 4096 T N 2 event 20 300 100 
# bench_v4_4096x4096x4096_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 4096 T N 2 event 20 300 100 
# bench_v1_8192x8192x8192_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 8192 8192 8192 N T 2 event 20 300 100 
# bench_v2_8192x8192x8192_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 8192 8192 8192 N T 2 event 20 300 100 
# bench_v3_8192x8192x8192_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 8192 8192 8192 N T 2 event 20 300 100 
# bench_v4_8192x8192x8192_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 8192 8192 8192 N T 2 event 20 300 100 
# bench_v5_8192x8192x8192_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 8192 8192 8192 N T 2 event 20 300 100 
# bench_v2_8192x8192x8192_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 8192 8192 8192 N T 2 event 20 300 100 
# bench_v3_8192x8192x8192_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 8192 8192 8192 N T 2 event 20 300 100 
# bench_v4_8192x8192x8192_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 8192 8192 8192 N T 2 event 20 300 100 
# bench_v5_8192x8192x8192_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 8192 8192 8192 N T 2 event 20 300 100 
# bench_v1_8192x8192x8192_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 8192 8192 8192 N T 2 event 20 300 100 
# bench_v3_8192x8192x8192_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 8192 8192 8192 N T 2 event 20 300 100 
# bench_v4_8192x8192x8192_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 8192 8192 8192 N T 2 event 20 300 100 
# bench_v5_8192x8192x8192_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 8192 8192 8192 N T 2 event 20 300 100 
# bench_v1_8192x8192x8192_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 8192 8192 8192 N T 2 event 20 300 100 
# bench_v2_8192x8192x8192_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 8192 8192 8192 N T 2 event 20 300 100 
# bench_v4_8192x8192x8192_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 8192 8192 8192 N T 2 event 20 300 100 
# bench_v5_8192x8192x8192_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 8192 8192 8192 N T 2 event 20 300 100 
# bench_v1_8192x8192x8192_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 8192 8192 8192 N T 2 event 20 300 100 
# bench_v2_8192x8192x8192_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 8192 8192 8192 N T 2 event 20 300 100 
# bench_v3_8192x8192x8192_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 8192 8192 8192 N T 2 event 20 300 100 
# bench_v5_8192x8192x8192_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 8192 8192 8192 N T 2 event 20 300 100 
# bench_v1_8192x8192x8192_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 8192 8192 8192 N T 2 event 20 300 100 
# bench_v2_8192x8192x8192_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 8192 8192 8192 N T 2 event 20 300 100 
# bench_v3_8192x8192x8192_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 8192 8192 8192 N T 2 event 20 300 100 
# bench_v4_8192x8192x8192_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 8192 8192 8192 N T 2 event 20 300 100 
# bench_v1_8192x8192x8192_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 8192 8192 8192 T N 2 event 20 300 100 
# bench_v2_8192x8192x8192_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 8192 8192 8192 T N 2 event 20 300 100 
# bench_v3_8192x8192x8192_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 8192 8192 8192 T N 2 event 20 300 100 
# bench_v4_8192x8192x8192_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 8192 8192 8192 T N 2 event 20 300 100 
# bench_v5_8192x8192x8192_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 8192 8192 8192 T N 2 event 20 300 100 
# bench_v2_8192x8192x8192_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 8192 8192 8192 T N 2 event 20 300 100 
# bench_v3_8192x8192x8192_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 8192 8192 8192 T N 2 event 20 300 100 
# bench_v4_8192x8192x8192_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 8192 8192 8192 T N 2 event 20 300 100 
# bench_v5_8192x8192x8192_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 8192 8192 8192 T N 2 event 20 300 100 
# bench_v1_8192x8192x8192_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 8192 8192 8192 T N 2 event 20 300 100 
# bench_v3_8192x8192x8192_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 8192 8192 8192 T N 2 event 20 300 100 
# bench_v4_8192x8192x8192_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 8192 8192 8192 T N 2 event 20 300 100 
# bench_v5_8192x8192x8192_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 8192 8192 8192 T N 2 event 20 300 100 
# bench_v1_8192x8192x8192_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 8192 8192 8192 T N 2 event 20 300 100 
# bench_v2_8192x8192x8192_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 8192 8192 8192 T N 2 event 20 300 100 
# bench_v4_8192x8192x8192_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 8192 8192 8192 T N 2 event 20 300 100 
# bench_v5_8192x8192x8192_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 8192 8192 8192 T N 2 event 20 300 100 
# bench_v1_8192x8192x8192_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 8192 8192 8192 T N 2 event 20 300 100 
# bench_v2_8192x8192x8192_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 8192 8192 8192 T N 2 event 20 300 100 
# bench_v3_8192x8192x8192_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 8192 8192 8192 T N 2 event 20 300 100 
# bench_v5_8192x8192x8192_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 8192 8192 8192 T N 2 event 20 300 100 
# bench_v1_8192x8192x8192_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 8192 8192 8192 T N 2 event 20 300 100 
# bench_v2_8192x8192x8192_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 8192 8192 8192 T N 2 event 20 300 100 
# bench_v3_8192x8192x8192_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 8192 8192 8192 T N 2 event 20 300 100 
# bench_v4_8192x8192x8192_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 8192 8192 8192 T N 2 event 20 300 100 
# bench_v1_4096x4096x256_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 256 N T 2 event 20 300 100 
# bench_v2_4096x4096x256_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 256 N T 2 event 20 300 100 
# bench_v3_4096x4096x256_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 256 N T 2 event 20 300 100 
# bench_v4_4096x4096x256_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 256 N T 2 event 20 300 100 
# bench_v5_4096x4096x256_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 256 N T 2 event 20 300 100 
# bench_v2_4096x4096x256_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 256 N T 2 event 20 300 100 
# bench_v3_4096x4096x256_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 256 N T 2 event 20 300 100 
# bench_v4_4096x4096x256_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 256 N T 2 event 20 300 100 
# bench_v5_4096x4096x256_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 256 N T 2 event 20 300 100 
# bench_v1_4096x4096x256_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 256 N T 2 event 20 300 100 
# bench_v3_4096x4096x256_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 256 N T 2 event 20 300 100 
# bench_v4_4096x4096x256_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 256 N T 2 event 20 300 100 
# bench_v5_4096x4096x256_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 256 N T 2 event 20 300 100 
# bench_v1_4096x4096x256_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 256 N T 2 event 20 300 100 
# bench_v2_4096x4096x256_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 256 N T 2 event 20 300 100 
# bench_v4_4096x4096x256_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 256 N T 2 event 20 300 100 
# bench_v5_4096x4096x256_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 256 N T 2 event 20 300 100 
# bench_v1_4096x4096x256_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 256 N T 2 event 20 300 100 
# bench_v2_4096x4096x256_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 256 N T 2 event 20 300 100 
# bench_v3_4096x4096x256_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 256 N T 2 event 20 300 100 
# bench_v5_4096x4096x256_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 256 N T 2 event 20 300 100 
# bench_v1_4096x4096x256_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 256 N T 2 event 20 300 100 
# bench_v2_4096x4096x256_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 256 N T 2 event 20 300 100 
# bench_v3_4096x4096x256_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 256 N T 2 event 20 300 100 
# bench_v4_4096x4096x256_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 256 N T 2 event 20 300 100 
# bench_v1_4096x4096x256_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 256 T N 2 event 20 300 100 
# bench_v2_4096x4096x256_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 256 T N 2 event 20 300 100 
# bench_v3_4096x4096x256_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 256 T N 2 event 20 300 100 
# bench_v4_4096x4096x256_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 256 T N 2 event 20 300 100 
# bench_v5_4096x4096x256_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 256 T N 2 event 20 300 100 
# bench_v2_4096x4096x256_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 256 T N 2 event 20 300 100 
# bench_v3_4096x4096x256_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 256 T N 2 event 20 300 100 
# bench_v4_4096x4096x256_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 256 T N 2 event 20 300 100 
# bench_v5_4096x4096x256_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 256 T N 2 event 20 300 100 
# bench_v1_4096x4096x256_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 256 T N 2 event 20 300 100 
# bench_v3_4096x4096x256_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 256 T N 2 event 20 300 100 
# bench_v4_4096x4096x256_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 256 T N 2 event 20 300 100 
# bench_v5_4096x4096x256_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 256 T N 2 event 20 300 100 
# bench_v1_4096x4096x256_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 256 T N 2 event 20 300 100 
# bench_v2_4096x4096x256_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 256 T N 2 event 20 300 100 
# bench_v4_4096x4096x256_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 256 T N 2 event 20 300 100 
# bench_v5_4096x4096x256_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 256 T N 2 event 20 300 100 
# bench_v1_4096x4096x256_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 256 T N 2 event 20 300 100 
# bench_v2_4096x4096x256_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 256 T N 2 event 20 300 100 
# bench_v3_4096x4096x256_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 256 T N 2 event 20 300 100 
# bench_v5_4096x4096x256_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 4096 4096 256 T N 2 event 20 300 100 
# bench_v1_4096x4096x256_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 4096 4096 256 T N 2 event 20 300 100 
# bench_v2_4096x4096x256_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 4096 4096 256 T N 2 event 20 300 100 
# bench_v3_4096x4096x256_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 4096 4096 256 T N 2 event 20 300 100 
# bench_v4_4096x4096x256_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 4096 4096 256 T N 2 event 20 300 100 
# bench_v1_16384x1024x4096_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 16384 1024 4096 N T 2 event 20 300 100 
# bench_v2_16384x1024x4096_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 16384 1024 4096 N T 2 event 20 300 100 
# bench_v3_16384x1024x4096_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 16384 1024 4096 N T 2 event 20 300 100 
# bench_v4_16384x1024x4096_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 16384 1024 4096 N T 2 event 20 300 100 
# bench_v5_16384x1024x4096_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 16384 1024 4096 N T 2 event 20 300 100 
# bench_v2_16384x1024x4096_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 16384 1024 4096 N T 2 event 20 300 100 
# bench_v3_16384x1024x4096_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 16384 1024 4096 N T 2 event 20 300 100 
# bench_v4_16384x1024x4096_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 16384 1024 4096 N T 2 event 20 300 100 
# bench_v5_16384x1024x4096_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 16384 1024 4096 N T 2 event 20 300 100 
# bench_v1_16384x1024x4096_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 16384 1024 4096 N T 2 event 20 300 100 
# bench_v3_16384x1024x4096_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 16384 1024 4096 N T 2 event 20 300 100 
# bench_v4_16384x1024x4096_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 16384 1024 4096 N T 2 event 20 300 100 
# bench_v5_16384x1024x4096_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 16384 1024 4096 N T 2 event 20 300 100 
# bench_v1_16384x1024x4096_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 16384 1024 4096 N T 2 event 20 300 100 
# bench_v2_16384x1024x4096_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 16384 1024 4096 N T 2 event 20 300 100 
# bench_v4_16384x1024x4096_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 16384 1024 4096 N T 2 event 20 300 100 
# bench_v5_16384x1024x4096_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 16384 1024 4096 N T 2 event 20 300 100 
# bench_v1_16384x1024x4096_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 16384 1024 4096 N T 2 event 20 300 100 
# bench_v2_16384x1024x4096_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 16384 1024 4096 N T 2 event 20 300 100 
# bench_v3_16384x1024x4096_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 16384 1024 4096 N T 2 event 20 300 100 
# bench_v5_16384x1024x4096_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 16384 1024 4096 N T 2 event 20 300 100 
# bench_v1_16384x1024x4096_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 16384 1024 4096 N T 2 event 20 300 100 
# bench_v2_16384x1024x4096_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 16384 1024 4096 N T 2 event 20 300 100 
# bench_v3_16384x1024x4096_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 16384 1024 4096 N T 2 event 20 300 100 
# bench_v4_16384x1024x4096_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 16384 1024 4096 N T 2 event 20 300 100 
# bench_v1_16384x1024x4096_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 16384 1024 4096 T N 2 event 20 300 100 
# bench_v2_16384x1024x4096_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 16384 1024 4096 T N 2 event 20 300 100 
# bench_v3_16384x1024x4096_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 16384 1024 4096 T N 2 event 20 300 100 
# bench_v4_16384x1024x4096_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 16384 1024 4096 T N 2 event 20 300 100 
# bench_v5_16384x1024x4096_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 16384 1024 4096 T N 2 event 20 300 100 
# bench_v2_16384x1024x4096_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 16384 1024 4096 T N 2 event 20 300 100 
# bench_v3_16384x1024x4096_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 16384 1024 4096 T N 2 event 20 300 100 
# bench_v4_16384x1024x4096_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 16384 1024 4096 T N 2 event 20 300 100 
# bench_v5_16384x1024x4096_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 16384 1024 4096 T N 2 event 20 300 100 
# bench_v1_16384x1024x4096_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 16384 1024 4096 T N 2 event 20 300 100 
# bench_v3_16384x1024x4096_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 16384 1024 4096 T N 2 event 20 300 100 
# bench_v4_16384x1024x4096_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 16384 1024 4096 T N 2 event 20 300 100 
# bench_v5_16384x1024x4096_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 16384 1024 4096 T N 2 event 20 300 100 
# bench_v1_16384x1024x4096_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 16384 1024 4096 T N 2 event 20 300 100 
# bench_v2_16384x1024x4096_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 16384 1024 4096 T N 2 event 20 300 100 
# bench_v4_16384x1024x4096_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 16384 1024 4096 T N 2 event 20 300 100 
# bench_v5_16384x1024x4096_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 16384 1024 4096 T N 2 event 20 300 100 
# bench_v1_16384x1024x4096_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 16384 1024 4096 T N 2 event 20 300 100 
# bench_v2_16384x1024x4096_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 16384 1024 4096 T N 2 event 20 300 100 
# bench_v3_16384x1024x4096_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 16384 1024 4096 T N 2 event 20 300 100 
# bench_v5_16384x1024x4096_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 16384 1024 4096 T N 2 event 20 300 100 
# bench_v1_16384x1024x4096_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 16384 1024 4096 T N 2 event 20 300 100 
# bench_v2_16384x1024x4096_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 16384 1024 4096 T N 2 event 20 300 100 
# bench_v3_16384x1024x4096_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 16384 1024 4096 T N 2 event 20 300 100 
# bench_v4_16384x1024x4096_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 16384 1024 4096 T N 2 event 20 300 100 
# bench_v1_1024x16384x4096_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 16384 4096 N T 2 event 20 300 100 
# bench_v2_1024x16384x4096_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 16384 4096 N T 2 event 20 300 100 
# bench_v3_1024x16384x4096_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 16384 4096 N T 2 event 20 300 100 
# bench_v4_1024x16384x4096_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 16384 4096 N T 2 event 20 300 100 
# bench_v5_1024x16384x4096_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 16384 4096 N T 2 event 20 300 100 
# bench_v2_1024x16384x4096_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 16384 4096 N T 2 event 20 300 100 
# bench_v3_1024x16384x4096_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 16384 4096 N T 2 event 20 300 100 
# bench_v4_1024x16384x4096_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 16384 4096 N T 2 event 20 300 100 
# bench_v5_1024x16384x4096_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 16384 4096 N T 2 event 20 300 100 
# bench_v1_1024x16384x4096_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 16384 4096 N T 2 event 20 300 100 
# bench_v3_1024x16384x4096_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 16384 4096 N T 2 event 20 300 100 
# bench_v4_1024x16384x4096_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 16384 4096 N T 2 event 20 300 100 
# bench_v5_1024x16384x4096_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 16384 4096 N T 2 event 20 300 100 
# bench_v1_1024x16384x4096_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 16384 4096 N T 2 event 20 300 100 
# bench_v2_1024x16384x4096_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 16384 4096 N T 2 event 20 300 100 
# bench_v4_1024x16384x4096_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 16384 4096 N T 2 event 20 300 100 
# bench_v5_1024x16384x4096_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 16384 4096 N T 2 event 20 300 100 
# bench_v1_1024x16384x4096_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 16384 4096 N T 2 event 20 300 100 
# bench_v2_1024x16384x4096_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 16384 4096 N T 2 event 20 300 100 
# bench_v3_1024x16384x4096_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 16384 4096 N T 2 event 20 300 100 
# bench_v5_1024x16384x4096_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 16384 4096 N T 2 event 20 300 100 
# bench_v1_1024x16384x4096_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 16384 4096 N T 2 event 20 300 100 
# bench_v2_1024x16384x4096_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 16384 4096 N T 2 event 20 300 100 
# bench_v3_1024x16384x4096_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 16384 4096 N T 2 event 20 300 100 
# bench_v4_1024x16384x4096_NT_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 16384 4096 N T 2 event 20 300 100 
# bench_v1_1024x16384x4096_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 16384 4096 T N 2 event 20 300 100 
# bench_v2_1024x16384x4096_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 16384 4096 T N 2 event 20 300 100 
# bench_v3_1024x16384x4096_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 16384 4096 T N 2 event 20 300 100 
# bench_v4_1024x16384x4096_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 16384 4096 T N 2 event 20 300 100 
# bench_v5_1024x16384x4096_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 16384 4096 T N 2 event 20 300 100 
# bench_v2_1024x16384x4096_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 16384 4096 T N 2 event 20 300 100 
# bench_v3_1024x16384x4096_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 16384 4096 T N 2 event 20 300 100 
# bench_v4_1024x16384x4096_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 16384 4096 T N 2 event 20 300 100 
# bench_v5_1024x16384x4096_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 16384 4096 T N 2 event 20 300 100 
# bench_v1_1024x16384x4096_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 16384 4096 T N 2 event 20 300 100 
# bench_v3_1024x16384x4096_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 16384 4096 T N 2 event 20 300 100 
# bench_v4_1024x16384x4096_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 16384 4096 T N 2 event 20 300 100 
# bench_v5_1024x16384x4096_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 16384 4096 T N 2 event 20 300 100 
# bench_v1_1024x16384x4096_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 16384 4096 T N 2 event 20 300 100 
# bench_v2_1024x16384x4096_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 16384 4096 T N 2 event 20 300 100 
# bench_v4_1024x16384x4096_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 16384 4096 T N 2 event 20 300 100 
# bench_v5_1024x16384x4096_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 16384 4096 T N 2 event 20 300 100 
# bench_v1_1024x16384x4096_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 16384 4096 T N 2 event 20 300 100 
# bench_v2_1024x16384x4096_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 16384 4096 T N 2 event 20 300 100 
# bench_v3_1024x16384x4096_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 16384 4096 T N 2 event 20 300 100 
# bench_v5_1024x16384x4096_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5 1024 16384 4096 T N 2 event 20 300 100 
# bench_v1_1024x16384x4096_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v1 1024 16384 4096 T N 2 event 20 300 100 
# bench_v2_1024x16384x4096_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v2 1024 16384 4096 T N 2 event 20 300 100 
# bench_v3_1024x16384x4096_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v3 1024 16384 4096 T N 2 event 20 300 100 
# bench_v4_1024x16384x4096_TN_r5.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4 1024 16384 4096 T N 2 event 20 300 100 
# env_gpu_after.txt
timeout --kill-after=30s 1800s nvidia-smi -i 0 -q 
