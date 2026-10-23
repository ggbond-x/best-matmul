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
# telemetry.csv
nvidia-smi -i 0 --query-gpu=timestamp\,pstate\,temperature.gpu\,power.draw\,power.limit\,clocks.sm\,clocks.mem\,utilization.gpu\,utilization.memory\,memory.used --format=csv --loop-ms=1000 
# bench_v4_m256_t128_4096x4096x4096_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 4096 4096 4096 N T 2 event 20 300 100 
# bench_v5_m256_t128_4096x4096x4096_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 4096 4096 4096 N T 2 event 20 300 100 
# bench_v5_m256_t256_4096x4096x4096_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 4096 4096 4096 N T 2 event 20 300 100 
# bench_v5_m256_t512_4096x4096x4096_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 4096 4096 4096 N T 2 event 20 300 100 
# bench_v5_m256_t128_4096x4096x4096_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 4096 4096 4096 N T 2 event 20 300 100 
# bench_v5_m256_t256_4096x4096x4096_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 4096 4096 4096 N T 2 event 20 300 100 
# bench_v5_m256_t512_4096x4096x4096_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 4096 4096 4096 N T 2 event 20 300 100 
# bench_v4_m256_t128_4096x4096x4096_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 4096 4096 4096 N T 2 event 20 300 100 
# bench_v5_m256_t256_4096x4096x4096_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 4096 4096 4096 N T 2 event 20 300 100 
# bench_v5_m256_t512_4096x4096x4096_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 4096 4096 4096 N T 2 event 20 300 100 
# bench_v4_m256_t128_4096x4096x4096_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 4096 4096 4096 N T 2 event 20 300 100 
# bench_v5_m256_t128_4096x4096x4096_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 4096 4096 4096 N T 2 event 20 300 100 
# bench_v5_m256_t512_4096x4096x4096_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 4096 4096 4096 N T 2 event 20 300 100 
# bench_v4_m256_t128_4096x4096x4096_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 4096 4096 4096 N T 2 event 20 300 100 
# bench_v5_m256_t128_4096x4096x4096_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 4096 4096 4096 N T 2 event 20 300 100 
# bench_v5_m256_t256_4096x4096x4096_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 4096 4096 4096 N T 2 event 20 300 100 
# bench_v4_m256_t128_4096x4096x4096_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 4096 4096 4096 T N 2 event 20 300 100 
# bench_v5_m256_t128_4096x4096x4096_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 4096 4096 4096 T N 2 event 20 300 100 
# bench_v5_m256_t256_4096x4096x4096_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 4096 4096 4096 T N 2 event 20 300 100 
# bench_v5_m256_t512_4096x4096x4096_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 4096 4096 4096 T N 2 event 20 300 100 
# bench_v5_m256_t128_4096x4096x4096_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 4096 4096 4096 T N 2 event 20 300 100 
# bench_v5_m256_t256_4096x4096x4096_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 4096 4096 4096 T N 2 event 20 300 100 
# bench_v5_m256_t512_4096x4096x4096_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 4096 4096 4096 T N 2 event 20 300 100 
# bench_v4_m256_t128_4096x4096x4096_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 4096 4096 4096 T N 2 event 20 300 100 
# bench_v5_m256_t256_4096x4096x4096_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 4096 4096 4096 T N 2 event 20 300 100 
# bench_v5_m256_t512_4096x4096x4096_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 4096 4096 4096 T N 2 event 20 300 100 
# bench_v4_m256_t128_4096x4096x4096_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 4096 4096 4096 T N 2 event 20 300 100 
# bench_v5_m256_t128_4096x4096x4096_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 4096 4096 4096 T N 2 event 20 300 100 
# bench_v5_m256_t512_4096x4096x4096_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 4096 4096 4096 T N 2 event 20 300 100 
# bench_v4_m256_t128_4096x4096x4096_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 4096 4096 4096 T N 2 event 20 300 100 
# bench_v5_m256_t128_4096x4096x4096_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 4096 4096 4096 T N 2 event 20 300 100 
# bench_v5_m256_t256_4096x4096x4096_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 4096 4096 4096 T N 2 event 20 300 100 
# bench_v4_m256_t128_4096x4096x256_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 4096 4096 256 N T 2 event 20 300 100 
# bench_v5_m256_t128_4096x4096x256_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 4096 4096 256 N T 2 event 20 300 100 
# bench_v5_m256_t256_4096x4096x256_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 4096 4096 256 N T 2 event 20 300 100 
# bench_v5_m256_t512_4096x4096x256_NT_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 4096 4096 256 N T 2 event 20 300 100 
# bench_v5_m256_t128_4096x4096x256_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 4096 4096 256 N T 2 event 20 300 100 
# bench_v5_m256_t256_4096x4096x256_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 4096 4096 256 N T 2 event 20 300 100 
# bench_v5_m256_t512_4096x4096x256_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 4096 4096 256 N T 2 event 20 300 100 
# bench_v4_m256_t128_4096x4096x256_NT_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 4096 4096 256 N T 2 event 20 300 100 
# bench_v5_m256_t256_4096x4096x256_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 4096 4096 256 N T 2 event 20 300 100 
# bench_v5_m256_t512_4096x4096x256_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 4096 4096 256 N T 2 event 20 300 100 
# bench_v4_m256_t128_4096x4096x256_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 4096 4096 256 N T 2 event 20 300 100 
# bench_v5_m256_t128_4096x4096x256_NT_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 4096 4096 256 N T 2 event 20 300 100 
# bench_v5_m256_t512_4096x4096x256_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 4096 4096 256 N T 2 event 20 300 100 
# bench_v4_m256_t128_4096x4096x256_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 4096 4096 256 N T 2 event 20 300 100 
# bench_v5_m256_t128_4096x4096x256_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 4096 4096 256 N T 2 event 20 300 100 
# bench_v5_m256_t256_4096x4096x256_NT_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 4096 4096 256 N T 2 event 20 300 100 
# bench_v4_m256_t128_4096x4096x256_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 4096 4096 256 T N 2 event 20 300 100 
# bench_v5_m256_t128_4096x4096x256_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 4096 4096 256 T N 2 event 20 300 100 
# bench_v5_m256_t256_4096x4096x256_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 4096 4096 256 T N 2 event 20 300 100 
# bench_v5_m256_t512_4096x4096x256_TN_r1.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 4096 4096 256 T N 2 event 20 300 100 
# bench_v5_m256_t128_4096x4096x256_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 4096 4096 256 T N 2 event 20 300 100 
# bench_v5_m256_t256_4096x4096x256_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 4096 4096 256 T N 2 event 20 300 100 
# bench_v5_m256_t512_4096x4096x256_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 4096 4096 256 T N 2 event 20 300 100 
# bench_v4_m256_t128_4096x4096x256_TN_r2.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 4096 4096 256 T N 2 event 20 300 100 
# bench_v5_m256_t256_4096x4096x256_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 4096 4096 256 T N 2 event 20 300 100 
# bench_v5_m256_t512_4096x4096x256_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 4096 4096 256 T N 2 event 20 300 100 
# bench_v4_m256_t128_4096x4096x256_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 4096 4096 256 T N 2 event 20 300 100 
# bench_v5_m256_t128_4096x4096x256_TN_r3.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 4096 4096 256 T N 2 event 20 300 100 
# bench_v5_m256_t512_4096x4096x256_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 4096 4096 256 T N 2 event 20 300 100 
# bench_v4_m256_t128_4096x4096x256_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v4_m256_t128 4096 4096 256 T N 2 event 20 300 100 
# bench_v5_m256_t128_4096x4096x256_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t128 4096 4096 256 T N 2 event 20 300 100 
# bench_v5_m256_t256_4096x4096x256_TN_r4.log
timeout --kill-after=30s 1800s /root/scg/best-matmul/build/h100_pcie/v5_m256_t256 4096 4096 256 T N 2 event 20 300 100 
# env_gpu_after.txt
timeout --kill-after=30s 1800s nvidia-smi -i 0 -q 
