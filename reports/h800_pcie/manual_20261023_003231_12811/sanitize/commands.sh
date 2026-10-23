# 命令审计清单；请使用 run.sh 重跑实验，不要整体执行此文件。
# 包含持续采样命令；日志重定向及后台进程回收由 run.sh 管理。
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-89384b53-c02b-0f66-c2ab-b966923d789f
# env_date.txt
timeout --kill-after=30s 7200s date --iso-8601=seconds 
# env_uname.txt
timeout --kill-after=30s 7200s uname -a 
# env_cpu.txt
timeout --kill-after=30s 7200s lscpu 
# env_gpu_list.txt
timeout --kill-after=30s 7200s nvidia-smi -L 
# env_gpu_before.txt
timeout --kill-after=30s 7200s nvidia-smi -i 0 -q 
# env_topology.txt
timeout --kill-after=30s 7200s nvidia-smi topo -m 
# env_processes.txt
timeout --kill-after=30s 7200s nvidia-smi --query-compute-apps=gpu_uuid\,pid\,process_name\,used_gpu_memory --format=csv 
# env_sanitizer_version.txt
timeout --kill-after=30s 7200s compute-sanitizer --version 
# v4.help.txt
timeout --kill-after=30s 7200s /root/scg/best-matmul/build/h100_pcie/v4 --help 
# v4.sha256.txt
timeout --kill-after=30s 7200s sha256sum /root/scg/best-matmul/build/h100_pcie/v4 
# v5.help.txt
timeout --kill-after=30s 7200s /root/scg/best-matmul/build/h100_pcie/v5 --help 
# v5.sha256.txt
timeout --kill-after=30s 7200s sha256sum /root/scg/best-matmul/build/h100_pcie/v5 
# v5_m256_t512.help.txt
timeout --kill-after=30s 7200s /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 --help 
# v5_m256_t512.sha256.txt
timeout --kill-after=30s 7200s sha256sum /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 
# sanitize_v4_NT_memcheck.log
timeout --kill-after=30s 7200s compute-sanitizer --tool memcheck --error-exitcode 99 /root/scg/best-matmul/build/h100_pcie/v4 256 256 448 N T 3 event 1 1 2 
# sanitize_v4_NT_racecheck.log
timeout --kill-after=30s 7200s compute-sanitizer --tool racecheck --error-exitcode 99 /root/scg/best-matmul/build/h100_pcie/v4 256 256 448 N T 3 event 1 1 2 
# sanitize_v4_NT_synccheck.log
timeout --kill-after=30s 7200s compute-sanitizer --tool synccheck --error-exitcode 99 /root/scg/best-matmul/build/h100_pcie/v4 256 256 448 N T 3 event 1 1 2 
# sanitize_v4_TN_memcheck.log
timeout --kill-after=30s 7200s compute-sanitizer --tool memcheck --error-exitcode 99 /root/scg/best-matmul/build/h100_pcie/v4 256 256 448 T N 3 event 1 1 2 
# sanitize_v4_TN_racecheck.log
timeout --kill-after=30s 7200s compute-sanitizer --tool racecheck --error-exitcode 99 /root/scg/best-matmul/build/h100_pcie/v4 256 256 448 T N 3 event 1 1 2 
# sanitize_v4_TN_synccheck.log
timeout --kill-after=30s 7200s compute-sanitizer --tool synccheck --error-exitcode 99 /root/scg/best-matmul/build/h100_pcie/v4 256 256 448 T N 3 event 1 1 2 
# sanitize_v5_NT_memcheck.log
timeout --kill-after=30s 7200s compute-sanitizer --tool memcheck --error-exitcode 99 /root/scg/best-matmul/build/h100_pcie/v5 256 256 448 N T 3 event 1 1 2 
# sanitize_v5_NT_racecheck.log
timeout --kill-after=30s 7200s compute-sanitizer --tool racecheck --error-exitcode 99 /root/scg/best-matmul/build/h100_pcie/v5 256 256 448 N T 3 event 1 1 2 
# sanitize_v5_NT_synccheck.log
timeout --kill-after=30s 7200s compute-sanitizer --tool synccheck --error-exitcode 99 /root/scg/best-matmul/build/h100_pcie/v5 256 256 448 N T 3 event 1 1 2 
# sanitize_v5_TN_memcheck.log
timeout --kill-after=30s 7200s compute-sanitizer --tool memcheck --error-exitcode 99 /root/scg/best-matmul/build/h100_pcie/v5 256 256 448 T N 3 event 1 1 2 
# sanitize_v5_TN_racecheck.log
timeout --kill-after=30s 7200s compute-sanitizer --tool racecheck --error-exitcode 99 /root/scg/best-matmul/build/h100_pcie/v5 256 256 448 T N 3 event 1 1 2 
# sanitize_v5_TN_synccheck.log
timeout --kill-after=30s 7200s compute-sanitizer --tool synccheck --error-exitcode 99 /root/scg/best-matmul/build/h100_pcie/v5 256 256 448 T N 3 event 1 1 2 
# sanitize_v5_m256_t512_NT_memcheck.log
timeout --kill-after=30s 7200s compute-sanitizer --tool memcheck --error-exitcode 99 /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 512 256 448 N T 3 event 1 1 2 
# sanitize_v5_m256_t512_NT_racecheck.log
timeout --kill-after=30s 7200s compute-sanitizer --tool racecheck --error-exitcode 99 /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 512 256 448 N T 3 event 1 1 2 
# sanitize_v5_m256_t512_NT_synccheck.log
timeout --kill-after=30s 7200s compute-sanitizer --tool synccheck --error-exitcode 99 /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 512 256 448 N T 3 event 1 1 2 
# sanitize_v5_m256_t512_TN_memcheck.log
timeout --kill-after=30s 7200s compute-sanitizer --tool memcheck --error-exitcode 99 /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 512 256 448 T N 3 event 1 1 2 
# sanitize_v5_m256_t512_TN_racecheck.log
timeout --kill-after=30s 7200s compute-sanitizer --tool racecheck --error-exitcode 99 /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 512 256 448 T N 3 event 1 1 2 
# sanitize_v5_m256_t512_TN_synccheck.log
timeout --kill-after=30s 7200s compute-sanitizer --tool synccheck --error-exitcode 99 /root/scg/best-matmul/build/h100_pcie/v5_m256_t512 512 256 448 T N 3 event 1 1 2 
# env_gpu_after.txt
timeout --kill-after=30s 7200s nvidia-smi -i 0 -q 
