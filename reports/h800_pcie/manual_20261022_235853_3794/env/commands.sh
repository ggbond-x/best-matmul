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
# env_ncu_version.txt
timeout --kill-after=30s 1800s ncu --version 
# env_ncu_sections.txt
timeout --kill-after=30s 1800s ncu --list-sections 
# env_ncu_sets.txt
timeout --kill-after=30s 1800s ncu --list-sets 
# env_ncu_metrics_gh100.txt
timeout --kill-after=30s 1800s ncu --query-metrics --chips gh100 --query-metrics-mode all 
# env_nsys_version.txt
timeout --kill-after=30s 1800s nsys --version 
# env_gpu_after.txt
timeout --kill-after=30s 1800s nvidia-smi -i 0 -q 
