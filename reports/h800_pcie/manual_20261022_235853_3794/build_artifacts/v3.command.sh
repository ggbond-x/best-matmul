#!/usr/bin/env bash
bash -x /root/scg/best-matmul/build_benchmark.sh --kernel v3 --cutlass /root/scg/cutlass --nvcc /usr/local/cuda-12.8/bin/nvcc --output /root/scg/best-matmul/build/h100_pcie/.v3.tmp.3809 -- -lineinfo -Xptxas=-v -DPEAK_CTA_M=128 -DPEAK_CTA_N=128 -DPEAK_CTA_K=64 -DPIPE=3 -DGMMA_INST_N=128 -DWGMMA_COMPUTE_THREADS=128 -DWGMMA_MAX_INFLIGHT=2 -DEPILOGUE_COALESCED=1 
