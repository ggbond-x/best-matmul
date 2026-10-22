# Archived v1–v5

These are the previous five entry points. The default path is `gemm/gemm_teach.cuh` and `gemm/gemm_fast.cuh` at the repository root.

| File | What it was |
|---|---|
| `kernels/wgmma_tma_fp16_v1.cuh` | One warpgroup, wait for each WGMMA batch (`inflight=1`), direct store |
| `kernels/wgmma_tma_fp16_pipeline_v2.cuh` | Two in-flight WGMMA batches |
| `kernels/wgmma_tma_fp16_epilogue_v3.cuh` | Pipeline plus coalesced store |
| `kernels/wgmma_tma_fp16_warp_specialized_v4.cuh` | Producer warpgroup + one compute warpgroup |
| `kernels/wgmma_tma_fp16_multi_warpgroup_v5.cuh` | Producer + configurable compute warpgroups (default 2) |

`teach` is the single-warpgroup pipeline and coalesced store (the v2/v3 schedule) with the fixed defaults. `fast` is the warp-specialized producer plus two compute warpgroups (the v5 default shape).

`GMMA_INST_N` in `kernels/detail/wgmma_config.cuh` now defaults to 128, the same value `experiments/build.sh` passes and the same instruction `gemm/config.cuh` fixes for `teach` / `fast`. The old header default of 64 is gone, so a direct archived build and the suite build no longer disagree. Pass `-DGMMA_INST_N=64` only when `PEAK_CTA_N` is a multiple of 64 and you explicitly want the narrower instruction.

`experiments/build.sh` compiles this directory. One include path in `kernels/detail/tma_wgmma_adapter.cuh` was adjusted when the files moved (`../../../../host/common.cuh`). The device schedules were not rewritten.
