# v1–v5 experiment suite

Optional. The default build is the repository root `build.sh` (`teach` and `fast`).

This directory compiles `archive/v1-v5/` and can run the old validate / bench / NCU / nsys / sanitizer flow. CUTLASS comes from `--cutlass`, `$CUTLASS_PATH`, or `<repo>/third_party/cutlass`.

```bash
cd experiments
export CUTLASS_PATH=/path/to/cutlass
export CUDACXX=/usr/local/cuda/bin/nvcc

bash build.sh --dry-run
bash build.sh --suite core --with-validation
bash run.sh --mode validate --gpu 0
```

`bash test.sh` is the long one-shot capture (15 configurations, NCU, sanitizer). It is not part of the default README path.

`experiments/build.sh` passes `-DGMMA_INST_N=128`. The archived header defaults to the same value, so `experiments/build_benchmark.sh --kernel v5` without an extra `-D` matches the suite.
