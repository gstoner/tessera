<!-- ===== MERGE_START Tessera Empirical Software Agent ===== -->
# Tessera Integration Points

- **IR Generation**: LLM proposals may materialize as **Tile IR** fragments (e.g., matmul/conv/attention kernels).
- **Lowering**: use your existing pipelines to Target IR (NVIDIA, ROCm, AVX/AMX, TPU).
- **Autotune**: when a proposal introduces tunables (tile sizes, swizzles, TMA row chunking), enqueue parameter sweeps and feed best score back to TS.
- **Profiler**: attach **Perfetto** + **roofline** overlays per run; export `reports/` into candidate artifact folder.
- **Pass Hook**: `-tessera-empirical-search` accepts `--task` and `--budget`, invoking this search over IR-level variants.

Interop sketch:
```
candidate/
  ir/seed.mlir
  ir/after-rewrite.mlir
  compile/
    target=sm90/
    target=gfx94/
  reports/
    roofline.csv
    perfetto.json
  score.json
```
<!-- ===== MERGE_END Tessera Empirical Software Agent ===== -->
