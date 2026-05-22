# Tessera EBT — v2.5

**New in this cut**
- Near-real MLIR rewrite scaffolds:
  - `models/ebt/passes/materialize_loops.{h,cc}`
  - `models/ebt/passes/select_grad_path.{h,cc}`
- CPU **K×T smoke** that writes `reports/roofline.csv`:
  - `tools/run_ebt_smoke.cpp`

## Build and run the smoke
cmake -S . -B build && cmake --build build -j
EBT_K=8 EBT_T=6 EBT_D=512 EBT_DEVICE=CPU ./build/run_ebt_smoke
# -> reports/roofline.csv (use v2.3 roofline.py to render HTML)

## Next wiring steps
- Replace TODO(mlir) with real includes and builders; register pipelines.
- Point your CI to compile the smoke and publish the CSV + roofline HTML artifact.
