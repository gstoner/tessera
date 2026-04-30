# Tessera EBT — v2.6

**What's new**
- `tessera-ebt-canonicalize` (real Pass + patterns).
- `tessera-ebt-materialize-loops` (emits `scf.for` K and T for @driver).
- `tessera-ebt-select-grad-path` (pattern swap → `ebt.energy_bilinear_jvp`).
- Optional CUDA/HIP smoke to label device in `reports/roofline.csv`.

## Build (CPU-only smoke)
cmake -S . -B build && cmake --build build -j
./build/run_smoke

## NVIDIA label
cmake -S . -B build -DSMOKE_WITH_CUDA=ON && cmake --build build -j && ./build/run_smoke

## AMD label
cmake -S . -B build -DSMOKE_WITH_HIP=ON && cmake --build build -j && ./build/run_smoke

## Pass demo (use your in-tree mlir-opt build)
mlir-opt models/ebt/ir/samples/driver_sample.mlir   -tessera-ebt-canonicalize -tessera-ebt-materialize-loops="K=2 T=3" -tessera-ebt-select-grad-path=true | FileCheck models/ebt/ir/samples/driver_sample.mlir
