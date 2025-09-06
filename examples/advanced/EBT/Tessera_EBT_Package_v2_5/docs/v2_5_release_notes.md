<!-- MERGE_START -->
# EBT v2.5 — Concrete rewrite sketches + CPU K×T smoke (roofline CSV)

**Additions**
- **Concrete MLIR-style rewrites (scaffold with near-real code)** for:
  - `tessera-ebt-materialize-loops`
  - `tessera-ebt-select-grad-path`
- **CPU K×T smoke executable** (`tools/run_ebt_smoke.cpp`) that runs a tiny inner loop,
  times kernels with `std::chrono`, and emits `reports/roofline.csv` you can feed to the v2.3 report tool.
- **End-to-end sample**: updated to print clear markers so FileCheck can verify loop emission & grad path.

These files are drop-in scaffolds—replace the `TODO(mlir)` sections with real includes and builders in-tree.
<!-- MERGE_END -->
