<!-- MERGE_START -->
# EBT v2.6 — Canonicalize Pass, Concrete MLIR Rewrites, CUDA/HIP smoke options

**Adds**
- `tessera-ebt-canonicalize` pass: normalizes `self_verify`, inlines trivial `decode_init`, hoists invariants.
- Concrete MLIR **pass classes** with real includes and minimal logic:
  - `MaterializeLoopsPass` (emits `scf.for` for K and T; attaches attrs).
  - `SelectGradPathPass` (pattern to replace `tessera.autodiff.grad_y` with `@energy_*_jvp` when enabled).
- **CUDA/HIP smoke** backends (optional via CMake flags) to label device in `roofline.csv`:
  - `-DSMOKE_WITH_CUDA=ON` builds a CUDA stub and sets `device=NVIDIA`.
  - `-DSMOKE_WITH_HIP=ON` builds a HIP stub and sets `device=AMD`.

This is still a scaffold, but now much closer to drop-in MLIR code.
<!-- MERGE_END -->
