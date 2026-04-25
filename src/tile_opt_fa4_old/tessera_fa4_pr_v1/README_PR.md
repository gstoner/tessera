# PR: FA‑4 Pipeline Features for Tessera

This PR adds first-class IR & lowering for warp specialization, persistent tile scheduling, Blackwell TMEM, and a softmax numerics policy inspired by FA‑4.
It includes:
- `tessera.schedule.warp`, `tessera.schedule.pipe`, `tessera.schedule.policy(persistent)`
- `tessera.numerics.softmax{exp = poly3, rescale_threshold = ...}`
- `!tile.memspace<"tmem", …>` + `tile.mma.tcgen05`, `tile.ld.tmem`, `tile.st.tmem`
- Autotuner knobs: `warp_counts`, `buffer_depths`, `scheduler`, `rescale_threshold`
- Tutorial notebook: *Life of a Tile (FA‑4 Edition)*

## Layout
- `dialects/…/*.td` — ODS additions
- `lib/Conversion/*` — lowering stubs with verified builders
- `test/*/*.mlir` — FileCheck tests
- `tools/autotune/schema_v2.json` — extended search space
- `docs/Life_of_a_Tile_FA4.md` and `notebooks/Life_of_a_Tile_FA4.ipynb` — tutorial

## Build
```
# CMake snippets are provided under cmake/
# Integrate by adding add_subdirectory() to your tree and linking tessera dialect libs.
```

## Status
These files are production‑grade scaffolds: they compile and register ops/passes with verifiers and printers, and contain concrete lowering hooks. The poly‑exp coefficients are placeholders (to be tuned per arch).
