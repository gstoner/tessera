# Tessera Neighbors & Halo Drop-in (v0.1)

This package adds a *Neighbors* topology abstraction, *@halo* management, reusable *stencil* operators,
pipelining directives, and dynamic topology behavior to Tessera.

It includes:
- **Ops**: none of its own. The `tessera.neighbors.*` ops (topology, halo
  region/exchange/pack/transport/unpack, neighbor read, stencil define/apply,
  pipeline config) are core `tessera` dialect ops, declared once in
  `src/compiler/ir/TesseraOps.td` and verified in `src/compiler/ir/TesseraOps.cpp`.
  An unbuilt `tessera_neighbors.td` and a hand-written `tessera.neighbors`
  dialect that re-declared the same names were deleted 2026-09-27 (Decision
  #31; sync `SMALL-CORRECTNESS-GAPS-2026-09-27`).
- **Passes**: `-tessera-halo-infer`, `-tessera-stencil-lower`, `-tessera-pipeline-overlap`, `-tessera-topology-dynamic`, and the boundary-condition / loop-materialize / halo-mesh / halo-transport lowerings
- **FileCheck tests**: `tests/tessera-ir/phase7/neighbors_*.mlir`
- **docs/Neighbors_and_Halo.md** (semantics & examples)
- **SPEC_Neighbors_and_Halo.md** (full spec write-up)

## Build (MLIR/LLVM style)

```cmake
# In your top-level CMakeLists.txt, add_subdirectory to this folder.
add_subdirectory(tessera_neighbors)
```

Then build as usual with your LLVM/MLIR build.
