# Tessera Neighbors & Halo Drop-in (v0.1)

This package adds a *Neighbors* topology abstraction, *@halo* management, reusable *stencil* operators,
pipelining directives, and dynamic topology behavior to Tessera.

It includes:
- **ODS**: `tessera_neighbors.td` (directions, Δ-relative addressing, stencil, pipeline ops)
- **Pass stubs**: `-tessera-halo-infer`, `-tessera-stencil-lower`, `-tessera-pipeline-overlap`, `-tessera-topology-dynamic`
- **8 FileCheck tests**
- **docs/Neighbors_and_Halo.md** (semantics & examples)
- **SPEC_Neighbors_and_Halo.md** (full spec write-up)

## Build (MLIR/LLVM style)

```cmake
# In your top-level CMakeLists.txt, add_subdirectory to this folder.
add_subdirectory(tessera-neighbors-dropin-v0_1)
```

Then build as usual with your LLVM/MLIR build.
