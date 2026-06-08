# Source Index

Generated inventory for the April 2026 canonical `/src` layout.
Reflects post-reorganization structure: compiler consolidation, `solvers/core/` addition,
`Operators/` placeholder, and archive additions from this session.

---

## Top-Level Folders

| Folder | Purpose |
|--------|---------|
| `Operators/` | Placeholder — operator library (Phase 4+) |
| `../archive/src/` | Superseded / retired source material — do not build |
| `collectives/` | Collective IR, ExecRuntime, NCCL/RCCL adapter stubs |
| `compiler/` | All compiler components (IR, codegen, passes, tooling) |
| `runtime/` | C ABI runtime — device, stream, buffer, kernel launch |
| `solvers/` | Scientific computing: linalg, spectral, scaling/resilience, tpp |
| `tessera.egg-info/` | Python packaging metadata present under `/src`; not a build component |
| `transforms/` | Graph IR canonicalization and lowering passes (Phases 1–3) |

---

## Summary

| | Count |
|-|-------|
| Active folders | 291 |
| Active files | 449 |
| Archived folders | 159 |
| Archived files | 173 |

---

## Active Folder Inventory

### `collectives/`
Collective communication IR and runtime. NCCL/RCCL adapters are stubs pending Phase 4.

```
collectives/
├── cmake/modules
├── docs
├── include/tessera/Dialect/Collective/
│   ├── IR
│   └── Runtime
├── lib/Dialect/Collective/
│   ├── IR
│   └── Runtime
├── test
└── tools/
    ├── tessera-collective-opt
    ├── tessera-exec-demo
    └── tessera-trace-demo
```

---

### `compiler/`
All compiler subsystems consolidated here. Sub-tree:

```
compiler/
├── autotuning/tessera/tools/autotune/
│   ├── configs
│   ├── docs
│   ├── examples
│   ├── include/tessera
│   ├── lib
│   ├── mlir
│   └── tessera_autotuner
│
├── codegen/                   ← Retained hardware backends
│   ├── Tessera_ROCM_Backend/
│   │   ├── ci
│   │   ├── docs
│   │   ├── external/ck_bridge
│   │   ├── include/TesseraROCM/IR
│   │   ├── lib/ (Conversion, IR)
│   │   ├── runtime/hip/examples
│   │   ├── test/rocm
│   │   └── tools
│   ├── tessera_gpu_backend_NVIDIA/
│   │   ├── bench
│   │   ├── docs
│   │   ├── include/tessera/gpu
│   │   ├── scripts
│   │   ├── src/ (kernels, runtime)
│   │   └── tests
│   └── tessera_x86_backend/
│       ├── docs
│       ├── include/tessera/x86
│       ├── scripts
│       ├── src/ (kernels, runtime)
│       └── tests
│
├── docs/pass_reference
│
├── ir/                        ← Graph IR ODS and tiling
│   ├── TesseraOps.td
│   ├── TesseraTiling.cpp
│   └── TilingInterface_NOTES.md
│
├── mlir/                      ← MLIR dialect integration
│   ├── include/Tessera/ (Graph, Schedule, Target)
│   └── lib/ (Graph, Schedule, Target)
│
├── programming_model/         ← Schedule IR, mesh/pipeline ODS, docs
│   ├── docs
│   ├── ir/ (cache, moe, schedule, tile)
│   ├── tests/pm_v1_1
│   └── tools/tessera-opt
│
├── tessera_neighbors/         ← Halo/neighbor exchange dialect
│   ├── docs
│   ├── include/tessera/Dialect/Neighbors/ (IR, Transforms)
│   ├── lib/Dialect/Neighbors/ (IR, Transforms)
│   └── test/Neighbors
│
└── tile_opt_fa4/              ← FA-4 Tile IR (warp spec, TMA, attention)
    ├── cmake
    ├── dialects/ (tessera_attn, tessera_queue)
    ├── docs
    ├── examples
    ├── include/tessera/Dialect/ (Attn, Queue)
    ├── lib/
    │   ├── Conversion/ (TesseraScheduleToTarget, TesseraTileToPTX)
    │   └── Dialect/ (Attn, Queue)
    └── test/ (attn, queue, tmem)
```

---

### `runtime/`
C ABI for device, stream, buffer, and kernel launch. Header-defined; full implementation is Phase 6.

```
runtime/
├── cuda/ (kernels, utils)
├── docs
├── include/tessera
├── src/ (backend, scheduler, util)
└── tests
```

---

### `solvers/`
Scientific computing suite. Reorganized this session: loose ODS and pass files moved into `core/`.

```
solvers/
├── core/                      ← Shared dialects and passes (moved from top-level loose folders)
│   ├── dialects/              ← tessera_rng.td, tessera_solver.td, tessera_sparse.td
│   └── passes/                ← RNGLegalize, SparseInspector, NewtonAutodiff, etc.
│
├── linalg/                    ← Dense linear algebra solver dialect
│   ├── cmake
│   ├── docs
│   ├── include/tessera/Dialect/Solver
│   ├── lib/ (Dialect/Solver, Passes)
│   ├── test/solver
│   └── tools/tessera-opt
│
├── scaling_resilience/        ← Phase 5 SR dialect + passes (canonical version)
│   ├── docs
│   ├── include/tessera/sr
│   ├── lib/sr/ (dialect, passes)
│   └── tests/sr
│
├── spectral/                  ← Spectral / FFT solver dialect
│   ├── benchmarks
│   ├── docs
│   ├── examples
│   ├── include/tessera/Spectral
│   ├── lib/ (Dialect/Spectral, Passes, TargetHooks/{AMD,CPU,NVIDIA})
│   ├── reports
│   ├── scripts
│   ├── test/ir
│   └── tools
│
└── tpp/                       ← Tensor Parallel Primitives dialect
    ├── ci
    ├── docs
    ├── include/tpp/Dialect/TPP
    ├── lib/ (Dialect/TPP, Passes)
    └── test/TPP
```

---

### `transforms/`
Graph IR canonicalization and lowering passes built in Phases 1–3.

```
transforms/
├── include/Tessera/Transforms
└── lib/
```

Key pass files in `lib/`:
- `CanonicalizeTesseraIR.cpp` — 4 Graph IR fusion patterns
- `VerifyTesseraIR.cpp` — module version attribute check
- `DistributionLoweringPass.cpp` — shard attrs → schedule.mesh (Phase 2)
- `EffectAnnotationPass.cpp` — effect inference (Phase 2)
- `TilingPass.cpp` — matmul → scf.for tile loops (Phase 2)
- `TileToX86Pass.cpp` — tiled BF16 → x86 AMX call (Phase 2)
- `TileIRLoweringPass.cpp` — schedule.mesh.region → Tile IR (Phase 3)
- `GPUCollectiveInsertionPass.cpp` — collective insertion (Phase 4, planned)
- `PipelineStageInsertionPass.cpp` — 1F1B schedule (Phase 4, planned)

---

## Archive Inventory
Superseded material — retained for reference only. Do not add build targets here.

| Folder | Superseded by |
|--------|--------------|
| `../archive/src/PDDL_Instruct/` | n/a — research scratch |
| `../archive/src/Sandbox_Toy_compilers/` | Full compiler stack |
| `../archive/src/codegen_prototypes/` | Promoted to proper modules or discarded |
| `../archive/src/tessera_scaling_resilience_v1/` | `solvers/scaling_resilience/` |
| `../archive/src/tile_opt_fa4_old/` | `compiler/tile_opt_fa4/` |
| `../archive/src/tpp_old/` | `solvers/tpp/` |

---

*Last updated: June 2026 — retained-backend focus, compiler consolidation, `Operators/`, `solvers/core`, and archive additions.*
