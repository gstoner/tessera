# Tessera Project Structure

This document reflects the canonical layout after the April 2026 reorganization.
All active directory names use underscores (no spaces). Each component has a
single authoritative version; superseded/experimental copies live under
`src/archive/` when retained for local reference.

---

## Directory Tree

```
tessera/
├── README.md                        # Project overview and quick-start
├── LICENSE                          # Apache 2.0
├── CONTRIBUTING.md                  # Contribution guidelines
├── CMakeLists.txt                   # Top-level build entry point
├── pyproject.toml                   # Python package metadata
├── requirements.txt                 # Python runtime dependencies
├── .gitignore
├── PROJECT_STRUCTURE.md             # This file
│
├── src/                             # Production C++/MLIR source
│   ├── Operators/                   # Operator library placeholder (Phase 4+)
│   ├── compiler/                    # Compiler IR, MLIR, codegen, autotuning, pass docs
│   ├── runtime/                     # C++ execution engine
│   ├── solvers/                     # Core sparse/RNG, linalg, SR, spectral, TPP solvers
│   ├── collectives/                 # Collective IR and runtime scaffolding
│   └── transforms/                  # Canonicalization and lowering passes
│
├── python/                          # Python frontend package
│   └── tessera/
│       ├── core/                    # Tensor, Module, fundamental abstractions
│       ├── nn/                      # Neural network layers and ops
│       └── ...
│
├── tests/                           # Test suite
│   ├── kernel_tests/                # System-level kernel + roofline tests
│   └── ...
│
├── docs/                            # All documentation
│   ├── architecture/                # System design docs (incl. Target IR doc)
│   ├── api/                         # API reference
│   ├── build/                       # CMake integration snippets & notes
│   └── tutorials/                   # Step-by-step guides (Flash Attention, etc.)
│
├── examples/                        # Runnable usage examples
│   ├── basic/
│   └── advanced/
│       └── power_retention/         # PowerAttention port (HIP, WGMMA, autotune)
│
├── benchmarks/                      # Performance benchmarks
│
├── tools/                           # Developer tooling
│   ├── profiler/                    # Profiler scripts (tprof_view.py, peaks_sample.yaml)
│   └── ...
│
├── scripts/                         # Build & CI utility scripts
├── cmake/                           # CMake find-modules and helpers
│
└── src/archive/                     # Superseded / pre-production work (not built)
    ├── PDDL_Instruct/               # PDDL-based instruction synthesis experiments
    ├── Sandbox_Toy_compilers/       # Frontend-to-backend sample compilers
    ├── tpp_old/                     # Superseded TPP snapshot
    └── tile_opt_fa4_old/            # Superseded FA4 snapshots
```

---

## Component Versioning Table

The table below records which version was promoted to canonical `src/` and why.
Superseded versions remain on disk as untracked local files but are removed from
the git index.

| Canonical path               | Promoted from                                    | Rationale |
|------------------------------|--------------------------------------------------|-----------|
| `src/compiler/ir`            | `src/ir`                                         | Core Tessera IR grouped under compiler infrastructure |
| `src/compiler/mlir`          | `src/mlir`                                       | MLIR dialect integration grouped under compiler infrastructure |
| `src/compiler/programming_model` | `src/tessera_pm_v1_1_memory_parallel`        | Programming-model IR grouped under compiler infrastructure |
| `src/compiler/tessera_neighbors` | `src/tessera-neighbors`                      | Neighbor topology/halo lowering grouped under compiler infrastructure |
| `src/compiler/tile_opt_fa4`  | `src/Tile_Optimization_FA4 /…/v1_3`              | FA4 tile optimization grouped under compiler infrastructure |
| `src/compiler/RubinCPX_Backend` | `src/Tessera_RubinCPX_Compiler`               | NV Rubin CPX target renamed and grouped under compiler infrastructure |
| `src/solvers/core`           | `src/src/solvers`                                | Sparse/RNG/nonlinear solver core grouped below solvers |
| `src/solvers/linalg`         | `src/tessera_linalg_solvers`                    | Linear algebra solver scaffold grouped under the solver stack |
| `src/solvers/scaling_resilience` | `src/tessera_scaling_resilience_v1_1`        | SR dialect and passes grouped below solvers |
| `src/solvers/spectral`       | `src/spectral`                                   | Spectral/FFT dialect grouped below solvers |
| `src/solvers/tpp`            | `src/tpp/tpp_v0_2`                               | TPP dialect grouped below solvers |
| `src/Operators`              | new                                              | Placeholder for operator library work (Phase 4+) |
| `tests/kernel_tests`         | `tests/tessera_kernels_scaffold`                 | Original retained: has system tests, roofline script, profile_ncu.sh |
| `examples/advanced/power_retention` | `examples/advanced/Power Retention/…/v0_9` | v0_9: HIP kernel, WGMMA, autotune, nlohmann_json integration |
| `src/archive/PDDL_Instruct`  | `src/PDDL_Instruct/pddl_instruct_tessera_v1`     | Experimental — archived out of the active `src/` surface |
| `src/archive/Sandbox_Toy_compilers` | `src/Sandbox_Toy_compilers/…`            | Experimental sample compilers — archived out of the active `src/` surface |
| `src/archive/tpp_old`        | `src/tpp_old`                                    | Superseded by canonical `src/solvers/tpp` |
| `src/archive/tile_opt_fa4_old` | `src/tile_opt_fa4_old`                         | Superseded by canonical `src/compiler/tile_opt_fa4` |
| `src/compiler/codegen/Tessera_TPU_Backend` | `src/compiler/codegen/Tessera_TPU_Backend_Starter_Advanced` | Advanced TPU backend promoted to single canonical TPU backend folder |
| `src/compiler/docs/pass_reference` | `src/compiler/passes`                     | Pass reference markdown grouped with compiler documentation |
| `docs/archive/pre_canonical/api` | `docs/api/Tessera_API_Vol*.md`               | Pre-canonical API volumes archived behind canonical API docs |
| `docs/archive/pre_canonical/model` | `docs/Tessera_Deep_Learning_Programming_Model.md` | Pre-canonical model guide archived due old API examples |
| `docs/architecture/`         | `src/compiler/tessera_target_ir_doc3b.md`        | Architecture doc migrated out of src/ |
| `docs/build/`                | `src/README_SRC_INTEGRATION.md` + `CMakeLists.add_this_snippet.txt` | Build notes migrated out of src/ |
| `docs/tutorials/Flash_Attention_in_Tessera.md` | `docs/tutorials/Flash Attention_in_Tessera.md` | Space in filename removed |

---

## Archive Policy

Files under `src/archive/` and `docs/archive/` are **not included in the
production build or active documentation lint**. They exist for reference, future
graduation to an active component, or deletion after review. New production work
should land in the canonical component folder, not beside an archived copy.

---

## Build System (next milestone)

The build system cleanup is the immediate follow-on to this reorganization.
Outstanding work:

- Keep `src/CMakeLists.txt` aligned as compiler and solver subtrees graduate
  from scaffold to production build targets.
- Add per-component `CMakeLists.txt` where missing.
- Validate MLIR dialect registration and tablegen targets for each component.
- Gate the build on a single `TESSERA_VERSION` variable defined in one place.
- Graduate `src/solvers/linalg` from opt-in scaffold to the parent solver build
  once its MLIR APIs are aligned with the canonical solver pass stack.
