# Tessera Project Structure

This document reflects the canonical layout after the April 2026 reorganization.
Canonical production component names use underscores where practical. Some
legacy example snapshots still keep their original names for reference; new
active work should land in the canonical folders listed here.

---

## Directory Tree

```
tessera/
├── README.md                        # Project overview and quick-start
├── LICENSE                          # Apache 2.0
├── CONTRIBUTING.md                  # Contribution guidelines
├── CMakeLists.txt                   # Top-level build entry point
├── pyproject.toml                   # Python package metadata
├── requirements.txt                 # Python runtime + development dependencies
├── .gitignore
├── PROJECT_STRUCTURE.md             # This file
├── .github/workflows/               # CI workflows (lint + CPU validation spine)
│
├── src/                             # Production C++/MLIR source
│   ├── Operators/                   # Operator library placeholder (Phase 4+)
│   ├── compiler/                    # Compiler IR, MLIR, codegen, autotuning, pass docs
│   ├── runtime/                     # C++ execution engine + CPU runtime backbone
│   ├── solvers/                     # Core sparse/RNG, linalg, SR, spectral, TPP solvers
│   ├── collectives/                 # Collective IR and runtime scaffolding
│   └── transforms/                  # Canonicalization and lowering passes
│
├── python/                          # Python frontend package
│   └── tessera/
│       ├── core/                    # Tensor, Module, fundamental abstractions
│       ├── compiler/                # JIT, constraints, effects, Graph IR, GPU target helpers
│       ├── distributed/             # Mesh, shard, domain, launch, region APIs
│       ├── nn/                      # Neural network layers and ops
│       ├── testing/                 # Mock collectives and Python test helpers
│       ├── cli/                     # tessera-mlir, tessera-prof, runtime smoke CLI
│       ├── runtime.py               # Python wrapper/helper over runtime C ABI
│       ├── profiler.py              # Runtime profiler facade
│       ├── autotune.py              # Public autotuning facade
│       ├── telemetry.py             # Shared telemetry event/report schema
│       └── ...
│
├── tests/                           # Test suite
│   ├── unit/                        # Python unit tests and CPU validation contracts
│   ├── integration/                 # Integration tests
│   ├── regression/                  # Regression tests
│   ├── tessera-ir/                  # MLIR/lit-style pipeline tests by phase
│   ├── kernel_tests/                # System-level kernel + roofline tests
│   ├── performance/                 # Optional performance/weekly sweep scaffolds
│   ├── tessera_numerical_validation/ # Numerical reference validation suite
│   └── ...
│
├── docs/                            # All documentation
│   ├── architecture/                # System design docs (incl. Target IR doc)
│   ├── api/                         # API reference
│   ├── operations/                  # Canonical operation catalog
│   ├── programming_guide/           # User-facing programming guide
│   ├── spec/                        # IR, ABI, language, and compiler specs
│   └── tutorials/                   # Step-by-step guides (Flash Attention, etc.)
│
├── examples/                        # Runnable usage examples
│   ├── getting_started/
│   ├── integration/
│   ├── optimization/
│   └── advanced/
│       └── power_retention/         # PowerAttention port (HIP, WGMMA, autotune)
│
├── benchmarks/                      # Performance benchmarks, telemetry gates, baselines
│   ├── baselines/                   # Deterministic CPU smoke perf-gate baselines
│   ├── common/                      # Shared benchmark compiler/correctness contracts
│   ├── Tessera_Operator_Benchmarks/ # Operator micro-benchmark suite
│   ├── Tessera_SuperBench/          # SuperBench-style benchmark suite
│   ├── benchmark_*.py               # GEMM/attention/collective benchmark models
│   ├── run_all.py                   # Unified benchmark orchestrator
│   └── perf_gate.py                 # Telemetry baseline gate
│
├── tools/                           # Developer tooling
│   ├── profiler/                    # Canonical tprof profiler runtime, CLI, reports
│   ├── roofline_tools/              # Roofline ingestion/report helpers
│   ├── tessera-opt/                 # MLIR opt-style driver
│   ├── tessera-translate/           # Translation tools placeholder
│   └── CLI/                         # CLI starter snapshots
│
├── scripts/                         # Build, validation, and CI utility scripts
│   ├── validate.sh                  # CPU-only validation spine
│   ├── check_versions.py            # CMake/Python/runtime version drift check
│   ├── build.sh
│   ├── test.sh
│   └── lint_docs.sh
├── cmake/                           # CMake find-modules and helpers
├── research/                        # Experimental research prototypes outside production build
│
└── src/archive/                     # Superseded / pre-production work (not built)
    ├── PDDL_Instruct/               # PDDL-based instruction synthesis experiments
    ├── Sandbox_Toy_compilers/       # Frontend-to-backend sample compilers
    ├── tpp_old/                     # Superseded TPP snapshot
    └── tile_opt_fa4_old/            # Superseded FA4 snapshots
```

Local/generated directories such as `build/`, `.venv/`, `__pycache__/`, and
`.pytest_cache/` may exist in a developer checkout but are not source layout
components.

---

## Component Versioning Table

The table below records notable promotions into canonical `src/` and why.
Superseded versions may remain under `src/archive/`, `docs/archive/`, or legacy
example snapshot folders when they are kept for reference.

| Canonical path               | Promoted from                                    | Rationale |
|------------------------------|--------------------------------------------------|-----------|
| `src/compiler/ir`            | `src/ir`                                         | Core Tessera IR grouped under compiler infrastructure |
| `src/compiler/mlir`          | `src/mlir`                                       | MLIR dialect integration grouped under compiler infrastructure |
| `src/compiler/programming_model` | `src/tessera_pm_v1_1_memory_parallel`        | Programming-model IR grouped under compiler infrastructure |
| `src/compiler/tessera_neighbors` | `src/tessera-neighbors`                      | Neighbor topology/halo lowering grouped under compiler infrastructure |
| `src/compiler/tile_opt_fa4`  | `src/Tile_Optimization_FA4 /…/v1_3`              | FA4 tile optimization grouped under compiler infrastructure |
| `src/compiler/codegen/Tessera_RubinCPX_Backend` | `src/Tessera_RubinCPX_Compiler`     | NV Rubin CPX target grouped with compiler codegen backends |
| `src/solvers/core`           | `src/src/solvers`                                | Sparse/RNG/nonlinear solver core grouped below solvers |
| `src/solvers/linalg`         | `src/tessera_linalg_solvers`                    | Linear algebra solver scaffold grouped under the solver stack |
| `src/solvers/scaling_resilience` | `src/tessera_scaling_resilience_v1_1`        | SR dialect and passes grouped below solvers |
| `src/solvers/spectral`       | `src/spectral`                                   | Spectral/FFT dialect grouped below solvers |
| `src/solvers/tpp`            | `src/tpp/tpp_v0_2`                               | TPP dialect grouped below solvers |
| `src/Operators`              | new                                              | Placeholder for operator library work (Phase 4+) |
| `tests/kernel_tests`         | `tests/tessera_kernels_scaffold`                 | Original retained: has system tests, roofline script, profile_ncu.sh |
| `examples/advanced/power_retention` | `examples/advanced/Power Retention/…/v0_9` | v0_9: HIP kernel, WGMMA, autotune, nlohmann_json integration |
| `research/pddl_instruct`     | `src/PDDL_Instruct/pddl_instruct_tessera_v1`     | Experimental research prototype moved out of the active `src/` surface |
| `research/sandbox_compilers` | `src/Sandbox_Toy_compilers/…`                    | Experimental sample compilers moved out of the active `src/` surface |
| `src/archive/tpp_old`        | `src/tpp_old`                                    | Superseded by canonical `src/solvers/tpp` |
| `src/archive/tile_opt_fa4_old` | `src/tile_opt_fa4_old`                         | Superseded by canonical `src/compiler/tile_opt_fa4` |
| `src/compiler/codegen/Tessera_TPU_Backend` | `src/compiler/codegen/Tessera_TPU_Backend_Starter_Advanced` | Advanced TPU backend promoted to single canonical TPU backend folder |
| `src/compiler/docs/pass_reference` | `src/compiler/passes`                     | Pass reference markdown grouped with compiler documentation |
| `docs/archive/pre_canonical/api` | `docs/api/Tessera_API_Vol*.md`               | Pre-canonical API volumes archived behind canonical API docs |
| `docs/archive/pre_canonical/model` | `docs/Tessera_Deep_Learning_Programming_Model.md` | Pre-canonical model guide archived due old API examples |
| `docs/architecture/`         | `src/compiler/tessera_target_ir_doc3b.md`        | Architecture doc migrated out of src/ |
| `docs/tutorials/Flash_Attention_in_Tessera.md` | `docs/tutorials/Flash Attention_in_Tessera.md` | Space in filename removed |
| `python/tessera/telemetry.py` | new                                             | Shared telemetry schema for profiler, autotune, benchmarks, and runtime smoke |
| `python/tessera/cli/runtime.py` | new                                           | `tessera-runtime-smoke` CLI for CPU runtime telemetry validation |
| `benchmarks/perf_gate.py`    | new                                              | Telemetry baseline gate for deterministic CPU smoke reports |
| `benchmarks/baselines/`      | new                                              | Baseline inputs for benchmark/runtime telemetry gates |
| `scripts/validate.sh`        | new                                              | CPU-only local validation spine across Python, runtime, profiler, and collectives |
| `scripts/check_versions.py`  | new                                              | Version consistency check across CMake, Python, and runtime headers |
| `.github/workflows/cpu-validation.yml` | new                                     | CI workflow for the CPU validation spine |

---

## Archive Policy

Files under `src/archive/` and `docs/archive/` are **not included in the
production build or active documentation lint**. Research prototypes under
`research/` and legacy example snapshots under `examples/advanced/` are likewise
outside the production build unless explicitly wired by CMake. They exist for
reference, future graduation to an active component, or deletion after review.
New production work should land in the canonical component folder, not beside an
archived copy.

---

## Build System Status

The April 2026 reorganization is now paired with a CPU-only validation spine.
Current active validation entry points:

- `scripts/validate.sh` runs version checks, Python unit tests, runtime smoke
  telemetry, benchmark smoke telemetry, standalone CPU runtime CMake/CTest,
  C++ profiler smoke build, and collectives runtime compile check.
- `.github/workflows/cpu-validation.yml` runs the same CPU validation spine in CI.
- `scripts/check_versions.py` gates CMake, Python package, and runtime header
  versions on one project version value.
- Standalone runtime tests are split into separate CTest executables so each
  test file owns its own `main()`.

Remaining build-system work:

- Keep `src/CMakeLists.txt` aligned as compiler, solver, backend, and collective
  subtrees graduate from scaffold to production build targets.
- Continue validating MLIR dialect registration and TableGen targets for each
  component in a full monorepo build when LLVM/MLIR are available.
- Expand CI beyond the CPU spine once CUDA/HIP/NCCL execution paths are real and
  deterministic enough for automated validation.
