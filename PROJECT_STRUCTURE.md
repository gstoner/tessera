# Tessera Project Structure

This document reflects the canonical layout after the April 2026 reorganization,
refreshed for the May 2026 surface additions (Apple GPU Phase 8, S-series
standalone-compiler track, audit-as-data infrastructure, M6/M7 domain cores).

Canonical production component names use underscores where practical. Some
legacy example snapshots still keep their original names for reference; new
active work should land in the canonical folders listed here.

---

## Directory Tree

```
tessera/
├── README.md                        # Project overview and quick-start
├── LICENSE                          # Apache 2.0
├── AGENTS.md                        # Agent / sub-agent operating manual
├── CLAUDE.md                        # Claude Code project context
├── CONTRIBUTING.md                  # Contribution guidelines
├── SECURITY.md                      # Security policy
├── CMakeLists.txt                   # Top-level build entry point
├── pyproject.toml                   # Python package metadata
├── requirements.txt                 # Python runtime + development dependencies
├── skills.md                        # Claude Code skill map
├── tessera_style_guide.md           # Coding style guide
├── .gitignore
├── PROJECT_STRUCTURE.md             # This file
├── .github/workflows/               # CI workflows (validate, codeql, pylint, codacy, python-quality)
│
├── src/                             # Production C++/MLIR source
│   ├── Operators/                   # Operator library placeholder (Phase 4+, empty today)
│   ├── compiler/                    # Compiler IR, MLIR, codegen, autotuning, pass docs
│   │   ├── ir/                      # Graph IR ODS + TilingInterface impls
│   │   ├── mlir/                    # MLIR dialect integration glue
│   │   ├── codegen/                 # Per-target backends — see "Backends" below
│   │   ├── diagnostics/             # ErrorReporter, ShapeInferencePass
│   │   ├── autotuning/              # Autotuner v1 framework
│   │   ├── docs/                    # Pass reference markdown grouped with compiler
│   │   ├── programming_model/       # Mesh/pipeline/region ODS + spec docs
│   │   ├── tessera_neighbors/       # Halo/stencil neighbor exchange dialect (Phase 7)
│   │   └── tile_opt_fa4/            # FA-4 Attn + Queue dialects, warp-spec / TMA passes
│   ├── runtime/                     # C++ execution engine + CPU/CUDA/HIP backends + C ABI
│   ├── solvers/                     # IR-level solver dialects + passes
│   │   ├── core/                    # Sparse / RNG / nonlinear (11 core passes)
│   │   ├── linalg/                  # MixedPrecision + IterativeRefinement
│   │   ├── scaling_resilience/      # SR dialect — InsertRecompute, OptimizerShard, ResilienceRestart
│   │   ├── spectral/                # Spectral/FFT dialect + 6 pass bodies + ts-spectral-opt
│   │   ├── tpp/                     # TPP dialect, 7 passes, tpp-space-time pipeline
│   │   ├── clifford/                # Clifford / GA solver scaffolding (M5)
│   │   └── ebm/                     # Energy-based-model solver scaffolding (M6)
│   ├── collectives/                 # Collective IR + NCCL/RCCL adapters + chunk planner
│   └── transforms/                  # Canonicalization, lowering, named pipelines (lib/Passes.cpp)
│
├── python/                          # Python frontend package
│   └── tessera/
│       ├── __init__.py              # Public surface — re-exports core, jit, dist, ops, dtype …
│       ├── core/                    # Tensor, Module, fundamental abstractions
│       ├── compiler/                # JIT, constraints, effects, Graph IR, target maps, audit modules
│       ├── distributed/             # Mesh, shard, domain, launch, region APIs;
│       │                            #   MoE router + distributed MegaMoE
│       │                            #   (expert-parallel 2x all-to-all,
│       │                            #   FP8xFP4, async comm/compute overlap)
│       ├── nn/                      # Stateful nn surface — module, layers, functional, utils
│       ├── autodiff/                # Tape-based reverse-mode (Tier 2) + vjps/jvps subpackages
│       ├── cache/                   # KVCacheHandle + MemoryStateHandle persistent state ABI
│       ├── ebm/                     # M6 — Energy-based-model primitives (energy/partition/Langevin)
│       ├── ga/                      # M5/M7 — Geometric Algebra (Clifford) primitives + rotors
│       ├── runtime/                 # Python runtime support package
│       ├── solvers/                 # Solver Python frontmatter (e.g. tpp.py status surface)
│       ├── state/                   # Pytree primitives + state-collection taxonomy
│       ├── testing/                 # Mock collectives and Python test helpers
│       ├── utils/                   # Shared helpers
│       ├── cli/                     # tessera-mlir, tessera-prof, tessera-translate, tessera-runtime-smoke,
│       │                            #   tessera-surface-audit, tessera-claim-lint, tessera-e2e-coverage,
│       │                            #   tessera-apple-target-map, tessera-gpu-target-map,
│       │                            #   tessera-operator-benchmarks-coverage, tessera-examples-audit,
│       │                            #   tessera-autotune
│       │
│       ├── runtime.py               # ctypes wrapper over runtime C ABI
│       ├── profiler.py              # Runtime profiler facade
│       ├── autotune.py              # Public autotuning facade
│       ├── telemetry.py             # Shared telemetry event/report schema
│       ├── diagnostics.py           # ErrorReporter, ShapeInferenceEngine, stable diagnostic codes
│       ├── debug.py                 # DebugTrace, GraphTrace, check_grad, check_determinism
│       ├── debug_env.py             # tessera-mlir diff console entry
│       ├── shape.py                 # Dim, Layout, Shape, ShapeConstraintGraph
│       ├── dtype.py                 # Canonical dtype enforcement + Dtype + result_type
│       ├── ops.pyi                  # Op-namespace type stub
│       │
│       ├── nn / autodiff sibling modules:
│       │   aot.py · checkpoint.py · custom.py · data.py · losses.py · memory.py ·
│       │   optim.py · quantization.py · rl.py · rng.py · sharding.py
│       │   (S2–S15 reference surface)
│       │
│       └── domain / advanced modules:
│           arch.py · bridge.py · collectives.py · complex.py ·
│           conformal_advanced.py · contour.py · control.py · distributions.py ·
│           elastic.py · energy.py · fault.py · flow.py · hyperbolic.py ·
│           riemann_surface.py · server.py · speculative.py ·
│           _apple_gpu_dispatch.py
│
├── tests/                           # Test suite (5,750 fast / ~6,530 total collected as of 2026-05-22)
│   ├── unit/                        # Python unit tests + CPU validation contracts
│   ├── integration/                 # Cross-component integration tests
│   ├── regression/                  # Locked-in past-bug regression cases
│   ├── tessera-ir/                  # MLIR/lit-style pipeline tests by phase
│   ├── kernel_tests/                # System-level kernel + roofline tests
│   ├── performance/                 # Deterministic roofline / proxy perf contracts
│   ├── tessera_numerical_validation/ # Reference-vs-runtime numerical comparisons
│   ├── tessera_tests/               # Shared test utilities and fixtures
│   ├── README.md                    # Test-suite quick-start + memory/performance pointer
│   ├── MEMORY_AND_PERFORMANCE.md    # Per-suite RAM/wall-clock contract
│   └── COMPILER_TEST_PLAN.md        # Tier 0–4 CI matrix, layering by IR stage
│
├── docs/                            # All documentation
│   ├── architecture/                # System design docs (target IR usage, kernel-compilation stages)
│   ├── api/                         # API reference index
│   ├── audit/                       # Audit reports + generated/ dashboards
│   │   └── generated/               # Auto-regenerated dashboards (drift-gated)
│   ├── benchmarks/                  # Benchmark-related documentation
│   ├── context/                     # Auto-generated context docs (graphify, etc.)
│   ├── guides/                      # 11 user-facing how-to guides
│   ├── operations/                  # Canonical operation catalog (Tessera_Standard_Operations.md)
│   ├── programming_guide/           # 11-chapter user manual
│   ├── reference/                   # Canonical references (tensor attributes, migration, API)
│   ├── spec/                        # IR, ABI, language, autodiff, compiler specs (14 files)
│   ├── status/                      # Milestone status docs (M6 EBM, M7 visual complex, etc.)
│   └── tutorials/                   # Flash Attention, performance tuning
│
├── examples/                        # Runnable usage examples
│   ├── getting_started/             # basic_tensor_ops, canonical @tessera.jit
│   ├── compiler/                    # Compiler-pipeline examples
│   ├── conformance/                 # Conformance-suite examples
│   ├── integration/                 # End-to-end integration scenarios
│   ├── optimization/                # Performance / autotune examples
│   └── advanced/                    # Research-shaped ports — see "Advanced examples" below
│
├── benchmarks/                      # Performance benchmarks, telemetry gates, baselines
│   ├── baselines/                   # Deterministic CPU smoke perf-gate baselines
│   ├── common/                      # Shared correctness/compiler-contract/artifact-schema libs
│   ├── Tessera_Operator_Benchmarks/ # CMake-built C++ operator micro-bench suite
│   ├── Tessera_SuperBench/          # Whole-model benchmark suite (~30 min, marked slow)
│   ├── DeepScholar-Bench/           # DeepScholar reference model port (CPU smoke)
│   ├── apple_cpu/                   # benchmark_execution_kind.py — execution_kind axis probe
│   ├── apple_gpu/                   # benchmark_fusion.py — matmul→softmax/gelu/rmsnorm fusion
│   ├── linalg/                      # cholesky / qr / svd / tri_solve reference suite
│   ├── spectral/                    # FFT correctness sentinel + runtime bench
│   ├── clifford_core/               # GA library workload (M5/M7) — tested via pytest contract
│   ├── corrdiff/                    # Diffusion grid core — standalone benchmark harness
│   ├── energy_core/                 # EBM library workload (M6) — tested via pytest contract
│   ├── grid_ai_core/                # Gridded-AI library workload — has matching MLIR fixture
│   ├── visual_complex_core/         # M7 cross-lane GA × EBM library — tested via pytest contract
│   ├── benchmark_gemm.py            # GEMM sweep — latency_ms, tflops, memory_bw
│   ├── benchmark_attention.py       # Attention sweep — tokens/sec, MFU
│   ├── benchmark_collective.py      # 2–128 ranks bus-bandwidth
│   ├── run_all.py                   # Unified benchmark orchestrator
│   └── perf_gate.py                 # Telemetry baseline gate
│
├── tools/                           # Developer tooling
│   ├── profiler/                    # Canonical tprof profiler runtime, CLI, reports
│   ├── roofline_tools/              # Roofline ingestion/report helpers (cli_v2)
│   ├── tessera-opt/                 # MLIR opt-style driver — 5 dialects + 70+ passes + 6 pipelines
│   ├── tessera-translate/           # Python export CLI + C++ MLIR/SPIR-V translation driver
│   └── CLI/                         # CLI starter snapshots
│
├── scripts/                         # Build, validation, and CI utility scripts
│   ├── validate.sh                  # CPU-only validation spine
│   ├── build.sh                     # Top-level build wrapper
│   ├── test.sh                      # Local test driver
│   ├── check_versions.py            # CMake/Python/runtime header version drift check
│   ├── format.sh                    # Code formatting
│   ├── lint_docs.sh / .py           # Documentation lint + Python helper
│   ├── mypy_ratchet.sh              # Strict-typing ratchet (now retired but kept)
│   ├── release_gate.py              # Per-target (e.g. apple_gpu) release-gate orchestrator
│   ├── run_sanitizers.sh            # ASan/TSan/UBSan driver
│   ├── generate_context_outputs.py  # Generate docs/context/ artifacts
│   ├── generate_mfma_table.py       # Emit src/.../mfma_table.inc from Python source-of-truth
│   ├── probe_collective_libs.py     # ctypes-probe NCCL/RCCL ≥ 2.22 symbols
│   ├── validate_nvcc_compile.py     # Compile-check PTX patterns against installed nvcc
│   ├── validate_hipcc_compile.py    # Compile-check AMDGCN intrinsics against installed hipcc
│   └── validation_env.py            # Validation environment helpers
│
├── cmake/                           # CMake find-modules + helpers (incl. TesseraToolchainPins.cmake)
├── research/                        # Experimental research prototypes outside production build
│   ├── pddl_instruct/               # PDDL instruction experiments
│   └── sandbox_compilers/           # Sample compiler experiments
│
└── archive/                         # Superseded / pre-production work (not built)
    ├── benchmarks/                  # Archived benchmark concepts
    ├── docs/                        # Pre-canonical documentation
    ├── examples/                    # Archived examples and source drops
    ├── research/                    # Archived research prototypes
    ├── src/                         # Superseded production-source snapshots
    └── tests/                       # Historical test suites
```

Local/generated directories such as `build/`, `build-asan/`, `build-tsan/`,
`build-ubsan/`, `build-nvidia/`, `build-rocm/`, `build-rocm-plan/`, `.venv/`,
`__pycache__/`, `.pytest_cache/`, `graphify-out/`, `results/`, and per-package
`*.egg-info/` may exist in a developer checkout but are not part of the source
layout. They should be gitignored.

---

## Backends (src/compiler/codegen/)

| Backend folder                           | Target                              | Status |
|------------------------------------------|-------------------------------------|--------|
| `tessera_x86_backend/`                   | x86 AMX BF16 + AVX512 GEMM          | ✅ End-to-end (Phase 2) |
| `tessera_gpu_backend_NVIDIA/`            | NVIDIA WGMMA/TMA (Hopper/Blackwell) | IR ready; execution Phase G |
| `Tessera_ROCM_Backend/`                  | AMD ROCm MFMA (gfx90a/94x/950/1100) | IR + per-arch MFMA table; execution Phase H |
| `Tessera_RubinCPX_Backend/`              | NV Rubin CPX (`tessera.target.cpx`) | ✅ 4 passes + `tessera-cpx-opt` |
| `Tessera_TPU_Backend/`                   | TPU StableHLO + Shardy export       | Built; runtime gated |
| `Tessera_Cerebras_backend/`              | Cerebras WSE-3 fabric (Phase 7)     | ~487 LOC, real impl |
| `Tessera_Metalium_Backend/`              | Tenstorrent Metalium (Phase 7)      | ~550 LOC, `tessera-lower-to-metalium` |
| `Tessera_Apple_Backend/`                 | Apple Silicon CPU + GPU             | ✅ Operational (Phases 8.2 → 8.4.7) |

---

## Advanced examples (examples/advanced/)

| Folder | Theme |
|---|---|
| `power_retention/` | PowerAttention port (HIP, WGMMA, autotune) |
| `kv_cache_serving/` | KV cache + paging on Apple/x86 |
| `long_context_attention/` | Long-context attention variants |
| `mla/` | Multi-head Latent Attention (DeepSeek-style) |
| `gumiho/` | Gumiho (ICML'25) hybrid speculative decoding on the Apple backend |
| `Diffusion_LLM/` | Diffusion-LLM hybrid surface |
| `Fast_dLLM_v2/` | Fast diffusion-LM port v2 |
| `Jet_nemotron/` | Jet-Nemotron port |
| `Nemotron_Nano_12B_v2/` | Nemotron Nano 12B port v2 |
| `rlvr_reasoning_suite/` | RLVR reasoning suite (PPO/GRPO/CISPO) |
| `Tessera_Empirical_Software_Agent/` | Empirical software-agent example |

---

## Component Versioning Table

The table below records notable promotions into canonical `src/` and why.
Superseded versions may remain under `archive/<area>/` when they are kept for
reference.

| Canonical path               | Promoted from                                    | Rationale |
|------------------------------|--------------------------------------------------|-----------|
| `src/compiler/ir`            | `src/ir`                                         | Core Tessera IR grouped under compiler infrastructure |
| `src/compiler/mlir`          | `src/mlir`                                       | MLIR dialect integration grouped under compiler infrastructure |
| `src/compiler/programming_model` | `src/tessera_pm_v1_1_memory_parallel`        | Programming-model IR grouped under compiler infrastructure |
| `src/compiler/tessera_neighbors` | `src/tessera-neighbors`                      | Neighbor topology/halo lowering grouped under compiler infrastructure |
| `src/compiler/tile_opt_fa4`  | `src/Tile_Optimization_FA4 /…/v1_3`              | FA-4 tile optimization grouped under compiler infrastructure |
| `src/compiler/codegen/Tessera_RubinCPX_Backend` | `src/Tessera_RubinCPX_Compiler`     | NV Rubin CPX target grouped with compiler codegen backends |
| `src/compiler/codegen/Tessera_Apple_Backend` | new (Phase 8)                       | Apple Silicon CPU + GPU backend (Accelerate, MPS, custom MSL) |
| `src/solvers/core`           | `src/src/solvers`                                | Sparse/RNG/nonlinear solver core grouped below solvers |
| `src/solvers/linalg`         | `src/tessera_linalg_solvers`                     | Linear algebra solver scaffold grouped under the solver stack |
| `src/solvers/scaling_resilience` | `src/tessera_scaling_resilience_v1_1`        | SR dialect and passes grouped below solvers |
| `src/solvers/spectral`       | `src/spectral`                                   | Spectral/FFT dialect grouped below solvers |
| `src/solvers/tpp`            | `src/tpp/tpp_v0_2`                               | TPP dialect, passes, and `tpp-space-time` pipeline grouped below solvers |
| `src/solvers/clifford`       | new (M5)                                         | Geometric Algebra / Clifford solver scaffolding |
| `src/solvers/ebm`            | new (M6)                                         | Energy-based-model solver scaffolding |
| `src/Operators`              | new                                              | Placeholder for operator library work (Phase 4+) |
| `tests/kernel_tests`         | `tests/tessera_kernels_scaffold`                 | Original retained: system tests, roofline script, profile_ncu.sh |
| `examples/advanced/power_retention` | `examples/advanced/Power Retention/…/v0_9` | v0_9: HIP kernel, WGMMA, autotune, nlohmann_json integration |
| `research/pddl_instruct`     | `src/PDDL_Instruct/pddl_instruct_tessera_v1`     | Experimental research prototype moved out of the active `src/` surface |
| `research/sandbox_compilers` | `src/Sandbox_Toy_compilers/…`                    | Experimental sample compilers moved out of the active `src/` surface |
| `archive/src/tpp_old`        | `src/tpp_old`                                    | Superseded by canonical `src/solvers/tpp` |
| `archive/src/tile_opt_fa4_old` | `src/tile_opt_fa4_old`                         | Superseded by canonical `src/compiler/tile_opt_fa4` |
| `src/compiler/codegen/Tessera_TPU_Backend` | `src/compiler/codegen/Tessera_TPU_Backend_Starter_Advanced` | Advanced TPU backend promoted to single canonical TPU folder |
| `src/compiler/docs/pass_reference` | `src/compiler/passes`                      | Pass reference markdown grouped with compiler documentation |
| `archive/docs/pre_canonical/api` | `docs/api/Tessera_API_Vol*.md`               | Pre-canonical API volumes archived behind canonical API docs |
| `archive/docs/pre_canonical/model` | `docs/Tessera_Deep_Learning_Programming_Model.md` | Pre-canonical model guide archived (old API examples) |
| `docs/architecture/`         | `src/compiler/tessera_target_ir_doc3b.md`        | Architecture doc migrated out of src/ |
| `docs/tutorials/Flash_Attention_in_Tessera.md` | `docs/tutorials/Flash Attention_in_Tessera.md` | Space in filename removed |
| `docs/audit/`                | reorganized                                     | Canonical audit index plus themed audit areas; root historical filenames are redirect stubs where needed |
| `docs/audit/compiler/`       | new                                             | Compiler architecture, IR handoff, lowering, correctness, and spec-gap audit authority |
| `docs/audit/backend/`        | new                                             | Shared backend/runtime audit authority with platform-specialized Apple, NVIDIA, and ROCm subfolders |
| `docs/audit/coverage/`       | new                                             | Primitive/op coverage, partial-op uplift, KV-cache, examples, and coverage-dashboard authority |
| `docs/audit/domain/`         | new                                             | GA/EBM, attention, CorrDiff/SciML, sharding, and autodiff historical/domain audit material |
| `docs/audit/roadmap/`        | new                                             | Execution roadmap, deferred items, and sprint/crosscut planning material |
| `docs/audit/generated/`      | new                                             | Auto-regenerated dashboards (test coverage, classification, runtime ABI, TSOL coverage, docs freshness, effect lattice, apple target map, support table, e2e op coverage, benchmarks/research/tools/tests/examples status, primitive coverage) |
| `docs/guides/`               | new (Phases A–F)                                 | 11 user-facing how-to guides (~3,400 LOC) |
| `docs/programming_guide/`    | new (Phases A–G)                                 | 11-chapter user manual (Ch.1–11 + Appendix NVL72) |
| `docs/reference/`            | new                                              | tessera_tensor_attributes.md (normative dtype/attribute reference), migration guide, tessera-api-reference |
| `docs/status/`               | new                                              | Per-milestone status docs (M6 EBM, M7 visual complex, GA/EBM milestone, ...) |
| `docs/spec/AUTODIFF_SPEC.md` | new (Tier 2)                                     | Tape-based reverse-mode autodiff design |
| `python/tessera/dtype.py`    | new (Sprint A0/F)                                | Canonical dtype enforcement + Dtype + result_type + promotion lattice |
| `python/tessera/autodiff/`   | new (Tier 2)                                     | Tape + VJPs + JVPs + mixed-precision + rematerialize |
| `python/tessera/cache/`      | new (Phase B2/E + Sprint D)                      | KVCacheHandle + MemoryStateHandle persistent state ABI |
| `python/tessera/ebm/`        | new (M6)                                         | Energy-based-model Python primitives |
| `python/tessera/ga/`         | new (M5/M7)                                      | Geometric Algebra / Clifford Python primitives |
| `python/tessera/state/`      | new (S3)                                         | Pytree primitives + 8-collection state taxonomy |
| `python/tessera/rl.py`       | new (S11)                                        | PPO / GRPO / CISPO post-training policy losses |
| `python/tessera/rng.py`      | new (S4)                                         | RNGKey + 12 samplers (Philox-backed, deterministic) |
| `python/tessera/aot.py`      | new (S14)                                        | AOT export + persistent compilation cache |
| `python/tessera/custom.py`   | new (S13)                                        | `@custom_primitive` decorator + custom-call escape hatch |
| `python/tessera/data.py`     | new (S15)                                        | Dataset combinators + 5 tokenizers |
| `python/tessera/memory.py`   | new (S7 + Sprint D)                              | Titans/Atlas memory primitives (read/write/evict) + vmap-axis registry |
| `python/tessera/sharding.py` | new (S6 + Sprint D)                              | shard_map, collectives library, MemoryShardSpec |
| `python/tessera/optim.py`    | new (S10)                                        | 9 functional optimizers + 7 schedules + 7 grad transforms |
| `python/tessera/losses.py`   | new (S11)                                        | 21 losses (regression/classification/distribution/contrastive/diffusion/sequence) |
| `python/tessera/control.py`  | new (S5)                                         | scan, associative_scan, while_loop, vmap, pmap, vjp, jvp, remat |
| `python/tessera/compiler/audit modules` | new (multi-sprint)                    | 17+ audit modules: primitive_coverage, backend_manifest, docs_manifest, effect_audit, runtime_abi_audit, test_coverage_audit, coverage_classification, verifier_coverage, dialects_manifest, pipeline_registry, pass_metadata, diagnostic_codes, surface_manifest, benchmarks_manifest, research_manifest, tools_manifest, tests_manifest, examples_manifest, tsol_coverage, e2e_coverage, apple_target_map |
| `python/tessera/cli/surface_audit.py` | new                                     | `tessera-surface-audit` CLI for examples/benchmarks/research/tools/tests manifests |
| `python/tessera/cli/claim_lint.py` | new                                        | `tessera-claim-lint` CLI scans READMEs for overclaim language |
| `python/tessera/cli/apple_target_map.py` | new (Apple plan A)                   | `tessera-apple-target-map` Apple GPU/CPU coverage dashboard |
| `python/tessera/cli/gpu_target_map.py` | new                                      | `tessera-gpu-target-map` NVIDIA/ROCm coverage dashboard |
| `python/tessera/cli/e2e_coverage.py` | new                                        | `tessera-e2e-coverage` end-to-end op coverage |
| `python/tessera/cli/examples_audit.py` | new                                      | `tessera-examples-audit` examples surface health |
| `python/tessera/cli/operator_benchmarks_coverage.py` | new                        | Operator benchmark coverage CLI |
| `python/tessera/cli/runtime.py` | new                                            | `tessera-runtime-smoke` CLI for CPU runtime telemetry validation |
| `python/tessera/cli/translate.py` | new                                          | `tessera-translate` console-script entry (StableHLO/GGUF/SafeTensors + mlir passthrough) |
| `tools/tessera-translate/`   | new                                              | C++ MLIR/SPIR-V translation driver (`tessera-translate-mlir`) |
| `python/tessera/telemetry.py` | new                                             | Shared telemetry schema for profiler, autotune, benchmarks, runtime smoke |
| `benchmarks/perf_gate.py`    | new                                              | Telemetry baseline gate for deterministic CPU smoke reports |
| `benchmarks/baselines/`      | new                                              | Baseline inputs for benchmark/runtime telemetry gates |
| `benchmarks/apple_cpu/`      | new (Apple plan B)                               | execution_kind axis microbench |
| `benchmarks/apple_gpu/`      | new (Phase 8.4.6)                                | Fusion sweep (matmul→softmax/gelu/rmsnorm + tiled large-N) |
| `benchmarks/linalg/`         | new (Phase D1)                                   | cholesky / qr / svd / tri_solve reference suite |
| `benchmarks/spectral/`       | new (Phase A1/A3/B1)                             | FFT correctness sentinel + runtime bench |
| `benchmarks/clifford_core/`  | new (M5)                                         | GA library workload — pytest-gated |
| `benchmarks/corrdiff/`       | new (Sub-5)                                      | Diffusion grid core standalone harness |
| `benchmarks/energy_core/`    | new (M6)                                         | EBM library workload — pytest-gated |
| `benchmarks/grid_ai_core/`   | new (Sub-5)                                      | Gridded-AI library workload + matching MLIR fixture |
| `benchmarks/visual_complex_core/` | new (M7)                                    | M7 cross-lane GA × EBM library |
| `benchmarks/DeepScholar-Bench/` | new                                            | DeepScholar reference model port (CPU smoke) |
| `scripts/validate.sh`        | new                                              | CPU-only local validation spine across Python, runtime, profiler, collectives |
| `scripts/check_versions.py`  | new                                              | Version consistency check across CMake, Python, runtime headers |
| `scripts/release_gate.py`    | new                                              | Per-target release-gate orchestrator (e.g. --target=apple_gpu) |
| `scripts/generate_mfma_table.py` | new (H-2)                                    | Generates `mfma_table.inc` from Python `_MFMA_VARIANTS` source-of-truth |
| `scripts/validate_nvcc_compile.py` | new (G-6)                                  | Compile-check PTX patterns against installed nvcc |
| `scripts/validate_hipcc_compile.py` | new (H-6)                                 | Compile-check AMDGCN intrinsics against installed hipcc |
| `scripts/probe_collective_libs.py` | new (G-9/H-8)                              | ctypes-probe NCCL/RCCL ≥ 2.22 symbols |
| `scripts/run_sanitizers.sh`  | new                                              | ASan/TSan/UBSan sweep driver |
| `cmake/TesseraToolchainPins.cmake` | new (G-6/H-6)                              | CMake helpers pinning CUDA 13.2 U1 / ROCm 7.2.4 toolchains |
| `.github/workflows/validate.yml` | renamed from `cpu-validation.yml`            | CI workflow for the CPU validation spine |
| `.github/workflows/codeql.yml` / `codacy.yml` / `pylint.yml` / `python-quality.yml` | new | Security + Python-quality CI lanes |

---

## Archive Policy

Files under `archive/` are **not included in the production build or active
documentation lint**. Research prototypes under `research/` and legacy example
snapshots under `examples/advanced/` are likewise outside the production build
unless explicitly wired by CMake. They exist for reference, future graduation to
an active component, or deletion after review. New production work should land
in the canonical component folder, not beside an archived copy.

---

## Audit-as-Data Surface

The audit surface that grew out of the April 2026 reorganization is now the
authoritative status truth for many of these directories. Five **manifest
families** sit under `python/tessera/compiler/`:

- `examples_manifest.py` → `docs/audit/generated/examples_status.md`
- `benchmarks_manifest.py` → `docs/audit/generated/benchmarks_status.md`
- `research_manifest.py` → `docs/audit/generated/research_status.md`
- `tools_manifest.py` → `docs/audit/generated/tools_status.md`
- `tests_manifest.py` → `docs/audit/generated/tests_status.md`

Each is paired with a drift-gate test in `tests/unit/` and with the shared
`tessera-surface-audit` / `tessera-claim-lint` CLIs.

A second tier of registries adds **op-level / compiler-level audit** on top:

- `primitive_coverage.py` → support table + standalone primitive coverage dashboards
- `backend_manifest.py` → per-op × per-target × per-dtype kernel matrix
- `verifier_coverage.py` → MLIR op verifier classification
- `dialects_manifest.py` → dialect registration drift gate
- `pipeline_registry.py` → named pass-pipeline registry
- `pass_metadata.py` → per-pass diagnostic-code cross-reference
- `diagnostic_codes.py` → unified MLIR + Python diagnostic code registry
- `docs_manifest.py` → documentation freshness audit
- `effect_audit.py` → effect-lattice audit
- `runtime_abi_audit.py` → C ABI surface audit
- `test_coverage_audit.py` + `coverage_classification.py` → per-op test coverage
- `tsol_coverage.py` → TSOL canonical-op coverage filter
- `apple_target_map.py` → Apple GPU/CPU target map

When you wonder "is X registered / tested / shipped?", the answer is in one of
these registries — and a drift-gate test enforces that the rendered dashboard
matches the live data.

---

## Build System Status

The April 2026 reorganization paired with a CPU-only validation spine remains
the active surface; the May 2026 additions extend it with the audit-as-data
and S-series surfaces. Current active validation entry points:

- `scripts/validate.sh` runs version checks, Python unit tests, runtime smoke
  telemetry, benchmark smoke telemetry, standalone CPU runtime CMake/CTest,
  C++ profiler smoke build, collectives runtime compile check, and the audit
  lane (claim-lint, surface-audit, generated-dashboard drift).
- `.github/workflows/validate.yml` runs the CPU validation spine in CI.
- `scripts/check_versions.py` gates CMake, Python package, and runtime header
  versions on one project version value.
- `scripts/release_gate.py` (`--target=<accel>`) is the per-target release gate
  — `--target=apple_gpu` is the canonical Apple release blocker.
- `cmake/TesseraToolchainPins.cmake` pins NVIDIA → CUDA 13.2 U1 and AMD →
  ROCm 7.2.4; `scripts/validate_{nvcc,hipcc}_compile.py` and
  `scripts/probe_collective_libs.py` enforce the pins.
- Standalone runtime tests are split into separate CTest executables so each
  test file owns its own `main()`.

Remaining build-system work:

- Keep `src/CMakeLists.txt` aligned as compiler, solver, backend, and collective
  subtrees graduate from scaffold to production build targets.
- Continue validating MLIR dialect registration and TableGen targets for each
  component in a full monorepo build when LLVM/MLIR are available.
- Expand CI beyond the CPU spine once CUDA/HIP/NCCL execution paths are real and
  deterministic enough for automated validation (Phase G/H/I — see
  `docs/audit/backend/BACKEND_AUDIT.md`).
