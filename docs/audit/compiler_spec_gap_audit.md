---
status: Informative
classification: Audit
last_updated: 2026-05-04
---

# Compiler Spec Gap Audit

This audit compares the active compiler implementation against the normative
spec set in `docs/spec/`. It excludes `src/archive/`, `docs/archive/`,
`research/`, and legacy example snapshots unless an active spec explicitly
points at them.

The companion matrix is `docs/audit/compiler_spec_gap_matrix.md`.

## Executive Summary

Tessera has a real compiler spine in place: Python frontend objects, Graph IR
emission, ODS-backed Graph IR ops, core transform passes, x86/CPU lowering, FA-4
Tile IR scaffolding, target-artifact backends, runtime ABI headers, CPU runtime
execution, and broad Python/lit coverage all exist in the active tree.

The main gap is not absence of a compiler; it is that several specs still read
as if phase-level plans are equivalent to production behavior. Some areas are
implemented and tested, some are lit-testable target artifacts, and some remain
intentional stubs. The highest-risk drift is around:

- Python API op status: several ops described as Phase 1 stubs now have numpy
  references, while distributed collectives remain single-rank/no-op.
- Runtime ABI status: the C ABI and CPU backend are implemented, and CUDA/HIP
  backend files exist behind compile flags, so `RUNTIME_ABI_SPEC.md` overstates
  that production wiring is only planned.
- Tile/Target IR claims: FA-4, queue, TMEM, NVIDIA, ROCm, Apple, TPU, Metalium,
  Cerebras, and Rubin CPX vary from ODS-backed and lit-testable to scaffolded or
  stubbed. Specs should avoid one status label for all target IR.
- Conformance phase claims: `CONFORMANCE.md` says Phases 1-3 complete and 4-6
  planned, but `COMPILER_REFERENCE.md`, active tests, and source show Phase 4-8
  components in mixed implemented/scaffolded/lit-testable states.
- Source-only behavior: Graph IR ODS and Python op catalog include newer ops
  such as spectral ops, RMSNorm, KV cache, architecture search, and all-to-all
  that are not fully reflected in the canonical spec files.

## Built, Scaffolded, And Missing

| Area | Current implementation evidence | Audit status | Action |
|------|---------------------------------|--------------|--------|
| Python frontend and public namespace | `python/tessera/__init__.py`, `python/tessera/compiler/jit.py`, `python/tessera/compiler/constraints.py`, `python/tessera/compiler/effects.py`, `tests/unit/test_constraints.py`, `tests/unit/test_effects.py` | implemented | Keep specs current with actual exported symbols. |
| Python Graph IR builder | `python/tessera/compiler/graph_ir.py`, `python/tessera/compiler/op_catalog.py`, `tests/unit/test_graph_ir.py`, `tests/unit/test_ir_spine_contract.py` | implemented | Update Graph/API specs for newer op catalog entries. |
| Graph IR ODS | `src/compiler/ir/TesseraOps.td`, `src/compiler/ir/TesseraOps.cpp`, `src/compiler/ir/TesseraTiling.cpp` | implemented / scaffolded | Mark tiling interface methods as scaffolded until TODOs are resolved. |
| Core transforms and x86 lowering | `src/transforms/lib/*.cpp`, `src/transforms/lib/Passes.cpp`, `tests/tessera-ir/phase2/` | implemented / lit-testable | Keep pipeline order anchored to `Passes.cpp`. |
| FA-4 Tile IR and queue dialects | `src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td`, `src/compiler/tile_opt_fa4/include/tessera/Dialect/Queue/Queue.td`, `tests/tessera-ir/phase3/`, `src/compiler/tile_opt_fa4/test/` | lit-testable | Clarify which ops have verifiers/lowering vs schema only. |
| TMEM / tcgen05 path | `src/compiler/tile_opt_fa4/test/tmem/tcgen05_ptx_body.mlir`, `src/compiler/tile_opt_fa4/lib/Conversion/TesseraTileToPTX/LowerTileToPTX.cpp` | stubbed / lit-testable | Keep Blackwell TMEM as artifact/stub until real PTX body and operands land. |
| NVIDIA backend | `src/compiler/codegen/tessera_gpu_backend_NVIDIA/`, `tests/tessera-ir/phase3/`, `src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia/` | lit-testable, with placeholder kernels | Avoid claiming native hardware-runtime for WGMMA placeholder paths. |
| ROCm backend | `src/compiler/codegen/Tessera_ROCM_Backend/`, ROCm lit tests, `runtime/hip/loader.cpp` stubs | lit-testable / scaffolded | Separate Target IR/ROCDL artifact support from HIP loader runtime. |
| TPU backend | `src/compiler/codegen/Tessera_TPU_Backend/`, `python/tessera/compiler/tpu_target.py`, `tests/tessera-ir/phase4/`, `tests/unit/test_tpu_lowering.py` | implemented / lit-testable | Note PJRT execute remains stubbed. |
| Apple backend | `src/compiler/codegen/Tessera_Apple_Backend/`, `tests/tessera-ir/phase8/apple_*` | lit-testable / artifact-only | Document as target artifact support. |
| Metalium backend | `src/compiler/codegen/Tessera_Metalium_Backend/`, Metalium tests, TODO in `TileToMetalium.cpp` | scaffolded / lit-testable | Keep matmul lowering incomplete in backlog. |
| Cerebras backend | `src/compiler/codegen/Tessera_Cerebras_backend/`, `src/compiler/codegen/Tessera_Cerebras_backend/tests/cerebras_lowering.test` | stubbed / scaffolded | Do not include in normative Target IR unless intentionally promoted. |
| Rubin CPX backend | `src/compiler/codegen/Tessera_RubinCPX_Backend/`, phase8 tests, CPX ODS | scaffolded / lit-testable | Add spec coverage or explicitly mark out-of-scope for current target spec. |
| Runtime C ABI and CPU backend | `src/runtime/include/tessera/*.h`, `src/runtime/src/tessera_runtime.cpp`, `src/runtime/src/backend/cpu_backend.cpp`, `tests/unit/test_runtime_abi.py` | implemented / mock-runtime | Update `RUNTIME_ABI_SPEC.md` phase table to reflect current CPU implementation. |
| CUDA/HIP runtime backends | `src/runtime/src/backend/cuda_backend.cpp`, `src/runtime/src/backend/hip_backend.cpp` | hardware-runtime when built, otherwise unavailable | Record compile-flag and device requirements explicitly. |
| Collectives and distributed planner | `src/collectives/`, `src/transforms/lib/GPUCollectiveInsertionPass.cpp`, `python/tessera/testing/mock_collective.py`, unit tests | implemented / scaffolded | Distinguish mock collectives from native NCCL/MPI execution. |
| Shape system | `python/tessera/shape.py`, `src/compiler/diagnostics/ShapeInferencePass.cpp`, shape/unit tests | implemented / scaffolded | Clarify which checks are Python-level vs MLIR verifier-level. |

## Spec Needs Update

| Spec | Issue | Recommended update |
|------|-------|--------------------|
| `docs/spec/PYTHON_API_SPEC.md` | The operations table labels `conv2d`, `all_reduce`, `reduce_scatter`, and `all_gather` as stubs. `conv2d` now has a reference NHWC implementation, while collectives are single-rank no-ops. | Split status into `numpy-reference`, `single-rank-mock`, `artifact-only`, and `planned-runtime`. |
| `docs/spec/PYTHON_API_SPEC.md` | Public/source ops include `all_to_all`, RNG helpers, spectral ops, RMSNorm, KV cache helpers, and an operator registry that are not fully covered by the spec symbol index. | Refresh the public symbol/op index from `python/tessera/__init__.py` and `python/tessera/compiler/op_catalog.py`. |
| `docs/spec/GRAPH_IR_SPEC.md` | Graph IR spec lists the original core op catalog, but ODS now includes layer norm, dropout, collectives, spectral ops, RMSNorm safe, KV cache, ring/cache page, and architecture-search ops. | Add a generated/current op catalog appendix or mark non-core ODS ops as extensions. |
| `docs/spec/LOWERING_PIPELINE_SPEC.md` | The pass list is close to source, but target/backend paths have mixed semantics: x86 lowering emits CPU calls, GPU lowering emits target artifacts and placeholder kernels in some paths. | Add a status column per pass: semantic transform, target artifact, stub emission, native runtime. |
| `docs/spec/TILE_IR.md` | `tshared.alloc` is specified, while active PM verifier code and tests use `tile.alloc_shared`; specs also mention `tile.barrier`, while source has mbarrier variants. | Normalize Tile IR op names or add an explicit compatibility note. |
| `docs/spec/TARGET_IR_SPEC.md` | Target IR focuses on Schedule, FA-4, queue, tile, and NVIDIA contracts, but current source also contains ROCm, Apple, TPU, Metalium, Cerebras, and Rubin CPX target dialects. | Split target spec into core/NVIDIA plus per-backend status appendices. |
| `docs/spec/RUNTIME_ABI_SPEC.md` | Phase table says CPU/CUDA/HIP production wiring is planned, but C ABI headers, CPU backend, CUDA/HIP backend sources, and Python wrapper exist. | Reclassify CPU as implemented/mock-runtime, CUDA/HIP as compile-flagged hardware-runtime where built and device-present. |
| `docs/spec/CONFORMANCE.md` | Phase status says Phases 4-6 planned, while active compiler reference and tests show Phase 4-8 partial implementations. | Replace phase-only conformance with profile/status labels from `docs/README.md`. |
| `docs/spec/MEMORY_MODEL_SPEC.md` | Memory model is stronger than current enforcement: atomics, device-wide fences, and deterministic mesh reduction are largely not verified in active compiler/runtime tests. | Mark enforcement requirements as planned unless a verifier/test evidence row exists. |
| `docs/spec/SHAPE_SYSTEM.md` | The spec covers Graph IR checker, schedule feasibility, Tile verifier, and runtime witnesses, but evidence is split across Python shape utilities, diagnostics pass, and tests. | Add an implementation map and identify MLIR verifier gaps explicitly. |

## Implementation Backlog

| Priority | Gap | Evidence | Recommended work |
|----------|-----|----------|------------------|
| P0 | Align conformance/spec status labels with active source. | `docs/spec/CONFORMANCE.md`, `docs/spec/COMPILER_REFERENCE.md`, `docs/README.md` disagree on phase status granularity. | Update conformance tables before adding new feature claims. |
| P0 | Refresh Graph/Python op catalogs. | `TesseraOps.td` and `op_catalog.py` contain more ops than `GRAPH_IR_SPEC.md` and parts of `PYTHON_API_SPEC.md`. | Generate or manually maintain a canonical op appendix. |
| P1 | Complete or explicitly gate TilingInterface TODOs. | `src/compiler/ir/TesseraTiling.cpp` has placeholder `failure()` paths. | Either implement tiling interface methods or mark interface support scaffolded. |
| P1 | Normalize Tile IR op naming. | Spec says `tshared.alloc`; PM verifier/tests use `tile.alloc_shared`; mbarrier naming differs from generic barrier text. | Pick canonical names and add migration aliases if needed. |
| P1 | Complete Metalium lowering semantics. | `TileToMetalium.cpp` reports matmul lowering not implemented and has TODOs for operands/types. | Make pass fail loudly for unsupported ops or implement real lowering. |
| P1 | Replace TMEM/tcgen05 placeholder PTX. | `LowerTileToPTX.cpp` and FA-4 docs call current body schematic. | Add real Blackwell PTX operands/constraints and target gating. |
| P2 | Strengthen memory model enforcement tests. | Specs require scoped atomics/fences/happens-before; active evidence is mostly async-copy/mbarrier oriented. | Add lit tests for illegal memory patterns and deterministic reduction contracts. |
| P2 | Separate mock collectives from native collective runtime. | Mock helpers and insertion pass exist; native multi-rank runtime remains scaffolded. | Add status badges and native-runtime acceptance criteria. |
| P2 | Document source-only backends. | Cerebras/Rubin CPX and parts of Apple/ROCm appear in active code but are thin or artifact-only. | Either add per-backend spec appendices or mark experimental/scaffolded. |

## Recommended Order

1. Correct the spec/status drift in `CONFORMANCE.md`, `PYTHON_API_SPEC.md`,
   `GRAPH_IR_SPEC.md`, and `RUNTIME_ABI_SPEC.md`.
2. Add or update tests that lock the corrected public claims, especially the
   Graph/Python op catalog and runtime ABI status.
3. Decide whether source-only target backends are normative, scaffolded
   extensions, or out-of-scope for `docs/spec/`.
4. Implement high-value missing behavior: TilingInterface methods, Tile IR name
   normalization, TMEM real PTX, Metalium matmul lowering, and memory model
   verifier tests.

## Validation Notes

Local validation run on 2026-05-04:

| Command | Result | Notes |
|---------|--------|-------|
| `scripts/lint_docs.sh` | passed | Initial missing path reference was fixed before the final run. |
| `.venv/bin/pytest tests/unit -v` | passed | `1836 passed, 1 skipped`. |
| `.venv/bin/python -m lit tests/tessera-ir/ -v` | skipped | Local venv does not have the `lit` module installed. |
| Backend lit suites | skipped | Same missing `lit` module; backend lit config files are present under `src/compiler/codegen/*/test*`. |
| `scripts/validate.sh` | passed | Includes version check, Python unit tests, runtime telemetry smoke, benchmark telemetry smoke, standalone CPU runtime CMake/CTest, C++ profiler smoke, collectives compile check, and existing build-tree CTest discovery. |
