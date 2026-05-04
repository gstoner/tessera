---
status: Normative
classification: Normative
last_updated: 2026-05-04
---

# Tessera Compiler Reference
**Status:** Normative - grounded in `src/`, `python/tessera/`, and `tests/`  
**Audience:** compiler engineers, pass authors, and documentation maintainers

This is the first document to read when working on compiler structure. It
defines canonical IR layer names, maps active pass and backend locations, and
records status using the labels in `docs/README.md`.

---

## 1. IR Stack

Use the full names: Graph IR, Schedule IR, Tile IR, and Target IR.

```text
Python API  (@tessera.jit, Region[...], tessera.domain, index_launch)
     |
     v  [python/tessera/compiler/graph_ir.py]
Graph IR    (tessera.* ops, effects, shard and privilege attrs)
     |
     v  [src/transforms/lib/*.cpp]
Schedule IR (schedule.mesh.*, pipeline regions, tiling structure)
     |
     v  [tile lowering, backend lowering, solver/target passes]
Tile IR     (tile.*, tessera.attn.*, tessera.queue.*)
     |
     v
Target IR   (backend-specific artifacts: x86, NVIDIA, ROCm, TPU, Apple, ...)
```

| Layer | Primary active files | Status |
|-------|----------------------|--------|
| Graph IR | `python/tessera/compiler/graph_ir.py`, `src/compiler/ir/TesseraOps.td`, `src/compiler/ir/TesseraTiling.cpp` | implemented |
| Schedule IR | `src/compiler/programming_model/ir/ScheduleOps.cpp`, `src/compiler/programming_model/` | implemented / scaffolded |
| Tile IR | `src/compiler/tile_opt_fa4/`, `src/transforms/lib/TileIRLoweringPass.cpp` | implemented / lit-testable |
| Target IR | `src/compiler/codegen/`, `python/tessera/compiler/matmul_pipeline.py` | implemented for CPU artifacts; lit-testable or scaffolded for non-CPU targets unless backend docs say otherwise |

---

## 2. Named Pipelines And Target Paths

| Pipeline or target path | Source | Coverage | Status |
|-------------------------|--------|----------|--------|
| `tessera-lower-to-x86` | `src/transforms/lib/Passes.cpp`, `src/transforms/lib/TileToX86Pass.cpp` | `tests/tessera-ir/phase2/`, `tests/unit/test_lowering_chain.py` | implemented |
| `tessera-lower-to-gpu` | `src/transforms/lib/TileIRLoweringPass.cpp`, `src/compiler/tile_opt_fa4/`, `src/compiler/codegen/tessera_gpu_backend_NVIDIA/` | `tests/tessera-ir/phase3/`, GPU target unit tests | implemented / lit-testable |
| `tessera-lower-to-rocm` | `src/compiler/codegen/Tessera_ROCM_Backend/`, `python/tessera/compiler/matmul_pipeline.py` | ROCm backend tests and target-contract tests | lit-testable / artifact-only |
| `tessera-lower-to-apple_cpu` | `src/compiler/codegen/Tessera_Apple_Backend/` | `tests/tessera-ir/phase8/apple_cpu_lowering.mlir` | lit-testable / artifact-only |
| `tessera-lower-to-apple_gpu` | `src/compiler/codegen/Tessera_Apple_Backend/` | `tests/tessera-ir/phase8/apple_gpu_lowering.mlir` | lit-testable / artifact-only |
| Metalium target artifacts | `src/compiler/codegen/Tessera_Metalium_Backend/`, `python/tessera/compiler/matmul_pipeline.py` | Metalium backend tests and target-contract tests | scaffolded / lit-testable |
| TPU StableHLO/Shardy target artifacts | `src/compiler/codegen/Tessera_TPU_Backend/`, `python/tessera/compiler/tpu_target.py` | `tests/unit/test_tpu_lowering.py`, `tests/tessera-ir/phase4/` | implemented / lit-testable |
| Solver and resilience pipelines | `src/solvers/` | `tests/unit/test_*solver*.py`, `tests/tessera-ir/phase5/` | implemented / lit-testable |
| Neighbor/halo/stencil passes | `src/compiler/tessera_neighbors/` | `tests/unit/test_neighbors_dialect.py`, `tests/tessera-ir/phase7/` | implemented / lit-testable |

Native execution support is separate from target artifact generation. The
Python lowering path emits inspectable Graph/Schedule/Tile/Target IR artifacts
for several targets; non-CPU native execution should only be called
hardware-runtime when the backend-specific docs and tests say so.

---

## 3. Core Pass Registry

| Pass or component | Active file(s) | Purpose | Status |
|-------------------|----------------|---------|--------|
| `CanonicalizeTesseraIR` | `src/transforms/lib/CanonicalizeTesseraIR.cpp` | Graph IR simplification and fusion | implemented |
| `VerifyTesseraIR` | `src/transforms/lib/VerifyTesseraIR.cpp` | Graph IR verification | implemented |
| `EffectAnnotationPass` | `src/transforms/lib/EffectAnnotationPass.cpp` | Infer and validate `tessera.effect` attrs | implemented |
| `DistributionLoweringPass` | `src/transforms/lib/DistributionLoweringPass.cpp` | Lower shard attrs into schedule mesh structure | implemented |
| `TilingPass` | `src/transforms/lib/TilingPass.cpp` | Lower static matmul into tiled loop structure | implemented |
| `TileToX86Pass` | `src/transforms/lib/TileToX86Pass.cpp` | Emit x86 C ABI calls for supported tiled ops | implemented |
| `TileIRLoweringPass` | `src/transforms/lib/TileIRLoweringPass.cpp` | Lower schedule bodies into Tile IR and FA-4 ops | implemented / lit-testable |
| `WarpSpecializationPass` | `src/compiler/tile_opt_fa4/lib/WarpSpecializationPass.cpp` | Producer/consumer warp roles and queue barriers | implemented / lit-testable |
| `AsyncCopyLoweringPass` | `src/compiler/tile_opt_fa4/lib/AsyncCopyLoweringPass.cpp` | Lower async copies to TMA or cp.async-style artifacts | implemented / lit-testable |
| NVIDIA WGMMA/TMA/FA emitter passes | `src/compiler/codegen/tessera_gpu_backend_NVIDIA/` | NVIDIA Target IR lowering contracts | implemented / lit-testable |
| `GPUCollectiveInsertionPass` | `src/transforms/lib/GPUCollectiveInsertionPass.cpp` | Insert collective communication structure | implemented / scaffolded |
| `PipelineStageInsertionPass` | `src/transforms/lib/PipelineStageInsertionPass.cpp` | Insert pipeline stage structure | implemented / scaffolded |
| TPU target support | `python/tessera/compiler/tpu_target.py`, `src/compiler/codegen/Tessera_TPU_Backend/` | TPU target metadata and Shardy/StableHLO artifacts | implemented / lit-testable |
| Runtime diagnostics | `src/compiler/diagnostics/ErrorReporter.cpp`, `src/compiler/diagnostics/ShapeInferencePass.cpp`, `python/tessera/diagnostics.py` | Error reporting and shape diagnostics | implemented |

---

## 4. Phase And Status Map

| Phase | Current status | Active source and tests |
|-------|----------------|-------------------------|
| Phase 1 | implemented | Python frontend, Graph IR, constraints, effects, distributed shape APIs; `tests/unit/test_constraints.py`, `test_effects.py`, `test_graph_ir.py` |
| Phase 2 | implemented | x86/CPU lowering spine; `src/transforms/`, `tests/tessera-ir/phase2/` |
| Phase 3 | implemented / lit-testable | NVIDIA SM90+ target artifacts, FA-4 and queue dialects; `tests/tessera-ir/phase3/` |
| Phase 4 | implemented / scaffolded / lit-testable | collectives, TPU target, cyclic distribution, distributed planners; `tests/tessera-ir/phase4/`, distributed unit tests |
| Phase 5 | implemented / lit-testable | checkpointing, optimizer shard, Bayesian/autotune foundations, sparse/RNG/solver passes; `src/solvers/`, `tests/tessera-ir/phase5/` |
| Phase 6 | mock-runtime / hardware-runtime where C runtime is built | runtime C ABI, Python runtime wrapper, diagnostics, benchmark smoke; `src/runtime/`, `python/tessera/runtime.py`, `tests/tessera-ir/phase6/` |
| Phase 7 | implemented / lit-testable | neighbors dialect and halo/stencil passes; `src/compiler/tessera_neighbors/`, `tests/tessera-ir/phase7/` |
| Phase 8 | scaffolded / lit-testable | hardware-free Target IR and Apple/ROCm/Metalium contracts; `tests/tessera-ir/phase8/`, `src/compiler/codegen/` |

---

## 5. Target Selection

`@tessera.jit(target=...)` accepts `None`, `GPUTargetProfile`, and string target
aliases handled by `python/tessera/compiler/matmul_pipeline.py`.

| Input | Normalized target | Status |
|-------|-------------------|--------|
| `None`, `"cpu"` | `cpu` | implemented / mock-runtime |
| `"cuda"`, `"nvidia"`, `"gpu"`, `"sm90"`, `"hopper"` | `nvidia_sm90` | implemented / lit-testable |
| `GPUTargetProfile(ISA.SM_80)` | `nvidia_sm80` | artifact target |
| `GPUTargetProfile(ISA.SM_90)` | `nvidia_sm90` | implemented / lit-testable |
| `GPUTargetProfile(ISA.SM_100)` | `nvidia_sm100` | artifact target |
| `"rocm"`, `"amd"`, `"hip"` | `rocm` | lit-testable / artifact-only |
| `"metalium"`, `"tt_metalium"`, `"tt"` | `metalium` | scaffolded / artifact-only |
| `"apple_cpu"`, `"macos_cpu"`, `"m_series_cpu"` | `apple_cpu` | lit-testable / artifact-only |
| `"apple_gpu"` | `apple_gpu` | lit-testable / artifact-only |

---

## 6. Architecture Decisions

These decisions are closed unless a new normative spec supersedes them.

1. The Python frontend is permanent. MLIR/C++ handles performance-critical
   lowering.
2. Tessera uses static compiler analysis, not a tracing JIT tier.
3. `Region[...]` is a type annotation, not a runtime wrapper.
4. Domains and distributions are separate objects.
5. Effects are inferred. User code declares determinism policy, not arbitrary
   effect labels.
6. Mock collectives use in-process test scaffolding so distributed API tests do
   not require NCCL/MPI.
7. `DistributedArray` carries logical shape and `ShardSpec`; it is not an
   alias for `numpy.ndarray`.
8. Native runtime support and compiler artifact support are distinct claims.

---

## 7. Test And Build Map

| Need | Command |
|------|---------|
| Python unit tests | `pytest tests/unit -v` |
| Optional performance tests | `TESSERA_RUN_PERFORMANCE_TESTS=1 scripts/test.sh` |
| Full CPU validation spine | `scripts/validate.sh` |
| MLIR lit tests | `python -m lit tests/tessera-ir/ -v` |
| C++/MLIR build | `scripts/build.sh` |
| Documentation lint | `scripts/lint_docs.sh` |

`pyproject.toml` configures `tests/unit` as the default pytest suite.
`tests/tessera-ir/` contains lit fixtures grouped by phase/path.

---

## 8. Key Source File Index

| What you need | Where to look |
|---------------|---------------|
| Python `@jit` implementation | `python/tessera/compiler/jit.py` |
| Python Graph IR builder | `python/tessera/compiler/graph_ir.py` |
| CPU and target artifact planner | `python/tessera/compiler/matmul_pipeline.py` |
| Public op catalog | `python/tessera/compiler/op_catalog.py` |
| GPU target profile | `python/tessera/compiler/gpu_target.py` |
| TPU target profile | `python/tessera/compiler/tpu_target.py` |
| Distributed domain/distribution API | `python/tessera/distributed/domain.py` |
| Mock collective testing | `python/tessera/testing/mock_collective.py` |
| Graph IR ODS | `src/compiler/ir/TesseraOps.td` |
| Transform passes | `src/transforms/lib/` |
| FA-4 Attn dialect | `src/compiler/tile_opt_fa4/dialects/tessera_attn/Attn.td` |
| Queue dialect | `src/compiler/tile_opt_fa4/dialects/tessera_queue/Queue.td` |
| x86 backend | `src/compiler/codegen/tessera_x86_backend/` |
| NVIDIA backend | `src/compiler/codegen/tessera_gpu_backend_NVIDIA/` |
| ROCm backend | `src/compiler/codegen/Tessera_ROCM_Backend/` |
| TPU backend | `src/compiler/codegen/Tessera_TPU_Backend/` |
| Apple backend | `src/compiler/codegen/Tessera_Apple_Backend/` |
| Metalium backend | `src/compiler/codegen/Tessera_Metalium_Backend/` |
| Collectives | `src/collectives/` |
| Runtime C ABI | `src/runtime/include/tessera/tessera_runtime.h` |
| Python runtime wrapper | `python/tessera/runtime.py` |
| Solvers and resilience | `src/solvers/` |
| Neighbor dialect | `src/compiler/tessera_neighbors/` |
