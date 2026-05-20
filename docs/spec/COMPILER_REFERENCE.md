---
status: Normative
classification: Normative
last_updated: 2026-05-06
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
Python API + textual DSL frontend
(@tessera.jit, module/func/kernel syntax, Region[...], tessera.domain)
     |
     v  [python/tessera/compiler/graph_ir.py, frontend/parser.py]
Graph IR    (tessera.* ops, effects, shape/dtype/layout metadata, diagnostics)
     |
     v  [python/tessera/compiler/schedule_ir.py, src/transforms/lib/*.cpp]
Schedule IR (schedule.mesh.*, pipeline regions, stages, tiling structure)
     |
     v  [python/tessera/compiler/tile_ir.py, tile lowering, backend lowering]
Tile IR     (tile.*, tessera.attn.* FA-4 ops, tessera.queue.* barriers)
     |
     v  [python/tessera/compiler/target_ir.py for CPU/NVIDIA/Apple/ROCm artifact paths]
Target IR   (backend-specific artifacts: x86, NVIDIA, ROCm, TPU, Apple, ...)
```

| Layer | Primary active files | Status |
|-------|----------------------|--------|
| Graph IR | `python/tessera/compiler/graph_ir.py`, `python/tessera/compiler/frontend/parser.py`, `src/compiler/ir/TesseraOps.td`, `src/compiler/ir/TesseraTiling.cpp` | implemented; Matmul TilingInterface conservative path implemented, Conv2D interface scaffolded |
| Schedule IR | `python/tessera/compiler/schedule_ir.py`, `src/compiler/programming_model/ir/ScheduleOps.cpp`, `src/compiler/programming_model/` | implemented / lit-testable |
| Tile IR | `python/tessera/compiler/tile_ir.py`, `src/compiler/tile_opt_fa4/`, `src/transforms/lib/TileIRLoweringPass.cpp` | implemented / lit-testable |
| Target IR | `python/tessera/compiler/target_ir.py`, `src/compiler/codegen/`, `python/tessera/compiler/matmul_pipeline.py` | CPU/NVIDIA/Apple/ROCm artifact paths implemented / lit-testable; other non-CPU targets vary by backend |

---

## 2. Named Pipelines And Target Paths

| Pipeline or target path | Source | Coverage | Status |
|-------------------------|--------|----------|--------|
| `tessera-lower-to-x86` | `src/transforms/lib/Passes.cpp`, `src/transforms/lib/TileToX86Pass.cpp` | `tests/tessera-ir/phase2/`, `tests/unit/test_lowering_chain.py` | implemented |
| `tessera-lower-to-gpu` | `src/transforms/lib/TileIRLoweringPass.cpp`, `src/compiler/tile_opt_fa4/`, `src/compiler/codegen/tessera_gpu_backend_NVIDIA/` | `tests/tessera-ir/phase3/`, GPU target unit tests | implemented / lit-testable |
| `tessera-lower-to-rocm` | `python/tessera/compiler/target_ir.py`, `src/compiler/codegen/Tessera_ROCM_Backend/`, `python/tessera/compiler/matmul_pipeline.py` | ROCm backend tests and target-contract tests | implemented / lit-testable / artifact-only |
| `tessera-lower-to-apple_cpu` | `python/tessera/compiler/target_ir.py`, `src/compiler/codegen/Tessera_Apple_Backend/` | `tests/unit/test_target_ir.py`, `tests/tessera-ir/phase8/apple_cpu_lowering.mlir` | implemented / lit-testable / artifact-only |
| `tessera-lower-to-apple_gpu` | `python/tessera/compiler/target_ir.py`, `src/compiler/codegen/Tessera_Apple_Backend/` | `tests/unit/test_target_ir.py`, `tests/tessera-ir/phase8/apple_gpu_lowering.mlir` | implemented / lit-testable / artifact-only |
| Metalium target artifacts | `src/compiler/codegen/Tessera_Metalium_Backend/`, `python/tessera/compiler/matmul_pipeline.py` | Metalium backend tests and target-contract tests | scaffolded / lit-testable |
| TPU StableHLO/Shardy target artifacts | `src/compiler/codegen/Tessera_TPU_Backend/`, `python/tessera/compiler/tpu_target.py` | `tests/unit/test_tpu_lowering.py`, `tests/tessera-ir/phase4/` | implemented / lit-testable |
| Solver and resilience pipelines | `src/solvers/` | `tests/unit/test_*solver*.py`, `tests/tessera-ir/phase5/` | implemented / lit-testable |
| Neighbor/halo/stencil passes | `src/compiler/tessera_neighbors/` | `tests/unit/test_neighbors_dialect.py`, `tests/tessera-ir/phase7/` | implemented / lit-testable |

Native execution support is separate from target artifact generation. The
Python lowering path uses object-backed Graph IR, Schedule IR, Tile IR, and
CPU/NVIDIA/Apple/ROCm Target IR layers with verifier checks before emitting inspectable
MLIR-like artifacts. Non-CPU native execution should only be called
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

TilingInterface support is intentionally classified separately from the
`TilingPass`: `TilingPass` is implemented/lit-testable for the supported static
matmul lowering path. `src/compiler/ir/TesseraTiling.cpp` now provides a
conservative ranked-tensor Matmul TilingInterface artifact path, while Conv2D
interface tiling remains explicitly scaffolded and must not be treated as
complete interface coverage.

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
| Phase 8 | implemented / lit-testable for Apple and ROCm artifacts; scaffolded / lit-testable for other targets | hardware-free Target IR and Apple/ROCm/Metalium contracts; `python/tessera/compiler/target_ir.py`, `tests/unit/test_target_ir.py`, `tests/tessera-ir/phase8/`, `src/compiler/codegen/` |

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
| `"rocm"`, `"amd"`, `"hip"` | `rocm` | implemented / lit-testable / artifact-only |
| `"metalium"`, `"tt_metalium"`, `"tt"` | `metalium` | scaffolded / artifact-only |
| `"apple_cpu"`, `"macos_cpu"`, `"m_series_cpu"` | `apple_cpu` | implemented / lit-testable / artifact-only |
| `"apple_gpu"` | `apple_gpu` | implemented / lit-testable / artifact-only |

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
| GA + EBM native Apple GPU health check | `python benchmarks/apple_gpu/benchmark_ga_ebm.py --ci` |
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
| Textual DSL parser/lowerer | `python/tessera/compiler/frontend/parser.py` |
| Python Schedule IR object model | `python/tessera/compiler/schedule_ir.py` |
| Python Tile IR object model | `python/tessera/compiler/tile_ir.py` |
| Python Apple/ROCm Target IR object model | `python/tessera/compiler/target_ir.py` |
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

## Constrained-lane Graph IR views (Phase B substrate)

The constrained math lanes (`@clifford_jit`, `@complex_jit`,
`@energy_jit`) keep their own narrower IR types — `CliffordIRProgram`,
`ComplexIRProgram`, `EnergyIRProgram`. Each is the source of truth
for execution dispatch.

For cross-cutting tooling (audit, explain, normalization passes),
each program type exposes a `to_graph_ir_view() -> GraphIRModule`
adapter. The view is a **projection**, not a normalization or
execution lowering.

### Contract (normative)

| Contract point | Required behavior | Rationale |
|---|---|---|
| **Op shape** | 1:1 projection: each op in the constrained IR's `ops` tuple maps to exactly one `IROp` in `view.functions[0].body`, in the same order. No reordering, merging, or splitting. | The view is for inspection, not optimization. Consumers (e.g., `support_table` audit) rely on the op count and order matching the constrained IR's `as_metadata()` output. |
| **Op naming** | Canonical names only. The view uses the op_name stored in the constrained `IROpCall` (e.g., `mobius`, `complex_mul`, `clifford_geometric_product`, `energy_quadratic`) — **never** backend-aliased names (`complex_mobius`, `complex_stereographic`). | The audit walker maintains `_M7_BACKEND_ALIASES` separately for backend-manifest lookup. Backend aliases are an artifact of C ABI symbol naming, not the public op vocabulary. |
| **Mutability** | Fresh deep copy of `IROp` / `GraphIRFunction` / `GraphIRModule` per `to_graph_ir_view()` call. Two successive calls must return distinct mutable objects. | Downstream passes (Phase C normalization) mutate ops in place. A shared reference would let one consumer corrupt another's view. |
| **Lane stamping** | `view.functions[0].lane` is **always** set to the source lane: `"clifford_jit"` / `"complex_jit"` / `"energy_jit"`. | Lane-aware passes (`tessera.compiler.lane_passes`) need to know which whitelist invariant holds for this function. |
| **Verification facts** | `view.functions[0].verification_facts` carries the lane's invariants — `{"ga_whitelisted"}` for Clifford, `{"holomorphic"}` for Complex (only when every op is in `HOLOMORPHIC_OPS`), `{"energy_whitelisted"}` for Energy. | Phase D optimization passes (deferred) will read these to choose lane-safe transforms. E.g., DBE consults `holomorphic` before eliminating `check_cauchy_riemann`. |
| **No execution** | The view is **never** the source of truth for runtime dispatch. Calling a constrained lane's compiled callable still consults the constrained IR + its lane-specific dispatcher; the view is read-only tooling. | Folding the view into execution would re-introduce the "constrained lanes lower into Graph IR" failure mode (see `docs/architecture/frontend_substrate_plan.md`). |

### Drift gate

`tests/unit/test_to_graph_ir_view_contract.py` exercises all three
adapters against the contract above:

  * 1:1 op-shape: `len(view.body) == len(program.ops)` + op names
    match in order.
  * Canonical names: no `complex_mobius` / `complex_stereographic`
    leakage in the view (rejected via a denylist).
  * Fresh deep copy: mutating one view's `body` doesn't affect a
    second view from the same program.
  * Lane stamping: each adapter's view carries the expected lane.
  * Verification facts: each adapter's view carries the expected
    invariant set.

Adding a new constrained lane requires both a `to_graph_ir_view()`
method and a corresponding row in the drift-gate's parametrize list.
