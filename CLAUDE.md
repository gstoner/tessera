# Tessera — Claude Code Project Context

> **Read this first.** This file gives you the full context you need to work on
> Tessera without re-deriving the architecture. It was generated after a
> comprehensive codebase audit (April 2026).

---

## What Tessera Is

Tessera is a **pre-alpha, tile-centric programming model and compiler** for
deep learning and HPC. The design goal is to make tiles, explicit memory
spaces, numerical precision, and parallelism **first-class IR objects** rather
than runtime heuristics.

Target hardware: NVIDIA (SM90 Hopper, SM100 Blackwell), AMD ROCm, Google TPU,
Cerebras WSE-3, Tenstorrent Metalium, and x86 AMX/AVX512.

---

## The Four-Layer IR Stack

```
Python API  (@jit, Region[...], tessera.domain, index_launch)
     │  [MISSING — Phase 1 builds this]
     ▼
Graph IR    (tessera dialect — TesseraOps.td, mathematical ops, effects, shapes)
     │  [lowering passes: MISSING]
     ▼
Schedule IR (schedule.* dialect — mesh regions, pipeline stages, optimizer sharding)
     │  [lowering passes: MISSING]
     ▼
Tile IR     (tile_opt_fa4 — warp specialization, TMEM, async copy, KV cache)
     │  [lowering passes: partial for attention; absent elsewhere]
     ▼
Target IR   (per-backend: NVRubinCPX, ROCm, TPU/StableHLO, Cerebras, x86)
             [x86 AMX/AVX512 backend EXISTS and works — use as CPU target]
```

Key source locations:

| Layer | Primary files |
|-------|--------------|
| Graph IR ops | `src/compiler/ir/TesseraOps.td`, `src/compiler/ir/TesseraTiling.cpp` |
| Graph IR passes | `src/transforms/lib/CanonicalizeTesseraIR.cpp` (4 patterns), `VerifyTesseraIR.cpp` |
| Schedule IR ODS | `src/compiler/programming_model/ir/schedule/ScheduleMeshPipelineOps.td` |
| Tile IR (FA-4) | `src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td`, `Queue.td` |
| x86 backend | `src/compiler/codegen/tessera_x86_backend/` (AMX BF16 + AVX512 GEMM — **works**) |
| RubinCPX backend | `src/compiler/codegen/Tessera_RubinCPX_Backend/` — `tessera.target.cpx` dialect; 4 passes; `tessera-cpx-opt` driver |
| Collectives IR | `src/collectives/include/tessera/Dialect/Collective/IR/CollectiveOps.td` |
| Scaling/Resilience | `src/solvers/scaling_resilience/lib/sr/passes/` (scaffold exists; bodies are stubs) |
| Neighbors/Halo | `src/compiler/tessera_neighbors/lib/` (HaloInferPass is stub) |
| Solver suite | `src/solvers/core/passes/` (pipeline wired; all 11 passes are TODO stubs) |
| Linalg solver | `src/solvers/linalg/lib/` (MixedPrecision + IterativeRefinement — TODO stubs) |
| Runtime C ABI | `src/runtime/src/tessera_runtime.cpp` (270 lines); CPU backend (139 lines, real thread pool) |
| Main tool | `tools/tessera-opt/tessera-opt.cpp` (wires all dialects + passes) |
| Python package | `python/tessera/` (Phases 1–3 complete; Phases 4–6 missing) |

---

## What Actually Works Today

| Component | Status |
|-----------|--------|
| `src/compiler/ir/TesseraOps.td` — MatmulOp, Conv2DNHWCOp, FlashAttnOp + TilingInterface | **Real ODS + verifiers** |
| `CanonicalizeTesseraIR.cpp` — 4 fusion patterns | **Implemented** |
| `VerifyTesseraIR.cpp` | **Checks module version attr** |
| `tessera-opt` tool | **Fully wired — all dialects + passes registered** |
| x86 AMX BF16/AVX512 GEMM kernels | **Work** |
| Autotuner v1 (Python, SQLite cache, Hyperband) | **Works standalone** |
| tprof profiler + Perfetto export | **Works standalone** |
| Roofline tooling | **Works standalone** |
| Runtime C ABI (`tessera_runtime.cpp`) | **270 lines implemented; CPU backend has real thread pool** |
| Runtime CUDA/HIP backends | **30-line stubs — dispatch not wired** |
| RubinCPX dialect (`tessera.target.cpx`) | **ODS-generated via NVRubinCPX.td; NVFP4/NVFP6 types; 7 ops registered** |
| RubinCPX `tessera-cpx-pipeline` | **4 passes wired: fuse-video-ingest → partition-longcontext → vectorize-nvfp4 → lower-kv-transport** |
| `FuseVideoIngestPass` | **Implemented: BFS chain detection, video.decode→attn.prefill_fused fusion into region** |
| `PartitionLongContextPass` | **Implemented: CPX/Rubin classification, kv.export after prefill, kv.import at Rubin boundary** |
| `NVFP4VectorizePass` | **Implemented: bf16/f16 matmul detection, tessera.cast insertion, nvfp4_accel attribute** |
| `LowerKVTransportPass` | **Implemented: kv.export/import/prefetch → runtime func.call to tessera_kv_{export_pcie,export_nvlink,import,prefetch}** |
| `tessera-cpx-opt` driver | **Wired: NVRubinCPXDialect registered, all passes and both pipelines registered** |
| Python runtime wrapper (`python/tessera/runtime.py`) | **Does not exist** |
| FA-4 Attn+Queue ODS | **Defined with verifiers (Phase 3)** |
| Phase 1–3 Python frontend | **Complete — 252 tests green** |
| Collectives IR (AllToAll, Await, QoS) | **ODS defined; AllReduce/ReduceScatter/AllGather ops missing** |
| NCCL/RCCL adapters | **Disabled stubs in `Adapters.h` only** |
| Scaling/Resilience dialect + 4 passes | **Scaffold exists; pass bodies are minimal/empty** |
| Solver core pipeline (11 passes) | **Pipeline wired; every pass body is `// TODO` stub** |
| Linalg solver (MixedPrecision, IterativeRefinement) | **Structure wired; bodies are `// TODO`** |
| `Cyclic` distribution | **Implemented — round-robin `np.take` strided sharding** |
| Phase 4 Python modules | **Complete** — `distributed_planner`, `pipeline_planner`, `tpu_target`, `moe` all implemented |
| Phase 4 test suite | **127 tests green** — `tests/phase4/` (cyclic, NCCL adapters, TPU target, DistributedPlan, PipelinePlan, MoE, GPU collective insertion) |
| Phase 4 lit tests | **4 lit tests** — `tests/tessera-ir/phase4/` (gpu_collective_insertion, pipeline_stage_insertion, tpu_attention, tpu_shardy_export) |
| Phase 5 Python modules | **Complete** — `autotune_v2.py` (BayesianAutotuner + grid fallback + SQLite cache), `checkpoint.py` (@checkpoint_jit + CheckpointIRAnnotator), `solver_config.py` (SolverConfig, ZeROConfig, ResilienceConfig, DeploymentManifest, RNGStreamPlan) |
| Phase 5 C++ passes | **Implemented** — 11 solver core (SparseInspector, SparsePrecond, SparseSolverSpecialize, RNGLegalize, RNGStreamAssign, NewtonAutodiff, TrigInit, PeriodicHalo, ParamBatchPlan, ContinuationGuard, ImplicitLower) + 2 linalg (MixedPrecision, IterativeRefinement) + 3 SR (InsertRecompute, OptimizerShard, ResilienceRestart) |
| Phase 5 test suite | **176 tests green** — `tests/phase5/` (sparse inspector, RNG, checkpoint, optimizer shard, resilience, deployment manifest, Bayesian autotuner) |
| Phase 5 lit tests | **4 lit tests** — `tests/tessera-ir/phase5/` (insert_recompute, optimizer_shard, resilience_restart, rng_legalize) |
| Phase 6 test suite | **170 tests green** — `tests/phase6/` (runtime ABI, CPU backend, shape inference, error reporter, GEMM benchmark, MFMA coverage) |
| `python/tessera/diagnostics.py` | **Complete** — ErrorReporter, ShapeInferenceEngine, TesseraShapeError/TargetError/TypeError hierarchy, SourceLocation |
| Benchmark runners | **Complete** — `benchmarks/benchmark_gemm.py`, `benchmark_attention.py`, `benchmark_collective.py`, `run_all.py` |
| C++ backends | **Complete** — `cuda_backend.cpp` (real CUDA calls), `hip_backend.cpp` (real HIP/ROCm calls) |
| C++ diagnostics passes | **Complete** — `src/compiler/diagnostics/ErrorReporter.cpp`, `ShapeInferencePass.cpp` |
| Phase 6 lit tests | **2 lit tests** — `tests/tessera-ir/phase6/` (shape_inference, error_reporter) |

---

## Phase 1 Mission

**Build the Python frontend and Graph IR lowering that makes this test pass:**

```python
import tessera

# 1. Domain & Distribution
D    = tessera.domain.Rect((4, 128, 256))
dist = tessera.dist.Block(mesh_axes=("dp", "tp"))
X    = tessera.array.from_domain(D, dtype="bf16", distribution=dist)
assert X.shard_spec.mesh_axes == ("dp", "tp")

# 2. Region privileges
@tessera.jit
def step(W: tessera.Region["read"], X: tessera.Region["read"],
         Y: tessera.Region["write"]):
    Y[:] = tessera.ops.gemm(X, W)

# 3. Index launch
@tessera.kernel
def tp_gemm(A: tessera.f16[..., ...], B: tessera.f16[..., ...],
            C: tessera.mut_f32[..., ...]):
    C[:] = tessera.ops.gemm(A, B)

tessera.index_launch(axis="tp")(tp_gemm)(
    X.parts("tp"), X.parts("tp"), X.parts("tp")
)

# 4. Constraint propagation
@tessera.jit
def aligned_gemm(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    tessera.require(tessera.constraint.Divisible("K", 64))  # catches misalignment early
    return tessera.ops.gemm(A, B)

# 5. Effect inference
@tessera.jit(deterministic=True, seed=42)
def stable_forward(x: tessera.Tensor["B", "D"]):
    return tessera.ops.layer_norm(x)   # RNG is seeded and deterministic
```

---

## Phase 1 File Map

**New files to create (all in `python/tessera/`):**

```
python/tessera/
├── distributed/
│   ├── __init__.py          ← re-exports: Region, domain, dist, array, index_launch
│   ├── region.py            ← Region["read"/"write"/"reduce_sum"] type annotation
│   ├── domain.py            ← Rect domain, Block/Cyclic/Blocked distributions
│   ├── shard.py             ← ShardSpec, MeshSpec
│   ├── array.py             ← DistributedArray.from_domain(), .parts()
│   └── launch.py            ← index_launch(), @kernel decorator
│
├── compiler/
│   ├── __init__.py
│   ├── constraints.py       ← ConstraintSolver: Divisible, Range, Equal predicates
│   ├── effects.py           ← EffectLattice: random, io, memory, deterministic
│   ├── graph_ir.py          ← Python → Graph IR lowering (emit MLIR text/objects)
│   └── jit.py               ← @jit and @kernel decorators
│
└── __init__.py              ← expose: jit, kernel, Region, domain, dist, array,
                               index_launch, constraint, ops, Tensor, f16, mut_f32
```

**New test files:**

```
tests/phase1/
├── __init__.py
├── test_distributed_api.py  ← Domain, ShardSpec, Region, index_launch
├── test_constraints.py      ← ConstraintSolver predicates
├── test_effects.py          ← EffectLattice propagation
├── test_graph_ir.py         ← Python → Graph IR lowering round-trips
└── conftest.py              ← shared fixtures
```

**Files to modify:**

```
python/tessera/__init__.py   ← add: from . import distributed, compiler
                                     from .distributed import Region, domain, dist, array, index_launch
                                     from .compiler import jit, kernel, constraint
```

---

## Key Design Contracts

### Region Privileges

`Region[mode]` is a **type annotation** on function parameters — not a runtime
wrapper. It lowers to a `tessera.effect` attribute on Graph IR function arguments.

Valid modes: `"read"`, `"write"`, `"reduce_sum"`, `"reduce_max"`, `"reduce_min"`

The effect system uses these to:
- Check exclusive-write conflicts at compile time
- Allow safe overlap of read-only and reduce regions
- Gate recompute insertion (only recompute-safe if inputs are `read`)

```python
# Conflict → compile error
def bad(X: Region["write"], Y: Region["write"]):
    X[:] = Y[:]   # ERROR: two write regions on overlapping data

# Safe → compiler overlaps reduce across ranks
def good(X: Region["read"], G: Region["reduce_sum"]):
    G += ops.gemm(X, X.T)
```

### Domain & Distribution

`Rect(dims)` is the **logical iteration space**. `Block(mesh_axes=...)` is the
**placement strategy**. They are always separate objects — algorithm vs placement.

```python
D    = tessera.domain.Rect((B, S, D_model))   # shape only
dist = tessera.dist.Block(mesh_axes=("dp",))   # partition dim-0 over dp axis
X    = tessera.array.from_domain(D, dtype="bf16", distribution=dist)
# X.shard_spec → ShardSpec(partition=(0,), mesh_axes=("dp",))
# X.parts("dp") → list of per-rank shard slices
```

Supported distributions (Phase 1 scope):
- `Block(mesh_axes)` — contiguous block partition along first N dims
- `Cyclic(mesh_axes)` — round-robin (needed for load-balanced MoE, Phase 2)
- `Replicated()` — no partition (weights replicated across ranks)

### ConstraintSolver

Constraints are checked **at `@jit` decoration time** (not at call time).
They are structural properties of the function's type signature.

```python
tessera.constraint.Divisible("K", 64)   # K % 64 == 0
tessera.constraint.Range("S", 1, 8192)  # 1 <= S <= 8192
tessera.constraint.Equal("D_in", "D_out")  # D_in == D_out
```

Violation → `TesseraConstraintError` with the offending dimension path.

### Effect Lattice

Effects flow **upward** through the call graph. A function that calls an RNG
op is tagged `random`. A function that calls a `write` collective is tagged
`io`. A `@jit(deterministic=True)` block **forbids** `random` unless wrapped.

Lattice order (least → most permissive):
`pure` < `random` < `memory` < `io` < `top`

---

## GPU-Only Tier — Skip in Phase 1

Do **not** implement these in Phase 1. They have no CPU analogue and will be
handled in Phase 3 (GPU backend swap):

- `tessera.schedule.warp` role assignments (FA-4 warp specialization)
- `tessera.tile.mma.tcgen05` (Blackwell TMEM MMA)
- `tile.async_copy` / `tile.wait_async` stage indexing
- `tessera.schedule.policy "persistent"` (persistent CTA scheduling)
- `tessera.queue.{create, push, pop}` (tile queue dialect)
- `tcgen05.mma` PTX inline asm

Gate these at IR level with:
```python
if target_profile.isa >= ISA.SM_90:
    # emit warp specialization ops
```

---

## CPU Collective Mock

For Phase 1 multi-rank testing, use **thread-based fake ranks** instead of
NCCL. Each "rank" is a Python thread sharing an in-process memory space.

```python
# tests/phase1/conftest.py
from tessera.testing import MockRankGroup
ranks = MockRankGroup(n=4, mesh_axes={"dp": 4})
# all_reduce → barrier + sum across thread-local buffers
# reduce_scatter → barrier + slice
```

This lives in `python/tessera/testing/mock_collective.py` (create in Phase 1).

---

## Build & Test Commands

```bash
# Install in dev mode (from repo root)
pip install -e ".[dev]"

# Run all Phase 1 tests
pytest tests/phase1/ -v

# Run with coverage
pytest tests/phase1/ --cov=tessera.distributed --cov=tessera.compiler -v

# Run a single test file
pytest tests/phase1/test_distributed_api.py -v

# Run MLIR lit tests (requires tessera-opt built)
python -m lit tests/tessera-ir/ -v

# Type check
mypy python/tessera/distributed/ python/tessera/compiler/

# Build the C++/MLIR stack (only needed for IR emission tests)
mkdir -p build && cd build
cmake .. -DTESSERA_ENABLE_CUDA=OFF -DTESSERA_CPU_ONLY=ON
make -j$(nproc)
```

---

## The First Test — Start Here

`tests/phase1/test_distributed_api.py` — make this pass first:

```python
def test_domain_and_distribution_basics():
    D = tessera.domain.Rect((4, 128, 256))
    assert D.shape == (4, 128, 256)
    assert D.rank == 3

def test_block_distribution():
    dist = tessera.dist.Block(mesh_axes=("dp", "tp"))
    assert dist.mesh_axes == ("dp", "tp")

def test_from_domain_creates_shard_spec():
    D    = tessera.domain.Rect((4, 128, 256))
    dist = tessera.dist.Block(mesh_axes=("dp", "tp"))
    X    = tessera.array.from_domain(D, dtype="bf16", distribution=dist)
    assert X.shard_spec.mesh_axes == ("dp", "tp")
    assert X.dtype == "bf16"
    assert X.shape == (4, 128, 256)

def test_region_annotation_read():
    reg = tessera.Region["read"]
    assert reg.mode == "read"

def test_region_annotation_write():
    reg = tessera.Region["write"]
    assert reg.mode == "write"

def test_region_annotation_reduce():
    reg = tessera.Region["reduce_sum"]
    assert reg.mode == "reduce_sum"
    assert reg.op == "sum"
```

---

## Architecture Decisions — Don't Revisit These

1. **CPU-first, then GPU.** Phase 1 targets x86 AMX via the existing
   `tessera_x86_backend`. All GPU-specific IR ops are gated behind
   `target_profile.isa >= SM_90`. This is deliberate.

2. **Region is a type annotation, not a runtime wrapper.** `Region["read"]`
   returns a `RegionType` object that participates in Python's type annotation
   system (`__class_getitem__`). It does NOT wrap tensors at runtime.

3. **Domains and distributions are always separate.** `Rect` describes shape,
   `Block/Cyclic/Replicated` describes placement. Never merge them.

4. **ConstraintSolver runs at decoration time.** `@jit` inspects annotations
   and calls `ConstraintSolver.check(signature)` before any IR is emitted.
   This gives early, precise errors before execution.

5. **Effects are inferred, not declared.** The `EffectLattice` walks the IR
   and infers effects; programmers only declare `@jit(deterministic=True)` and
   `@jit(seed=N)` at the top level. The compiler infers everything else.

6. **Mock collectives use threads, not processes.** Phase 1 multi-rank tests
   run in-process with Python threads as fake ranks. This avoids NCCL/MPI
   dependency in the Python frontend test suite.

7. **`tessera.array` is not `numpy.ndarray`.** `DistributedArray` carries a
   `ShardSpec` and a logical shape. Physical storage is backend-dependent. In
   Phase 1, it is an eagerly-evaluated numpy array on CPU.

---

## Key Reference Files

| What you need | Where to look |
|---------------|--------------|
| Existing Graph IR op definitions | `src/ir/TesseraOps.td` |
| Existing Graph IR canonicalizations | `src/transforms/lib/CanonicalizeTesseraIR.cpp` |
| Schedule IR op definitions | `src/programming_model/ir/schedule/ScheduleMeshPipelineOps.td` |
| Mesh + pipeline design docs | `src/programming_model/docs/Parallelism_Constructs_v1_1.md` |
| Memory model docs | `src/programming_model/docs/Memory_Execution_Model_v1_1.md` |
| Effect system design | `src/programming_model/docs/Tessera_Programming_Model_v1_1_Plan_20250917_212640.md` §1.2 |
| Constraint design | Same doc §1.1 |
| Collective IR types | `src/collectives/include/tessera/Dialect/Collective/IR/CollectiveOps.td` |
| Collectives overlap design | `src/collectives/docs/Tessera_Collectives_Overlap_Design.md` |
| Programming guide Ch.4 (execution model) | `docs/programming_guide/Tessera_Programming_Guide_Chapter4_Execution_Model.md` |
| Programming guide Ch.10 (domain/dist) | `docs/programming_guide/Tessera_Programming_Guide_Chapter10_Portability.md` |
| x86 backend reference | `src/compiler/codegen/tessera_x86_backend/` |
| Autotune architecture | `src/compiler/autotuning/tessera/tools/autotune/docs/` |
| Full deep dive analysis (April 2026) | `docs/Tessera_Deep_Dive_Analysis.docx` (see also project folder) |

---

## Phase 1 Completion Criteria

- [ ] `pytest tests/phase1/ -v` — all tests green
- [ ] `mypy python/tessera/distributed/ python/tessera/compiler/` — no errors
- [ ] `tessera.domain.Rect`, `tessera.dist.Block`, `tessera.array.from_domain` importable
- [ ] `Region["read"]`, `Region["write"]`, `Region["reduce_sum"]` work as type annotations
- [ ] `@tessera.jit` decorator captures function, runs ConstraintSolver, emits Graph IR
- [ ] `@tessera.kernel` + `tessera.index_launch` fanout works with MockRankGroup(n=4)
- [ ] `ConstraintSolver` catches `Divisible`, `Range`, `Equal` violations at decoration time
- [ ] `EffectLattice` infers `random`/`memory`/`pure` through a 3-level call graph
- [ ] `@jit(deterministic=True)` rejects a function that calls an RNG op without seed
- [ ] One end-to-end: Python GEMM with ShardSpec → Graph IR text emitted → x86 execution

---

## Status as of April 2026

| Phase | Status | Key deliverables |
|-------|--------|-----------------|
| Phase 1 | ✅ **Complete** | Python frontend — Region, Domain, ShardSpec, DistributedArray, @jit, ConstraintSolver, EffectLattice, GraphIRBuilder |
| Phase 2 | ✅ **Complete** | C++ lowering chain — DistributionLoweringPass, EffectAnnotationPass, TilingPass, TileToX86Pass; `tessera-lower-to-x86` pipeline |
| Phase 3 | ✅ **Complete** | NVIDIA GPU backend — GPUTargetProfile, TileIRLoweringPass, WarpSpecializationPass, AsyncCopyLoweringPass, NVWGMMALoweringPass, NVTMADescriptorPass, NVFlashAttnKernelEmitter; `tessera-lower-to-gpu` pipeline; FA-4 Attn dialect (ScaledDotProduct, OnlineSoftmax, LseAccumulate, DropoutMask, CausalMask) |
| Cross-phase infra | ✅ **Added** | Core IR ODS with real verifiers (TesseraOps.td v2); Queue dialect QueueOps.cpp; `tessera-opt` fully wired; solver pipeline scaffold; SR pass scaffold; runtime C ABI implemented; tprof consolidated |
| RubinCPX backend | ✅ **Built** | `tessera.target.cpx` dialect (NVRubinCPX ODS — NVFP4/NVFP6 types, 7 ops, 6 attrs); 4 compiler passes with real implementations; `tessera-cpx-pipeline` + `tessera-cpx-context-pipeline` named pipelines; `tessera-cpx-opt` driver; wired into main build via `TESSERA_BUILD_RUBINCPX_BACKEND` option |
| Phase 4 | ✅ **Complete** | Distributed training: Cyclic distribution, NCCL/RCCL adapters, CollectiveInsertionPass, PipelineStageInsertionPass, TPU target + quantized dot, DistributedPlan, PipelinePlan, MoE helpers — 127 tests green |
| Phase 5 | ✅ **Complete** | Solver pass bodies (11 core + 2 linalg + 3 SR), Autotuner v2 (BayesianAutotuner / grid fallback / SQLite cache), checkpoint decorator, solver_config.py — 176 tests green |
| Phase 6 | ✅ **Complete** | Runtime Python wrapper (TesseraRuntime + _MockBackend), CUDA/HIP backends (real CUDA/HIP calls), ROCm MFMA coverage (gfx90a/gfx94x/gfx120x), benchmark runners (GEMM/Attention/Collective/run_all), diagnostics (ErrorReporter + ShapeInferenceEngine), C++ shape inference pass — 170 tests green |

---

## Remaining Tasks — Ordered by Phase

### Phase 4 — Distributed Training (START HERE)

#### Python — create these files

| File | What to implement |
|------|-------------------|
| `python/tessera/distributed/domain.py` | Implement `Cyclic.make_shard_spec()` — currently raises `NotImplementedError` |
| `python/tessera/distributed/array.py` | Add cyclic branch in `parts()` — strided round-robin slicing |
| `python/tessera/distributed/shard.py` | Add `cyclic: bool = False` field to `ShardSpec` |
| `python/tessera/distributed/moe.py` | `MoEConfig`, `route_tokens()`, `plan_all_to_all()` |
| `python/tessera/compiler/tpu_target.py` | `TPUTargetProfile` (mxu_tile=128, mesh_axes, `validate_matmul_dims`) |
| `python/tessera/compiler/distributed_planner.py` | `DistributedPlan`, `LayerSpec`, dp/tp/pp assignment, `to_mlir_attrs()` |
| `python/tessera/compiler/pipeline_planner.py` | `PipelinePlan`, 1F1B schedule builder, `schedule_steps()` |

#### C++ — create these files

| File | What to implement |
|------|-------------------|
| `src/collectives/include/tessera/Dialect/Collective/IR/CollectiveOps.td` | Add `AllReduceOp`, `ReduceScatterOp`, `AllGatherOp` (currently only `AllToAllOp` exists) |
| `src/collectives/include/tessera/Dialect/Collective/Runtime/Adapters.h` | Expand `NCCLAdapter`/`RCCLAdapter` with `all_reduce`, `reduce_scatter`, `all_gather`, `all_to_all` methods |
| `src/collectives/lib/NCCLAdapter.cpp` | Full mock NCCL adapter (real NCCL call sites + in-memory mock path) |
| `src/collectives/lib/RCCLAdapter.cpp` | RCCL equivalent — same interface as NCCL |
| `src/collectives/lib/ChunkPlanner.cpp` | `ChunkPlanner::chunk_bytes(Topology)` — NVLink=512KiB, PCIe=128KiB, RDMA=256KiB |
| `src/collectives/lib/CollectiveScheduler.cpp` | Credit-based link scheduler; `CollectiveScheduler::submit()` |
| `src/transforms/lib/GPUCollectiveInsertionPass.cpp` | Inserts `collective.reduce_scatter` at DP mesh boundaries; reads `tessera.effect = "memory"` |
| `src/transforms/lib/PipelineStageInsertionPass.cpp` | 1F1B micro-batch schedule; partitions `schedule.pipeline.region` stages across ranks |
| `src/compiler/codegen/Tessera_TPU_Backend/src/passes/TPUQuantizedDotPass.cpp` | `tessera.matmul` → `stablehlo.dot_general` with bf16/int8 precision attrs |
| `src/compiler/codegen/Tessera_TPU_Backend/src/passes/TesseraShardyExport.cpp` | Complete Shardy export — translate `tessera.shard` attrs to `sdy.tensor_sharding` per func arg |

#### Tests — create these directories and files

| Path | Tests |
|------|-------|
| `tests/phase4/__init__.py` | Empty |
| `tests/phase4/conftest.py` | 8-rank `MockRankGroup`, `NCCLMock` fixture |
| `tests/phase4/test_nccl_adapter.py` | all_reduce, reduce_scatter, all_gather mock correctness |
| `tests/phase4/test_chunk_planner.py` | 512KiB NVLink, 128KiB PCIe |
| `tests/phase4/test_cyclic_distribution.py` | `Cyclic.make_shard_spec`, `parts()` round-robin slicing |
| `tests/phase4/test_tpu_lowering.py` | TPUTargetProfile validation; flash_attn→stablehlo; shardy round-trip |
| `tests/phase4/test_distributed_plan.py` | `DistributedPlan` 4-layer MLP over dp=4, tp=2; anchoring IR test |
| `tests/phase4/test_gpu_collective_insertion.py` | `collective.reduce_scatter` at dp boundary in IR |
| `tests/phase4/test_pipeline_stage_insertion.py` | 1F1B 4 stages, 8 micro-batches |
| `tests/tessera-ir/phase4/gpu_collective_insertion.mlir` | Lit test |
| `tests/tessera-ir/phase4/pipeline_stage_insertion.mlir` | Lit test |
| `tests/tessera-ir/phase4/tpu_attention.mlir` | Lit test |
| `tests/tessera-ir/phase4/tpu_shardy_export.mlir` | Lit test |

---

### Phase 5 — Solver Implementations + Autotuner v2

#### C++ — implement these stub bodies

| File | Current state | What to implement |
|------|--------------|-------------------|
| `src/solvers/core/passes/SparseInspector.cpp` | `// TODO` stub | Walk tensor ops; tag fill-fraction < 5% as `tessera_solver.sparse_hint` |
| `src/solvers/core/passes/SparsePrecond.cpp` | `// TODO` stub | Select Jacobi/ILU/AMG preconditioner; attach `tessera_solver.precond` attr |
| `src/solvers/core/passes/SparseSolverSpecialize.cpp` | `// TODO` stub | Specialize sparse solver variant based on precond + sparsity |
| `src/solvers/core/passes/RNGLegalize.cpp` | `// TODO` stub | Legalize `tessera_rng.*` ops; assign stream IDs |
| `src/solvers/core/passes/RNGStreamAssign.cpp` | `// TODO` stub | Assign Philox/Threefry streams; `stream_id = global_seed * ranks + rank` |
| `src/solvers/core/passes/NewtonAutodiff.cpp` | `// TODO` stub | Generate VJP/JVP for `tessera_solver.implicit` ops |
| `src/solvers/core/passes/TrigInit.cpp` | `// TODO` stub | Initialize trig tables for spectral solver ops |
| `src/solvers/core/passes/PeriodicHalo.cpp` | `// TODO` stub | Infer periodic halo exchange patterns |
| `src/solvers/core/passes/ParamBatchPlan.cpp` | `// TODO` stub | Batch parameter sweeps; plan multi-config execution |
| `src/solvers/core/passes/ContinuationGuard.cpp` | `// TODO` stub | Insert guard ops at continuation/checkpoint boundaries |
| `src/solvers/core/passes/ImplicitLower.cpp` | `// TODO` stub | Lower `tessera_solver.implicit` to explicit residual + Newton loop |
| `src/solvers/linalg/lib/Passes/MixedPrecision.cpp` | `// TODO` body | Insert `tessera.quantize/dequantize` stubs around factor/solve boundaries |
| `src/solvers/linalg/lib/Passes/IterativeRefinement.cpp` | `// TODO` body | Wrap solve region with residual compute + convergence loop |
| `src/solvers/scaling_resilience/lib/sr/passes/InsertRecomputePass.cpp` | Tags only | Greedy memory-budget scan; insert `tessera_sr.checkpoint` + `recompute_hint` |
| `src/solvers/scaling_resilience/lib/sr/passes/OptimizerShardPass.cpp` | Stub | ZeRO-2: partition momentum/variance across dp mesh |
| `src/solvers/scaling_resilience/lib/sr/passes/ResilienceRestartPass.cpp` | Stub | Wrap body in `tessera_sr.resilience_region`; insert save/restore hooks |
| `src/solvers/scaling_resilience/lib/sr/passes/ExportDeploymentManifestPass.cpp` | Stub | Emit `deployment_manifest.json` with mesh + shard + collective + checkpoint keys |

#### Python — create these files

| File | What to implement |
|------|-------------------|
| `python/tessera/compiler/autotune_v2.py` | `BayesianAutotuner` (Optuna TPE + Hyperband pruning); warm-start from SQLite cache |
| `python/tessera/compiler/checkpoint.py` | `CollectiveCheckpointConfig`; `@jit(checkpoint=True)` extension |

#### Tests — create `tests/phase5/`

`test_insert_recompute.py`, `test_optimizer_shard.py`, `test_resilience_restart.py`, `test_deployment_manifest.py`, `test_sparse_inspector.py`, `test_rng_legalize.py`, `test_bayesian_autotuner.py`, `test_checkpoint_decorator.py` + lit tests in `tests/tessera-ir/phase5/`

---

### Phase 6 — Runtime + ROCm + Benchmarks + Diagnostics

#### Python — create these files

| File | What to implement |
|------|-------------------|
| `python/tessera/runtime.py` | `TesseraRuntime` ctypes wrapper around `tessera_runtime.cpp` C ABI |
| `python/tessera/diagnostics.py` | `TesseraShapeError` with Python source location; `TesseraTargetError` |
| `benchmarks/benchmark_gemm.py` | `GEMMBenchmark` — sweeps M/N/K; reports latency_ms, tflops, memory_bw |
| `benchmarks/benchmark_attention.py` | `FlashAttnBenchmark` — B/H/S/D sweep; tokens/sec, MFU |
| `benchmarks/benchmark_collective.py` | `CollectiveBenchmark` — 2–128 ranks; bus bandwidth |
| `benchmarks/run_all.py` | Orchestrates all benchmarks; emits `tessera_benchmarks_*.json` |

#### C++ — create/complete these files

| File | What to implement |
|------|-------------------|
| `src/runtime/src/backend/cuda_backend.cpp` | Wire `cudaMalloc/cudaMemcpy/cudaStream` — currently 30-line stub |
| `src/runtime/src/backend/hip_backend.cpp` | Wire `hipMalloc/hipMemcpy/hipStream` — currently 30-line stub |
| `src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TesseraTargetToROCDL.cpp` | Verify MFMA shape table covers gfx90a/gfx94/gfx1200 (check `MFMTables.cpp`) |
| `src/compiler/diagnostics/ErrorReporter.cpp` | Walk MLIR op `loc` chain; attach Python file + line to shape errors |
| `src/compiler/diagnostics/ShapeInferencePass.cpp` | Forward-propagate static shapes; catch mismatches early |

#### Tests — create `tests/phase6/`

`test_runtime_abi.py`, `test_runtime_cpu_backend.py`, `test_mfma_full_coverage.py`, `test_shape_inference.py`, `test_error_reporter.py`, `test_benchmark_gemm.py` + lit tests in `tests/tessera-ir/phase6/`

---

## Phase 3 Deliverables (complete — do not re-implement)

### Python files

| File | Purpose |
|------|---------|
| `python/tessera/compiler/gpu_target.py` | `GPUTargetProfile` + `ISA` enum (SM_80–SM_100) |
| `python/tessera/compiler/attn_lower.py` | `FlashAttnLoweringConfig` (tile_q, tile_kv, dropout, causal) |
| `python/tessera/compiler/jit.py` | Extended with `target=` + `attn_config=` params |
| `python/tessera/core/__init__.py` | Added `Tensor.__class_getitem__` for subscript annotation syntax |

### C++ passes (registered in `tessera-lower-to-gpu` named pipeline)

| Pass | File | Purpose |
|------|------|---------|
| `TileIRLoweringPass` | `src/transforms/lib/TileIRLoweringPass.cpp` | `tessera.flash_attn` → `tessera.attn.*` + `tile.async_copy` + `tile.mma` |
| `WarpSpecializationPass` | `src/compiler/tile_opt_fa4/lib/WarpSpecializationPass.cpp` | Assigns producer/consumer roles; inserts `tessera.queue` barriers |
| `AsyncCopyLoweringPass` | `src/compiler/tile_opt_fa4/lib/AsyncCopyLoweringPass.cpp` | `tile.async_copy` → `tessera.tma.*` (SM_90) or `tessera.cp_async.*` |
| `NVWGMMALoweringPass` | `src/.../NVWGMMALoweringPass.cpp` | `tile.mma` → `tessera.nvgpu.wgmma.mma_async` PTX (SM_90+) or WMMA |
| `NVTMADescriptorPass` | `src/.../NVTMADescriptorPass.cpp` | Hoists TMA descriptors to kernel preamble; assigns mbarrier slots |
| `NVFlashAttnKernelEmitter` | `src/.../NVFlashAttnKernelEmitter.cpp` | Resolves scale sentinel, emits mbarrier arrive/wait, attaches launch bounds |

### FA-4 Attn dialect (Attn.td v2.0 additions)

New ops: `ScaledDotProductOp`, `OnlineSoftmaxOp`, `LseAccumulateOp`, `DropoutMaskOp`, `CausalMaskOp`
Verifiers: `AttnOps.cpp` (FA-2 shape invariants)

### Tests

Python: `tests/phase3/` — 56 tests, all green.
Lit: `tests/tessera-ir/phase3/` — 4 MLIR lit files (tile_ir_lowering, warp_specialization, nvwgmma_lowering, flash_attn_full).

---

## Phase 2 Deliverables (complete — do not re-implement)

Passes live in `src/transforms/lib/`. All 4 are registered and wired into the
`tessera-lower-to-x86` named pipeline.

| Pass | File | Purpose |
|------|------|---------|
| `DistributionLoweringPass` | `DistributionLoweringPass.cpp` | `tessera.shard` attrs → `schedule.mesh.define` + `schedule.mesh.region` |
| `EffectAnnotationPass` | `EffectAnnotationPass.cpp` | Infers `pure/random/memory/io`, annotates `func.func` |
| `TilingPass` | `TilingPass.cpp` | `tessera.matmul` → `scf.for` M×N tile loops with `tensor.extract/insert_slice` |
| `TileToX86Pass` | `TileToX86Pass.cpp` | Tiled BF16 matmul → `func.call @tessera_x86_amx_gemm_bf16` via raw i64 pointers |

`schedule.yield` was added to `ScheduleMeshPipelineOps.td` as the region terminator.
Phase 2 lit tests: `tests/tessera-ir/phase2/`.
Phase 2 Python tests: `tests/phase2/test_lowering_chain.py`.

---

## Phase 3 — NVIDIA GPU Backend + Tile IR FA-4 + FlashAttention

### Mission

Enable end-to-end GPU execution of FlashAttention on NVIDIA SM90 (Hopper).  Wire
the full FA-4 Tile IR dialect (warp specialization, TMA async copy, mbarrier
double-buffering, online softmax), lower `tessera.flash_attn` through Schedule IR
→ Tile IR → NVGPU intrinsics → PTX.

This is the first phase that produces code that runs on real GPU hardware.

### Anchoring test — make this pass before declaring Phase 3 done

```python
# tests/phase3/test_gpu_flash_attn.py

import tessera
from tessera.compiler.gpu_target import GPUTargetProfile, ISA

@tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4))
def flash_attn_fwd(
    Q: tessera.Tensor["B", "H", "S", "D"],
    K: tessera.Tensor["B", "H", "S", "D"],
    V: tessera.Tensor["B", "H", "S", "D"],
) -> tessera.Tensor["B", "H", "S", "D"]:
    tessera.require(tessera.constraint.Divisible("D", 64))
    return tessera.ops.flash_attn(Q, K, V, causal=True)

# Verifies that:
# 1. The emitted IR contains tessera.attn.scaled_dot_product
# 2. The Schedule IR wraps it in a schedule.mesh.region
# 3. The Tile IR lowering emits tile.async_copy + tile.wait_async
# 4. The NVGPU lowering emits wgmma.mma_async inline PTX for SM_90
ir = flash_attn_fwd.graph_ir.to_mlir()
assert "tessera.flash_attn" in ir
assert "tessera.effect" in ir
```

### File map

**New C++ files:**

```
src/transforms/lib/
  TileIRLoweringPass.cpp        ← schedule.mesh.region → Tile IR ops
                                   (tessera.attn.*, tile.async_copy, tile.wait_async)

src/compiler/tile_opt_fa4/
  dialects/tessera_attn/
    AttnOps.cpp                 ← FA-4 attention op implementations:
                                   scaled_dot_product, online_softmax,
                                   lse_accumulate, dropout_mask, causal_mask
  lib/
    WarpSpecializationPass.cpp  ← assigns warp roles (producer/consumer),
                                   inserts tile.queue barriers between roles
    AsyncCopyLoweringPass.cpp   ← tile.async_copy → TMA descriptor + cp.async PTX

src/compiler/codegen/tessera_gpu_backend_NVIDIA/
  NVWGMMALoweringPass.cpp       ← tile.mma → wgmma.mma_async PTX for SM_90+
  NVTMADescriptorPass.cpp       ← emit TMA descriptor setup + mbarrier init
  NVFlashAttnKernelEmitter.cpp  ← full FA fwd kernel: QKV tiles + online softmax
                                   + LSE accumulation + epilogue
```

**New Python files:**

```
python/tessera/compiler/
  gpu_target.py     ← GPUTargetProfile(isa, warps_per_cta, shared_mem_bytes)
                       ISA enum: SM_80, SM_86, SM_89, SM_90, SM_100
  attn_lower.py     ← FlashAttnLoweringConfig(tile_q, tile_kv, pipeline_stages)
                       Integration with @jit for flash_attn lowering decisions

python/tessera/compiler/jit.py  ← extend @jit to accept target= keyword;
                                   route to GPU lowering pipeline when
                                   target.isa >= ISA.SM_90
```

**New test files:**

```
tests/phase3/
  __init__.py
  conftest.py                   ← GPUTargetProfile fixtures, mock WGMMA verifier
  test_tile_ir_lowering.py      ← TileIRLoweringPass: schedule.mesh.region → Tile IR
  test_warp_specialization.py   ← WarpSpecializationPass: producer/consumer roles
  test_async_copy.py            ← AsyncCopyLoweringPass: tile.async_copy emission
  test_flash_attn_lowering.py   ← end-to-end flash_attn: Graph IR → Tile IR
  test_gpu_target.py            ← GPUTargetProfile validation

tests/tessera-ir/phase3/
  tile_ir_lowering.mlir         ← lit: schedule.mesh.region → tile.async_copy
  warp_specialization.mlir      ← lit: warp role assignment
  nvwgmma_lowering.mlir         ← lit: tile.mma → wgmma.mma_async PTX
  flash_attn_full.mlir          ← lit: end-to-end flash_attn pipeline
```

### Key design contracts

1. **Warp role separation is structural, not advisory.**
   `WarpSpecializationPass` emits `tessera.schedule.warp {role="producer"}` and
   `tessera.schedule.warp {role="consumer"}` regions. These are hard boundaries —
   the backend allocates different register files and barrier slots per role.

2. **TMA descriptors are generated once per kernel, not per tile.**
   `NVTMADescriptorPass` hoists descriptor setup to kernel preamble.
   Tile loops call `cp.async.bulk.tensor` referencing the descriptor.

3. **Online softmax uses the flash-attention 2 algorithm.**
   `AttnOps.cpp` implements the two-pass online softmax from the FA-2 paper:
   running max, running sum, correction factor applied at LSE accumulation.
   Do NOT implement a simpler batch softmax — it will OOM on long sequences.

4. **Tile sizes for SM_90:**
   Default for FlashAttention: `tile_q=64, tile_kv=64, pipeline_stages=2`.
   These are AMPs (autotunable) — store them as `tessera.tile_q` and
   `tessera.tile_kv` attributes so the autotuner (Phase 5) can sweep them.

5. **Gate all WGMMA code behind `target_profile.isa >= ISA.SM_90`.**
   Below SM_90, fall back to WMMA (existing NVIDIA backend).

### Phase 3 completion criteria

- [ ] `pytest tests/phase3/ -v` — all tests green
- [ ] `python -m lit tests/tessera-ir/phase3/ -v` — all lit tests pass
- [ ] `TileIRLoweringPass` lowers `schedule.mesh.region { tessera.flash_attn }` to Tile IR ops
- [ ] `WarpSpecializationPass` assigns producer/consumer roles with queue barriers
- [ ] `NVWGMMALoweringPass` emits `wgmma.mma_async.sync.aligned` PTX for BF16 on SM_90
- [ ] `NVTMADescriptorPass` emits `cp.async.bulk.tensor` with correct descriptor args
- [ ] `GPUTargetProfile(isa=ISA.SM_90)` accepted by `@jit(target=...)` without error
- [ ] `flash_attn_fwd` Graph IR text contains `tessera.flash_attn` with `causal=true`
- [ ] FlashAttention lit test verifies online softmax ops appear in Tile IR
- [ ] WGMMA fallback to WMMA path when `isa < SM_90` (verified by lit test)

### Key reference files for Phase 3

| What you need | Where to look |
|---------------|--------------|
| FA-4 Tile IR ODS (attn + queue) | `src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td`, `Queue.td` |
| Memory model (tile.async_copy spec) | `src/programming_model/docs/Memory_Execution_Model_v1_1.md` |
| Warp specialization design | `src/programming_model/docs/Parallelism_Constructs_v1_1.md` |
| NVIDIA WGMMA backend skeleton | `src/compiler/codegen/tessera_gpu_backend_NVIDIA/lowering_nvvm_mlir.cpp` |
| NVIDIA WMMA kernels (reference) | `src/compiler/codegen/tessera_gpu_backend_NVIDIA/src/` |
| CUDA driver wrapper | `src/compiler/codegen/tessera_gpu_backend_NVIDIA/include/tessera/cuda/cuda_driver.h` |
| Compilation stages overview | `docs/Tessera_Kernel_Compilation_Stages_Overview.md` |
| Tile IR spec | `docs/spec/04_tile_ir.md` |
| Phase 2 TilingPass (reference impl) | `src/transforms/lib/TilingPass.cpp` |

---

## Phase 4 — Distributed Training: Collectives + TPU Backend

### Mission

Enable real multi-device GPU training via NCCL/RCCL. Complete the TPU StableHLO
backend for matmul + attention + convolution. Add Cyclic distribution for MoE
load balancing. Wire the full mesh parallelism pipeline: data-parallel (DP),
tensor-parallel (TP), and pipeline-parallel (PP).

### Anchoring test — make this pass before declaring Phase 4 done

```python
# tests/phase4/test_distributed_training.py

from tessera.testing import MockRankGroup

ranks = MockRankGroup(n=8, mesh_axes={"dp": 4, "tp": 2})

@tessera.jit
def dp_tp_step(
    W: tessera.Region["read"],
    X: tessera.Region["read"],
    grad: tessera.Region["reduce_sum"],
):
    Y = tessera.ops.gemm(X, W)
    return Y

# Verify that the emitted IR contains both dp and tp mesh regions
# and reduce_scatter + all_gather collective ops at boundaries.
ir = dp_tp_step.graph_ir.to_mlir()
assert "schedule.mesh.region" in ir
assert "tessera.collective" in ir or "collective.reduce_scatter" in ir
```

### File map

**New C++ files:**

```
src/collectives/lib/
  NCCLAdapter.cpp               ← real NCCL all_reduce / reduce_scatter /
                                   all_gather / all_to_all; wraps ncclComm_t
  RCCLAdapter.cpp               ← ROCm RCCL equivalent; same interface as NCCL
  ChunkPlanner.cpp              ← selects chunk_bytes + max_inflight based on
                                   Topology (NVLink vs PCIe vs RDMA bandwidth)
  CollectiveScheduler.cpp       ← credit-based link scheduler; issues chunks
                                   to ExecRuntime in policy-determined order
  PipelineOverlapPass.cpp       ← overlaps reduce_scatter(grads) with
                                   all_gather(next layer weights)

src/transforms/lib/
  GPUCollectiveInsertionPass.cpp ← inserts collective.reduce_scatter at
                                    DP mesh boundaries after backward pass;
                                    inserts collective.all_gather at TP boundaries
  PipelineStageInsertionPass.cpp ← partitions schedule.pipeline.region stages
                                    across ranks, inserts send/recv micro-batch ops

src/compiler/codegen/Tessera_TPU_Backend_Starter_Advanced/lib/
  TPUAttentionLoweringPass.cpp  ← tessera.flash_attn → stablehlo composite
                                   (scaled_dot_product + mask + softmax)
  TPUConvLoweringPass.cpp       ← tessera.conv2d → stablehlo.convolution (NHWC)
  TPUQuantizedDotPass.cpp       ← bf16/int8 dot_general with precision attrs
  TPUShardyExportPass.cpp       ← complete Shardy export (mesh + sharding rules)
```

**New Python files:**

```
python/tessera/distributed/
  cyclic.py         ← Cyclic distribution full implementation (Phase 1 was stub):
                       ShardSpec with stride > 1, round-robin fanout in .parts()
  moe.py            ← MoE-specific helpers: token routing, expert assignment,
                       all_to_all bucket planning

python/tessera/compiler/
  distributed_planner.py  ← DistributedPlan: decides dp/tp/pp split from
                              mesh axes, emits collective insertion annotations
  pipeline_planner.py     ← PipelinePlan: 1F1B / interleaved schedule builder,
                              computes micro-batch counts and stage latencies
  tpu_target.py           ← TPUTargetProfile (MXU tile size, mesh axes)
```

**New test files:**

```
tests/phase4/
  __init__.py
  conftest.py                      ← 8-rank MockRankGroup, NCCLMock adapter
  test_nccl_adapter.py             ← NCCLAdapter interface + NCCLMock correctness
  test_chunk_planner.py            ← ChunkPlanner sizes for NVLink / PCIe topologies
  test_gpu_collective_insertion.py ← GPUCollectiveInsertionPass: reduce_scatter placement
  test_pipeline_stage_insertion.py ← PipelineStageInsertionPass: 1F1B schedule
  test_cyclic_distribution.py      ← Cyclic.make_shard_spec + .parts() fanout
  test_tpu_lowering.py             ← TPU attention + conv + quantized dot lowering
  test_distributed_plan.py         ← DistributedPlan dp/tp/pp assignment

tests/tessera-ir/phase4/
  gpu_collective_insertion.mlir    ← lit: reduce_scatter at dp boundary
  pipeline_stage_insertion.mlir    ← lit: 1F1B micro-batch schedule
  tpu_attention.mlir               ← lit: flash_attn → stablehlo composite
  tpu_shardy_export.mlir           ← lit: Shardy mesh + sharding rules export
```

### Key design contracts

1. **NCCL/RCCL adapters share a single `CollectiveAdapter` interface.**
   `NCCLAdapter` and `RCCLAdapter` both inherit from `CollectiveAdapter` defined
   in `src/collectives/include/tessera/Dialect/Collective/Runtime/Adapters.h`.
   The pass pipeline selects the adapter at compile time via a target option.

2. **Collective insertion happens after effect annotation.**
   `GPUCollectiveInsertionPass` reads `tessera.effect = "memory"` on write-region
   args to identify gradient tensors needing reduce_scatter. It must run AFTER
   `EffectAnnotationPass` in the pipeline.

3. **Cyclic distribution fanout preserves global tensor semantics.**
   `Cyclic.parts("dp")` returns a list of per-rank slices where element `i` is on
   rank `i % dp_size`. The `distributed_planner.py` must emit an all_to_all
   rebalance whenever Cyclic and Block tensors interact in the same op.

4. **TPU MXU tile constraint is 128×128.**
   All matmul M/N/K dims must be multiples of 128 on TPU.
   `TPUTargetProfile` stores this and `@jit(target=tpu)` injects a
   `Divisible("M", 128)` + `Divisible("N", 128)` + `Divisible("K", 128)` check
   automatically.

5. **Pipeline parallelism uses 1F1B scheduling by default.**
   `PipelineStageInsertionPass` implements the 1F1B (one-forward-one-backward)
   schedule from GPipe/PipeDream. The `schedule="interleaved"` option implements
   the interleaved variant (requires `micro_batches >= 2 * num_stages`).

### Phase 4 completion criteria

- [ ] `pytest tests/phase4/ -v` — all tests green
- [ ] `python -m lit tests/tessera-ir/phase4/ -v` — all lit tests pass
- [ ] `NCCLAdapter` passes mock all_reduce, reduce_scatter, all_gather tests
- [ ] `ChunkPlanner` selects 512 KiB chunks for NVLink, 128 KiB for PCIe
- [ ] `GPUCollectiveInsertionPass` inserts `collective.reduce_scatter` at all dp mesh boundaries
- [ ] `PipelineStageInsertionPass` emits correct 1F1B schedule for 4 stages, 8 micro-batches
- [ ] `Cyclic.make_shard_spec` is implemented (no longer raises `NotImplementedError`)
- [ ] `Cyclic.parts("dp")` returns correct per-rank slices for n=4 ranks
- [ ] TPU attention lowering test: `tessera.flash_attn` → `stablehlo.composite`
- [ ] TPU Shardy export: mesh axis annotations survive round-trip
- [ ] `DistributedPlan` correctly partitions a 4-layer MLP over dp=4, tp=2 mesh

### Key reference files for Phase 4

| What you need | Where to look |
|---------------|--------------|
| Collective IR + runtime design | `src/collectives/include/tessera/Dialect/Collective/IR/CollectiveOps.td` |
| Overlap design spec | `src/collectives/docs/Tessera_Collectives_Overlap_Design.md` |
| ExecRuntime + Policy + Topology | `src/collectives/include/tessera/Dialect/Collective/Runtime/` |
| TPU backend skeleton | `src/compiler/codegen/Tessera_TPU_Backend_Starter_Advanced/` |
| Parallelism constructs | `src/programming_model/docs/Parallelism_Constructs_v1_1.md` |
| Schedule IR ODS | `src/programming_model/ir/schedule/ScheduleMeshPipelineOps.td` |
| Phase 2 DistributionLoweringPass | `src/transforms/lib/DistributionLoweringPass.cpp` |
| Collective adapter stubs | `src/collectives/include/tessera/Dialect/Collective/Runtime/Adapters.h` |

---

## Phase 5 — Scaling, Resilience + Autotuner v2 + Solver Suite

### Mission

Make Tessera production-grade. Implement all four scaling/resilience passes
(recompute insertion, optimizer sharding, checkpoint/restart, manifest export).
Upgrade the autotuner from Hyperband grid search to Bayesian optimization.
Implement the scientific computing solver suite (sparse, RNG, Newton autodiff).

### Anchoring test — make this pass before declaring Phase 5 done

```python
# tests/phase5/test_resilience.py

@tessera.jit(checkpoint=True, checkpoint_interval=2)
def transformer_block(
    x:   tessera.Region["read"],
    W_q: tessera.Region["read"],
    W_k: tessera.Region["read"],
    W_v: tessera.Region["read"],
    W_o: tessera.Region["read"],
):
    """4-layer transformer block with activation checkpointing."""
    q = tessera.ops.gemm(x, W_q)
    k = tessera.ops.gemm(x, W_k)
    v = tessera.ops.gemm(x, W_v)
    a = tessera.ops.flash_attn(q, k, v, causal=True)
    return tessera.ops.gemm(a, W_o)

ir = transformer_block.graph_ir.to_mlir()
# checkpoint regions must appear every 2 ops
assert "tessera_sr.checkpoint" in ir
assert "tessera_sr.recompute_hint" in ir

# Autotuner v2: Bayesian search finds a better config faster than grid
from tessera.compiler.autotune_v2 import BayesianAutotuner, GEMMWorkload
tuner = BayesianAutotuner(GEMMWorkload(M=4096, N=4096, K=4096))
result = tuner.run(max_trials=30)
assert result.tflops > 0
```

### File map

**New C++ files:**

```
src/scaling_resilience/lib/
  InsertRecomputePass.cpp         ← identifies recomputable segments (pure ops
                                     between two write checkpoints), inserts
                                     tessera_sr.checkpoint + recompute_hint
  OptimizerShardPass.cpp          ← ZeRO-stage-2: partitions optimizer state
                                     (momentum, variance) across dp mesh axes;
                                     emits tessera_sr.optimizer_shard annotations
  ResilienceRestartPass.cpp       ← wraps tessera_sr.resilience_region around
                                     compute blocks; inserts save/restore hooks
                                     calling the runtime C ABI tsrCheckpointSave/Load
  ExportDeploymentManifestPass.cpp ← emits deployment_manifest.json: mesh topology,
                                      shard assignments, collective routes,
                                      checkpoint intervals, restart policy

src/solvers/lib/
  SparseInspectorPass.cpp         ← analyzes tensor fill patterns; tags sparse
                                     ops with tessera_solver.sparse_hint attrs
  SparsePrecondPass.cpp           ← selects Jacobi/ILU/AMG preconditioner for
                                     sparse solver ops
  RNGLegalizePass.cpp             ← legalizes tessera_rng.* ops; assigns stream
                                     IDs for deterministic parallel RNG
  RNGStreamAssignPass.cpp         ← assigns philox/threefry streams to RNG ops
                                     such that per-worker streams are independent
  NewtonAutodiffPass.cpp          ← generates VJP/JVP for tessera_solver.implicit
                                     ops (implicit function theorem differentiation)
```

**New Python files:**

```
python/tessera/compiler/
  autotune_v2.py          ← BayesianAutotuner: wraps Optuna TPE sampler;
                              pruning via Hyperband; constraint-aware search
                              (memory footprint, latency bounds);
                              upgrades existing SQLite cache schema to v2
  checkpoint.py           ← @jit(checkpoint=True) extension:
                              CollectiveCheckpointConfig(interval, storage_path)
  solver_config.py        ← SolverConfig: sparse tolerance, RNG seed policy,
                              Newton max_iter, convergence criteria
```

**New test files:**

```
tests/phase5/
  __init__.py
  conftest.py                          ← fixtures: small transformer, sparse matrix
  test_insert_recompute.py             ← InsertRecomputePass: checkpoint placement
  test_optimizer_shard.py              ← OptimizerShardPass: ZeRO shard assignment
  test_resilience_restart.py           ← ResilienceRestartPass: save/restore hooks
  test_deployment_manifest.py          ← ExportDeploymentManifestPass: JSON schema
  test_sparse_inspector.py             ← SparseInspectorPass: sparsity detection
  test_rng_legalize.py                 ← RNGLegalizePass: stream assignment
  test_bayesian_autotuner.py           ← BayesianAutotuner convergence + caching
  test_checkpoint_decorator.py         ← @jit(checkpoint=True) integration

tests/tessera-ir/phase5/
  insert_recompute.mlir                ← lit: checkpoint + recompute_hint placement
  optimizer_shard.mlir                 ← lit: ZeRO-2 shard annotations
  resilience_restart.mlir              ← lit: resilience_region + restart hooks
  deployment_manifest.mlir             ← lit: manifest export op
  rng_legalize.mlir                    ← lit: rng stream assignment
```

### Key design contracts

1. **Recompute insertion is guided by memory budget, not heuristics.**
   `InsertRecomputePass` accepts a `--memory-budget-mb` option. It uses a greedy
   algorithm: scan the op sequence, accumulate live tensor sizes, insert a
   `tessera_sr.checkpoint` whenever the live-set exceeds the budget.
   The `recompute_hint` marks which ops between two checkpoints can be re-run
   during backward (only pure ops with `tessera.effect = "pure"` qualify).

2. **Optimizer sharding is ZeRO stage 2 by default.**
   `OptimizerShardPass` partitions momentum and variance across `dp` mesh axes.
   ZeRO stage 3 (parameter sharding) is out of scope for Phase 5.

3. **Bayesian autotuner is Optuna-based but cache-compatible.**
   The existing SQLite cache from Phase 1's autotuner uses key =
   `hash(device_class + kernel_id + config)`. Phase 5 adds a `v2` schema with
   Optuna trial IDs. The `BayesianAutotuner` can seed from existing cache entries
   (warm start) — this is important for iteration speed.

4. **RNG streams are per-worker, seeded from global seed + rank.**
   `RNGStreamAssignPass` assigns `stream_id = global_seed * num_ranks + rank`.
   This ensures per-rank streams are statistically independent (Philox counter
   offsets are non-overlapping for 2^128 elements).

5. **Solver passes only lower if the op has a `tessera_solver.*` dialect op.**
   Do not try to lower `tessera.matmul` in solver passes — that is handled by
   TilingPass and TileToX86Pass from Phase 2.

### Phase 5 completion criteria

- [ ] `pytest tests/phase5/ -v` — all tests green
- [ ] `python -m lit tests/tessera-ir/phase5/ -v` — all lit tests pass
- [ ] `InsertRecomputePass` places checkpoints every N ops, only on pure segments
- [ ] `OptimizerShardPass` partitions optimizer state correctly for dp=4 mesh
- [ ] `ResilienceRestartPass` wraps body in `tessera_sr.resilience_region`
- [ ] `ExportDeploymentManifestPass` emits valid JSON with mesh + shard keys
- [ ] `SparseInspectorPass` tags matmul with fill-fraction < 5% as sparse
- [ ] `RNGLegalizePass` assigns unique stream IDs to all `tessera_rng.*` ops
- [ ] `BayesianAutotuner` finds a better config than `GridSearch` in ≤30 trials on 4096×4096 GEMM
- [ ] `@jit(checkpoint=True)` produces IR with `tessera_sr.checkpoint` every 2 ops
- [ ] `BayesianAutotuner` warm-starts from existing SQLite cache entries

### Key reference files for Phase 5

| What you need | Where to look |
|---------------|--------------|
| Scaling/Resilience dialect ODS | `src/scaling_resilience/lib/sr/dialect/SROps.td` |
| SR pass declarations | `src/scaling_resilience/include/tessera/sr/Passes.h` |
| Solver pass declarations | `src/solvers/passes/SolversPasses.h` |
| Autotuner v1 framework | `src/compiler/autotuning/tessera/tools/autotune/` |
| Memory/execution model spec | `src/programming_model/docs/Memory_Execution_Model_v1_1.md` |
| Runtime checkpoint ABI | `src/runtime/include/tessera/tessera_runtime.h` |
| Phase 2 EffectAnnotationPass | `src/transforms/lib/EffectAnnotationPass.cpp` |

---

## Phase 6 — ROCm Backend + Runtime C ABI + Production Hardening + Benchmarks

### Mission

Full AMD GPU support via the ROCm backend with complete MFMA instruction coverage
and Composable Kernels (CK) bridge. Implement the tessera runtime C ABI
(`tsrContextCreate`, `tsrTileGraphLaunch`, etc.). Harden diagnostics with precise
shape-error attribution. Build a comprehensive benchmark suite linked to the
roofline tool.

### Anchoring test — make this pass before declaring Phase 6 done

```python
# tests/phase6/test_runtime_abi.py

from tessera.runtime import TesseraRuntime

rt = TesseraRuntime()
assert rt.device_count() >= 0   # works even with no GPU (returns 0)

ctx = rt.create_context(device_id=0)
stream = rt.create_stream(ctx)

# Allocate + fill + copy
buf_a = rt.malloc(ctx, size_bytes=128 * 256 * 2)   # 128×256 BF16
rt.memset(buf_a, 0, 128 * 256 * 2)
rt.memcpy(buf_a, src=bytes(128 * 256 * 2), stream=stream)
rt.synchronize(stream)

rt.destroy_stream(stream)
rt.destroy_context(ctx)

# Benchmarks run and produce non-zero TFLOPS
from benchmarks.benchmark_gemm import run_gemm_benchmark
results = run_gemm_benchmark(backends=["x86"], sizes=[(512, 512, 512)])
assert results[0].tflops > 0.1
```

### File map

**New C++ files:**

```
src/compiler/codegen/Tessera_ROCM_Backend/lib/
  MFMAFullCoveragePass.cpp  ← extends MFMA instruction table to all AMD GPU
                               variants: gfx90a (MI210/250), gfx94 (MI300),
                               gfx1200 (RDNA4); picks optimal MFMA shape
                               (16×16×4, 32×32×4, etc.) per target
  CKBridgePass.cpp          ← wraps CK (Composable Kernels) for GEMM + conv
                               fallback; detects at runtime if CK library
                               is available and dispatches to it
  ROCmAsyncCopyPass.cpp     ← tessera tile.async_copy → HIP async copy API
                               using hipMemcpy2DAsync + LDS double-buffering
  ROCmFlashAttnPass.cpp     ← full FlashAttention kernel for ROCm using
                               MFMA matrix multiply + LDS staging

src/runtime/lib/
  tessera_runtime.cpp       ← full implementation of the tessera runtime C ABI:
                               tsrInit/Shutdown, tsrGetDevice, tsrCreateStream,
                               tsrMalloc/Free/Memset/Memcpy, tsrLaunchHostTileKernel
                               Dispatches to CUDA/HIP/CPU backend based on device type
  tessera_runtime_cpu.cpp   ← CPU-only backend for the runtime ABI (used in tests
                               without a GPU; executes tile kernels as C++ lambdas)

src/compiler/diagnostics/
  ErrorReporter.cpp         ← attaches source location (file:line) to shape errors;
                               formats constraint violation messages with the
                               offending dim path highlighted
  ShapeInferencePass.cpp    ← forward-propagates static shapes through the Graph IR;
                               catches shape mismatches early with line-attributed errors
```

**New Python files:**

```
python/tessera/
  runtime.py          ← TesseraRuntime Python wrapper: thin ctypes/cffi binding
                         to tessera_runtime.cpp C API;
                         create_context / malloc / memcpy / synchronize / launch

benchmarks/
  __init__.py
  benchmark_gemm.py       ← GEMMBenchmark: sweeps M/N/K sizes across backends
                              (x86 AMX, NVIDIA WGMMA, ROCm MFMA, TPU MXU);
                              reports latency_ms, tflops, memory_bw_gb_s;
                              generates roofline-compatible JSON
  benchmark_attention.py  ← FlashAttnBenchmark: sweeps B/H/S/D configs;
                              reports tokens_per_second, MFU (model FLOP utilization)
  benchmark_collective.py ← CollectiveBenchmark: all_reduce / reduce_scatter / all_gather
                              sweep for 2–128 ranks; reports bus bandwidth
  run_all.py              ← orchestrates all benchmarks, produces HTML report
                              linking into Perfetto trace and roofline chart

python/tessera/
  diagnostics.py      ← user-facing error message improvements:
                         TesseraShapeError (wraps MLIR verifier errors with
                         Python source location), TesseraTargetError (backend
                         capability mismatch with suggested fix)
```

**New test files:**

```
tests/phase6/
  __init__.py
  conftest.py                        ← TesseraRuntime fixture (CPU backend only)
  test_runtime_abi.py                ← full C ABI: context, stream, malloc, memcpy, launch
  test_runtime_cpu_backend.py        ← CPU-only backend correctness
  test_mfma_full_coverage.py         ← MFMAFullCoveragePass for gfx90a/gfx94/gfx1200
  test_ck_bridge.py                  ← CKBridgePass fallback path (mock CK library)
  test_rocm_flash_attn.py            ← ROCm FlashAttention kernel correctness
  test_shape_inference.py            ← ShapeInferencePass forward propagation
  test_error_reporter.py             ← ErrorReporter: constraint errors have source loc
  test_benchmark_gemm.py             ← benchmark produces valid tflops > 0

tests/tessera-ir/phase6/
  mfma_full_coverage.mlir            ← lit: MFMA shape selection per gfx target
  ck_bridge_fallback.mlir            ← lit: CK dispatch annotation
  rocm_flash_attn.mlir               ← lit: ROCm FlashAttention Tile IR
  shape_inference.mlir               ← lit: shape propagation + error reporting
```

### Key design contracts

1. **The runtime C ABI is single-threaded per stream.**
   `tsrLaunchHostTileKernel` is synchronous w.r.t. the calling thread but
   asynchronous w.r.t. the stream. Multiple streams are independent.
   The CPU backend implements streams as `std::thread` pools; CUDA uses
   `cudaStream_t`; HIP uses `hipStream_t`.

2. **MFMA shape selection is target-specific and encoded in a lookup table.**
   `MFMAFullCoveragePass` reads a static table `mfma_table.inc` that maps
   `(gfx_target, element_type, M, N, K)` to the best MFMA variant.
   Do NOT hardcode shapes in pass logic — keep them in the table.

3. **CK bridge is an optional dependency.**
   `CKBridgePass` emits a `tessera.ck.gemm` op only when the CK library is
   detected at configure time (`TESSERA_USE_CK=ON`). Otherwise it is a no-op.
   Downstream passes must tolerate both paths.

4. **Benchmark JSON schema is stable across phases.**
   `run_all.py` emits `tessera_benchmarks_<date>.json` with schema v1.
   Fields: `backend`, `op`, `shape`, `dtype`, `latency_ms`, `tflops`,
   `memory_bw_gb_s`, `device`, `tessera_version`.
   The roofline tool at `tools/roofline/` reads this schema directly.

5. **`TesseraShapeError` always includes a Python source location.**
   The `ErrorReporter` must walk the MLIR op's `loc` attribute chain to find
   the originating Python file and line. If `loc` is unavailable, it emits
   `"<unknown location>"` — never suppress the error.

### Phase 6 completion criteria

- [ ] `pytest tests/phase6/ -v` — all tests green
- [ ] `python -m lit tests/tessera-ir/phase6/ -v` — all lit tests pass
- [ ] `TesseraRuntime` Python wrapper: context/stream/malloc/memcpy/launch work on CPU backend
- [ ] `tessera_runtime_cpu.cpp` executes a `tsrLaunchHostTileKernel` with correct tile coord callbacks
- [ ] `MFMAFullCoveragePass` selects correct MFMA shape for gfx90a / gfx94 / gfx1200
- [ ] `CKBridgePass` is a no-op when `TESSERA_USE_CK=OFF` (verified by lit test)
- [ ] `benchmark_gemm.py` runs on x86 backend and reports tflops > 0
- [ ] `benchmark_attention.py` runs on x86 backend with mock GPU fallback
- [ ] `ShapeInferencePass` catches shape mismatch in a 3-op chain
- [ ] `ErrorReporter` produces error with Python source file + line number
- [ ] `run_all.py` produces valid `tessera_benchmarks_*.json` with correct schema
- [ ] All Phase 1–6 test suites pass together (`pytest tests/ -v`)

### Key reference files for Phase 6

| What you need | Where to look |
|---------------|--------------|
| ROCm backend (existing) | `src/compiler/codegen/Tessera_ROCM_Backend/` |
| ROCm ABI + MFMA tables | `src/compiler/codegen/Tessera_ROCM_Backend/include/` |
| Runtime C ABI header | `src/runtime/include/tessera/tessera_runtime.h` |
| Kernel context header | `src/runtime/include/tessera/tsr_kernel.h` |
| tprof + Perfetto (reference) | `tools/tprof/` |
| Roofline tool | `tools/roofline/` |
| Autotuner benchmark workloads | `src/compiler/autotuning/tessera/tools/autotune/` |
| Phase 5 BayesianAutotuner | `python/tessera/compiler/autotune_v2.py` |
| x86 C API reference | `src/compiler/codegen/tessera_x86_backend/include/tessera/x86/target.h` |

---

## Build & Test Commands (all phases)

```bash
# Python dev install
pip install -e ".[dev]"

# Run a specific phase's tests
pytest tests/phase1/ -v
pytest tests/phase2/ -v
pytest tests/phase3/ -v   # requires GPU or mock
pytest tests/phase4/ -v
pytest tests/phase5/ -v
pytest tests/phase6/ -v

# Run all phases
pytest tests/ -v

# Coverage for a phase
pytest tests/phase3/ --cov=tessera.compiler.gpu_target -v

# MLIR lit tests (all phases)
python -m lit tests/tessera-ir/ -v

# MLIR lit tests (single phase)
python -m lit tests/tessera-ir/phase3/ -v

# C++ build (CPU only)
mkdir -p build && cd build
cmake .. -DTESSERA_ENABLE_CUDA=OFF -DTESSERA_CPU_ONLY=ON
make -j$(nproc)

# C++ build (with CUDA for Phase 3+)
cmake .. -DTESSERA_ENABLE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
make -j$(nproc)

# C++ build (with ROCm for Phase 6)
cmake .. -DTESSERA_ENABLE_HIP=ON -DHIP_ROOT_DIR=/opt/rocm
make -j$(nproc)

# Type check Python
mypy python/tessera/

# Run benchmarks (Phase 6)
python benchmarks/run_all.py --backends x86 --output tessera_benchmarks.json
```

---

*Last updated: April 2026 — Phases 1–3 complete (252 tests green). Cross-phase infra added (Core IR ODS, tessera-opt, solver scaffold, runtime C ABI, profiler). RubinCPX backend built out: `tessera.target.cpx` dialect (ODS-generated, NVFP4/NVFP6, 7 ops), 4 passes (FuseVideoIngest, PartitionLongContext, NVFP4Vectorize, LowerKVTransport) all implemented, `tessera-cpx-opt` driver wired, `TESSERA_BUILD_RUBINCPX_BACKEND` CMake option added. Phase 4 is next: Cyclic distribution, NCCL/RCCL, CollectiveInsertionPass, PipelineStageInsertionPass, TPU quantized dot, DistributedPlan, PipelinePlan, MoE helpers. Phases 5–6 are solver body implementations, Autotuner v2, runtime wrappers, and benchmarks.*
