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
| Graph IR ops | `src/ir/TesseraOps.td`, `src/ir/TesseraTiling.cpp` |
| Graph IR passes | `src/transforms/lib/CanonicalizeTesseraIR.cpp` (4 patterns), `VerifyTesseraIR.cpp` |
| Schedule IR ODS | `src/programming_model/ir/schedule/ScheduleMeshPipelineOps.td` |
| Tile IR (FA-4) | `src/tile_opt_fa4/dialects/tessera_attn/Attn.td`, `tessera_queue/Queue.td` |
| x86 backend | `src/compiler/codegen/tessera_x86_backend/` (AMX BF16 + AVX512 GEMM — **works**) |
| Collectives IR | `src/collectives/include/tessera/Dialect/Collective/IR/CollectiveOps.td` |
| Scaling/Resilience | `tessera_scaling_resilience_v1/` (passes are TODO stubs) |
| Neighbors/Halo | `src/tessera_neighbors/` (HaloInferPass is stub) |
| Python package | `python/tessera/` (exports only Tensor, Module, NumericalPolicy — **thin**) |

---

## What Actually Works Today

| Component | Status |
|-----------|--------|
| `src/ir/TesseraOps.td` — matmul, conv2d, flash_attn, fused_epilogue, cast, transpose | **Defined** |
| `CanonicalizeTesseraIR.cpp` — 4 fusion patterns | **Implemented** |
| `VerifyTesseraIR.cpp` | **Checks module version attr only** |
| `tessera_runtime.h` — C ABI for device/stream/buffer/kernel | **Header only, not wired** |
| x86 AMX BF16/AVX512 GEMM kernels | **Work** |
| Autotuner (Python, SQLite cache, Hyperband) | **Works standalone** |
| tprof profiler + Perfetto export | **Works standalone** |
| Roofline tooling | **Works standalone** |
| FA-4 warp specialization ODS | **Defined; lowering NOT implemented** |
| Collectives IR + ExecRuntime + TokenLimiter | **Designed; NCCL adapter incomplete** |
| All scaling/resilience passes | **Empty TODO stubs** |
| Python distributed API (Region, domain, index_launch) | **Does not exist** |
| Effect type propagation pass | **Does not exist** |
| ConstraintSolver | **Does not exist** |
| Autodiff (Graph IR VJP/JVP) | **Does not exist** |
| Numerics policy ODS + cast/round pass | **Does not exist** |

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

## Phase 2 Preview (don't start yet)

Phase 2 wires the lowering chain:
Graph IR → Schedule IR → Tile IR → x86 Target IR → CPU execution.

The passes to implement are:
- `DistributionLoweringPass` — ShardSpec attrs → `schedule.mesh.define` + `mesh.region` ops
- `EffectAnnotationPass` — EffectLattice results → Graph IR function attrs
- `TilingPass` — `schedule.mesh.region` → tiled loop nest in Tile IR
- `CPUCollectiveInsertionPass` — inserts mock collective ops at mesh boundaries
- `TileToX86Pass` — wires into existing `tessera_x86_backend` AMX kernels

---

*Last updated: April 2026. Generated from full codebase audit.*
