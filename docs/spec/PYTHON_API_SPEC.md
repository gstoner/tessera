---
status: Normative
classification: Normative
last_updated: 2026-05-22
---

# Tessera Python API Specification
**Status:** Normative — grounded in `python/tessera/` Phase 1–3 implementation plus S-series standalone compiler updates
**Last updated:** May 22, 2026
**Authority:** This document specifies every public Python symbol in Tessera Phases 1–3. For naming disputes, `docs/CANONICAL_API.md` is the final arbiter. For compiler internals (pass pipeline, IR layers), see `docs/spec/COMPILER_REFERENCE.md`.

---

## Documentation refresh (2026-05-22)

The 2026-05-06 spec gap audit (`docs/audit/compiler/COMPILER_AUDIT.md`)
flagged the developer-tool surface as ahead of the prose in this spec.
Resolution as of 2026-05-22:

- **Debug surface** — Beyond §16 (Error Types), the public `tessera.debug`
  module now exports `DebugTrace`, `GraphTrace`, `summarize_tensor`,
  `debug_trace`, `trace_graph`, `export_graphviz`, `debug_value`,
  `debug_artifact`, `debug_barrier`, `replay_capture`, `replay_manifest`,
  `save_replay_manifest`, `check_grad`, and `check_determinism`. These
  are the canonical developer contracts; their normative source is
  `python/tessera/debug.py` (526 LOC) and the documented user-facing
  surface is `docs/guides/Tessera_Debugging_Tools_Guide.md`.
- **Profiling and autotuning** — `tessera.profiler` (wraps
  `tools/profiler/`) and `tessera.autotune` (wraps
  `compiler/autotune_v2.py`) are public; CLIs are `tessera-prof` and
  `tessera-autotune`. Their normative behaviour is locked by
  `tests/unit/test_cli_debug_profile_commands.py` and
  `tests/unit/test_profiling_autotuning_foundation.py`.
- **MLIR CLI** — `tessera-mlir` (entry: `python/tessera/cli/mlir.py`, 425
  LOC) emits metadata, diagnostics, Chrome trace JSON, GraphViz, and
  supports `--mode=compile_artifact --symbol=name` to inspect real JIT
  artifacts without launching tensors.
- **Translate CLI** — `tessera-translate` (entry:
  `python/tessera/cli/translate.py`) routes through `tessera.aot` (S14)
  for StableHLO / GGUF / SafeTensors export plus an `mlir` subcommand
  that pass-throughs to the C++ `tessera-translate-mlir` binary.
- **S-series surfaces (S2–S15)** — `tessera.rng`, `tessera.state`,
  `tessera.control`, `tessera.sharding`, `tessera.losses`,
  `tessera.optim`, `tessera.quantization`, `tessera.data`,
  `tessera.aot`, `tessera.custom`, `tessera.memory`, and `tessera.rl`
  (PPO/GRPO/CISPO) are all public, with normative contracts in their
  respective modules and primitive coverage tracked by the auto-generated
  `docs/audit/generated/support_table.md`. Do not duplicate counts here;
  treat the generated dashboard as the source of truth.

This refresh keeps the spec body authoritative for Phases 1–3 surfaces
and cross-links the additional Python surfaces above to their
implementation source rather than restating them in normative prose.
The 2026-05-22 sharding mock-mesh proofs (Sprints #17–#20) live in
`tests/unit/test_*_sharding_mock_mesh.py` and are the source of truth
for sharding-rule contract claims; they do not introduce new public
Python symbols beyond what is already specified here.

---

## Table of Contents

1. [Module Hierarchy](#1-module-hierarchy)
2. [Decorators](#2-decorators)
3. [Region Privileges](#3-region-privileges)
4. [Domain API](#4-domain-api)
5. [Distribution API](#5-distribution-api)
6. [Array API](#6-array-api)
7. [ShardSpec](#7-shardspec)
8. [Index Launch](#8-index-launch)
9. [Constraint API](#9-constraint-api)
10. [Effect System](#10-effect-system)
11. [GPU Target API](#11-gpu-target-api)
12. [FlashAttention Lowering Config](#12-flashattention-lowering-config)
13. [Operations Namespace](#13-operations-namespace)
14. [Tensor Annotations](#14-tensor-annotations)
15. [Dtype Annotations](#15-dtype-annotations)
16. [Error Types](#16-error-types)
17. [Testing Utilities](#17-testing-utilities)

---

## 1. Module Hierarchy

```
tessera/
├── __init__.py                   # top-level namespace (see §1.1)
├── core/
│   ├── __init__.py               # Tensor, NumericalPolicy, Module
│   └── tensor.py                 # Tensor base + __class_getitem__
├── distributed/
│   ├── __init__.py               # re-exports all distributed symbols
│   ├── region.py                 # Region, RegionType, RegionMeta
│   ├── domain.py                 # domain.Rect
│   ├── shard.py                  # ShardSpec, MeshSpec
│   ├── array.py                  # DistributedArray
│   └── launch.py                 # index_launch, @kernel
├── compiler/
│   ├── __init__.py               # re-exports compiler symbols
│   ├── jit.py                    # @jit, JitFn
│   ├── constraints.py            # ConstraintSolver, Divisible, Range, Equal
│   ├── effects.py                # Effect, EffectLattice
│   ├── graph_ir.py               # GraphIRBuilder
│   ├── gpu_target.py             # GPUTargetProfile, ISA
│   └── attn_lower.py             # FlashAttnLoweringConfig
├── shape.py                      # Dim, Shape, layout/shard checks, runtime witnesses
├── debug.py                      # graph tracing, tensor summaries, grad/determinism checks
├── profiler.py                   # profiling sessions, reports, Chrome trace export
├── autotune.py                   # public autotune facade and roofline cost model
├── fault.py                      # fault policies, preemption hooks, failure injection
├── elastic.py                    # elastic rendezvous and reshard planning
├── checkpoint.py                 # runtime checkpoint manifests and load/save helpers
├── optim.py                      # functional optimizers, schedules, grad transforms
├── losses.py                     # standalone loss / criterion helpers
├── rl.py                         # PPO/GRPO/CISPO reasoning post-training helpers
├── server.py                     # inference package, scheduler, KV cache, app registry
├── dtype.py                      # canonical dtype names, aliases, Dtype wrapper, promotion helpers
├── ops/
│   └── __init__.py               # tessera.ops.* namespace
└── testing/
    └── mock_collective.py        # MockRankGroup, MockRank, MockCollectiveError
```

### 1.1 Top-Level Namespace

All of the following are importable directly from `tessera`:

```python
import tessera

# Decorators
tessera.jit          # @tessera.jit
tessera.kernel       # @tessera.kernel

# Region
tessera.Region       # Region["read"], Region["write"], etc.

# Domain & Distribution namespaces
tessera.domain       # tessera.domain.Rect(dims)
tessera.dist         # tessera.dist.Block / Cyclic / Replicated
tessera.array        # tessera.array.from_domain(...)

# Index launch
tessera.index_launch # tessera.index_launch(axis=...)

# Constraints namespace
tessera.constraint   # tessera.constraint.Divisible / Range / Equal
tessera.require      # tessera.require(constraint) inside @jit body

# Shape system
tessera.sym          # symbolic dimensions: B, N, D = tessera.sym("B N D")
tessera.check_shapes # marks functions for shape-system validation
tessera.shape        # tessera.shape.ShapeConstraintGraph / RuntimeShapeWitness / helpers

# Debugging
tessera.debug        # tessera.debug.trace_graph / check_grad / check_determinism
tessera.graph        # trace / debug_trace / debug_value / export_graphviz / replay_capture

# Profiling and autotuning
tessera.profiler     # tessera.profiler.session / record / timeline
tessera.autotune     # callable autotune facade; also .load / .cache_key

# Developer commands
tessera-mlir         # static/debug IR dumps and opt-in compile-artifact inspection
tessera-prof         # profiling report, telemetry JSON, Chrome trace, autotune submode
tessera-autotune     # GEMM/matmul tuning and schedule artifact writer

# Fault tolerance and elasticity
tessera.fault        # on_failure / on_preempt / inject
tessera.elastic      # configure / elastic / reshard
tessera.checkpoint   # runtime checkpoint save/load/manifest helpers
tessera.optim        # Adam/AdamW/Adafactor/Momentum/Lion and training utilities
tessera.optimizers   # compatibility namespace: Adam / AdamW stateful wrappers
tessera.losses       # regression/classification/contrastive/diffusion losses
tessera.rl           # PPO/GRPO/CISPO post-training helpers

# Inference serving
tessera.server       # App / load_package / scheduler / KVCacheManager

# Canonical dtype helpers
tessera.dtype        # Dtype, canonicalize_dtype, result_type, planned-gated checks

# Ops namespace
tessera.ops          # tessera.ops.gemm / layer_norm / dropout / etc.
tessera.arange       # compatibility alias for tessera.ops.arange
tessera.gather       # compatibility alias for tessera.ops.gather
tessera.clip         # compatibility alias for tessera.ops.clip
tessera.einsum       # compatibility alias for tessera.ops.einsum
tessera.masked_fill  # compatibility alias for tessera.ops.masked_fill

# Tensor type annotations
tessera.Tensor       # tessera.Tensor["M", "K"]
tessera.f16          # tessera.f16[..., ...]
tessera.bf16         # tessera.bf16[..., ...]
tessera.f32          # tessera.f32[..., ...]
tessera.mut_f32      # tessera.mut_f32[..., ...]
tessera.fp8_e4m3     # low-precision annotation shorthand
tessera.fp8_e5m2     # low-precision annotation shorthand
tessera.fp6          # alias for fp6_e3m2 annotation shorthand
tessera.fp4          # alias for fp4_e2m1 annotation shorthand
tessera.nvfp4        # block-scaled fp4 annotation shorthand
```

---

## 2. Decorators

### 2.1 `@tessera.jit`

**Module:** `tessera.compiler.jit`  
**Import:** `from tessera import jit` or `import tessera; @tessera.jit`

Compiles a Python function into Tessera Graph IR. At decoration time:
1. Inspects type annotations for `Region[...]` and `Tensor[...]` parameters.
2. Extracts `tessera.require(...)` calls via AST parsing (`_ConstraintExtractor`).
3. Runs the `ConstraintSolver` against any concrete `bindings`.
4. Infers `Effect` via `EffectLattice` analysis.
5. Emits Graph IR via `GraphIRBuilder`.

**Signature (no-argument form):**
```python
@tessera.jit
def fn(...): ...
```

**Signature (keyword-argument form):**
```python
@tessera.jit(
    deterministic: bool = False,
    seed: int | None = None,
    bindings: dict[str, int] | None = None,
    target: GPUTargetProfile | None = None,
    attn_config: FlashAttnLoweringConfig | None = None,
    cpu_tile: tuple[int, int, int] = (128, 128, 64),
    source: str | None = None,
    source_path: str | None = None,
)
def fn(...): ...
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deterministic` | `bool` | `False` | If `True`, raises `TesseraEffectError` if the function body contains any unseeded `random` effect op (e.g. `dropout` without `seed`). |
| `seed` | `int \| None` | `None` | RNG seed. Required when `deterministic=True` and the body calls a `random` effect op. |
| `bindings` | `dict[str, int] \| None` | `None` | Optional concrete dimension bindings for early constraint checking at decoration time. Example: `{"K": 128, "M": 512}`. When omitted, symbolic tensor annotations are resolved from call-time argument shapes and constraints are checked before execution. |
| `target` | `GPUTargetProfile \| str \| None` | `None` | Target lowering profile. `None` routes to the CPU/interpreted path. `GPUTargetProfile(ISA.SM_90)` selects NVIDIA Hopper artifacts; `ISA.SM_100` and `ISA.SM_120` select Blackwell artifacts. String aliases include `cuda`, `nvidia`, `gpu`, `sm90`, `sm100`, and `sm120`. |
| `attn_config` | `FlashAttnLoweringConfig \| None` | `None` | Flash attention tile sizes and pipeline configuration. If `None` and `target.isa >= ISA.SM_90`, `SM90_DEFAULT` is used automatically. |
| `cpu_tile` | `tuple[int, int, int]` | `(128, 128, 64)` | CPU matmul/GEMM Schedule IR tile `(tile_m, tile_n, tile_k)` for the narrow end-to-end CPU compiler path. |
| `source` | `str \| None` | `None` | Optional Python source text for functions created from `stdin`, notebooks, or `exec(...)` where `inspect.getsource()` cannot recover the body. |
| `source_path` | `str \| None` | `None` | Optional file path containing Python source text for AST lowering. Mutually exclusive with `source`. |

**Returns:** `JitFn` — a callable wrapper with the following additional attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.graph_ir` | `GraphIRBuilder` | The emitted Graph IR. Call `.to_mlir()` to get MLIR text. |
| `.effect` | `Effect` | Inferred effect of the compiled function. |
| `.constraints` | `ConstraintSolver` | Solver containing all constraints extracted from the function body. |
| `.target` | `GPUTargetProfile \| None` | The target profile passed at decoration time. |
| `.cpu_plan` | `MatmulCPUPlan \| None` | Lowered CPU/reference plan for supported straight-line ops. |
| `.source_origin` | `str` | Source origin used by AST lowering: `inspect`, `explicit`, `file:<path>`, or `unavailable`. |
| `.schedule_ir` | `str \| None` | Schedule IR text for supported lowered functions. |
| `.tile_ir` | `str \| None` | Tile IR text for supported lowered functions. |
| `.target_ir` | `str \| None` | Target IR text for supported lowered functions. |
| `.lowering_artifacts()` | `tuple[LoweringArtifact, ...]` | Graph/Schedule/Tile/Target artifacts for supported lowered functions. |
| `.execution_kind` | `str` | One of `reference_cpu`, `native_cpu`, `native_gpu`, `artifact_only`, or `fallback_eager`. |
| `.is_executable` | `bool` | `True` for launchable reference or native execution. |
| `.is_reference_execution` | `bool` | `True` for NumPy/reference CPU execution. |
| `.is_native_execution` | `bool` | `True` for native CPU/GPU runtime execution. |
| `.lowering_diagnostics` | `tuple[JitDiagnostic, ...]` | Compile/fallback decision diagnostics. |
| `.explain_lowering()` | `str` | Human-readable compile/fallback explanation. |

**Exceptions raised at decoration time:**

| Exception | Condition |
|-----------|-----------|
| `TesseraConstraintError` | A constraint is violated (only when concrete `bindings` are provided). |
| `TesseraEffectError` | `deterministic=True` and the body contains an unseeded `random` op. |
| `TesseraJitError` | Graph IR emission pipeline failure. |
| `TesseraTargetError` | `target` parameter is an invalid `GPUTargetProfile`. |
| `TesseraAttnConfigError` | `attn_config` has invalid tile sizes or pipeline parameters. |

**Example:**
```python
import tessera

@tessera.jit(deterministic=True, seed=42, bindings={"K": 128})
def stable_gemm(
    A: tessera.Tensor["M", "K"],
    B: tessera.Tensor["K", "N"],
) -> tessera.Tensor["M", "N"]:
    tessera.require(tessera.constraint.Divisible("K", 64))
    return tessera.ops.gemm(A, B)
```

**Current CPU end-to-end path:**

```python
import numpy as np
import tessera as ts

@ts.jit
def mm(A, B):
    return ts.ops.matmul(A, B)

Y = mm(np.ones((2, 3), dtype=np.float32), np.ones((3, 4), dtype=np.float32))
print(mm.schedule_ir)
print(mm.tile_ir)
print(mm.target_ir)
```

256x128 GEMM schedule:

```python
@ts.jit(cpu_tile=(256, 128, 64))
def gemm_256x128(A, B):
    return ts.ops.gemm(A, B)

print(gemm_256x128.schedule_ir)
```

This path supports returned straight-line dataflow made from CPU-backed ops
including `ops.matmul`, `ops.gemm`, `ops.relu`, `ops.sigmoid`, `ops.softmax`,
`ops.reduce`, `ops.sum`, `ops.tanh`, `ops.sin`, and functional `ops.adam`. It
exposes Graph IR, Schedule IR, Tile IR, and Target IR artifacts before executing
on CPU. Other functions continue to use the eager Python fallback and expose the
reason through `.lowering_diagnostics` and `.explain_lowering()`.

**S-series sprint S2 — additional CPU-backed ops (landed 2026-05-10).** The
following primitives now have numpy reference implementations and are
addressable via `ops.<name>`:

- *Reductions:* `mean`, `prod`, `amax`, `amin`, `var`, `std`, `argmax`,
  `argmin`, `cumsum`, `cumprod`. All accept `axis=` and `keepdims=`.
- *Numerical-stability primitives:* `logsumexp`, `log_softmax`, `log1p`,
  `expm1`, `softplus`, `sigmoid_safe`. Each lowers to `tessera.<name>` and
  registers a VJP for reverse-mode autodiff.
- *Numeric helpers:* `clamp`, `where`, `absolute`, `sign`, `minimum`,
  `maximum`, `isnan`, `isinf`, `isfinite`.
- *Comparisons:* `eq`, `ne`, `lt`, `le`, `gt`, `ge` — return a boolean
  tensor; not differentiable.

Each appears in `python/tessera/compiler/op_catalog.py` (graph names
`tessera.mean`, `tessera.logsumexp`, etc.) and `tessera.compiler.primitive_coverage`
imports them automatically as partial coverage entries. VJPs are
registered in `python/tessera/autodiff/vjp.py` for the differentiable
ones; comparisons and `isnan`/`isinf`/`isfinite` deliberately skip VJP
registration.

For functions created dynamically from `stdin` or `exec(...)`, pass explicit
source so the AST frontend can compile them:

```python
src = """
def mm(A, B):
    return ts.ops.gemm(A, B)
"""

ns = {}
exec(src, {"ts": ts}, ns)
mm = ts.jit(ns["mm"], source=src, cpu_tile=(256, 128, 64))

assert mm.is_executable
assert mm.source_origin == "explicit"
```

If no source can be inspected or supplied, `.explain_lowering()` includes
`JIT_SOURCE_UNAVAILABLE` and the wrapper uses the eager fallback.

---

### 2.2 `@tessera.kernel`

**Module:** `tessera.distributed.launch`  
**Import:** `from tessera import kernel` or `import tessera; @tessera.kernel`

Marks a function as a tile-level kernel dispatched by `index_launch`. The decorated function receives per-rank tensor shards and executes one tile operation per launch invocation.

**Signature:**
```python
@tessera.kernel
def fn(
    arg1: tessera.f16[..., ...],
    arg2: tessera.f16[..., ...],
    out: tessera.mut_f32[..., ...],
):
    ...
```

**Returns:** `KernelFn` — a callable wrapper. Kernel functions are not called directly; they are passed to `index_launch(axis)(fn)(shard_lists...)`.

**Constraints:**
- Kernel function parameters must use dtype annotations (`f16`, `bf16`, `f32`, `mut_f32`) or `Region[...]` annotations. Plain Python type hints are accepted but not validated.
- Kernel bodies should not contain `tessera.require(...)` calls (constraints belong in `@jit` functions).

**Example:**
```python
@tessera.kernel
def tp_gemm(
    A: tessera.f16[..., ...],
    B: tessera.f16[..., ...],
    C: tessera.mut_f32[..., ...],
):
    C[:] = tessera.ops.gemm(A, B)
```

---

## 3. Region Privileges

**Module:** `tessera.distributed.region`  
**Import:** `from tessera import Region` or `tessera.Region`

`Region[mode]` is a **type annotation only** — it does not wrap tensors at runtime. It lowers to a `tessera.effect` attribute on Graph IR function arguments. The `@jit` decorator inspects these annotations to enforce privilege contracts.

**Syntax:**
```python
tessera.Region["read"]        # → RegionType(mode="read")
tessera.Region["write"]       # → RegionType(mode="write")
tessera.Region["reduce_sum"]  # → RegionType(mode="reduce_sum", op="sum")
tessera.Region["reduce_max"]  # → RegionType(mode="reduce_max", op="max")
tessera.Region["reduce_min"]  # → RegionType(mode="reduce_min", op="min")
```

**Mode table:**

| Mode string | `RegionType.exclusive` | `RegionType.reduces` | Reduction op |
|-------------|----------------------|---------------------|--------------|
| `"read"` | `False` | `False` | — |
| `"write"` | `True` | `False` | — |
| `"reduce_sum"` | `False` | `True` | `"sum"` |
| `"reduce_max"` | `False` | `True` | `"max"` |
| `"reduce_min"` | `False` | `True` | `"min"` |

**`RegionType` attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `.mode` | `str` | One of the five mode strings above. |
| `.exclusive` | `bool` | `True` for `"write"` — no other region may overlap. |
| `.reduces` | `bool` | `True` for all `"reduce_*"` modes. |
| `.op` | `str \| None` | Reduction op string (`"sum"`, `"max"`, `"min"`) or `None`. |

**Privilege invariants enforced by `@jit`:**
- Two `Region["write"]` parameters on the same tensor → `TesseraPrivilegeError` at decoration time.
- `Region["read"]` and `Region["reduce_sum"]` on the same tensor → allowed (reduce does not conflict with read).
- Any `Region["write"]` combined with any `Region["reduce_*"]` on the same tensor → `TesseraPrivilegeError`.

**Exceptions:**

| Exception | Module | Raised when |
|-----------|--------|-------------|
| `ValueError` | `tessera.distributed.region` | Invalid mode string passed to `Region[...]`. |
| `TesseraPrivilegeError` | `tessera.distributed.region` | Conflicting write regions at `@jit` decoration time. |

**Example:**
```python
@tessera.jit
def grad_accumulate(
    X: tessera.Region["read"],
    G: tessera.Region["reduce_sum"],
):
    G += tessera.ops.gemm(X, X.T)
```

---

## 4. Domain API

**Module:** `tessera.distributed.domain`  
**Import:** `tessera.domain.Rect`

Domains describe the **logical shape** of a tensor. They are always separate from distributions (placement). See §5 for distributions.

### 4.1 `tessera.domain.Rect`

Represents a dense rectangular (N-dimensional box) domain.

**Signature:**
```python
tessera.domain.Rect(dims: tuple[int, ...]) -> Rect
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `dims` | `tuple[int, ...]` | Shape of the domain. All values must be positive integers. |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `.shape` | `tuple[int, ...]` | The `dims` tuple passed at construction. |
| `.rank` | `int` | Number of dimensions (`len(dims)`). |
| `.volume` | `int` | Total element count (product of all dims). |

**Exceptions:**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | Any dimension ≤ 0. |
| `ValueError` | `dims` is empty. |

**Example:**
```python
D = tessera.domain.Rect((4, 128, 256))
assert D.shape == (4, 128, 256)
assert D.rank == 3
assert D.volume == 4 * 128 * 256
```

---

## 5. Distribution API

**Module:** `tessera.distributed.domain`  
**Import:** `tessera.dist.Block`, `tessera.dist.Cyclic`, `tessera.dist.Replicated`

Distributions describe the **placement strategy** — how a domain is partitioned across mesh ranks.

### 5.1 `tessera.dist.Block`

Contiguous block partition. Assigns contiguous chunks of each partitioned dimension to successive ranks.

**Signature:**
```python
tessera.dist.Block(mesh_axes: tuple[str, ...]) -> Block
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mesh_axes` | `tuple[str, ...]` | Non-empty tuple of mesh axis names. The first N dims of the domain are partitioned — one dim per axis name. |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `.mesh_axes` | `tuple[str, ...]` | The mesh axis names passed at construction. |

**Exceptions:**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `mesh_axes` is empty. |

**Phase:** 1–3 (fully implemented).

---

### 5.2 `tessera.dist.Cyclic`

Round-robin (cyclic/interleaved) partition. Element `i` goes to rank `i % rank_count`. Required for load-balanced Mixture-of-Experts.

**Signature:**
```python
tessera.dist.Cyclic(mesh_axes: tuple[str, ...]) -> Cyclic
```

**Attributes:** Same as `Block`.

**Phase:** Stub in Phase 1–3. `make_shard_spec()` raises `NotImplementedError` until Phase 4.

---

### 5.3 `tessera.dist.Replicated`

No partition — tensor is replicated identically on all ranks.

**Signature:**
```python
tessera.dist.Replicated() -> Replicated
```

**Attributes:** None. `make_shard_spec()` returns `ShardSpec.replicate()`.

**Phase:** 1–3 (fully implemented).

---

## 6. Array API

**Module:** `tessera.distributed.array`  
**Import:** `tessera.array.from_domain`

### 6.1 `tessera.array.from_domain`

Creates a `DistributedArray` from a domain and a distribution.

**Signature:**
```python
tessera.array.from_domain(
    domain: Rect,
    dtype: str,
    distribution: Block | Cyclic | Replicated,
    fill: str = "zeros",
    mesh: MeshSpec | None = None,
) -> DistributedArray
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `domain` | `Rect` | — | The logical domain (shape). |
| `dtype` | `str` | — | Storage dtype string. Accepted values: `"f16"`, `"bf16"`, `"f32"`, `"f64"`, `"i32"`, `"i64"`. |
| `distribution` | `Block \| Cyclic \| Replicated` | — | Placement strategy. |
| `fill` | `str` | `"zeros"` | Initial fill strategy. Accepted: `"zeros"`, `"ones"`, `"random"`. |
| `mesh` | `MeshSpec \| None` | `None` | Explicit mesh specification. If `None`, a default single-device mesh is used. |

**Returns:** `DistributedArray`

**Exceptions:**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | Unsupported `dtype` string. |
| `ValueError` | Unsupported `fill` strategy. |
| `NotImplementedError` | `distribution` is `Cyclic` (Phase 4 only). |

---

### 6.2 `DistributedArray`

The primary distributed tensor type. In Phase 1, backed by an eagerly-evaluated numpy array on CPU. Not a `numpy.ndarray` — it carries a `ShardSpec` and a logical shape.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `.shape` | `tuple[int, ...]` | Logical (global) shape — independent of how data is sharded. |
| `.dtype` | `str` | Storage dtype string (e.g. `"bf16"`). |
| `.shard_spec` | `ShardSpec` | Partition metadata describing how the tensor is distributed. |
| `.ndim` | `int` | Number of dimensions (`len(shape)`). |
| `.numel` | `int` | Total element count (product of all shape dims). |

**Methods:**

#### `.parts(axis: str) -> list[DistributedArray]`

Returns per-rank slices along the named mesh axis.

| Parameter | Type | Description |
|-----------|------|-------------|
| `axis` | `str` | Mesh axis name (must be present in `shard_spec.mesh_axes`). |

Returns a list of `DistributedArray` objects, one per rank along the given axis. The list length equals the axis size from the mesh.

Raises `KeyError` if `axis` is not a partitioned axis in `shard_spec`.

#### `.numpy() -> np.ndarray`

Returns the backing numpy array. Phase 1 CPU only. Raises `RuntimeError` if the array is on a GPU device (Phase 3+).

**Example:**
```python
D    = tessera.domain.Rect((4, 128, 256))
dist = tessera.dist.Block(mesh_axes=("dp", "tp"))
X    = tessera.array.from_domain(D, dtype="bf16", distribution=dist)

assert X.shape == (4, 128, 256)
assert X.dtype == "bf16"
assert X.shard_spec.mesh_axes == ("dp", "tp")
assert X.ndim == 3
assert X.numel == 4 * 128 * 256

shards = X.parts("tp")   # list of per-rank slices along tp axis
```

---

## 7. ShardSpec

**Module:** `tessera.distributed.shard`

Describes how a tensor is partitioned across a device mesh.

### 7.1 Construction

```python
ShardSpec(
    partition: tuple[int, ...],
    mesh_axes: tuple[str, ...],
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `partition` | `tuple[int, ...]` | Logical dimension indices that are partitioned. |
| `mesh_axes` | `tuple[str, ...]` | Mesh axis names, one per element of `partition`. Must have the same length as `partition`. |

**Class method:**
```python
ShardSpec.replicate() -> ShardSpec
```
Returns a `ShardSpec` with no partition (fully replicated).

### 7.2 Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.partition` | `tuple[int, ...]` | Logical dimension indices that are partitioned. |
| `.mesh_axes` | `tuple[str, ...]` | Mesh axis names, one per partitioned dim. |
| `.replicated` | `bool` | `True` if the tensor is fully replicated (no partition). |

**Example:**
```python
spec = ShardSpec(partition=(0, 1), mesh_axes=("dp", "tp"))
assert spec.partition == (0, 1)
assert spec.mesh_axes == ("dp", "tp")
assert not spec.replicated

rep = ShardSpec.replicate()
assert rep.replicated
```

---

## 8. Index Launch

**Module:** `tessera.distributed.launch`

`index_launch` dispatches a `@tessera.kernel` function once per rank along a named mesh axis. In Phase 1, ranks execute sequentially. In Phase 3+, they execute in parallel on separate GPU streams.

### 8.1 `tessera.index_launch`

**Signature:**
```python
tessera.index_launch(axis: str) -> IndexLauncher
```

Returns an `IndexLauncher` configured for the given mesh axis.

### 8.2 `IndexLauncher.__call__`

```python
launcher(kernel_fn: KernelFn) -> _ShardDispatcher
```

Binds the kernel to the launcher. Returns a `_ShardDispatcher`.

### 8.3 `_ShardDispatcher.__call__`

```python
dispatcher(*shard_lists: list[DistributedArray]) -> None
```

Executes `kernel_fn` once per rank, passing the corresponding element from each shard list. The number of `shard_lists` must match the number of parameters in `kernel_fn`.

**Full usage pattern:**
```python
tessera.index_launch(axis="tp")(my_kernel)(
    A.parts("tp"),   # list of per-rank shards for arg A
    B.parts("tp"),   # list of per-rank shards for arg B
    C.parts("tp"),   # list of per-rank shards for arg C
)
```

**Exceptions:**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | Number of shard lists does not match kernel parameter count. |
| `ValueError` | Shard list length does not match rank count for the axis. |

---

## 9. Constraint API

**Module:** `tessera.compiler.constraints`  
**Import:** `tessera.constraint.Divisible`, `tessera.constraint.Range`, `tessera.constraint.Equal`

Constraints are checked **at `@jit` decoration time**, not at call time. They are structural assertions about dimension relationships and are extracted from the function body via AST parsing (`_ConstraintExtractor`).

### 9.1 `tessera.require`

```python
tessera.require(constraint: Constraint) -> None
```

Registers a constraint inside a `@jit` function body. The `@jit` decorator extracts these calls via AST parsing before executing the function.

**Note:** `tessera.require(...)` is a no-op at Python runtime — it has no effect when the function is called. Its purpose is purely to communicate constraints to the `@jit` decorator at decoration time.

---

### 9.2 `tessera.constraint.Divisible`

Asserts that a named dimension is divisible by a constant.

```python
tessera.constraint.Divisible(dim: str, divisor: int) -> Divisible
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim` | `str` | Symbolic dimension name (e.g. `"K"`). |
| `divisor` | `int` | Positive integer divisor. |

**Checks:** `dim % divisor == 0`

---

### 9.3 `tessera.constraint.Range`

Asserts that a named dimension falls within an inclusive range.

```python
tessera.constraint.Range(dim: str, lo: int, hi: int) -> Range
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim` | `str` | Symbolic dimension name. |
| `lo` | `int` | Inclusive lower bound. |
| `hi` | `int` | Inclusive upper bound. |

**Checks:** `lo <= dim <= hi`

---

### 9.4 `tessera.constraint.Equal`

Asserts that two named dimensions are equal.

```python
tessera.constraint.Equal(dim_a: str, dim_b: str) -> Equal
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim_a` | `str` | First symbolic dimension name. |
| `dim_b` | `str` | Second symbolic dimension name. |

**Checks:** `dim_a == dim_b`

---

### 9.5 `ConstraintSolver`

Internal class used by `@jit`. Not part of the public API but documented here for reference:

```python
solver = ConstraintSolver()
solver.add(tessera.constraint.Divisible("K", 64))
solver.check({"K": 128})    # → None (passes)
solver.check({"K": 100})    # → raises TesseraConstraintError
solver.check_all({"K": 128, "M": 512})  # checks all added constraints
```

---

### 9.6 Constraint example

```python
@tessera.jit(bindings={"K": 128, "M": 512})
def aligned_gemm(
    A: tessera.Tensor["M", "K"],
    B: tessera.Tensor["K", "N"],
) -> tessera.Tensor["M", "N"]:
    tessera.require(tessera.constraint.Divisible("K", 64))
    tessera.require(tessera.constraint.Range("M", 1, 8192))
    tessera.require(tessera.constraint.Equal("K", "K"))   # identity — always true
    return tessera.ops.gemm(A, B)
# Raises TesseraConstraintError immediately if K % 64 != 0 (given bindings)
```

---

### 9.7 Shape-system helpers

**Module:** `tessera.shape`  
**Imports:** `tessera.sym`, `tessera.dim`, `tessera.check_shapes`, `tessera.shape.*`

The shape-system helpers are the Python mirror of `docs/spec/SHAPE_SYSTEM.md`.
They support symbolic dimensions, derived dimension products, broadcasting,
reshape element-count checks, logical-dimension sharding checks, schedule tile
feasibility, and runtime shape witnesses.

```python
B, N, M, D = tessera.sym("B N M D")

@tessera.check_shapes
def attention(q: tessera.Tensor[B, N, D],
              k: tessera.Tensor[B, M, D]) -> tessera.Tensor[B, N, M]:
    ...
```

Public helpers:

| Symbol | Purpose |
|--------|---------|
| `Dim`, `dim`, `sym` | Symbolic/concrete dimensions |
| `Shape`, `Layout`, `ShapeShard` | Tensor shape metadata |
| `ShapeConstraintGraph` | Equality, derived, divisibility, and range checks |
| `broadcast_shape` | NumPy-style symbolic broadcast rule |
| `matmul_shape` | Matmul result-shape rule with batch broadcasting |
| `reshape_shape` | Reshape element-count validation |
| `check_shard` | Logical dimension divisibility against mesh axes |
| `check_schedule_tile` | Schedule divisibility check with padding suggestion |
| `RuntimeShapeWitness` | Runtime refinement for dynamic dimensions |

These helpers do not replace the MLIR verifier; they make the same contracts
available to the frontend, tests, and IDE-facing tooling.

---

### 9.8 Debugging helpers

**Module:** `tessera.debug`  
**Convenience namespace:** `tessera.graph`

Debugging helpers provide Graph IR inspection, numerical tracing, gradient
checking, and determinism checks. The behavior is described in
`docs/guides/Tessera_Debugging_Tools_Guide.md`.

| Symbol | Purpose |
|--------|---------|
| `trace_graph(value, ir_level="graph")` | Return a printable/exportable graph trace |
| `export_graphviz(value)` | Return GraphViz DOT for a trace |
| `debug_trace(samples=0, stream=None, metadata=None)` | Context manager for numerical summaries |
| `trace_value(name, value)` | Record a tensor-like value in the active trace |
| `summarize_tensor(value)` | Compute shape/dtype/mean/std/min/max/finite summary |
| `TensorSummary.to_dict()` | JSON-native bounded tensor summary |
| `DebugTrace.to_dict()` / `.to_json()` | Structured debug trace export |
| `GraphTrace.to_dict()` / `.to_json()` | Structured graph trace export |
| `debug_value(name, value, metadata=None)` | Named capture point; returns `value` unchanged |
| `debug_artifact(name, artifact=None, metadata=None)` | Schedule-artifact debug descriptor |
| `debug_barrier(name, queue_id=None, scope="block", metadata=None)` | Tile barrier debug descriptor |
| `replay_manifest(value=None, **metadata)` | Bounded replay manifest for artifacts/JIT wrappers |
| `save_replay_manifest(path, value=None, **metadata)` | Write replay manifest JSON |
| `replay_capture(value=None, **metadata)` | Convenience alias for replay manifest capture |
| `check_grad(fn, inputs, analytic_grads=...)` | Finite-difference gradient check |
| `check_determinism(fn, runs=5)` | Repeated-run reproducibility check |

`tessera.graph.trace`, `tessera.graph.debug_trace`, and
`tessera.graph.export_graphviz` are aliases for the graph-oriented debug
helpers. `tessera.graph.debug_value` maps to `tessera.debug.debug_value`, and
`tessera.graph.replay_capture` maps to `tessera.debug.replay_capture`.

Replay manifests and full tensor capture are separate contracts: manifests
include artifact hashes, metadata, and selected environment switches, but do not
include full tensor values unless callers attach bounded summaries explicitly.

---

### 9.9 Profiling and autotuning helpers

**Modules:** `tessera.profiler`, `tessera.autotune`  
**Guide:** `docs/guides/Tessera_Profiling_And_Autotuning_Guide.md`

Profiling helpers:

| Symbol | Purpose |
|--------|---------|
| `profiler.session()` | Context manager that collects profile events |
| `ProfileSession.record(...)` | Record latency, FLOPs, bytes, counters |
| `ProfileSession.measure(...)` | Measure a callable region |
| `profiler.measure_backend(..., backend="cpu" \| "apple_cpu")` | Wall-clock backend measurement with telemetry |
| `ProfileSession.report()` | Render a tabular text report |
| `ProfileSession.timeline(path)` | Write Chrome Trace Event JSON |

Autotuning helpers:

| Symbol | Purpose |
|--------|---------|
| `autotune(op, shapes=(M,N,K), ...)` | Tune a GEMM-like op and persist results |
| `autotune.load(op, shapes, ...)` | Load the best cached result |
| `autotune.cache_key(...)` | Return public cache key tuple |
| `autotune.schedule_artifact(...)` | Build a schedule artifact from a result |
| `autotune.RooflineCostModel` | Analytical FLOPs/bytes cost model |

The public autotune facade currently supports GEMM/matmul shapes and delegates
search and SQLite persistence to `tessera.compiler.autotune_v2`. Synthetic
roofline tuning and schedule artifact generation are implemented. CPU and Apple
CPU `method="on_device"` runs use wall-clock measurement. CUDA/HIP/NVIDIA/ROCm
device timers remain planned and report `unmeasured` or `backend_unavailable`
instead of silently pretending to benchmark hardware.

Developer commands:

| Command | Purpose |
|---------|---------|
| `tessera-mlir my_model.py --emit=graph-ir --debug` | Dump static/debug IR with source locations |
| `tessera-mlir my_model.py --emit=metadata` | Emit source-inspection metadata JSON |
| `tessera-mlir my_model.py --emit=diagnostics` | Emit diagnostics JSON |
| `tessera-mlir my_model.py --emit=trace` | Emit Chrome Trace Event JSON |
| `tessera-mlir my_model.py --emit=graphviz` | Emit GraphViz DOT |
| `tessera-mlir my_model.py --emit=all --artifacts-dir out` | Write all static debug artifacts |
| `tessera-mlir my_model.py --mode=compile_artifact --symbol=step --emit=all` | Opt-in import of a selected JIT symbol and artifact inspection without launching tensors |
| `tessera-prof my_model.py --metrics=flops,bandwidth,occupancy` | Print a profiling report |
| `tessera-prof my_model.py --trace=trace.json` | Write Chrome Trace Event JSON |
| `tessera-prof my_model.py --emit=json --autotune` | Emit profiling JSON and GEMM tuning artifact metadata |
| `tessera-autotune --op=matmul --shapes=128,128,128` | Run public GEMM/matmul tuner and write cache/artifacts |

---

### 9.10 Fault tolerance and elasticity helpers

**Modules:** `tessera.fault`, `tessera.elastic`, `tessera.checkpoint`  
**Guide:** `docs/guides/Tessera_Fault_Tolerance_And_Elasticity_Guide.md`

Fault helpers:

| Symbol | Purpose |
|--------|---------|
| `fault.on_failure(policy=...)` | Attach drain/resume, fail-fast, or manual policy |
| `fault.on_preempt(grace_s=..., action=...)` | Attach scheduler preemption policy |
| `fault.inject(...)` | Controlled failure injection context |

Elastic helpers:

| Symbol | Purpose |
|--------|---------|
| `elastic.configure(...)` | Configure rendezvous backend and rank bounds |
| `elastic.elastic(...)` | Context manager for elastic membership |
| `elastic.reshard(...)` | Build a mesh resharding plan |
| `elastic.on_topology_change(...)` | Kernel/autotune invalidation policy |

Runtime checkpoint helpers:

| Symbol | Purpose |
|--------|---------|
| `checkpoint.save(tag=..., atomic=True)` | Write sharded checkpoint manifest |
| `checkpoint.load(tag, remap_to=...)` | Load committed manifest and remap mesh |
| `checkpoint.enable_async(...)` | Configure async checkpointing policy |
| `checkpoint.last_committed(...)` | Find latest committed checkpoint tag |

`tessera.checkpoint` is for runtime checkpoint/restart. Activation
checkpointing/rematerialization remains under `tessera.compiler.checkpoint`.

---

### 9.11 Inference server helpers

**Module:** `tessera.server`  
**Guide:** `docs/guides/Tessera_Inference_Server_Guide.md`

Inference helpers define the Python foundation for production serving:

| Symbol | Purpose |
|--------|---------|
| `server.load_package(path)` | Load and validate `.tspkg/manifest.yaml` |
| `server.ModelManifest` | Normative package manifest representation |
| `server.KVCacheConfig` | Paged KV cache settings |
| `server.KVCacheManager` | KV page accounting and hit-rate helper |
| `server.scheduler(...)` | Continuous/sequence/priority scheduler metadata |
| `server.App` | In-process model/route registry for future HTTP/gRPC binding |
| `server.capabilities(...)` | Runtime placement capability descriptor |

The server module is not a network transport. It defines validated contracts for
package loading, scheduling, KV cache accounting, health, and metrics.

---

## 10. Effect System

**Module:** `tessera.compiler.effects`

Effects are **inferred, not declared** (except the `deterministic` flag on `@jit`). The `EffectLattice` walks the Graph IR call graph and propagates effects upward.

### 10.1 `Effect` Enum

```python
from tessera.compiler.effects import Effect

Effect.pure    # value 0 — no side effects; recompute-safe
Effect.random  # value 1 — calls RNG; result varies between runs
Effect.movement   # value 2 — explicit data movement / async copy-wait
Effect.state      # value 3 — compiler-visible state, such as KV cache
Effect.collective # value 4 — device/rank communication
Effect.memory     # value 5 — writes mutable tensors or host-visible aliases
Effect.io         # value 6 — host I/O or unknown external work
Effect.top        # value 7 — unknown / unconstrained
```

**Lattice order (least → most permissive):**

```
pure(0) < random(1) < movement(2) < state(3)
  < collective(4) < memory(5) < io(6) < top(7)
```

**Lattice join:**
```python
effect_a.join(effect_b)   # → max(effect_a.value, effect_b.value)
```

The join of two effects is the more permissive one. A function that calls a `random` op has at minimum `Effect.random`, regardless of how pure its other ops are.

### 10.2 Effect Mapping for `tessera.ops`

| Op | Effect |
|----|--------|
| `gemm`, `matmul`, `layer_norm`, `softmax`, `reduce`, `sum`, `gelu`, `tanh`, `relu`, `add`, `mul` | `pure` |
| `transpose`, `cast` | `pure` |
| `dropout` (when `training=True`) | `random` |
| `prefetch`, `async_copy`, `await_movement` | `movement` |
| `kv_cache_create`, `kv_cache_append`, `kv_cache_prune`, `kv_cache_read`, `kv_cache_write`, `flash_attn` | `state` |
| `conv2d` | `pure` |
| `all_reduce`, `reduce_scatter`, `all_gather`, `all_to_all` | `collective` |
| `fused_epilogue` | `pure` |

### 10.3 `deterministic=True` contract

When `@jit(deterministic=True)` is applied:
- If the inferred effect is `Effect.random` and no `seed` is provided → `TesseraEffectError`.
- If `seed` is provided, random ops are considered seeded and deterministic → allowed.
- `Effect.movement`, `Effect.state`, and `Effect.collective` are allowed when represented in Tessera IR, where ordering contracts can be enforced.
- `Effect.io` and `Effect.top` are rejected because host I/O and unknown external work cannot be made deterministic by Tessera.

```python
# OK: pure effect, deterministic is fine
@tessera.jit(deterministic=True)
def norm(x): return tessera.ops.layer_norm(x)

# OK: random but seeded
@tessera.jit(deterministic=True, seed=42)
def drop(x): return tessera.ops.dropout(x, p=0.1)

# ERROR: random without seed
@tessera.jit(deterministic=True)
def bad(x): return tessera.ops.dropout(x, p=0.1)  # TesseraEffectError
```

---

## 11. GPU Target API

**Module:** `tessera.compiler.gpu_target`  
**Import:** `from tessera.compiler.gpu_target import GPUTargetProfile, ISA`

**Phase:** 3+

### 11.1 `ISA` Enum

```python
class ISA(enum.IntEnum):
    SM_80  = 80    # NVIDIA A100
    SM_86  = 86    # NVIDIA RTX 30xx
    SM_89  = 89    # NVIDIA RTX 40xx
    SM_90  = 90    # NVIDIA H100 / GH200
    SM_100 = 100   # NVIDIA B100 / GB200
    SM_120 = 120   # Rubin-family placeholder until final CC numbering is published
```

WGMMA, TMA, and mbarrier transaction-barrier support are available on `SM_90`
and above. `SM_120` is a Tessera planning placeholder for Rubin-family targets,
not a claim about final NVIDIA compute capability numbering.

### 11.2 `GPUTargetProfile`

Controls how the GPU lowering pipeline emits code.

**Signature:**
```python
GPUTargetProfile(
    isa: ISA = ISA.SM_90,
    warps_per_cta: int = 4,
    shared_mem_bytes: int | None = None,
    prefer_ptx: bool = True,
    pipeline_stages: int = 2,
) -> GPUTargetProfile
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `isa` | `ISA` | `ISA.SM_90` | GPU ISA target. Determines which backend lowering path is selected. |
| `warps_per_cta` | `int` | `4` | Warps per Cooperative Thread Array. Must be a power of 2. |
| `shared_mem_bytes` | `int \| None` | `None` | Shared memory budget in bytes. `None` = use SM default maximum. |
| `prefer_ptx` | `bool` | `True` | Emit PTX inline asm for WGMMA/TMA rather than LLVM IR intrinsics. |
| `pipeline_stages` | `int` | `2` | Software pipeline depth for double-buffering in Tile IR. |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `.supports_wgmma` | `bool` | `True` when `isa >= ISA.SM_90`. |
| `.supports_tma` | `bool` | `True` when `isa >= ISA.SM_90`. |
| `.supports_mbarrier` | `bool` | `True` when `isa >= ISA.SM_90`. |
| `.tensor_core_dtypes` | `frozenset[str]` | Tensor Core dtype names for the target. |
| `.cuda_core_dtypes` | `frozenset[str]` | CUDA-core scalar dtype names for the target. |
| `.runtime_arch` | `str` | CUDA architecture string such as `sm_90a`, `sm_100a`, or `sm_120`. |
| `.supports_tcgen05` | `bool` | True for Blackwell SM100+ TCGEN05 contracts. |
| `.supports_tmem` | `bool` | True for Blackwell SM100+ Tensor Memory contracts. |
| `.supports_cta_pairs` | `bool` | True when CTA-pair metadata is available. |
| `.supports_block_scaled_mma` | `bool` | True for Blackwell block-scaled MMA dtypes. |

**Exceptions:**

| Exception | Condition |
|-----------|-----------|
| `TesseraTargetError` | `warps_per_cta` is not a power of 2. |
| `TesseraTargetError` | `pipeline_stages < 1`. |
| `TesseraTargetError` | `isa` is not a valid `ISA` enum value. |

**Example:**
```python
from tessera.compiler.gpu_target import GPUTargetProfile, ISA

profile = GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4)
assert profile.supports_wgmma   # True
assert profile.supports_tma     # True

@tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4))
def flash_attn_fwd(Q, K, V):
    tessera.require(tessera.constraint.Divisible("D", 64))
    return tessera.ops.flash_attn(Q, K, V, causal=True)
```

---

## 12. FlashAttention Lowering Config

**Module:** `tessera.compiler.attn_lower`  
**Import:** `from tessera.compiler.attn_lower import FlashAttnLoweringConfig`

**Phase:** 3+

Controls how `tessera.ops.flash_attn` is lowered to FA-4 Tile IR. Consumed by `TileIRLoweringPass` (C++) via MLIR attributes.

### 12.1 `FlashAttnLoweringConfig`

**Signature (dataclass):**
```python
FlashAttnLoweringConfig(
    tile_q: int = 64,
    tile_kv: int = 64,
    pipeline_stages: int = 2,
    causal: bool = False,
    dropout_p: float = 0.0,
    seed: int | None = None,
) -> FlashAttnLoweringConfig
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tile_q` | `int` | `64` | Q tile size — rows of Q processed per outer loop step. Must be a positive power of 2. |
| `tile_kv` | `int` | `64` | KV tile size — columns processed per inner loop step. Must be a positive power of 2. |
| `pipeline_stages` | `int` | `2` | Software double-buffer pipeline stages. Must be ≥ 1. |
| `causal` | `bool` | `False` | If `True`, emits `CausalMaskOp` in the inner loop. |
| `dropout_p` | `float` | `0.0` | Dropout probability in `[0, 1)`. If `> 0`, emits `DropoutMaskOp`. |
| `seed` | `int \| None` | `None` | RNG seed for dropout. Required when `dropout_p > 0`. |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `.has_dropout` | `bool` | `True` when `dropout_p > 0`. |

**Methods:**

#### `.to_mlir_attrs() -> str`

Returns an inline MLIR attribute dictionary string for the `tessera.flash_attn` op. Format:
```
{tessera.tile_q = 64 : i32, tessera.tile_kv = 64 : i32, tessera.pipeline_stages = 2 : i32, causal = false}
```

**Exceptions:**

| Exception | Condition |
|-----------|-----------|
| `TesseraAttnConfigError` | `tile_q` is not a positive power of 2. |
| `TesseraAttnConfigError` | `tile_kv` is not a positive power of 2. |
| `TesseraAttnConfigError` | `dropout_p` not in `[0.0, 1.0)`. |
| `TesseraAttnConfigError` | `dropout_p > 0` and `seed` is `None`. |
| `TesseraAttnConfigError` | `pipeline_stages < 1`. |

**Pre-built constant:**
```python
from tessera.compiler.attn_lower import SM90_DEFAULT
# FlashAttnLoweringConfig(tile_q=64, tile_kv=64, pipeline_stages=2, causal=False)
```

---

## 13. Operations Namespace

**Module:** `tessera.ops`  
**Import:** `tessera.ops.<name>`  
**Library contract:** `docs/operations/Tessera_Standard_Operations.md`

Phase 1 implementations are numpy-backed references or single-rank mock
helpers. Phase 3 dispatches to compiled MLIR artifacts via the GPU lowering
pipeline where that target path is implemented. Native hardware runtime support
is a separate claim and is recorded in `RUNTIME_ABI_SPEC.md`.
The Tessera Standard Operator Library reserves a broad set of compiler-visible
operator names. This table lists the current Python runtime and registry
surface: implemented entries have NumPy/reference behavior, while artifact-only
entries still have stable names, Graph IR spellings, effects, and lowering
metadata for frontend and compiler tests.

Additional S-series runtime catalog entries are part of the same public
`tessera.ops` namespace and are guarded against spec drift:
`acos`, `argsort`, `asin`, `atan`, `atan2`, `bitwise_and`, `bitwise_not`,
`bitwise_or`, `bitwise_xor`, `ceil`, `cosh`, `digamma`, `dynamic_slice`,
`dynamic_update_slice`, `erf`, `erfc`, `flatten`, `flip`, `floor`,
`floor_div`, `index_select`, `index_update`, `lgamma`, `logical_and`,
`logical_not`, `logical_or`, `logical_xor`, `nonzero`, `permute`,
`reciprocal`, `rsqrt`, `scatter_add`, `scatter_reduce`, `sinh`, `sort`,
`squeeze`, `stack`, `trunc`, and `unsqueeze`.

Optimizer ops share the `tessera.optim` mixed-precision policy:
`compute_dtype="fp32"` and `state_dtype="fp32"` by default, optional
`master_dtype` for fp16/bf16/quantized-adjacent training, and returned
parameter leaves cast back to their original storage dtype unless
`cast_updates_to_param_dtype=False`. The tree API accepts
`compute_dtype`, `state_dtype`, `master_dtype`, and
`cast_updates_to_param_dtype` across Adam, AdamW, Momentum, Adafactor, and
Lion. Hyperparameters, integer steps, dtype policy fields, and casts are
treated as nondifferentiable; parameter/state tensor outputs participate in
VJP/JVP rules.

Attention-family ops support fp32 accumulation for fp16/bf16 inputs. Linear
and recurrent variants (`lightning_attention`, `gated_deltanet`,
`kimi_delta_attention`, `modified_delta_attention`, and hybrid policies) accept
optional recurrent `state`; when `return_state=True`, they return
`(output, new_state)` and default recurrent state storage to fp32.

Reasoning RL helpers are public as both `tessera.rl.<name>` and
`tessera.ops.<name>`. `RolloutBatch` stores `logp_new`, `logp_old`, optional
`ref_logp`, rewards, mask, and free-form metadata for post-training loops.
PPO/GRPO/CISPO losses support masks, reference-policy KL, and
`reduction={"none","sum","mean"}`; CISPO clips importance weights directly and
uses the clipped weight as a detached multiplier on the log-prob objective.

| Operation | Signature | Effect | Current behavior |
|-----------|-----------|--------|------------------|
| `gemm(A, B)` | `(array, array) → array` | `pure` | `np.matmul(A, B)` |
| `matmul(A, B)` | `(array, array) → array` | `pure` | Alias for `gemm` |
| `batched_gemm(A, B)` | `(array, array) → array` | `pure` | Batched `np.matmul` reference |
| `einsum(spec, *tensors)` | `(str, arrays...) → array` | `pure` | `np.einsum` reference |
| `factorized_matmul(A, B, rank)` | `(array, array) → array` | `pure` | Low-rank SVD reference over `A @ B` |
| `tri_solve(A, b, lower=True)` | `(array, array) → array` | `pure` | Triangular `np.linalg.solve` reference |
| `cholesky(A)` | `(array) → array` | `pure` | `np.linalg.cholesky` reference |
| `qr(A)` | `(array) → tuple` | `pure` | `np.linalg.qr` reference |
| `svd(A)` | `(array) → tuple` | `pure` | `np.linalg.svd(..., full_matrices=False)` reference |
| `layer_norm(x, eps=1e-5)` | `(array) → array` | `pure` | NumPy layer norm |
| `softmax(x, axis=-1)` | `(array) → array` | `pure` | NumPy softmax |
| `softmax_safe(x, axis=-1)` | `(array) → array` | `pure` | Stable NumPy softmax alias |
| `reduce(x, op="sum", axis=None, keepdims=False)` | `(array) → array` | `pure` | NumPy sum reduction; non-sum reductions planned |
| `sum(x, axis=None, keepdims=False)` | `(array) → array` | `pure` | Alias for `reduce(..., op="sum")` |
| `gelu(x)` | `(array) → array` | `pure` | NumPy GELU |
| `tanh(x)` | `(array) → array` | `pure` | NumPy tanh |
| `add(x, y=None, scalar=None)` | `(array, array/scalar) → array` | `pure` | NumPy addition; also used by Python frontend binary `+` lowering |
| `mul(x, y=None, scalar=None)` | `(array, array/scalar) → array` | `pure` | NumPy multiplication; also used by Python frontend binary `*` lowering |
| `relu(x)` | `(array) → array` | `pure` | NumPy ReLU |
| `silu(x)` | `(array) → array` | `pure` | NumPy SiLU |
| `silu_mul(a, b)` | `(array, array) → array` | `pure` | Fused `silu(a) * b` primitive — anchors the SwiGLU `matmul → silu_mul → matmul` 3-op chain that the Schedule IR fusion recognizer collapses into `tessera.swiglu_fused` for backends with a fused MLP-block kernel |
| `sigmoid(x)` | `(array) → array` | `pure` | NumPy sigmoid |
| `sin(x)` | `(array) → array` | `pure` | NumPy sine |
| `adam(param, grad, moment1, moment2, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, step=1)` | `(array,array,array,array) → tuple` | `pure` | Functional NumPy Adam step; math/state default to fp32 and outputs cast back to param dtype |
| `adamw(params, grads, state=None, ...)` | `(tree,tree,dict?) → tuple` | `pure` | Tree-structured AdamW optimizer step from `tessera.optim` |
| `momentum(params, grads, state=None, lr, momentum=0.9, ...)` | `(tree,tree,dict?) → tuple` | `pure` | Tree-structured SGD with momentum |
| `adafactor(params, grads, state=None, ...)` | `(tree,tree,dict?) → tuple` | `pure` | Memory-efficient Adafactor with factored fp32 accumulator slots for matrix leaves |
| `lion(params, grads, state=None, ...)` | `(tree,tree,dict?) → tuple` | `pure` | Lion optimizer with stop-gradient sign update and fp32 momentum slots |
| `linear_general(x, W, bias=None, axis=-1)` | `(array,array,array?) → array` | `pure` | S7 axis-flexible linear projection; Graph IR op `tessera.linear_general` |
| `sgd(params, grads, lr)` | `(array,array) → array` | `pure` | S10 functional SGD reference; Graph IR op `tessera.sgd` |
| `mse_loss(pred, target, reduction="mean")` | `(array,array) → scalar/array` | `pure` | S11 MSE criterion; Graph IR op `tessera.loss.mse` |
| `mae_loss(pred, target, reduction="mean")` | `(array,array) → scalar/array` | `pure` | S11 MAE criterion; Graph IR op `tessera.loss.mae` |
| `huber_loss(pred, target, delta=1.0, reduction="mean")` | `(array,array) → scalar/array` | `pure` | S11 Huber criterion; Graph IR op `tessera.loss.huber` |
| `smooth_l1_loss(pred, target, beta=1.0, reduction="mean")` | `(array,array) → scalar/array` | `pure` | S11 SmoothL1 criterion; Graph IR op `tessera.loss.smooth_l1` |
| `log_cosh_loss(pred, target, reduction="mean")` | `(array,array) → scalar/array` | `pure` | S11 log-cosh criterion; Graph IR op `tessera.loss.log_cosh` |
| `cross_entropy_loss(logits, targets, reduction="mean")` | `(array,array) → scalar/array` | `pure` | S11 cross-entropy criterion; Graph IR op `tessera.loss.cross_entropy` |
| `binary_cross_entropy_loss(logits, targets, reduction="mean")` | `(array,array) → scalar/array` | `pure` | S11 BCE-with-logits criterion; Graph IR op `tessera.loss.binary_cross_entropy` |
| `ddpm_noise_pred_loss(pred_noise, true_noise, reduction="mean")` | `(array,array) → scalar/array` | `pure` | S11 diffusion noise-prediction criterion; Graph IR op `tessera.loss.ddpm_noise_pred` |
| `score_matching_loss(score, target_score, reduction="mean")` | `(array,array) → scalar/array` | `pure` | S11 score-matching criterion; Graph IR op `tessera.loss.score_matching` |
| `vlb_loss(terms, reduction="mean")` | `(array) → scalar/array` | `pure` | S11 diffusion VLB reducer; Graph IR op `tessera.loss.vlb` |
| `normalize_group_advantages(rewards, group_axis=1)` | `(array) → array` | `pure` | Reasoning RL helper; normalizes rewards within each prompt/group |
| `ppo_policy_loss(logp_new, logp_old, advantages, ...)` | `(array,array,array) → scalar/array` | `pure` | PPO clipped surrogate loss with optional mask, entropy, and reference-policy KL |
| `grpo_policy_loss(logp_new, logp_old, rewards=None, advantages=None, ...)` | `(array,array,optional array) → scalar/array` | `pure` | GRPO policy loss; derives group-normalized advantages from rewards when needed |
| `cispo_policy_loss(logp_new, logp_old, rewards=None, advantages=None, ...)` | `(array,array,optional array) → scalar/array` | `pure` | CISPO loss; clips importance-sampling weights directly and detaches the clipped weight from the log-prob gradient |
| `rmsnorm(x, eps=1e-5)` | `(array) → array` | `pure` | NumPy RMSNorm reference |
| `rmsnorm_safe(x, eps=1e-6)` | `(array) → array` | `pure` | NumPy RMSNorm reference with safer default epsilon |
| `transpose(x, axes=None)` | `(array) → array` | `pure` | `np.transpose(x, axes)` |
| `cast(x, dtype)` | `(array, str) → array` | `pure` | `x.astype(dtype)` |
| `arange(start, stop=None, step=1, dtype="fp32")` | `(...) → 1-D array` | `pure` | Theme 9 — `numpy.arange` over `[start, stop)`. Single-arg form starts at 0 |
| `gather(x, indices, axis=0)` | `(array, int-array) → array` | `pure` | Theme 9 — `numpy.take(x, indices, axis=axis)`. VJP scatters via `np.add.at` (correct under repeated indices) |
| `clip(x, min_val=None, max_val=None)` | `(array, float?, float?) → array` | `pure` | Theme 9 — element-wise clamp; either bound may be `None`. Straight-through gradient |
| `masked_fill(x, mask, value)` | `(array, bool-array, scalar) → array` | `pure` | Theme 9 — replace `x` where `mask` is True. Used by attention masks and softmax `-inf` fill |
| `quantize_fp8(x, format="e4m3", scale=None)` | `(array) → (array, float)` | `pure` | Theme 10 — per-tensor symmetric fp8 quantization. `format` is `"e4m3"` (max 448) or `"e5m2"` (max 57344). Returns `(x_q_as_fp32, scale)`. Native cast via `ml_dtypes` when installed; pure-numpy mantissa-snap fallback otherwise |
| `dequantize_fp8(x_q, scale, format="e4m3")` | `(array, float) → array` | `pure` | Theme 10 — inverse of `quantize_fp8`. Pair-wise op so the IR layer can intercept (quantize→dequantize) for fusion |
| `quantize_fp6(x, format="e3m2", scale=None)` | `(array) → (array, float)` | `pure` | Item 2 — fp6 quantization. `format` is `"e2m3"` (max ±7.5; precision-favored) or `"e3m2"` (max ±28; range-favored) |
| `dequantize_fp6(x_q, scale, format="e3m2")` | `(array, float) → array` | `pure` | Item 2 — inverse of `quantize_fp6` |
| `quantize_fp4(x, format="e2m1", scale=None)` | `(array) → (array, float)` | `pure` | Item 2 — fp4 quantization. Only `"e2m1"` supported (Blackwell hardware format) |
| `dequantize_fp4(x_q, scale, format="e2m1")` | `(array, float) → array` | `pure` | Item 2 — inverse of `quantize_fp4` |
| `quantize_nvfp4(x, block_size=16)` | `(array) → (array, scales)` | `pure` | Item 2 — block-scaled fp4 (Blackwell convention). One fp32 scale per `block_size`-element block along the last axis |
| `dequantize_nvfp4(x_q, scales, block_size=16)` | `(array, array) → array` | `pure` | Item 2 — inverse of `quantize_nvfp4` |
| `latent_kv_compress(x, w_dkv)` | `(array, array) → array` | `pure` | Theme 5 — MLA latent compression `c = x @ W_dkv`. Distinct op_name anchors the FlashMLA target-pass match |
| `latent_kv_expand_k(c, w_uk)` / `latent_kv_expand_v(c, w_uv)` | `(array, array) → array` | `pure` | Theme 5 — expand cached latent back to K / V at attention time. Will be absorbed into the score kernel by Phase G FlashMLA pass |
| `rope_split(x, rope_dim)` | `(array) → (array, array)` | `pure` | Theme 5 — split last dim into `(rope_part, no_rope_part)`. Used by MLA's decoupled-RoPE design so positional encoding only touches the small rope_dim slice |
| `rope_merge(rope_part, no_rope_part)` | `(array, array) → array` | `pure` | Theme 5 — inverse of `rope_split`; concatenate along last dim |
| `alibi(num_heads, seq_len, slopes=None)` | `(int,int) → array` | `pure` | S7 ALiBi positional-bias helper |
| `ntk_rope(x, theta, scale=1.0)` | `(array,array) → array` | `pure` | S7 NTK-scaled RoPE wrapper |
| `dropout(x, p=0.1, training=True)` | `(array) → array` | `random` | Bernoulli mask, numpy rng |
| `conv2d(x, weight, bias=None, stride=1, padding=0)` | `(NHWC, HWIO) → NHWC` | `pure` | NumPy NHWC/HWIO reference |
| `conv3d(x, weight, bias=None, stride=1, padding=0)` | `(NDHWC, DHWIO) → NDHWC` | `pure` | NumPy NDHWC/DHWIO reference |
| `qkv_projection(x, W_qkv)` | `(array, array) → tuple` | `pure` | Matmul and split into Q/K/V references |
| `flash_attn(Q, K, V, scale=None, causal=False, dropout_p=0.0, seed=None)` | `(array,array,array) → array` | `pure` / `random` when dropout is active | Naive O(S²) Phase 1; FA-4 Phase 3 |
| `linear_attn(Q, K, V, *, feature_map="elu", state=None, chunk_size=None, decay=None, causal=True)` | `(array,array,array) → array` | `state` | Linear / kernel-feature attention. Recurrent or chunk-parallel forms; optional decay (RetNet/GLA/Mamba2-selective). Returns just `O` — pair with `linear_attn_state` for chained-chunk inference (see attention_variants_plan LA-1) |
| `linear_attn_state(Q, K, V, ...)` | `(array,array,array) → array` | `state` | Companion to `linear_attn` returning the post-update state `(B, H, D_qk, D_v)` |
| `power_attn(Q, K, V, *, state, window=None, deg=2, causal=True)` | `(array,array,array) → array` | `state` | LA-4 — Symmetric power attention (linear-cost). Promoted from `examples/advanced/power_retention/` |
| `retention(Q, K, V, *, log_g=None, deg=2, chunk=128, causal=True)` | `(array,array,array) → array` | `state` | LA-4 — RetNet-style retention with multiplicative decay |
| `lightning_attention(Q, K, V, ...)` | `(array,array,array) → array/tuple` | `state` | Lightning-style identity-feature linear attention; optional fp32 recurrent state |
| `gated_attention(Q, K, V, gate, ...)` | `(array,array,array,array) → array` | `state` | Softmax attention multiplied by a learned gate |
| `gated_deltanet(Q, K, V, gate=None, beta=None, decay=None, ...)` | `(array,array,array,optional array...) → array/tuple` | `state` | Gated DeltaNet recurrent attention reference |
| `kimi_delta_attention(Q, K, V, ...)` | `(array,array,array,optional array...) → array/tuple` | `state` | Kimi Delta Attention recurrence |
| `modified_delta_attention(Q, K, V, ...)` | `(array,array,array,optional array...) → array/tuple` | `state` | Bounded modified delta-attention recurrence |
| `hybrid_attention(Q, K, V, pattern="...", layer_index=0, ...)` | `(array,array,array) → array/tuple` | `state` | Named Ling/Kimi hybrid attention policy wrapper |
| `multi_head_attention(Q, K, V, num_heads, ...)` | `(array,array,array) → array` | `state` | S7 multi-head attention wrapper over `flash_attn` |
| `gqa_attention(Q, K, V, num_query_heads, num_kv_heads, ...)` | `(array,array,array) → array` | `state` | S7 grouped-query attention wrapper |
| `mqa_attention(Q, K, V, ...)` | `(array,array,array) → array` | `state` | S7 multi-query attention wrapper |
| `mla_decode(Q, K_latent, V_latent, W_k=None, W_v=None, ...)` | `(array,array,array,optional array,optional array) → array` | `state` | S7 MLA decode wrapper over latent expansion and `flash_attn` |
| `mla_decode_fused(x, w_dkv, w_uk, w_uv, q, *, scale=None, causal=False)` | `(array,array,array,array,array) → array` | `state` | MLA-1 — DeepSeek MLA decode block as a single op (result of the Schedule IR `tessera-mla-fusion` pass) |
| `attn_sliding_window(Q, K, V, *, window_size, causal=True)` | `(array,array,array) → array` | `state` | NSA-1 branch — sliding-window dense local attention |
| `attn_local_window_2d(Q, K, V, *, window=(1,1))` | `(array,array,array) → array` | `state` | Gap 4 — 2D local-window attention for spatial grids (weather/climate, ViT-style local bias); Q/K/V rank-5 `(B, H, Hq, Wq, D)` |
| `attn_compressed_blocks(Q, K_c, V_c)` | `(array,array,array) → array` | `state` | NSA-1 branch — attention over per-block compressed K/V summaries |
| `attn_top_k_blocks(Q, K, V, *, scores, top_k, block_size, causal=True)` | `(array,array,array) → array` | `state` | NSA-1 branch — top-k block-selected attention |
| `deepseek_sparse_attention(Q, K, V, gate_logits=None, ...)` | `(array,array,array,optional array) → array` | `state` | DeepSeek/NSA wrapper composing sliding, compressed, and top-k branches |
| `compress_blocks(K, V, *, block_size, w_compress=None)` | `(array,array) → tuple` | `pure` | NSA-2 — chunk K/V into block_size groups; returns `(K_c, V_c)` per-block summaries (mean or learnable projection) |
| `mor_router(x, w_router, *, max_depth)` | `(array, array) → int64-array` | `pure` | Phase F-MoR — token-choice depth router for Mixture of Recursions. Argmax of the router logits + 1; returns ints in [1, max_depth] |
| `mor_partition(x, depth, *, step)` | `(array, int64-array) → bool-array` | `pure` | Phase F-MoR — bool mask of tokens whose target depth ≥ step (1-indexed) |
| `mor_scatter(full, updated, mask)` | `(array, array, bool-array) → array` | `pure` | Phase F-MoR — write `updated` values into `full` at masked positions (frozen-token semantics for the unselected rows) |
| `moe(x, experts)`, `moe_dispatch(x, route)`, `moe_combine(partials, inverse_route)` | `(array, ...) → array` | `collective` | Reference/mock transport helpers; distributed execution planned |
| `all_reduce(x, op="sum")` | `(array) → array` | `collective` | Single-rank/mock no-op unless using explicit mock collective helpers |
| `reduce_scatter(x, op="sum", axis=0)` | `(array) → array` | `collective` | Single-rank/mock no-op unless using explicit mock collective helpers |
| `all_gather(x, axis=0)` | `(array) → array` | `collective` | Single-rank/mock no-op unless using explicit mock collective helpers |
| `all_to_all(x, axis=0)` | `(array) → array` | `collective` | Single-rank/mock no-op; MoE planner tests cover planning behavior |
| `rng_uniform(shape, dtype="fp32", seed=None, lo=0.0, hi=1.0)` | `(shape) → array` | `random` | NumPy generator helper |
| `rng_normal(shape, dtype="fp32", seed=None, mean=0.0, std=1.0)` | `(shape) → array` | `random` | NumPy generator helper |
| `fused_epilogue(x, bias=None, activation="linear")` | `(array) → array` | `pure` | Applies bias + activation |
| `fft(x, axis=-1)`, `ifft(xf, axis=-1)`, `rfft(x, axis=-1)`, `irfft(xf, axis=-1, n=None)` | `(array) → array` | `pure` | NumPy FFT helpers |
| `stft(x, win, hop)`, `istft(xf, win, hop)` | `(array, array) → array` | `pure` | NumPy FFT-derived windowed transform references |
| `spectral_filter(Xf, Hf)` | `(array, array) → array` | `pure` | Frequency-domain multiply reference |
| `dct(x, type=2, axis=-1)` | `(array) → array` | `pure` | NumPy FFT-derived DCT helper |
| `spectral_conv(x, w)` | `(array, array) → array` | `pure` | NumPy spectral convolution helper |
| `spmm_coo(A_coo, B)` | `(sparse/dense, array) → array` | `pure` | Dense fallback or simple COO tuple reference |
| `spmm_csr(A_csr, B)` | `(sparse/dense, array) → array` | `pure` | Dense fallback or simple CSR tuple reference |
| `sddmm(A, B, mask)` | `(array, array, array) → array` | `pure` | Dense sampled matmul reference |
| `bsmm(X, W_bsr)` | `(array, array) → array` | `pure` | Dense fallback block-sparse matmul reference |
| `segment_reduce(x, seg_ids, op="sum")` | `(array, array) → array` | `pure` | NumPy segment reduction reference |
| `rearrange(x, layout)`, `pack(x, layout)`, `unpack(x)`, `tile_view(x, BM, BN, BK=None)` | `(array, ...) → array` | `pure` / `movement` for materialized pack/unpack | Reference layout helpers; compiler-visible layout metadata |
| `kv_cache_append(cache, key, value)` | `(cache, array, array) → cache` | `state` | Reference in-process KV cache helper |
| `kv_cache_prune(cache, max_entries=None, max_seq=None)` | `(cache) → cache` | `state` | Reference in-process KV cache helper |
| `selective_ssm(x, A, B, C, D=None, initial_state=None)` | `(array, array, array, array, [array, array]) → array` | `state` | Mamba2-style selective state-space scan; closed-form JVP through the recurrence shipped (see `tessera.selective_ssm` Graph IR op) |
| `complex_mul(z, w)` / `complex_div(z, w, eps=1e-12)` / `complex_exp(z)` / `complex_log(z)` / `complex_sqrt(z, branch=0)` / `complex_pow(z, w)` | `(complex, complex) → complex` | `pure` | M7 Visual Complex Analysis pointwise math (Graph IR ops `tessera.complex_mul` / `complex_div` / `complex_exp` / `complex_log` / `complex_sqrt` / `complex_pow`). Fused Apple GPU MSL kernel ships for `complex_mul`/`complex_exp` (fp32). Long-tail (`complex_div`/`log`/`sqrt`/`pow`) runs via Python reference today (fp32-only); fp16/bf16 are planned target dtypes for the NVIDIA WGMMA / ROCm MFMA / Apple GPU MSL kernels that land in Phase G / H / M7 follow-up. |
| `complex_conjugate(z)` / `complex_abs(z)` / `complex_arg(z)` | `(complex) → complex/real` | `pure` | M7 pointwise complex helpers (Graph IR ops `tessera.complex_conjugate` / `complex_abs` / `complex_arg`). CPU reference today is fp32-only; apple_gpu / nvidia_sm90+ / rocm slots are reserved with fp16/bf16 in the planned dtype matrix — those are target dtypes for the future native kernel, not currently-runnable dtypes. |
| `mobius(z, a, b, c, d)` / `mobius_from_three_points(src, dst)` / `stereographic(x, y, z)` | Möbius / projective | `pure` | M7 Möbius transformation family (Graph IR ops `tessera.mobius` / `mobius_from_three_points` / `stereographic`). Fused Apple GPU MSL kernels ship for `mobius` + `stereographic` (fp32). `mobius_from_three_points` runs via Python reference today; native kernel slots reserved on every GPU target with fp32/fp16/bf16 listed as the planned kernel-target dtype set. |
| `cross_ratio(z1, z2, z3, z4)` / `is_concyclic(z1, z2, z3, z4)` / `check_cauchy_riemann(f)` | `(complex×4) → scalar` / certificate | `pure` | M7 projective invariants + holomorphicity certificate (Graph IR ops `tessera.cross_ratio` / `tessera.is_concyclic` / `tessera.check_cauchy_riemann`). Python reference today is fp32; fp16/bf16 on the planned GPU slots are kernel target dtypes only. |
| `dz(f, z0, h=1e-5)` / `dbar(f, z0, h=1e-5)` / `laplacian_2d(f, ...)` | stencils | `pure` | M7 Wirtinger derivatives + Laplacian (Graph IR ops `tessera.dz` / `tessera.dbar` / `tessera.laplacian_2d`, `lowering="stencil"`). Python reference today is fp32; planned GPU dtype matrix (fp32/fp16/bf16) describes the future stencil kernel, not what runs now. |
| `conformal_jacobian(f, z0)` / `conformal_energy_on_sphere(f, ...)` | conformal | `pure` | M7 conformal Jacobian + spherical conformal energy (Graph IR ops `tessera.conformal_jacobian` / `tessera.conformal_energy_on_sphere`). Same dtype semantics as the rest of the M7 long-tail: fp32 reference today; fp16/bf16 are planned kernel-target dtypes. |

> **Reading the dtype matrices.** Across the M7 family — and the rest
> of the backend manifest — a dtype tuple is interpreted **against the
> entry's status**: ``status="reference"`` lists dtypes that run
> today via the Python path; ``status="fused"`` lists dtypes the
> native kernel actually supports today; ``status="planned"`` lists
> the target dtype matrix for the **unbuilt** native kernel. fp16 and
> bf16 in a ``planned`` row are explicit kernel-author contracts for
> Phase G / H / M7 follow-up, never claims about what runs in the
> current runtime. See ``BackendKernelEntry.dtypes`` for the
> normative interpretation.

**Operator registry behavior:** `tessera.ops.registry` tracks reference,
lowering, and runtime-kernel handlers. Current public runtime names are mirrored
by `python/tessera/compiler/op_catalog.py`; tests guard this table against drift.

**`flash_attn` parameter details:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Q` | array | — | Query tensor, shape `[B, H, S, D]` |
| `K` | array | — | Key tensor, shape `[B, H, S, D]` |
| `V` | array | — | Value tensor, shape `[B, H, S, D]` |
| `scale` | `float \| None` | `None` | Attention scale factor. `None` = `1 / sqrt(D)`. |
| `causal` | `bool` | `False` | Apply causal (lower-triangular) mask. |
| `dropout_p` | `float` | `0.0` | Dropout probability in `[0, 1)`. If `> 0`, applies inverted dropout to attention weights. |
| `seed` | `int \| None` | `None` | Optional NumPy RNG seed for deterministic dropout masks. |

**`fused_epilogue` activation values:** `"linear"` (identity), `"relu"`, `"gelu"`.

---

## 14. Tensor Annotations

**Module:** `tessera.core.tensor`  
**Import:** `tessera.Tensor`

`tessera.Tensor[dim_name, ...]` is a type annotation for `@jit` function parameters. Symbolic dimension names are used by the `ConstraintSolver`.

**Syntax:**
```python
tessera.Tensor["M", "K"]      # 2D tensor with symbolic dims M, K
tessera.Tensor["B", "H", "S", "D"]  # 4D tensor
```

The subscript creates a `TensorType` with `.dim_names` set to the provided strings. Dimension names must be strings or `...` (Ellipsis, meaning "any size").

**Interaction with constraints:** The string names in `Tensor["M", "K"]` must match the `dim` strings used in `tessera.constraint.Divisible("K", 64)` etc.

---

## 15. Dtype Annotations

**Module:** `tessera.core.tensor` (exported via `tessera.__init__`)

Dtype annotations are used in `@tessera.kernel` parameter signatures to express both the storage type and the read/write privilege. The canonical tensor-attribute and dtype vocabulary is `docs/reference/tessera_tensor_attributes.md`.

| Annotation | Storage dtype | Privilege | Use in |
|------------|---------------|-----------|--------|
| `tessera.f16[..., ...]` | FP16 | Read-only | `@kernel` params |
| `tessera.bf16[..., ...]` | BF16 | Read-only | `@kernel` params |
| `tessera.f32[..., ...]` | FP32 | Read-only | `@kernel` params |
| `tessera.mut_f32[..., ...]` | FP32 | Write-privileged | `@kernel` output params |

The `[..., ...]` subscript specifies dimensionality via Ellipsis (arbitrary shape). These annotations participate in Python's `__class_getitem__` protocol and are validated at `@kernel` decoration time.

### 15.1 Canonical Dtype Helper Module

**Module:** `tessera.dtype`  
**Import:** `import tessera; tessera.dtype` or `from tessera.dtype import ...`

The dtype helper module is the implementation source for canonical dtype
membership, accepted aliases, planned/gated dtype names, and helper-level
promotion inspection.

| API | Purpose |
|-----|---------|
| `Dtype(value)` | Str-compatible wrapper around a canonical dtype name. Aliases normalize at construction. |
| `canonicalize_dtype(value, allow_planned_gated=False)` | Normalize aliases such as `"f32"` / `"float32"` to canonical spellings such as `"fp32"`. |
| `assert_canonical_dtype(value, context=None)` | Canonicalize or raise `TesseraDtypeError` with caller context. |
| `result_type(*dtypes, mode="standard")` | Compute standard helper-level promotion or reject mixed dtypes in `mode="strict"`. |
| `canonical_dtypes()` | Return the canonical dtype set. |
| `planned_gated_dtypes()` | Return recognized but gated dtype names. |
| `dtype_aliases()` | Return accepted alias spellings. |
| `is_canonical_dtype(value)` / `is_planned_gated_dtype(value)` / `is_known_dtype(value)` | Predicate helpers for validation and audits. |

TF32 is not a storage dtype. Use `numeric_policy.math_mode = "tf32"` on
`fp32` storage instead of `dtype="tf32"`.

---

## 16. Error Types

All Tessera exception types are importable from their respective modules. The table below gives the canonical import path for each.

| Exception | Module | Raised when |
|-----------|--------|-------------|
| `TesseraConstraintError` | `tessera.compiler.constraints` | A structural constraint (Divisible/Range/Equal) is violated at `@jit` decoration time when concrete `bindings` are provided. |
| `TesseraEffectError` | `tessera.compiler.effects` | `@jit(deterministic=True)` is applied to a function with unseeded RNG, host I/O, or unknown external work. |
| `TesseraJitError` | `tessera.compiler.jit` | Graph IR emission pipeline failure — malformed IR, unsupported op, or internal compiler error. |
| `TesseraTargetError` | `tessera.compiler.gpu_target` | Invalid `GPUTargetProfile` parameters (e.g. non-power-of-2 `warps_per_cta`, invalid ISA). |
| `TesseraAttnConfigError` | `tessera.compiler.attn_lower` | Invalid `FlashAttnLoweringConfig` parameters (e.g. non-power-of-2 tile size, missing seed for dropout). |
| `TesseraPrivilegeError` | `tessera.distributed.region` | Conflicting `Region` privileges on the same tensor (two `write` regions, or `write` + `reduce`). Raised at `@jit` decoration time. |
| `MockCollectiveError` | `tessera.testing.mock_collective` | A mock collective fails (shape mismatch) or a rank thread raises an exception or times out. |

---

## 17. Testing Utilities

**Module:** `tessera.testing.mock_collective`  
**Import:** `from tessera.testing import MockRankGroup`

Provides thread-based fake multi-rank execution for testing distributed programs without NCCL, MPI, or CUDA. Phase 1 CPU (numpy) only.

### 17.1 `MockRankGroup`

Simulates `n` ranks using Python threads sharing in-process numpy buffers.

**Signature:**
```python
MockRankGroup(
    n: int,
    mesh_axes: dict[str, int] | None = None,
) -> MockRankGroup
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | — | Total number of ranks (`world_size`). Must be ≥ 1. |
| `mesh_axes` | `dict[str, int] \| None` | `None` | Logical mesh mapping axis name → count. Product of all values must equal `n`. If `None`, defaults to `{"dp": n}`. |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `.world_size` | `int` | Total rank count. |
| `.mesh_axes` | `dict[str, int]` | Mesh axis configuration. |
| `.ranks` | `list[MockRank]` | List of `MockRank` objects, one per rank. |

**Methods:**

#### `.run(fn, *, timeout=30.0) -> list[Any]`

Executes `fn(rank: MockRank)` on each rank in a separate Python thread.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fn` | `Callable[[MockRank], Any]` | — | Worker function. Receives a `MockRank` for this rank's collective operations. |
| `timeout` | `float` | `30.0` | Per-thread timeout in seconds. |

Returns a list of per-rank results in rank order. Raises `MockCollectiveError` if any rank raises an exception or times out.

**Exceptions:**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `n < 1`. |
| `ValueError` | `mesh_axes` product != `n`. |
| `MockCollectiveError` | Any rank raises an exception. |
| `MockCollectiveError` | Any rank times out. |

---

### 17.2 `MockRank`

Per-rank view returned to each worker function by `MockRankGroup.run()`.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `.rank` | `int` | This rank's 0-based index. |
| `.world_size` | `int` | Total number of ranks. |
| `.mesh_axes` | `dict[str, int]` | Mesh axis configuration (shared from group). |

**Methods:**

#### `.all_reduce(tensor, op="sum") -> np.ndarray`

Reduces `tensor` across all ranks using `op`. All ranks receive the result.

Supported ops: `"sum"`, `"max"`, `"min"`, `"prod"`.

#### `.reduce_scatter(tensor, axis=0, op="sum") -> np.ndarray`

All-reduces `tensor` then returns this rank's `1/world_size` slice along `axis`.

Raises `MockCollectiveError` if `tensor.shape[axis]` is not divisible by `world_size`.

#### `.all_gather(tensor, axis=0) -> np.ndarray`

Gathers `tensor` from all ranks and concatenates along `axis`. Returns the full concatenated tensor.

#### `.barrier() -> None`

Waits until all ranks reach this barrier.

---

### 17.3 Usage example

```python
from tessera.testing import MockRankGroup
import numpy as np

group = MockRankGroup(n=4, mesh_axes={"dp": 4})

def worker(rank):
    local = np.ones((256,), dtype=np.float32) * rank.rank
    result = rank.all_reduce(local, op="sum")
    # result == 256 elements, each = 0+1+2+3 = 6
    assert result[0] == 6.0
    return rank.rank

results = group.run(worker)
assert results == [0, 1, 2, 3]
```

**Multi-axis mesh:**
```python
group = MockRankGroup(n=8, mesh_axes={"dp": 4, "tp": 2})
assert group.world_size == 8
assert group.mesh_axes["dp"] == 4
assert group.mesh_axes["tp"] == 2
```

---

## Appendix A: Public Symbol Index

Quick-lookup table of all public symbols and their canonical module paths.

| Symbol | Canonical import |
|--------|-----------------|
| `@tessera.jit` | `tessera.jit` |
| `@tessera.kernel` | `tessera.kernel` |
| `tessera.Region` | `tessera.distributed.region.Region` |
| `tessera.domain.Rect` | `tessera.distributed.domain.Rect` |
| `tessera.dist.Block` | `tessera.distributed.domain.Block` |
| `tessera.dist.Cyclic` | `tessera.distributed.domain.Cyclic` |
| `tessera.dist.Replicated` | `tessera.distributed.domain.Replicated` |
| `tessera.array.from_domain` | `tessera.distributed.array.from_domain` |
| `tessera.index_launch` | `tessera.distributed.launch.index_launch` |
| `tessera.require` | `tessera.require` (top-level no-op; extracted by AST) |
| `tessera.constraint.Divisible` | `tessera.compiler.constraints.Divisible` |
| `tessera.constraint.Range` | `tessera.compiler.constraints.Range` |
| `tessera.constraint.Equal` | `tessera.compiler.constraints.Equal` |
| `tessera.Tensor` | `tessera.core.tensor.Tensor` |
| `tessera.f16` | `tessera.core.tensor` |
| `tessera.bf16` | `tessera.core.tensor` |
| `tessera.f32` | `tessera.core.tensor` |
| `tessera.mut_f32` | `tessera.core.tensor` |
| `GPUTargetProfile` | `tessera.compiler.gpu_target.GPUTargetProfile` |
| `ISA` | `tessera.compiler.gpu_target.ISA` |
| `FlashAttnLoweringConfig` | `tessera.compiler.attn_lower.FlashAttnLoweringConfig` |
| `TesseraConstraintError` | `tessera.compiler.constraints.TesseraConstraintError` |
| `TesseraEffectError` | `tessera.compiler.effects.TesseraEffectError` |
| `TesseraJitError` | `tessera.compiler.jit.TesseraJitError` |
| `TesseraTargetError` | `tessera.compiler.gpu_target.TesseraTargetError` |
| `TesseraAttnConfigError` | `tessera.compiler.attn_lower.TesseraAttnConfigError` |
| `TesseraPrivilegeError` | `tessera.distributed.region.TesseraPrivilegeError` |
| `MockRankGroup` | `tessera.testing.mock_collective.MockRankGroup` |
| `MockRank` | `tessera.testing.mock_collective.MockRank` |
| `MockCollectiveError` | `tessera.testing.mock_collective.MockCollectiveError` |
