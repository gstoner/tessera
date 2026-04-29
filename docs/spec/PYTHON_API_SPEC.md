---
status: Normative
classification: Normative
last_updated: 2026-04-28
---

# Tessera Python API Specification
**Status:** Normative — grounded in `python/tessera/` Phase 1–3 implementation  
**Last updated:** April 28, 2026  
**Authority:** This document specifies every public Python symbol in Tessera Phases 1–3. For naming disputes, `docs/CANONICAL_API.md` is the final arbiter. For compiler internals (pass pipeline, IR layers), see `docs/spec/COMPILER_REFERENCE.md`.

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
├── server.py                     # inference package, scheduler, KV cache, app registry
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
tessera.graph        # tessera.graph.trace / debug_trace / export_graphviz

# Profiling and autotuning
tessera.profiler     # tessera.profiler.session / record / timeline
tessera.autotune     # callable autotune facade; also .load / .cache_key

# Developer commands
tessera-mlir         # static/debug IR dumps: --emit=graph-ir|schedule-ir|tile-ir|target-ir
tessera-prof         # profiling report and Chrome trace export

# Fault tolerance and elasticity
tessera.fault        # on_failure / on_preempt / inject
tessera.elastic      # configure / elastic / reshard
tessera.checkpoint   # runtime checkpoint save/load/manifest helpers

# Inference serving
tessera.server       # App / load_package / scheduler / KVCacheManager

# Ops namespace
tessera.ops          # tessera.ops.gemm / layer_norm / dropout / etc.

# Tensor type annotations
tessera.Tensor       # tessera.Tensor["M", "K"]
tessera.f16          # tessera.f16[..., ...]
tessera.bf16         # tessera.bf16[..., ...]
tessera.f32          # tessera.f32[..., ...]
tessera.mut_f32      # tessera.mut_f32[..., ...]
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
)
def fn(...): ...
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deterministic` | `bool` | `False` | If `True`, raises `TesseraEffectError` if the function body contains any unseeded `random` effect op (e.g. `dropout` without `seed`). |
| `seed` | `int \| None` | `None` | RNG seed. Required when `deterministic=True` and the body calls a `random` effect op. |
| `bindings` | `dict[str, int] \| None` | `None` | Concrete dimension bindings for early constraint checking at decoration time. Example: `{"K": 128, "M": 512}`. If `None`, constraint checking is deferred to the first call where sizes are known. |
| `target` | `GPUTargetProfile \| None` | `None` | GPU lowering target. `None` routes to the CPU/interpreted path. When set to a profile with `isa >= ISA.SM_90`, the GPU lowering pipeline (`tessera-lower-to-gpu`) is selected. |
| `attn_config` | `FlashAttnLoweringConfig \| None` | `None` | Flash attention tile sizes and pipeline configuration. If `None` and `target.isa >= ISA.SM_90`, `SM90_DEFAULT` is used automatically. |

**Returns:** `JitFn` — a callable wrapper with the following additional attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.graph_ir` | `GraphIRBuilder` | The emitted Graph IR. Call `.to_mlir()` to get MLIR text. |
| `.effect` | `Effect` | Inferred effect of the compiled function. |
| `.constraints` | `list[Constraint]` | All constraints extracted from the function body. |
| `.target` | `GPUTargetProfile \| None` | The target profile passed at decoration time. |

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

The shape-system helpers are the Python mirror of `docs/spec/shape-system.md`.
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
| `debug_trace(samples=0, stream=None)` | Context manager for numerical summaries |
| `trace_value(name, value)` | Record a tensor-like value in the active trace |
| `summarize_tensor(value)` | Compute shape/dtype/mean/std/min/max/finite summary |
| `check_grad(fn, inputs, analytic_grads=...)` | Finite-difference gradient check |
| `check_determinism(fn, runs=5)` | Repeated-run reproducibility check |

`tessera.graph.trace`, `tessera.graph.debug_trace`, and
`tessera.graph.export_graphviz` are aliases for the graph-oriented debug
helpers.

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
search and SQLite persistence to `tessera.compiler.autotune_v2`.

Developer commands:

| Command | Purpose |
|---------|---------|
| `tessera-mlir my_model.py --emit=graph-ir --debug` | Dump static/debug IR with source locations |
| `tessera-prof my_model.py --metrics=flops,bandwidth,occupancy` | Print a profiling report |
| `tessera-prof my_model.py --trace=trace.json` | Write Chrome Trace Event JSON |

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
Effect.memory  # value 2 — reads or writes mutable state (KV cache, etc.)
Effect.io      # value 3 — collective communication or host I/O
Effect.top     # value 4 — unknown / unconstrained
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
| `gemm`, `matmul`, `layer_norm`, `softmax`, `gelu`, `relu` | `pure` |
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
- `Effect.memory` and `Effect.io` are not restricted by `deterministic` — only `random` is.

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

Phase 1 implementations are numpy-backed stubs. Phase 3 dispatches to compiled MLIR kernels via the GPU lowering pipeline.
The Tessera Standard Operator Library reserves additional operator names for
planned compiler/runtime paths; this table lists only the current Phase 1-3
runtime surface.

| Operation | Signature | Effect | Phase 1 behavior |
|-----------|-----------|--------|-----------------|
| `gemm(A, B)` | `(array, array) → array` | `pure` | `np.matmul(A, B)` |
| `matmul(A, B)` | `(array, array) → array` | `pure` | Alias for `gemm` |
| `layer_norm(x, eps=1e-5)` | `(array) → array` | `pure` | NumPy layer norm |
| `softmax(x, axis=-1)` | `(array) → array` | `pure` | NumPy softmax |
| `gelu(x)` | `(array) → array` | `pure` | NumPy GELU |
| `relu(x)` | `(array) → array` | `pure` | NumPy ReLU |
| `transpose(x, axes=None)` | `(array) → array` | `pure` | `np.transpose(x, axes)` |
| `cast(x, dtype)` | `(array, str) → array` | `pure` | `x.astype(dtype)` |
| `dropout(x, p=0.1, training=True)` | `(array) → array` | `random` | Bernoulli mask, numpy rng |
| `conv2d(x, weight, bias=None, stride=1, padding=0)` | stub | `pure` | Returns zeros |
| `flash_attn(Q, K, V, scale=None, causal=False)` | `(array,array,array) → array` | `pure` | Naive O(S²) Phase 1; FA-4 Phase 3 |
| `all_reduce(x, op="sum")` | stub | `io` | No-op Phase 1 |
| `reduce_scatter(x, op="sum", axis=0)` | stub | `io` | No-op Phase 1 |
| `all_gather(x, axis=0)` | stub | `io` | No-op Phase 1 |
| `fused_epilogue(x, bias=None, activation="linear")` | `(array) → array` | `pure` | Applies bias + activation |

**`flash_attn` parameter details:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Q` | array | — | Query tensor, shape `[B, H, S, D]` |
| `K` | array | — | Key tensor, shape `[B, H, S, D]` |
| `V` | array | — | Value tensor, shape `[B, H, S, D]` |
| `scale` | `float \| None` | `None` | Attention scale factor. `None` = `1 / sqrt(D)`. |
| `causal` | `bool` | `False` | Apply causal (lower-triangular) mask. |

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

Dtype annotations are used in `@tessera.kernel` parameter signatures to express both the storage type and the read/write privilege.

| Annotation | Storage dtype | Privilege | Use in |
|------------|---------------|-----------|--------|
| `tessera.f16[..., ...]` | FP16 | Read-only | `@kernel` params |
| `tessera.bf16[..., ...]` | BF16 | Read-only | `@kernel` params |
| `tessera.f32[..., ...]` | FP32 | Read-only | `@kernel` params |
| `tessera.mut_f32[..., ...]` | FP32 | Write-privileged | `@kernel` output params |

The `[..., ...]` subscript specifies dimensionality via Ellipsis (arbitrary shape). These annotations participate in Python's `__class_getitem__` protocol and are validated at `@kernel` decoration time.

---

## 16. Error Types

All Tessera exception types are importable from their respective modules. The table below gives the canonical import path for each.

| Exception | Module | Raised when |
|-----------|--------|-------------|
| `TesseraConstraintError` | `tessera.compiler.constraints` | A structural constraint (Divisible/Range/Equal) is violated. Raised at `@jit` decoration time when concrete `bindings` are provided, or at first call when sizes become known. |
| `TesseraEffectError` | `tessera.compiler.effects` | `@jit(deterministic=True)` is applied to a function with an unseeded `random` effect (e.g. `dropout` without a `seed`). |
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
