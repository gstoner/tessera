---
status: Normative
classification: Normative
last_updated: 2026-04-26
---

# Tessera Canonical API Quick Reference
**Status:** Normative — grounded in `python/tessera/` Phase 1–3 implementation  
**Last updated:** April 26, 2026  
**Use this document** to resolve any naming disagreement in other docs. If something here conflicts with another doc, this file wins.

---

## Core Rule: One Name per Concept

Other docs use `@tessera.function`, `@ts.kernel`, `@jit`, and other variants. **None of those are correct.** The table below is authoritative.

---

## Decorators

| Decorator | Import path | Purpose |
|-----------|-------------|---------|
| `@tessera.jit` | `tessera.jit` | Compile a Python function: run ConstraintSolver, infer effects, emit Graph IR |
| `@tessera.kernel` | `tessera.kernel` | Mark a tile-level kernel function dispatched by `index_launch` |

```python
import tessera

@tessera.jit
def step(W: tessera.Region["read"], X: tessera.Region["read"],
         Y: tessera.Region["write"]):
    Y[:] = tessera.ops.gemm(X, W)

@tessera.kernel
def tp_gemm(A: tessera.f16[..., ...], B: tessera.f16[..., ...],
            C: tessera.mut_f32[..., ...]):
    C[:] = tessera.ops.gemm(A, B)
```

### `@tessera.jit` parameters

```python
@tessera.jit(
    deterministic: bool = False,     # if True, forbid unseeded random effects
    seed: int | None = None,         # RNG seed; required when using dropout under deterministic=True
    bindings: dict[str,int] | None = None,  # concrete dim sizes for early constraint checking
    target: GPUTargetProfile | None = None, # GPU target; None = CPU/interpreted path
    attn_config: FlashAttnLoweringConfig | None = None,  # FA-4 tile sizes; auto-set on SM_90+
)
```

---

## Region Privileges

`Region[mode]` is a **type annotation only** — not a runtime wrapper. It lowers to `tessera.effect` attributes on Graph IR function arguments.

| Syntax | Mode | Exclusive? | Reduces? |
|--------|------|-----------|---------|
| `Region["read"]` | read-only | No | No |
| `Region["write"]` | exclusive write | Yes | No |
| `Region["reduce_sum"]` | parallel sum | No | Yes (`op="sum"`) |
| `Region["reduce_max"]` | parallel max | No | Yes (`op="max"`) |
| `Region["reduce_min"]` | parallel min | No | Yes (`op="min"`) |

```python
@tessera.jit
def grad_step(X: tessera.Region["read"], G: tessera.Region["reduce_sum"]):
    G += tessera.ops.gemm(X, X.T)
```

Invalid mode → `ValueError` at annotation time. Conflicting write regions → `TesseraPrivilegeError` at `@jit` decoration time.

---

## Domain & Distribution API

Domains and distributions are **always separate objects** — shape vs. placement. Never merge them.

```python
# 1. Define the logical shape
D = tessera.domain.Rect((4, 128, 256))      # dims: (batch, seq, hidden)

# 2. Define the placement strategy  
dist = tessera.dist.Block(mesh_axes=("dp", "tp"))

# 3. Create the distributed array
X = tessera.array.from_domain(D, dtype="bf16", distribution=dist)

assert X.shape == (4, 128, 256)              # logical (global) shape
assert X.shard_spec.mesh_axes == ("dp", "tp")
assert X.dtype == "bf16"
```

### `tessera.domain`

| Symbol | Type | Description |
|--------|------|-------------|
| `tessera.domain.Rect(dims)` | `Rect` | Dense rectangular domain. `dims` is a tuple of positive ints. |

### `tessera.dist`

| Symbol | Phase | Description |
|--------|-------|-------------|
| `tessera.dist.Block(mesh_axes)` | 1–3 | Contiguous block partition. `mesh_axes` is a non-empty tuple of strings. |
| `tessera.dist.Cyclic(mesh_axes)` | 4 | Round-robin partition. `make_shard_spec` raises `NotImplementedError` until Phase 4. |
| `tessera.dist.Replicated()` | 1–3 | No partition — tensor replicated on all ranks. |

### `tessera.array`

| Symbol | Description |
|--------|-------------|
| `tessera.array.from_domain(domain, dtype, distribution, fill="zeros", mesh=None)` | Create `DistributedArray` |

### `DistributedArray`

| Attribute / Method | Type | Description |
|--------------------|------|-------------|
| `.shape` | `tuple[int,...]` | Logical (global) shape |
| `.dtype` | `str` | Storage dtype string |
| `.shard_spec` | `ShardSpec` | Partition metadata |
| `.ndim` | `int` | Number of dimensions |
| `.numel` | `int` | Total element count |
| `.parts(axis)` | `list[DistributedArray]` | Per-rank slices along mesh axis |
| `.numpy()` | `np.ndarray` | Backing numpy array (Phase 1 CPU only) |

---

## `ShardSpec`

```python
ShardSpec(partition=(0, 1), mesh_axes=("dp", "tp"))
ShardSpec.replicate()   # fully replicated
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `.partition` | `tuple[int,...]` | Logical dimension indices that are partitioned |
| `.mesh_axes` | `tuple[str,...]` | Mesh axis names, one per partitioned dim |
| `.replicated` | `bool` | True if tensor is fully replicated |

---

## Constraint API

Constraints are checked at `@jit` decoration time, not at call time.

```python
@tessera.jit
def aligned_gemm(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    tessera.require(tessera.constraint.Divisible("K", 64))
    tessera.require(tessera.constraint.Range("M", 1, 8192))
    tessera.require(tessera.constraint.Equal("D_in", "D_out"))
    return tessera.ops.gemm(A, B)
```

| Symbol | Args | Checks |
|--------|------|--------|
| `tessera.constraint.Divisible(dim, divisor)` | `str, int` | `dim % divisor == 0` |
| `tessera.constraint.Range(dim, lo, hi)` | `str, int, int` | `lo <= dim <= hi` |
| `tessera.constraint.Equal(dim_a, dim_b)` | `str, str` | `dim_a == dim_b` |

Violation → `TesseraConstraintError` (at decoration time if concrete bindings exist; else deferred to first call where sizes are known via `bindings=` kwarg).

---

## Effect System

Effects are **inferred, not declared** (except the `deterministic` flag).

| Effect | Value | Meaning |
|--------|-------|---------|
| `Effect.pure` | 0 | No side effects; recompute-safe |
| `Effect.random` | 1 | Calls RNG; result varies |
| `Effect.memory` | 2 | Reads/writes mutable state (KV cache, etc.) |
| `Effect.io` | 3 | Collective communication or host I/O |
| `Effect.top` | 4 | Unknown / unconstrained |

Lattice join: `effect_a.join(effect_b)` → `max(a, b)`.

```python
@tessera.jit(deterministic=True, seed=42)
def stable_fwd(x):
    return tessera.ops.layer_norm(x)   # pure → OK under deterministic

@tessera.jit(deterministic=True)
def bad_fwd(x):
    return tessera.ops.dropout(x, p=0.1)  # random → TesseraEffectError
```

---

## Index Launch

```python
tessera.index_launch(axis="tp")(my_kernel)(
    A.parts("tp"),    # list of per-rank shards
    B.parts("tp"),
    C.parts("tp"),
)
```

`index_launch(axis)` returns an `IndexLauncher`. Calling it with a kernel returns a `_ShardDispatcher`. Calling that with shard lists executes the kernel once per rank (sequentially in Phase 1, parallel in Phase 3+).

---

## GPU Target API (Phase 3+)

```python
from tessera.compiler.gpu_target import GPUTargetProfile, ISA

@tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4))
def flash_attn_fwd(Q, K, V):
    tessera.require(tessera.constraint.Divisible("D", 64))
    return tessera.ops.flash_attn(Q, K, V, causal=True)
```

| `ISA` value | Hardware | WGMMA | TMA |
|-------------|----------|-------|-----|
| `ISA.SM_80` | A100 | ❌ | ❌ |
| `ISA.SM_86` | RTX 30xx | ❌ | ❌ |
| `ISA.SM_89` | RTX 40xx | ❌ | ❌ |
| `ISA.SM_90` | H100 / GH200 | ✅ | ✅ |
| `ISA.SM_100` | B100 / GB200 | ✅ | ✅ |

`GPUTargetProfile` key parameters: `isa`, `warps_per_cta` (default 4, must be power of 2), `shared_mem_bytes` (None = SM default), `pipeline_stages` (default 2).

Key properties: `.supports_wgmma` → `isa >= SM_90`, `.supports_tma` → `isa >= SM_90`.

---

## Dtype Annotations

| Annotation | Dtype | Notes |
|------------|-------|-------|
| `tessera.f16[..., ...]` | FP16 | Read-only tensor annotation |
| `tessera.bf16[..., ...]` | BF16 | Read-only tensor annotation |
| `tessera.f32[..., ...]` | FP32 | Read-only tensor annotation |
| `tessera.mut_f32[..., ...]` | FP32 | Write-privileged tensor annotation |

---

## `tessera.ops` Namespace

Phase 1 implementations are numpy-backed stubs. Phase 3 dispatches to compiled MLIR kernels.

| Op | Signature | Notes |
|----|-----------|-------|
| `tessera.ops.gemm(A, B)` | `(array, array) → array` | Matrix multiply via `np.matmul` |
| `tessera.ops.matmul(A, B)` | alias for `gemm` | |
| `tessera.ops.layer_norm(x, eps=1e-5)` | `(array) → array` | Pure effect |
| `tessera.ops.softmax(x, axis=-1)` | `(array) → array` | Pure effect |
| `tessera.ops.gelu(x)` | `(array) → array` | Pure effect |
| `tessera.ops.relu(x)` | `(array) → array` | Pure effect |
| `tessera.ops.transpose(x, axes=None)` | `(array) → array` | Pure effect |
| `tessera.ops.cast(x, dtype)` | `(array, str) → array` | Pure effect |
| `tessera.ops.dropout(x, p=0.1, training=True)` | `(array) → array` | `random` effect |
| `tessera.ops.conv2d(x, weight, bias=None, stride=1, padding=0)` | stub | Returns zeros Phase 1 |
| `tessera.ops.flash_attn(Q, K, V, scale=None)` | `(array,array,array) → array` | Naive Phase 1; FA-4 Phase 3 |
| `tessera.ops.all_reduce(x, op="sum")` | stub | No-op Phase 1 |
| `tessera.ops.reduce_scatter(x, op="sum", axis=0)` | stub | No-op Phase 1 |
| `tessera.ops.all_gather(x, axis=0)` | stub | No-op Phase 1 |
| `tessera.ops.fused_epilogue(x, bias=None, activation="linear")` | `(array) → array` | |

---

## Error Types

| Exception | Module | Raised when |
|-----------|--------|-------------|
| `TesseraConstraintError` | `tessera.compiler.constraints` | Structural constraint violated at decoration time |
| `TesseraEffectError` | `tessera.compiler.effects` | `deterministic=True` + unseeded random op |
| `TesseraJitError` | `tessera.compiler.jit` | Graph IR emission pipeline failure |
| `TesseraTargetError` | `tessera.compiler.gpu_target` | Invalid `GPUTargetProfile` parameters |
| `TesseraAttnConfigError` | `tessera.compiler.attn_lower` | Invalid `FlashAttnLoweringConfig` parameters |

---

## Phase Implementation Status

| API Symbol | Implemented | Phase |
|-----------|-------------|-------|
| `@tessera.jit` | ✅ | 1 |
| `@tessera.kernel` | ✅ | 1 |
| `tessera.Region[...]` | ✅ | 1 |
| `tessera.domain.Rect` | ✅ | 1 |
| `tessera.dist.Block` | ✅ | 1 |
| `tessera.dist.Replicated` | ✅ | 1 |
| `tessera.dist.Cyclic` | ✅ (stub) | 4 (full) |
| `tessera.array.from_domain` | ✅ | 1 |
| `DistributedArray.parts()` | ✅ | 1 |
| `tessera.constraint.*` | ✅ | 1 |
| `EffectLattice` | ✅ | 1 |
| `GPUTargetProfile` / `ISA` | ✅ | 3 |
| `FlashAttnLoweringConfig` | ✅ | 3 |
| `MockRankGroup` (testing) | ✅ | 1 |
| NCCL/RCCL collectives | 🔲 | 4 |
| TPU backend | 🔲 | 4 |
| Runtime C ABI (Python wrapper) | 🔲 | 6 |
