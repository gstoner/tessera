---
status: Normative
classification: Normative
authority: Language surface semantics; defers all symbol names and signatures to docs/spec/PYTHON_API_SPEC.md
last_updated: 2026-04-26
---

# Tessera Language Specification (Normative)

**Version:** 0.3.0  
**Authority:** This document specifies the Python surface language semantics of Tessera. For the authoritative list of every public symbol, signature, and error type, see `docs/spec/PYTHON_API_SPEC.md`. For canonical names, see `docs/CANONICAL_API.md`.

---

## 1. Scope

This document specifies:

- The decoration protocol for `@tessera.jit` and `@tessera.kernel`
- The type annotation system (`Region`, `Tensor`, dtype annotations)
- The constraint language (`tessera.require`, `Divisible`, `Range`, `Equal`)
- The effect system and `EffectLattice` semantics
- The kernel dispatch model (`index_launch`)
- The concurrency and synchronization model at the Python surface

It does not specify IR-level semantics (see `GRAPH_IR_SPEC.md`, `TARGET_IR_SPEC.md`) or runtime ABI (see `RUNTIME_ABI_SPEC.md`).

---

## 2. Decoration Protocol

### 2.1 `@tessera.jit`

`@tessera.jit` is a **decoration-time compiler** — not a trace-and-replay system. All
structural analysis (constraint solving, effect inference, Graph IR emission) runs at the
moment the decorator executes, before the function is ever called.

**Decoration sequence (normative):**

1. Extract `tessera.require(...)` calls from the function body via AST analysis
2. Instantiate the corresponding `Constraint` objects and register them with `ConstraintSolver`
3. Check all constraints against any concrete `bindings` provided to `@jit`; raise `TesseraConstraintError` on first violation
4. Walk the AST via `EffectLattice` to infer the function's effect level
5. If `deterministic=True`: validate the inferred effect against the deterministic contract (see §5.3); raise `TesseraEffectError` on violation
6. Resolve `attn_config`: if `target.isa >= ISA.SM_90` and no config provided, use `SM90_DEFAULT`
7. Emit Graph IR via `GraphIRBuilder`; raise `TesseraCompileError` on IR emission failure
8. Return a `JitFn` wrapper

Steps 1–7 happen exactly once, at decoration time. Subsequent calls to the decorated function execute the wrapped Python function eagerly (Phase 1) or dispatch to the compiled kernel (Phase 3+).

**Decorator forms — both are equivalent:**

```python
# Bare form (no arguments)
@tessera.jit
def step(W: tessera.Region["read"], X: tessera.Region["read"], Y: tessera.Region["write"]):
    Y[:] = tessera.ops.gemm(X, W)

# Keyword-argument form
@tessera.jit(deterministic=True, seed=42)
def stable_forward(x: tessera.Tensor["B", "D"]):
    return tessera.ops.layer_norm(x)
```

**Accepted keyword arguments:**

| Argument | Type | Default | Meaning |
|----------|------|---------|---------|
| `deterministic` | `bool` | `False` | Enforce no unseeded random effects |
| `seed` | `int \| None` | `None` | RNG seed; allows random ops under `deterministic=True` |
| `bindings` | `dict[str, int] \| None` | `None` | Concrete dimension sizes for constraint checking at decoration time |
| `target` | `GPUTargetProfile \| None` | `None` | Route to GPU lowering pipeline when set |
| `attn_config` | `FlashAttnLoweringConfig \| None` | `None` | Flash attention tile config; auto-set for SM_90+ |

### 2.2 `@tessera.kernel`

`@tessera.kernel` marks a function as a tile kernel — a lower-level compute unit
dispatched by `tessera.index_launch`. Unlike `@tessera.jit`, `@kernel` functions use
explicit dtype annotations (`tessera.f16[..., ...]`, `tessera.mut_f32[..., ...]`) rather
than `Region` privilege annotations.

`@tessera.kernel` performs the same decoration-time analysis as `@tessera.jit` with the
following differences:

- No `deterministic`, `seed`, or `target` keyword arguments
- Dtype annotations are lowered to typed memref args in Graph IR (not effect attributes)
- The function is not callable directly; it must be dispatched via `tessera.index_launch`

```python
@tessera.kernel
def tp_gemm(A: tessera.f16[..., ...],
            B: tessera.f16[..., ...],
            C: tessera.mut_f32[..., ...]):
    C[:] = tessera.ops.gemm(A, B)
```

---

## 3. Type Annotation System

### 3.1 `tessera.Region`

`Region` is a **type annotation**, not a runtime wrapper. It participates in Python's
`__class_getitem__` annotation protocol and lowers to a `tessera.effect` attribute on
Graph IR function arguments.

**Syntax:**

```python
tessera.Region["read"]        # read-only privilege
tessera.Region["write"]       # exclusive write
tessera.Region["reduce_sum"]  # commutative accumulation (sum)
tessera.Region["reduce_max"]  # commutative accumulation (max)
tessera.Region["reduce_min"]  # commutative accumulation (min)
```

**Privilege semantics:**

| Mode | Compiler guarantee | Collective inference |
|------|-------------------|---------------------|
| `"read"` | Argument not modified; safe to overlap across ranks | None — reads are local |
| `"write"` | Exclusive write; no other arg may alias this memory | None — writes are local |
| `"reduce_sum"` | Commutative accumulation; safe to insert `all_reduce(sum)` | `collective.all_reduce(op=sum)` at DP boundary (Phase 4+) |
| `"reduce_max"` | Commutative accumulation; safe to insert `all_reduce(max)` | `collective.all_reduce(op=max)` at DP boundary (Phase 4+) |
| `"reduce_min"` | Commutative accumulation; safe to insert `all_reduce(min)` | `collective.all_reduce(op=min)` at DP boundary (Phase 4+) |

**Conflict rule (normative):** Two `"write"` annotations on arguments that may alias the
same underlying storage is a **conflict** and **shall** raise `TesseraPrivilegeError` at
decoration time. Read and reduce annotations never conflict with each other.

**Graph IR lowering:** `Region["read"]` on argument `x` becomes:

```mlir
func.func @step(%x: memref<?xf32> {tessera.effect = "read"}, ...)
```

### 3.2 `tessera.Tensor`

`tessera.Tensor` is a symbolic-dimension annotation for use with `@tessera.jit`. Dimension
names are strings; their values are resolved at decoration time (if `bindings` are
provided) or remain symbolic.

```python
def aligned_gemm(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    tessera.require(tessera.constraint.Divisible("K", 64))
    return tessera.ops.gemm(A, B)
```

`Tensor["M", "K"]` lowers to a `memref<?x?xf32>` (or the appropriate dtype) in Graph IR,
with `tessera.dim_names = ["M", "K"]` as an attribute for use by the constraint solver.

### 3.3 Dtype Annotations

Used in `@tessera.kernel` functions. The annotation encodes element type and mutability.

| Annotation | Meaning | Graph IR type |
|-----------|---------|---------------|
| `tessera.f16[..., ...]` | Read-only f16 tensor (any rank) | `memref<?x...xf16>` |
| `tessera.bf16[..., ...]` | Read-only bf16 tensor | `memref<?x...xbf16>` |
| `tessera.f32[..., ...]` | Read-only f32 tensor | `memref<?x...xf32>` |
| `tessera.mut_f16[..., ...]` | Mutable f16 tensor | `memref<?x...xf16>` + write effect |
| `tessera.mut_f32[..., ...]` | Mutable f32 tensor | `memref<?x...xf32>` + write effect |
| `tessera.mut_bf16[..., ...]` | Mutable bf16 tensor | `memref<?x...xbf16>` + write effect |

The `...` ellipsis means "any number of dimensions." Fixed-rank forms like
`tessera.f16[128, 256]` are also accepted and lower to static-size memrefs.

---

## 4. Constraint Language

### 4.1 `tessera.require`

`tessera.require(predicate)` registers a structural constraint on the enclosing `@jit`
function. It **shall** appear as a bare expression statement (not in a conditional or
nested scope) in the function body for the AST extractor to detect it.

```python
@tessera.jit
def aligned_gemm(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    tessera.require(tessera.constraint.Divisible("K", 64))
    tessera.require(tessera.constraint.Range("M", 1, 65536))
    return tessera.ops.gemm(A, B)
```

At **decoration time**: constraints are extracted and checked against any concrete
`bindings`. At **call time**: `require()` is a no-op — constraints were already checked.

### 4.2 Predicates

**`Divisible(dim, divisor)`**

Asserts `dim % divisor == 0`. Used to enforce alignment requirements for tensor-core
GEMM (multiples of 64 for BF16 WGMMA), TMA tile sizes, and vectorization.

```python
tessera.constraint.Divisible("K", 64)   # K must be a multiple of 64
```

**`Range(dim, lo, hi)`**

Asserts `lo <= dim <= hi` (both inclusive). Used to bound sequence lengths, batch sizes,
or tile counts.

```python
tessera.constraint.Range("S", 1, 8192)  # 1 <= S <= 8192
```

**`Equal(dim_a, dim_b)`**

Asserts `dim_a == dim_b`. Used to catch inner-dimension mismatches at decoration time
rather than at runtime.

```python
tessera.constraint.Equal("D_in", "D_out")  # D_in must equal D_out
```

### 4.3 ConstraintSolver Semantics

- Constraints are checked in registration order
- `ConstraintSolver.check(bindings)` raises on the **first** violation
- `ConstraintSolver.check_all(bindings)` returns **all** violations (for diagnostic tools)
- A dimension not present in `bindings` is **symbolic** — the predicate is skipped, not failed
- A negative `divisor` in `Divisible` is a programmer error and raises `ValueError` at
  predicate construction time, not at constraint-check time

---

## 5. Effect System

### 5.1 Effect Lattice

Every Tessera function has an **inferred effect level** that is the least upper bound of
the effects of all ops it calls. The lattice is totally ordered:

```
pure (0) < random (1) < memory (2) < io (3) < top (4)
```

| Level | Meaning |
|-------|---------|
| `pure` | No side effects; output depends only on inputs; safe to recompute |
| `random` | Calls an RNG op; result varies across otherwise-identical inputs |
| `memory` | Reads or writes mutable state (KV cache, attention state) |
| `io` | Performs collective communication or host I/O |
| `top` | Conservative fallback; used when source cannot be inspected |

### 5.2 Effect Inference

`EffectLattice.infer(fn)` infers the effect level by walking the function's AST and
looking up each `tessera.ops.*` call in a static op→effect table. Inference is:

- **Intra-procedural in Phase 1** — only direct calls in the function body are examined
- **Inter-procedural in Phase 2+** — the IR call graph is walked transitively

The effect table (normative for Phase 1):

| Op | Effect |
|----|--------|
| `gemm`, `matmul`, `conv2d`, `layer_norm`, `softmax`, `gelu`, `relu`, `transpose`, `cast`, `fused_epilogue` | `pure` |
| `dropout`, `randn`, `rand`, `bernoulli`, `normal` | `random` |
| `flash_attn`, `kv_cache_read`, `kv_cache_write` | `memory` |
| `all_reduce`, `reduce_scatter`, `all_gather`, `send`, `recv`, `barrier` | `io` |

Functions whose source cannot be retrieved (C extensions, built-ins) are conservatively
assigned `top`.

### 5.3 Deterministic Contract

`@tessera.jit(deterministic=True)` imposes:

- The function's inferred effect **shall** be `pure` or `random`
- If the effect is `random`, a `seed` **must** be provided; the seed is bound to a
  `tessera.deterministic = {seed = N}` attribute on the emitted Graph IR module
- If the effect is `memory` or `io`, the contract is **unconditionally violated** — seeding
  does not make collective communication or cache mutation deterministic

Violations raise `TesseraEffectError` at decoration time.

```python
# Allowed: no random effects
@tessera.jit(deterministic=True)
def forward(x: tessera.Tensor["B", "D"]):
    return tessera.ops.layer_norm(x)

# Allowed: random effects with seed
@tessera.jit(deterministic=True, seed=42)
def with_dropout(x: tessera.Tensor["B", "D"]):
    return tessera.ops.dropout(x, p=0.1)

# REJECTED at decoration time: random without seed
@tessera.jit(deterministic=True)
def bad(x: tessera.Tensor["B", "D"]):
    return tessera.ops.dropout(x, p=0.1)   # TesseraEffectError

# REJECTED at decoration time: io effect
@tessera.jit(deterministic=True)
def also_bad(x: tessera.Region["read"], g: tessera.Region["reduce_sum"]):
    g += tessera.ops.gemm(x, x)            # TesseraEffectError (reduce_sum → io)
```

### 5.4 Graph IR Effect Attribute

The inferred effect is attached to `func.func` ops in Graph IR:

```mlir
func.func @step(%W: memref<?x?xf32> {tessera.effect = "read"},
                %X: memref<?x?xf32> {tessera.effect = "read"},
                %Y: memref<?x?xf32> {tessera.effect = "write"})
    attributes {tessera.function_effect = "pure"} {
  ...
}
```

For deterministic functions with a seed:

```mlir
module attributes {tessera.version = "0.3", tessera.deterministic = {seed = 42}} {
  ...
}
```

---

## 6. Kernel Dispatch Model

### 6.1 `tessera.index_launch`

`index_launch` dispatches a `@tessera.kernel` function across shards of a distributed
mesh. It is the primary way to express tensor-parallel (TP) or data-parallel (DP) fanout.

**Syntax:**

```python
tessera.index_launch(axis="tp")(kernel_fn)(arg0, arg1, ...)
```

- `axis`: the mesh axis name to iterate over (string; must match a `mesh_axes` key in the `DistributedArray` arguments)
- `kernel_fn`: a `@tessera.kernel`-decorated function
- Arguments: `DistributedArray` instances; `.parts(axis)` is called automatically on each

**Normative dispatch rule:**

For each rank `r` in `range(mesh_size[axis])`:
1. Slice each `DistributedArray` argument using `.parts(axis)[r]`
2. Dispatch `kernel_fn` with those slices
3. In Phase 1 (mock/eager): execute sequentially per rank using `MockRankGroup`
4. In Phase 4+ (NCCL): execute concurrently across real ranks

### 6.2 `MockRankGroup`

`MockRankGroup(n, mesh_axes)` provides a thread-based fake multi-rank environment for
testing without NCCL. Each "rank" is a Python thread sharing the in-process memory space.

```python
from tessera.testing import MockRankGroup
ranks = MockRankGroup(n=4, mesh_axes={"dp": 4})
```

Collective semantics under `MockRankGroup`:

| Collective | Behavior |
|-----------|---------|
| `all_reduce` | Thread barrier + reduction across thread-local buffers |
| `reduce_scatter` | Thread barrier + per-rank slice of reduced result |
| `all_gather` | Thread barrier + concatenation of per-rank buffers |

`MockRankGroup` is the required testing mechanism for Phases 1–5. Phase 6 introduces real
NCCL execution for T2 conformance.

---

## 7. Concurrency and Synchronization

### 7.1 Intra-kernel Synchronization

Within a `@tessera.kernel`, synchronization is expressed through Tile IR ops (see
`04_tile_ir.md`). These ops are not directly callable from Python — they are emitted by
the lowering pipeline based on the structure of the kernel:

- `tile.async_copy {stage=N}` / `tile.wait_async {stage=N}` — explicit double-buffering
- `tile.barrier` — warp-level barrier within a CTA
- `tessera.queue.push` / `tessera.queue.pop` — producer/consumer token ordering

### 7.2 Inter-rank Synchronization

At the `@tessera.jit` level, inter-rank synchronization is implicit in `Region` privilege
annotations. No explicit `barrier()` call is available at the `@jit` level — synchronization
is always derived from data-flow and privilege annotations.

`Region["reduce_sum/max/min"]` signals that a collective **may** be inserted at the DP
mesh boundary. This is a Phase 4+ planned feature (`GPUCollectiveInsertionPass`).

### 7.3 Forward Progress Guarantees

A conformant implementation **shall** guarantee forward progress for:

- All `@tessera.jit` functions that contain only `pure` or `memory`-effect ops
- All `@tessera.kernel` functions that do not call collective ops

Functions with `io` effect have forward progress contingent on the NCCL runtime
completing the collective. Deadlock detection is not in scope for Phases 1–3.

---

## 8. Valid Program Structure

A valid Tessera program:

1. Imports `tessera` at the module level
2. Optionally defines domains, distributions, and distributed arrays
3. Decorates one or more functions with `@tessera.jit` or `@tessera.kernel`
4. Calls decorated functions directly or dispatches them via `tessera.index_launch`

A program is **invalid** and **shall** be rejected if:

- A `@tessera.jit` function contains a `Region["write"]` conflict on aliased args
- A `@tessera.kernel` function is called directly (not via `index_launch`)
- A `@tessera.jit(deterministic=True)` function body infers an `io` effect
- Any `tessera.require()` predicate is violated with a concrete binding

---

## 9. Versioning

The language surface version tracks the `tessera.version` Graph IR attribute (`"0.3"` as
of this document). The Python package version and IR version **shall** always agree; a
mismatch causes the Graph IR verifier to raise `TesseraCompileError`.

---

## Appendix A — Decoration-Time Checklist (Informative)

When `@tessera.jit` decorates a function, the following checks run in order. A failure at
any step aborts decoration with the indicated error type:

| Step | Action | Error on failure |
|------|--------|-----------------|
| 1 | Extract `tessera.require(...)` calls via AST | — (silent if unparseable) |
| 2 | Check constraints against `bindings` | `TesseraConstraintError` |
| 3 | Infer effect level via `EffectLattice` | — (falls back to `top`) |
| 4 | Validate deterministic contract | `TesseraEffectError` |
| 5 | Resolve `attn_config` for SM_90+ | — (auto-set to `SM90_DEFAULT`) |
| 6 | Emit Graph IR via `GraphIRBuilder` | `TesseraCompileError` |
| 7 | Construct and return `JitFn` wrapper | — |
