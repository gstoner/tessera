---
status: Normative
classification: Normative
authority: Shape constraint system; defers predicate implementations to docs/spec/PYTHON_API_SPEC.md §11 and constraint source
last_updated: 2026-04-26
---

# Tessera Shape System Specification (Normative)

**Version:** 0.3.0  
**Authority:** This document specifies the symbolic dimension system, constraint predicates,
solver algorithm, error attribution, and interaction with distribution. For canonical predicate
signatures and the ConstraintSolver implementation, see `docs/spec/PYTHON_API_SPEC.md §11`
and `python/tessera/compiler/constraints.py`.

---

## 1. Scope

This document specifies:

- The symbolic dimension model: how tensor dimensions are named and tracked
- The three built-in constraint predicates (`Divisible`, `Range`, `Equal`)
- The `ConstraintSolver` algorithm and evaluation order
- How symbolic dimensions interact with `ShardSpec` (distributed tensors)
- The error attribution model — what information a `TesseraConstraintError` carries
- The `ShapeInferencePass` forward-propagation model (Phase 6)

It does not specify runtime shape checking (which happens in the tensor execution backend)
or MLIR verifier rules (which are in `GRAPH_IR_SPEC.md §6`).

---

## 2. Symbolic Dimension Model

### 2.1 Dimension Names

A **symbolic dimension** is a named, compile-time unknown integer. Dimension names are
Python strings used as subscripts in `tessera.Tensor` annotations:

```python
def gemm(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    ...
```

Here `"M"`, `"K"`, and `"N"` are symbolic dimensions. The shared name `"K"` expresses
that the inner dimension of `A` and the outer dimension of `B` are the **same symbolic
variable**.

**Naming rules (normative):**
- A dimension name is any non-empty Python string
- Two annotations using the same name assert the corresponding dimensions are equal
- Dimension names are scoped to a single `@tessera.jit` or `@tessera.kernel` function —
  they do not cross function boundaries
- The empty string `""` and names containing whitespace are reserved and **shall not** be
  used by user code; the compiler may use them internally

### 2.2 Concrete Bindings

A **concrete binding** is a mapping from dimension name to a positive integer. Bindings
are supplied to `@tessera.jit` via the `bindings=` keyword argument:

```python
@tessera.jit(bindings={"K": 128, "M": 4096})
def aligned_gemm(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    tessera.require(tessera.constraint.Divisible("K", 64))
    return tessera.ops.gemm(A, B)
```

Dimensions present in `bindings` are **concrete** and will be checked. Dimensions absent
from `bindings` are **symbolic** and will be skipped by the solver until a concrete value
is available.

### 2.3 Symbolic vs. Concrete

| State | When | Solver behavior |
|-------|------|-----------------|
| Concrete | Dimension present in `bindings` at decoration time | Constraint is checked; error raised on violation |
| Symbolic | Dimension absent from `bindings` | Constraint is skipped (deferred) |

Symbolic dimensions that are never made concrete are **unchecked** — the constraint is
registered but never evaluated. This is intentional: Tessera is an AOT compiler, and many
shapes are only known at runtime (e.g., dynamic sequence length). Shape errors for purely
symbolic dimensions are caught downstream by the MLIR verifier or at runtime.

---

## 3. Constraint Predicates

### 3.1 `Divisible(dim, divisor)`

**Semantics:** Asserts that the dimension `dim` is evenly divisible by `divisor`.

```
Divisible("K", d)  iff  K % d == 0
```

**Motivation:** Required for tensor-core alignment:
- BF16 WGMMA on SM_90: K must be a multiple of 64 (or 32 for WMMA)
- TMA async copy: tile sizes must align to 16 bytes
- AVX-512 GEMM: K must be a multiple of 32

**Invariants (normative):**
- `divisor` must be a positive integer; `divisor <= 0` raises `ValueError` at construction time
- `dim` must be a non-empty string; empty raises `ValueError` at construction time
- If `dim` is symbolic (not in bindings), the predicate returns `None` (no error)
- If `K % divisor != 0`, raises `TesseraConstraintError` with `dim_name="K"` and `actual=K`

### 3.2 `Range(dim, lo, hi)`

**Semantics:** Asserts that `lo <= dim <= hi` (both bounds inclusive).

```
Range("S", lo, hi)  iff  lo <= S <= hi
```

**Motivation:** Bounds sequence lengths, batch sizes, and tile counts to prevent OOM
or hardware-specific overflow (e.g., attention score overflow for S > 65536 in F16).

**Invariants (normative):**
- `lo` and `hi` must both be integers with `lo <= hi`; violated at construction raises `ValueError`
- `lo < 0` is permitted (allows negative-dimension assertions for internal use)
- If `dim` is symbolic, returns `None`
- If `S < lo` or `S > hi`, raises `TesseraConstraintError`

### 3.3 `Equal(dim_a, dim_b)`

**Semantics:** Asserts that two named dimensions have the same concrete value.

```
Equal("D_in", "D_out")  iff  D_in == D_out
```

**Motivation:** Catches shape mismatches early — e.g., attention head dimension must match
across Q, K, V; residual connections require matching hidden dimensions.

**Invariants (normative):**
- Both `dim_a` and `dim_b` must be non-empty strings
- If either dimension is symbolic (not in bindings), returns `None` — the check is skipped
  even if the other dimension is concrete
- If both are concrete and unequal, raises `TesseraConstraintError` with
  `dim_name="D_in/D_out"` and `actual=(a, b)`

---

## 4. ConstraintSolver Algorithm

### 4.1 Registration

Constraints are registered at decoration time by the `_ConstraintExtractor` AST visitor,
which scans the function body for `tessera.require(...)` calls. Each call instantiates a
predicate and adds it to the solver's constraint list via `ConstraintSolver.add()`.

Registration order is source order (top to bottom in the function body).

### 4.2 Evaluation

`ConstraintSolver.check(bindings)` iterates constraints in registration order and calls
`predicate.check(bindings)` on each. The algorithm (normative):

```
for each constraint C in registration order:
    error = C.check(bindings)
    if error is not None:
        raise error      # stop on first violation
```

`ConstraintSolver.check_all(bindings)` collects all violations without stopping:

```
errors = []
for each constraint C in registration order:
    error = C.check(bindings)
    if error is not None:
        errors.append(error)
return errors
```

`check_all` is intended for diagnostic tools that want to surface all problems at once.

### 4.3 Symbolic Dimension Handling

A predicate that references a dimension not in `bindings` **shall** return `None` (not an
error). The predicate is **not** called with a partial binding — the full `bindings` dict
is passed and the predicate decides which keys it needs.

For `Equal(dim_a, dim_b)`: if **either** key is absent, the predicate returns `None`. Both
must be concrete for the equality to be checked.

### 4.4 Decoration-Time vs. Call-Time

Constraints are checked **once**, at decoration time. `tessera.require()` inside a
`@jit`-decorated function body is a no-op when the function is called — the constraint was
already evaluated by the `_ConstraintExtractor` before the function was ever invoked.

This means:
- Constraints with purely symbolic dimensions at decoration time are **never automatically
  re-checked** with the actual call-time shapes in Phase 1
- Phase 5+ may introduce call-time shape specialization, but this is not part of the
  current constraint contract

---

## 5. Shape and Distribution Interaction

### 5.1 ShardSpec Dimension Semantics

When a tensor is distributed with `tessera.array.from_domain`, its **logical shape** (the
full tensor before sharding) is used for constraint checking. The per-shard shape is a
derived quantity and is not directly visible in `@jit` annotations.

```python
D    = tessera.domain.Rect((4, 128, 256))   # logical shape (4, 128, 256)
dist = tessera.dist.Block(mesh_axes=("dp", "tp"))
X    = tessera.array.from_domain(D, dtype="bf16", distribution=dist)
# X.shape        → (4, 128, 256)   ← logical shape
# X.shard_spec   → ShardSpec(partition=(0, 1), mesh_axes=("dp", "tp"))
# X.parts("dp")  → per-rank slices along axis 0
```

Constraints on a sharded tensor are stated against the **logical** dimension:

```python
@tessera.jit(bindings={"B": 4, "S": 128, "D": 256})
def step(X: tessera.Tensor["B", "S", "D"]):
    tessera.require(tessera.constraint.Divisible("D", 64))
    # Checks D=256, which is 256%64=0 → passes
```

### 5.2 `ShardSpec.partition` is Integer-Indexed

`ShardSpec.partition` is a tuple of **integer axis indices** (0-based), not string names.
This is distinct from `mesh_axes` (which are string names):

```python
spec = X.shard_spec
spec.partition   # → (0, 1)    — axes 0 and 1 are partitioned
spec.mesh_axes   # → ("dp", "tp")  — the corresponding mesh axis names
```

Constraints reference **logical dimension names** from `Tensor["B", "S", "D"]`
annotations, not `partition` indices. The mapping between annotation names and partition
axes is maintained by the `GraphIRBuilder` when emitting `tessera.shard` attributes.

### 5.3 TPU Automatic Constraints

When `@tessera.jit(target=TPUTargetProfile(...))` is used (Phase 4+), the TPU target
profile automatically injects `Divisible("M", 128)`, `Divisible("N", 128)`, and
`Divisible("K", 128)` constraints, reflecting the TPU MXU tile constraint of 128×128. These
injected constraints run before user-supplied constraints and cannot be overridden.

---

## 6. Error Attribution Model

### 6.1 `TesseraConstraintError` Fields

| Field | Type | Content |
|-------|------|---------|
| `constraint` | `Constraint` | The predicate that failed |
| `dim_name` | `str` | The dimension name (or `"dim_a/dim_b"` for `Equal`) |
| `actual` | `int \| tuple[int,int] \| None` | The concrete value(s) that violated the constraint |
| `message` | `str` | Human-readable description with the offending value and requirement |

### 6.2 Error Message Format

The message format is (normative):

**`Divisible`:**
```
Dimension 'K' = 100 is not divisible by 64.
Required: K % 64 == 0
```

**`Range`:**
```
Dimension 'S' = 0 is out of range [1, 8192].
Required: 1 <= S <= 8192
```

**`Equal`:**
```
Dimension equality constraint violated: 'D_in' = 512 != 'D_out' = 256
```

### 6.3 Source Location Attribution

In Phase 6, `ShapeInferencePass` and `ErrorReporter` attach the originating Python
`file:line` to shape errors by walking the MLIR `loc` attribute chain on the failing op.
When `loc` is unavailable, the error reports `"<unknown location>"`.

The `TesseraConstraintError` at the Python layer does not yet carry source location
(Phase 1–5). Source location attribution is a Phase 6 deliverable.

---

## 7. `ShapeInferencePass` (Phase 6 Planned)

`ShapeInferencePass` is a forward-propagation pass that runs after Graph IR emission and
before the lowering pipeline. It is not yet implemented; the specification is normative for
Phase 6.

**Algorithm:**
1. Start from the function's argument types (which carry concrete or symbolic shapes from annotations)
2. Propagate shapes forward through each Graph IR op using the op's shape-inference interface
3. For each op whose output shape conflicts with a downstream consumer's expected shape, emit a `TesseraShapeError` with the MLIR `loc` attribute of the violating op
4. If all output shapes are consistent, annotate each op result with its inferred shape as an attribute (`tessera.shape = [128, 256]`)

**Op shape rules (normative for Phase 6):**

| Op | Output shape |
|----|-------------|
| `tessera.matmul(A[M,K], B[K,N])` | `[M, N]` |
| `tessera.conv2d(X[B,H,W,C], W[Kh,Kw,C,F])` | `[B, H', W', F]` where H', W' depend on stride/padding |
| `tessera.cast(X[...], dtype)` | same shape as X |
| `tessera.transpose(X[M,N])` | `[N, M]` |
| `tessera.flash_attn(Q[B,H,S,D], K[B,H,S,D], V[B,H,S,D])` | `[B, H, S, D]` |

---

## 8. Diagnostics Reference

### 8.1 Error Codes (Informative)

| Code | Meaning | Example trigger |
|------|---------|----------------|
| `E1001` | Constraint violation: Divisible | `K=100`, `Divisible("K", 64)` |
| `E1002` | Constraint violation: Range | `S=0`, `Range("S", 1, 8192)` |
| `E1003` | Constraint violation: Equal | `D_in=512`, `D_out=256` |
| `E1101` | Unknown dimension in binding | Typo in `bindings={"KK": 64}` where annotation uses `"K"` |
| `E1201` | Effect contract violation | `deterministic=True` on function with `dropout` and no `seed` |
| `E1301` | Shape mismatch (Phase 6) | Matmul inner dim K mismatch between A and B |
| `E1302` | Broadcast violation (Phase 6) | Non-broadcastable shapes in element-wise op |

### 8.2 Example Error Output

```
TesseraConstraintError [E1001]: Dimension 'K' = 100 is not divisible by 64.
  Required: K % 64 == 0
  Constraint: Divisible('K', 64)
  Function: aligned_gemm
  Registered at: decoration time

TesseraConstraintError [E1003]: Dimension equality constraint violated:
  'D_in' = 512 != 'D_out' = 256
  Constraint: Equal('D_in', 'D_out')
  Function: residual_block
  Registered at: decoration time
```

Phase 6 `ShapeInferencePass` errors will additionally include:

```
TesseraShapeError [E1301]: Shape mismatch in tessera.matmul:
  A inner dimension K=128 != B outer dimension K=256
  at aligned_gemm (aligned_gemm.py:14)
```

---

## Appendix A — Adding a New Constraint Predicate (Informative)

To add a new predicate type `Congruent("K", mod, remainder)` (asserting `K % mod == remainder`):

1. Subclass `Constraint` in `python/tessera/compiler/constraints.py`
2. Implement `check(bindings)` and `dim_names()`
3. Add the constructor to `_ConstraintExtractor._PREDICATE_CTORS` in `jit.py`
4. Add an entry to the error code table above
5. Add tests to `tests/phase1/test_constraints.py`
6. Update `PYTHON_API_SPEC.md §11` with the new predicate signature
