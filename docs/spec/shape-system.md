---
status: Normative
classification: Normative
authority: Shape, layout, shard, and schedule feasibility contract
last_updated: 2026-04-28
---

# Tessera Shape System Specification

**Version:** 0.4.0  
**Status:** Normative for compiler behavior unless marked Informative.

The Tessera Shape System extends the type system with tensor dimensions,
layouts, distributed shards, and schedule constraints. Its purpose is to catch
shape mismatches before lowering, prune impossible schedules before autotuning,
and preserve precise diagnostics when dynamic dimensions must be checked at
runtime.

The central design rule is separation of concerns:

- **Logical shape:** rank and dimensions of the mathematical tensor.
- **Physical layout:** memory order, tiling, packing, alignment, and address space.
- **Shard map:** how logical dimensions are partitioned over mesh axes.
- **Runtime witness:** a single inserted assertion/refinement for dimensions that
  are unknown at compile time.

These concepts may appear together in a tensor type, but compiler passes shall
reason about them independently and join their constraints only at pass
boundaries.

---

## 1. Shape Objects

### 1.1 Dimensions

A dimension is one of:

- A concrete positive integer, such as `128`.
- A symbolic dimension, such as `B`, `N`, or `D`.
- A derived dimension expression, currently a product such as `H * Dh`.

Symbolic dimensions are scoped to one function/module specialization. Reusing a
name asserts equality within that scope.

```tessera
dim B, N, M, D

fn attention(q: Tensor[B, N, D],
             k: Tensor[B, M, D]) -> Tensor[B, N, M] {
  let scores = q @ transpose(k, axes=(0, 2, 1));
  return scores * (1.0 / sqrt(D));
}
```

The compiler shall reject:

```text
error[shape-mismatch]: matmul inner dimensions differ
  left:  q  : Tensor[B, N, D]
  right: kT : Tensor[B, K, M]
  need:  D == K
```

### 1.2 Derived Dimensions

Derived dimensions express factorization constraints that are common in deep
learning: head splitting, tensor-parallel partitioning, vector packs, and tile
groups.

```tessera
dim B, T, H, Dh
let D = H * Dh

fn mha(q: Tensor[B, T, D],
       k: Tensor[B, T, D],
       v: Tensor[B, T, D]) -> Tensor[B, T, D]
where D = H * Dh {
  let qh = reshape(q, [B, T, H, Dh]);
  let kh = reshape(k, [B, T, H, Dh]);
  let vh = reshape(v, [B, T, H, Dh]);
  let out = attention(qh, kh, vh);
  return reshape(out, [B, T, D]);
}
```

Verification shall check the product when bindings are known. If `D`, `H`, or
`Dh` is dynamic, the compiler shall defer the check to a runtime witness unless
the target requires a static value before lowering.

---

## 2. Layout and Shard Metadata

### 2.1 Layout

Layout is a first-class contract attached to the tensor, not a comment on the
shape. The logical shape remains stable across layout casts.

Examples:

- `row_major`
- `col_major`
- `nhwc`
- `tile(BM=128, BN=128, BK=64)`
- `bsr(block_m=16, block_n=16)`
- `packed(dtype=fp8_e4m3, vector=8)`

An operator shall declare its accepted input layouts and produced output layout.
If a producer and consumer disagree, Schedule IR must either insert an explicit
`layout_cast` or reject the program.

### 2.2 Shard Map

A shard map partitions logical dimensions across mesh axes.

```tessera
mesh M(tp=8, dp=4)

let X: Tensor[B, N, D] @shard({B: dp, D: tp})
```

The compiler shall check:

- `B % dp == 0` when `B` is known.
- `D % tp == 0` when `D` is known.
- Mesh axis names exist.
- Collective operands agree on logical shape, layout, and shard map for the
  collective axis.

If the dimension is dynamic, a runtime witness may guard the divisibility before
the first compiled collective that needs it.

---

## 3. Constraint Language

The built-in constraint predicates are:

| Predicate | Meaning |
|-----------|---------|
| `Equal(A, B)` | Dimensions or expressions evaluate to the same integer |
| `Divisible(D, k)` | `D % k == 0` |
| `Range(D, lo, hi)` | `lo <= D <= hi` |
| `Derived(D, Expr)` | Alias for `Equal(D, Expr)` |

Constraints operate over concrete bindings when available. Purely symbolic
constraints are preserved in module metadata and rechecked when a later pass
specializes the dimensions.

The implementation may use an affine/Presburger subset. Products are allowed
only for derived dimension declarations and are expected to be fully resolved
before Tile IR lowering.

---

## 4. Broadcasting

Elementwise operators may declare NumPy-style broadcasting:

```tessera
Tensor[B, 1, D] + Tensor[B, N, D] -> Tensor[B, N, D]
```

Broadcasting is legal when each trailing dimension pair is equal or one side is
`1`. Operators that do not declare broadcasting must require exact shape
equality.

Schedule IR should materialize broadcasts only when required by a consumer. A
broadcasted logical singleton may be lowered as a zero-stride read rather than a
copy when the target layout permits it.

---

## 5. Compiler Implementation

### 5.1 Graph IR Type Checker

The Graph IR type checker shall:

1. Collect symbolic dimensions from function signatures and op result types.
2. Build a constraint graph from op contracts, `where` clauses, shard maps, and
   layout requirements.
3. Solve equalities, divisibility, ranges, and derived dimension constraints
   when bindings are concrete.
4. Preserve deferred symbolic constraints in module metadata.
5. Emit diagnostics with source spans, operand names, logical shapes, and
   suggested repairs.

Graph IR is responsible for correctness of mathematical shapes. It should not
decide tile sizes or invent physical layouts.

### 5.2 Schedule Feasibility

Schedule IR shall combine logical shape constraints with schedule candidates.
Before autotuning, it shall prune infeasible points:

```text
BM=128, BN=256 invalid for N=2305
  violated: N % 256 == 0
  suggestion: pad N from 2305 to 2560 or choose BN=128
```

`N=2304` with `BN=256` is valid because `2304 % 256 == 0`.

Schedule artifacts must record the shape bindings, layout, shard map, numeric
policy, movement plan, and tile knobs used to produce the tuned result.

### 5.3 Tile Verifier

Tile IR shall verify target-level constraints:

- Tensor Core fragment sizes are supported by the target architecture.
- Shared-memory layouts satisfy bank and vector alignment constraints.
- `ldmatrix`, `wgmma`, `tcgen05`, and corresponding AMD/RISC-V/NPU intrinsics
  receive legal layouts and dtypes.
- Shared memory, tensor memory, and register usage stay within target limits.
- Async copy, TMA, and mbarrier usage obey the memory model.

Tile IR may assume Graph and Schedule IR shape contracts have already been
checked, but it must still reject target-incompatible concrete instances.

### 5.4 Runtime Shape Witnesses

When a dimension is dynamic but required by a later lowering step, Graph or
Schedule IR shall insert one runtime witness:

```text
witness attention.input:
  shape = [B, N, D]
  constraints = [Divisible(N, 128), Equal(D, H * Dh)]
```

After the witness succeeds, the refined bindings are recorded in module metadata
so subsequent kernels can reuse the fact without rechecking at every op.

Runtime witnesses should be hoisted to the earliest point where all referenced
dimensions are known and before any operation that relies on the refinement.

---

## 6. Python and IDE Surface

The Python frontend exposes a lightweight shape model:

```python
import tessera as ts

B, N, M, D = ts.sym("B N M D")

@ts.check_shapes
def attention(q: ts.Tensor[B, N, D],
              k: ts.Tensor[B, M, D]) -> ts.Tensor[B, N, M]:
    return ts.ops.matmul(q, ts.ops.transpose(k, (0, 2, 1))) * (1.0 / ts.sqrt(D))
```

The Python surface is intentionally a mirror of the compiler semantics:

- `ts.sym(...)` creates symbolic dimensions.
- `Dim` products represent derived dimensions.
- `ShapeConstraintGraph` models equality, divisibility, ranges, and derived
  constraints.
- `broadcast_shape`, `matmul_shape`, and `reshape_shape` expose op shape rules.
- `check_shard` validates logical dimensions against mesh axes.
- `check_schedule_tile` reports padding suggestions for schedule candidates.
- `RuntimeShapeWitness` represents dynamic-shape refinement.

IDEs should use `.pyi` annotations and, later, a mypy/Pyright plugin to display
operator result shapes and shape diagnostics in tooltips.

---

## 7. Diagnostic Requirements

Shape diagnostics shall include:

- Stable error code.
- Op name and IR level.
- Source location when available.
- Left and right operand shapes.
- The exact violated constraint.
- A suggested repair when one is mechanically obvious.

Examples:

```text
error[shape-mismatch]: matmul inner dimensions differ
  left:  q : Tensor[B=8, N=1024, D=128]
  right: kT: Tensor[B=8, K=256, M=1024]
  need:  D == K
```

```text
error[tile-constraints]: schedule BN=256 invalid for N=2305
  violated: N % 256 == 0
  suggestion: pad N from 2305 to 2560
```

---

## 8. Pass Pipeline Placement

The shape system participates in the compiler as follows:

| Stage | Responsibility |
|-------|----------------|
| Frontend | Capture symbolic annotations and user constraints |
| Graph IR | Infer logical result shapes and solve shape/shard contracts |
| Autodiff | Preserve shape variables across primal/adjoint values |
| Schedule IR | Prune infeasible tile/layout/movement candidates |
| Autotuner | Key schedule artifacts by shape, layout, shard, dtype, target, and numerics policy |
| Tile IR | Verify target fragment, memory, and alignment legality |
| Runtime | Execute dynamic witnesses once per specialization |

This makes shape checking useful at every level without letting any one pass own
too much of the model.
