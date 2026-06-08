---
status: Normative
classification: Normative
authority: Shape, layout, shard, and schedule feasibility contract
last_updated: 2026-05-22
---

# Tessera Shape System Specification

**Version:** 0.4.1
**Status:** Normative for compiler behavior unless marked Informative.

## Documentation refresh (2026-05-22)

The 2026-05-06 spec gap audit asked this spec to **identify MLIR verifier
gaps explicitly**. Section 11 below was added in response — it maps
each contract to where it is currently checked (Python decoration-time,
Python call-time, MLIR pass, runtime witness) or names it as a verifier
gap with the file path that would own the check.

Section 9 (Current Implementation Map) and Section 10 (Dynamic Shape
Support Matrix) remain authoritative for layer-by-layer status; this
refresh adds the per-contract granularity the audit requested.

Locked by `tests/unit/test_shape_verifier_gap_map.py` (structural
guard) — the per-contract table below must agree with the
implementation inventory it cites.

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

## 9. Current Implementation Map

| Layer | Current status | Active evidence | Notes |
|-------|----------------|-----------------|-------|
| Python shape objects and helpers | implemented | `python/tessera/shape.py`, `tests/unit/test_shape_system_foundation.py` | Strongest current implementation surface. |
| Constraint predicates and solver | implemented | `python/tessera/compiler/constraints.py`, `tests/unit/test_constraints.py`, `tests/unit/test_dynamic_shapes.py` | Decoration-time concrete bindings and call-time symbolic shape bindings are checked. |
| Graph IR shape inference | implemented / scaffolded | `src/compiler/diagnostics/ShapeInferencePass.cpp`, `tests/unit/test_shape_inference.py` | Coverage is focused on core ops and diagnostics. |
| Schedule feasibility | implemented / scaffolded | `check_schedule_tile`, shape-system tests | Python helper reports padding/feasibility; full MLIR verifier coverage is incremental. |
| Tile verifier integration | scaffolded / lit-testable | Tile IR and backend lit tests where available | Alignment/resource checks exist for selected paths; full memory-model legality is planned. |
| Runtime shape witnesses | implemented / mock-runtime | `RuntimeShapeWitness`, shape-system tests | Dynamic witnesses refine shapes in Python/runtime-facing tests. |

Specs or guides that claim stronger MLIR verifier coverage should cite a test or
downgrade that claim to `planned`.

## 10. Dynamic Shape Support Matrix (audited 2026-05-09)

How well do symbolic dims (`Dim("S")`, `tessera.Tensor["B", "S", "D"]`) flow
through to each backend?

| Backend | Symbolic dims at call time | Decoration-time constraint check (`bindings=`) | Call-time constraint enforcement | Notes |
|---------|----------------------------|------------------------------------------------|----------------------------------|-------|
| CPU reference (no `target=`) | ✅ accepted; numpy reference handles concrete call shapes | ✅ | ✅ | `JitFn.__call__` resolves symbolic dims from actual arguments and raises `TesseraConstraintError` for violations |
| `target="apple_cpu"` (Accelerate) | ✅ accepted; rank-2 / rank-3 GEMM dispatch reads runtime shape | ✅ | ✅ | Tested 4×8 ⊗ 8×16 and 4×16 ⊗ 16×16 with one decorator |
| `target="apple_gpu"` (MPS + MSL) | ✅ accepted; MPS matrix descriptors built from concrete shape | ✅ | ✅ | Verified in `tests/unit/test_dynamic_shapes.py` |
| `target="rocm"` / `"nvidia"` | n/a — artifact-only, no execution to test | ✅ | n/a | Symbolic dims appear in emitted IR text; runtime semantics undefined until backend executes |

### Call-time constraint enforcement

`@tessera.jit(bindings={"K": 7})` on a body with `tessera.require(Divisible("K", 8))`
raises `TesseraConstraintError` at decoration time. The same function called
without eager bindings is checked at call time:

```python
@tessera.jit
def aligned_gemm(A: ts.Tensor["M", "K"], B: ts.Tensor["K", "N"]):
    ts.require(ts.constraint.Divisible("K", 8))
    return ts.ops.gemm(A, B)

aligned_gemm(np.random.randn(4, 7), np.random.randn(7, 16))  # K=7 → TesseraConstraintError
```

`JitFn.__call__` binds symbolic dimensions from tensor annotations, verifies
that repeated symbols resolve consistently across arguments, then re-runs the
constraint solver. Passing shapes are cached so repeated calls with the same
shape tuple avoid redundant solver work.

### Recommended user pattern today

For early errors on a known specialization, declare bindings at decoration
time:

```python
@tessera.jit(bindings={"K": K_VALUE})
def gemm(A: ts.Tensor["M", "K"], B: ts.Tensor["K", "N"]):
    tessera.require(ts.constraint.Divisible("K", 8))
    return ts.ops.gemm(A, B)
```

For polymorphic call sites, omit `bindings=` and let call-time enforcement
validate each new concrete shape.

---

## 11. MLIR Verifier Gap Enumeration (Sprint S5, 2026-05-22)

Per the 2026-05-06 spec gap audit's "identify MLIR verifier gaps
explicitly" ask, this section maps each shape-system contract to where
it is enforced today vs. where it is a gap. The legend:

- **PY-DT** = Python decoration-time check (decorator inspects bindings + signature)
- **PY-CT** = Python call-time check (`JitFn.__call__` resolves symbolic dims from actual args)
- **MLIR-PASS** = MLIR pass-level check (typically `ShapeInferencePass` or per-pass verifier)
- **MLIR-VERIFIER** = MLIR ODS op-level verifier (`mlir::OpTrait` / `verify()`)
- **RT-WITNESS** = Runtime shape witness (`RuntimeShapeWitness` + refinement)
- **GAP** = no enforcement today; named with the file that would own it

### 11.1 Per-contract enforcement matrix

| Contract | Where checked today | Evidence | Gap (if any) |
|----------|--------------------|---------|---------------|
| Symbolic dim equality (`Equal(A, B)`) | PY-DT, PY-CT | `python/tessera/compiler/constraints.py::ConstraintSolver.check`; `tests/unit/test_constraints.py` | **MLIR-PASS** — symbolic-dim equality is not re-checked at the MLIR level after Graph IR lowering; would belong in `src/compiler/diagnostics/ShapeInferencePass.cpp` |
| Concrete dim divisibility (`Divisible(D, k)`) | PY-DT, PY-CT | `constraints.py::Divisible`; `tests/unit/test_constraints.py`, `test_dynamic_shapes.py` | **MLIR-PASS** — schedule feasibility check uses Python `check_schedule_tile`; MLIR equivalent not yet a verifier |
| Range constraints (`Range(D, lo, hi)`) | PY-DT, PY-CT | `constraints.py::Range`; `tests/unit/test_constraints.py` | **MLIR-PASS** — same gap pattern as divisibility |
| Derived dim products (`D = H * Dh`) | PY-DT, PY-CT | `constraints.py::Derived`; `shape.py::DimProduct`; `tests/unit/test_shape_system_foundation.py` | **MLIR-PASS** — product factorization not verified at MLIR level after Graph IR; only inference-by-binding today |
| Mesh axis name existence | PY-DT, PY-CT | `shape.py::check_shard`; `distributed/shard.py::MeshSpec` | **MLIR-PASS** — `tessera.shard` ops do not have an ODS verifier that checks mesh axis name against the active `MeshSpec` (`src/compiler/ir/TesseraOps.td`) |
| Shard divisibility (`B % dp == 0`) | PY-DT | `shape.py::check_shard`; `tests/unit/test_shape_system_foundation.py` | **MLIR-VERIFIER** — would belong on the `tessera.shard` op; today the check fires before Graph IR emission |
| Layout cast insertion (producer/consumer disagreement) | PY (frontend) | `compiler/graph_ir.py` inserts `layout_cast` ops | **MLIR-PASS** — no pass-level verifier enforces "every producer layout matches consumer accept-set"; would belong as a `LayoutLegalityPass` |
| NumPy-style broadcasting | PY (op shape rule) | `shape.py::broadcast_shape`; verified via Graph IR `inferElementwise` | **MLIR-VERIFIER** — elementwise ops do not have an ODS verifier for the broadcasting rule (only inference); per-op `verify()` would catch shape mismatches at parse time |
| Matmul inner-dim equality | PY (op rule) + MLIR-PASS | `shape.py::matmul_shape`; `ShapeInferencePass.cpp::inferMatmul` | **MLIR-VERIFIER** — `tessera.matmul` ODS does not have a custom verifier; inference catches mismatches but only when both shapes are concrete |
| FlashAttn shape rule | PY (op rule) + MLIR-PASS | `inferFlashAttn` in ShapeInferencePass | **MLIR-VERIFIER** — same gap pattern as matmul |
| Reshape / Transpose / Concat / Slice / Reduce shape rules | MLIR-PASS | 8 per-op rules in `ShapeInferencePass.cpp` (inferReshape, inferTranspose, inferConcat, inferSlice, inferReduce) | **MLIR-VERIFIER** for per-op verifiers; inference handles correctness today |
| Tile fragment size legality (WGMMA/MFMA/TCgen05) | MLIR-PASS + lit | FA-4 Tile IR lowering tests under `tests/tessera-ir/phase3/cuda13/`; lit FileCheck on emitted PTX | **MLIR-VERIFIER** — `tile.mma` / `tile.wgmma` ODS verifiers do not enforce fragment-size legality per target; relies on the lowering pass to gate |
| Shared-memory bank/vector alignment | structural verifier (lit) | Tile IR lowering tests for `tile.alloc_shared` | **MLIR-PASS** — no dedicated alignment legality pass; falls out of lowering failure if violated |
| Async copy / TMA / mbarrier memory-model legality | PY-VERIFIER + lit | `compiler/memory_verifier.py` (Sprint M4 + M5); `tests/unit/test_memory_verifier.py` (46 tests) | None — memory-model verifier covers the Tile IR layer; backend lowering is Phase G/H/I |
| Runtime shape witnesses | RT-WITNESS | `shape.py::RuntimeShapeWitness`; `JitFn.__call__` re-runs ConstraintSolver | None — the witness contract IS the runtime check |

### 11.2 Summary of gaps

The audit's "MLIR verifier gaps" ask, condensed.  **Sprint V1+V2+V3
(2026-05-22) corrected the framing and closed three of the four
items:**

1. **ODS-level shape verifiers on `tessera.*` ops.** Original gap
   wording was inaccurate — 15 ops in `TesseraOps.td` already had
   `let hasVerifier = 1;` + a `verify()` body in `TesseraOps.cpp`
   (`MatmulOp`, `Conv2DNHWCOp`, `FlashAttnOp`, `FusedEpilogueOp`,
   `AttnLocalWindow2DOp`, plus 10 archsearch / state ops).  Coverage
   was uneven across the dialect.  **Sprint V1 closure (2026-05-22):**
   added `hasVerifier = 1;` + real `verify()` for `TransposeOp`,
   `LayerNormOp`, `MoeDispatchOp` with 9 stable diagnostic phrases
   and 9 lit-fixture cases (`tests/tessera-ir/phase2/sprint_v1_verifiers.mlir`).

   **Sprint V4b closure (2026-05-22):** added shape-preserve + attribute-
   bounds verifiers to the long-tail per-op set: `CastOp` (rank +
   static-dim preservation; element type may differ), `SoftmaxOp`
   (rank + static-dim preservation; optional `axis` must satisfy
   `-rank <= axis < rank`), `RopeOp` (rank + element type + static-dim
   preservation), and `DropoutOp` (probability must satisfy
   `0.0 <= p < 1.0`; shape preserved).  Previously `DropoutOp::verify()`
   was a trivial `return success();` stub; V4b replaces it with the
   real check.  Stable diagnostic phrases: `cast must preserve {rank,dim}`,
   `softmax must preserve {rank,dim}`, `axis out of range`,
   `rope must preserve {rank,element type,dim}`,
   `dropout probability must satisfy 0.0 <= p < 1.0`.  Lit fixture
   `tests/tessera-ir/phase2/sprint_v4b_per_op_verifiers.mlir` carries
   4 positive cases and 7 negative cases across all four ops.
   The "uneven coverage" subgap remains for any further dialect ops
   (op-by-op work as needs arise).  **Status: partially closed.**
2. **MLIR-level symbolic dim equality re-check after lowering.**
   **Sprint V5 closure (2026-05-22):**
   `src/transforms/lib/SymbolicDimEqualityPass.cpp` ships as
   `--tessera-symdim-equality`.  It reads function-level
   `tessera.dim_bindings` (ArrayAttr of equation strings like
   `"D = H * Dh"`) and `tessera.dim_sizes` (DictionaryAttr of
   symbol → i64), validates each equation when both sides are
   bound, and walks `tessera.reshape` / `tessera.transpose` /
   `tessera.matmul` ops checking the per-op dim-name contract.
   Four stable diagnostic codes: `SYMDIM_BINDING_VIOLATION`,
   `SYMDIM_RESHAPE_VIOLATION`, `SYMDIM_TRANSPOSE_VIOLATION`,
   `SYMDIM_MATMUL_CONTRACT_VIOLATION`.

   **Sprint V6a closure (2026-05-22):** `tessera.reshape`
   registered as a proper ODS op in `TesseraOps.td` +
   `TesseraOps.cpp` with element-count-preserving verifier; the
   V5 reshape branch is now exercised end-to-end through the
   real C++ binary (lit fixture grew from 1 positive + 2
   negative → 1 positive + 3 negative covering all 3 stable
   diagnostic codes whose ops are now registered).

   **Sprint V6b closure (2026-05-22):**
   `--tessera-symdim-equality` is now inserted into the three
   named lowering pipelines (`tessera-lower-to-x86`,
   `tessera-lower-to-gpu`, `tessera-nvidia-pipeline` family)
   AFTER `DistributionLoweringPass`, so broken `where D = H * Dh`
   clauses are caught automatically mid-flight.  Lit fixture
   `tests/tessera-ir/phase2/sprint_v6b_symdim_in_pipeline.mlir`
   proves the pass fires inside the named pipeline.

   **Sprint V2-flow closure (2026-05-22):**
   SSA-value dim-name propagation added.  The pass now reads
   function-level `tessera.arg_dim_names` (ArrayAttr-of-ArrayAttr
   naming each dim of each function argument) and propagates
   through `tessera.transpose / matmul / reshape` ops without
   requiring per-op `dim_names_in/out` annotations.  Inferred
   dim-names cross-check against any explicit per-op annotation
   and emit `SYMDIM_FLOW_INCONSISTENCY` on disagreement.  Lit
   fixture: `tests/tessera-ir/phase2/sprint_v2_flow_propagation.mlir`
   (1 positive + 1 negative + 1 backward-compat).  V1's behaviour
   is preserved as the fall-through when no `tessera.arg_dim_names`
   is declared — existing V5/V6a/V6b functions keep working
   unchanged.

   **Sprint V3a closure (2026-05-22):** Affine reasoning for
   non-product bindings landed.  The binding parser now accepts
   sum-of-products RHSes such as `D = H * Dh + K` and bare-symbol
   terms such as `Total = A + B + C`.  Multi-term bindings render
   `value of RHS (sum of products) = N`; single-term bindings keep
   V5's `product of RHS = N` wording for backward compatibility.
   Lit fixture: `tests/tessera-ir/phase2/sprint_v3a_affine_bindings.mlir`
   (1 multi-term holds + 1 multi-term broken + 1 V5 single-product
   still works + 1 sum-of-3-singletons).

   **Sprint V3b closure (2026-05-22):** Interprocedural dim-name
   tracking landed.  The pass builds a module-level `SymbolTable`,
   walks `func.call` ops, and cross-checks each caller's propagated
   dim-names against the callee's declared `tessera.arg_dim_names`
   position-by-position.  Mismatch ⇒ `SYMDIM_CALL_ARG_MISMATCH`.
   The pass also reads `tessera.ret_dim_names` on the callee and
   seeds the call result values, so dim-names flow ACROSS the call
   boundary into subsequent ops in the caller.  Lit fixture:
   `tests/tessera-ir/phase2/sprint_v3b_interprocedural.mlir`
   (1 caller-callee match + 1 mismatch + 1 ret_dim_names propagation
   + 1 backward-compat no-callee-decl).

   **Sprint V3c closure (2026-05-22):** SSA flow through `scf.for`
   and `scf.if` region bodies landed.  For `scf.for`, iter_args
   inherit dim-names from init operands, the body is walked
   recursively, and the `scf.yield` operands must match the
   iter_args' names (loop must be name-invariant) ⇒
   `SYMDIM_LOOP_YIELD_MISMATCH` on conflict.  For `scf.if`, both
   branches' yields must agree ⇒ `SYMDIM_IF_BRANCH_MISMATCH` on
   conflict; the scf.if result inherits the (matching) yield names.
   Lit fixture: `tests/tessera-ir/phase2/sprint_v3c_scf_propagation.mlir`
   (scf.for invariant + scf.for yield mismatch + scf.if branches
   agree + scf.if branches disagree).

   **Status: V1 + V2-flow + V3a + V3b + V3c all shipped.  Further
   affine/Presburger reasoning beyond sum-of-products (subtraction,
   parens, integer literals) remains as backlog.**
3. **`LayoutLegalityPass` skeleton + first rule.** **Sprint V2
   closure (2026-05-22):** `src/transforms/lib/LayoutLegalityPass.cpp`
   ships with the canonical 8-name layout accept-set
   (`row_major / col_major / nhwc / nchw / bhsd / tile / bsr / packed`),
   first rule (`tessera.cast` with non-canonical `tessera.layout`
   attribute emits `LAYOUT_LEGALITY_UNKNOWN_LAYOUT` and fails the
   pass), registered as `--tessera-layout-legality` in
   `Passes.cpp`, wired into the `TesseraPasses` CMake target, and
   exercised by `tests/tessera-ir/phase2/sprint_v2_layout_legality.mlir`
   (positive: row_major, bhsd, no-attr; negative: exotic_block_format).
   **Sprint V4a closure (2026-05-22):** Producer/consumer accept-set
   rule shipped.  `--tessera-layout-legality` now also walks each
   `tessera.matmul` operand's def-using op for a `tessera.layout`
   attribute and rejects layouts outside matmul's stricter
   `{row_major, col_major}` accept-set with stable code
   `LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH`.  Lit fixture
   `tests/tessera-ir/phase2/sprint_v4a_layout_producer_consumer.mlir`
   (positive: row_major OK + no-producer-attr OK; negative: bsr-lhs,
   packed-rhs).  Identity-cascade folding remains a comment-only
   placeholder.  **Status: skeleton + 2 rules closed; identity-cascade
   folding planned.**
4. **Target-aware verifier on the canonical attention/MMA op family.**
   **Sprint V3 closure (2026-05-22):** `FlashAttnOp::verify()` extended
   to walk the parent op chain for a `tessera.target_sm` attribute and
   enforce the per-SM `head_dim` ceiling from
   `docs/nvidia_cuda13_kernel_inventory.md` (sm_70/75/80/86/89 ≤ 128;
   sm_90/100/120 ≤ 256; no SM tag ⇒ no limit applied — CPU path
   unaffected).  Diagnostic phrase `head_dim=N exceeds the SM <sm> ...`
   exercised by `tests/tessera-ir/phase3/sprint_v3_flash_attn_target_aware.mlir`
   (positive: sm_90 at 256; negative: sm_90 at 257; negative: sm_80 at
   256).  Generalizing this pattern to `tile.mma` / `tile.wgmma` /
   `tessera.attn.scaled_dot_product` for tile-shape legality is the
   next slice.  **Status: head_dim ceiling closed; tile-shape legality
   planned.**

### 11.3 Active evidence pointers

Specs that claim stronger MLIR-verifier coverage should cite tests; the
following are the canonical evidence files:

| Surface | File |
|---------|------|
| Python constraint solver | `python/tessera/compiler/constraints.py`; `tests/unit/test_constraints.py` |
| Python shape objects + helpers | `python/tessera/shape.py`; `tests/unit/test_shape_system_foundation.py` |
| MLIR per-op shape inference | `src/compiler/diagnostics/ShapeInferencePass.cpp`; `tests/unit/test_shape_inference.py` |
| Dynamic shape call-time enforcement | `python/tessera/compiler/jit.py::JitFn._enforce_call_time_constraints`; `tests/unit/test_dynamic_shapes.py` |
| Memory-model verifier (Tile IR layer) | `python/tessera/compiler/memory_verifier.py`; `tests/unit/test_memory_verifier.py` (46 tests) |
| Runtime shape witnesses | `python/tessera/shape.py::RuntimeShapeWitness`; `tests/unit/test_shape_system_foundation.py` |

A primitive that claims a contract not in §11.1 above must add a row
here with the file path that owns the check, or downgrade the claim to
`planned`.
