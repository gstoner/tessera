---
last_updated: 2026-09-03
audit_role: reference
scope: python/tessera/compiler/{graph_ir,trace,jit,driver}.py, emit/kernel_emitter.py, structured_cfg.py, presburger.py, src/transforms/lib/{SymbolicDimEqualityPass,TesseraToLinalgPass}.cpp, the Graph→Target text boundary
companions: FRONTEND_GRAPH_SCHEDULE_REVIEW.md · IR_STACK_INTEGRATION_REVIEW.md · COMPILER_THEORY_OF_OPERATION.md · TARGET_IR_REVIEW.md
queue: INTEGRATED_COMPILER_PLAN.md · FRONTEND-IR-MEDIUM-1
---

# Front-End Lowering Assessment — KGEN as a Yardstick

Prompted by a first-principles walk through Modular's now-open-source **KGEN**
compiler (Mojo → LIT → KGEN/POP/HLCF → elaboration → LLVM dialect → LLVM IR).
The exercise is compare-and-contrast, not adoption: where does KGEN's discipline
expose an assumption in Tessera's Python→MLIR front half worth challenging?

Status truth stays with the generated dashboards (Decision #26). This is a
reference; global ordering defers to `INTEGRATED_COMPILER_PLAN.md`
(**FRONTEND-IR-MEDIUM-1**).

All evidence below was reproduced 2026-09-03 on the Mac against the checked-in
`build/tools/tessera-opt`. Every probe is host-independent (no device lane), so
it is valid on any fleet box.

---

## 0. A correction that removes the easy explanation

An earlier reading leaned on **G2** from
[`FRONTEND_GRAPH_SCHEDULE_REVIEW.md`](FRONTEND_GRAPH_SCHEDULE_REVIEW.md) — "type
inference is a five-case if-chain defaulting to `operand_types[0]`, fail-open."
That is **stale**. The current
[`graph_ir._infer_result_type`](../../../python/tessera/compiler/graph_ir.py#L3738)
is a **72-rule registry** keyed by `shape_rule_for(op_name)` that **fails loud**:

```python
# graph_ir.py:3763
if fn is None:
    raise KeyError(
        f"shape rule {rule!r} is declared for {op_name!r} in op_catalog "
        f"but has no implementation in graph_ir._SHAPE_RULES")
```

```
$ PYTHONPATH=python python3 -c "from tessera.compiler.graph_ir import _SHAPE_RULES; print(len(_SHAPE_RULES))"
72
```

The surviving `operand_types[0]` lines are now *named* rules (`same_as_first`),
not a default. Per the amendment protocol, the review is a doc-bug against the
code, not a live finding — corrected in that review with a dated banner on
2026-09-03.

The correction matters because it removes the tempting explanation ("the
frontend is just weak") and forces the structural one below.

---

## (a) The finding: Tessera runs on two memories; KGEN runs on one

KGEN's thesis, stripped of its dialect list, is **the IR is the single medium of
record.** Every fact a pass needs — lifetime (`!lit.ref<T, origin>`),
parametricity (a KGEN generator), loop structure (`hlcf.for`), debug lineage —
is a type/attr/op *on the IR*, and passes read those facts *from the IR*.
Lowering is the act of deciding when a fact has finished its job and may be
dropped (`LowerLIT` turns a checked reference into a pointer). There is one
memory, and it is the IR.

Tessera, as built, keeps **two memories.** Below the Graph IR boundary the MLIR
text is authoritative — the compiled route feeds canonical Graph IR to
`tessera-opt`, which then owns Schedule→Tile→Target
([`driver.py:2130`](../../../python/tessera/compiler/driver.py#L2130)). But *at
and above* that boundary, and for everything cross-cutting, the MLIR text is a
**lossy projection of a richer Python object graph.** The computation lowers
through MLIR; the *facts about the computation* ride in Python beside it.

### Evidence table

| Fact | KGEN | Tessera — where it actually lives | Probe |
|---|---|---|---|
| Symbolic shapes | IR type, parametric | Python `IRType`; **does not survive to `tessera-opt`** | `Tensor['M','K']` → `tensor<?x?x?>` (rank inflated, dtype gone) → parse error `expected 'x' in dimension list`; a named-dim render `tensor<MxNxf32>` → `expected non-function type` |
| Source location | `loc` on every op | **Nowhere in the MLIR text** — **tracer closed 2026-09-03** (repo-relative `loc` in the canonical render, parser-verified; the AST frontend still emits none) | `'loc(' in f.graph_ir.to_mlir()` → `False` |
| Numeric policy | — (LLVM has no field) | Carrier `to_mlir_attr` exists, but a plain `matmul` emits **none** — opt-in, not universal | `'numeric_policy' in to_mlir()` → `False` for `ops.matmul` |
| Provenance / route | — | Python descriptor objects; `provenance` appears in **46** compiler modules, **~1** as an MLIR attr string | `grep -rln provenance python/tessera/compiler/*.py \| wc -l` → 46 |
| Arbiter decision (Decision #28) | — | Python (`composition_cost.provenance`, `autotune_v2.timing_provenance`), never an IR attribute | source read |

The precise claim — narrower and sharper than "two disconnected compilers":
**the system of record is the Python object graph, and MLIR is downstream of
it.** `tessera-opt` optimizes the projection; the Python spine reasons over the
record; the two do not share a memory. The Apple "two compilers" seam and the
Python-packager seam are *symptoms* of this, not the disease.

### Why this is the right frame for the north star

Decision #28's arbiter needs provenance/accuracy/determinism to pick a kernel;
Decision #13 needs `loc` so a human can read a diagnostic; codegen needs
`numeric_policy` to pick an instruction. Those are **three consumers with three
definitions of "useful information"** — exactly KGEN's debug-info tension, one
consumer wider. Today Tessera satisfies the arbiter (in Python) and the human
(in Python) but starves the *MLIR passes* of both. The asymmetry is telling: we
remember what the Python arbiter needs and forget what the C++ passes need,
because only one of them shares the record.

E2E-REAL-6's "one compiler authority" currently means **one frontend** (promote
the tracer, delete `_OpExtractor`). The KGEN comparison earns a stronger
reading: **one medium** — every fact a pass reads is an attribute on the IR the
C++ passes see. That is the governing principle this assessment proposes, and
Decision #29 ("a declaration must have a consumer") is its enforcement arm run
in reverse: *a consumer must read from the declared medium, not a shadow.*

---

## (b) Elaboration is entirely pre-MLIR — the perf/rigor cost, and the substrate already present

KGEN optimizes the **parametric** program (`add<N>`) *before* elaboration
collapses it to `add<42>`, because elaboration multiplies code: clean the recipe
before stamping it N times. Passes appear on both sides of the boundary because
elaboration reveals new facts (a branch on a compile-time parameter becomes
constant).

Tessera elaborates **entirely before MLIR exists.** Confirmed by construction:

```
decoration-time module (symbolic):  tensor<?x?x?>     → tessera-opt: parse error
traced concrete instance:           tensor<8x16xf32>  → parses, round-trips clean
```

So `tessera-opt` only ever receives **post-elaboration instances.** There is no
parametric level *in* MLIR and therefore **no pre-elaboration optimization
tier** — the stage KGEN spends most of its effort on.

**Performance cost, tied directly to the arbiter.** Decision #28 sweeps
`(op, shape-bucket, dtype, target)`. Today each bucket is a *separately lowered
concrete instance*; shape-independent optimization is re-run per bucket or
approximated by the Python `SpecPolicy.BUCKET` key in
[`emit/kernel_emitter.py:50`](../../../python/tessera/compiler/emit/kernel_emitter.py#L50)
— again outside MLIR. The KGEN move is: optimize the parametric recipe **once**,
in the IR, then elaborate per bucket. Two payoffs: (1) compile-time — one
optimization pass instead of N; (2) *comparison integrity* — the arbiter
compares buckets of **one optimized recipe** instead of N independently-lowered
programs, so a bucket-to-bucket regression is a real difference, not a lowering
artifact. This is the strongest single reason to challenge the
"specialize-in-Python-then-lower" assumption.

**Rigor cost.** KGEN carries a compile-time *interpreter* — principled partial
evaluation. Tessera's
[`specialize_module_from_values`](../../../python/tessera/compiler/graph_ir.py)
is value substitution, not a typed partial evaluator over a symbolic domain. The
rigorous form is a **total, symbolic shape/type semantics** where a shape fact is
a theorem about `matmul<M,K,N>`, not a per-instance check — which is also what
would let the *symbolic* program be verified in MLIR rather than only its
concrete shadow.

**The substrate is already in tree — this is not greenfield.**
`presburger.py` (`PresburgerSystem`) is carried through
[`structured_cfg.py`](../../../python/tessera/compiler/structured_cfg.py) as the
"typed Presburger-system digest" every block inherits so region consumers cannot
lose shape constraints at a branch. That is precisely the substrate a parametric
shape/type semantics needs. And there is a **Decision #29 violation hiding here**
to retire as part of the same work:
[`SymbolicDimEqualityPass`](../../../src/transforms/lib/SymbolicDimEqualityPass.cpp#L554)
consumes `tessera.dim_names`, but the frontend cannot emit a parseable symbolic
program to feed it (the `tensor<MxNxf32>` / `tensor<?x?x?>` renders both fail to
parse). A verify pass whose producer is broken is a declared consumer with no
live producer path — the mirror image of #29's usual failure.

---

## (c) Raising is the missing capability — and it is how user math reaches the arbiter

KGEN's `RaiseForLoops` breaks the one-way-elevator model: recognize structure in
a lower form and *raise* it to a richer one because the richer form makes the
next optimization easier, then lower again.

Tessera has lowering — [`TesseraToLinalgPass`](../../../src/transforms/lib/TesseraToLinalgPass.cpp),
`TilingPass`, the Tile→Target chains — and **no inverse.** `structured_cfg.py`
raises a *trace* to a block graph, but there is no pass that lifts a
hand-written loop nest back to `tessera.matmul` / `tessera.flash_attn`.

For a compiler whose north star is "**leads set the ceiling** via named
high-performance kernels" (Decision #28: hand-emitted `wgmma`/`mma.sync`/MFMA
stay first-class arbiter candidates), the inability to *raise* means **user-written
math can never enter the fast tier.** A user who writes an attention loop by hand
gets the generic synthesizer (Tier 1) at best; the arbiter's Tier-3 hand-tuned
`flash_attn` candidate is unreachable because nothing recognizes the loop as
attention. Raising / idiom-recognition is the on-ramp from arbitrary user code
into the arbiter's high-performance population. This is the algorithmic gap with
the clearest ceiling attached to it.

This composes with (b): a parametric idiom (`matmul<M,K,N>` recognized from a
loop nest) raised *before* elaboration is optimized once and elaborated per
bucket — raising and parametric-optimize are the same program viewed from two
ends.

---

## (d) The down-payment — items 1/2/3

These are correct, small, and independently landable. They are the surface of
the same debt, so framing them as one down-payment on "one medium" keeps them
from being read as one-offs.

1. **Fail closed on an unresolved element type.**
   [`tensor_ir_type`](../../../python/tessera/compiler/graph_ir.py#L445) renders
   `tensor<{dims}x{dtype}>` and substitutes `?` for an unknown dtype, so a rank-2
   `Tensor['M','K']` becomes a malformed `tensor<?x?x?>`. A missing dtype is a
   **semantic key** (Decision #21a) and must fail closed, not degrade into an
   unparseable type. **Corrected 2026-09-03 on
   implementation — the front door is *not* silent.** `compile_graph_module`
   calls the reason-returning `_lower_apple_value_target_ir` and records the
   failure as `apple_value_target_ir_error` on the bundle (the S4 "observable,
   never silent" contract,
   [`driver.py`](../../../python/tessera/compiler/driver.py#L758)). The real
   defect was the *quality* of that reason: it was the parser's symptom
   (`expected 'x' in dimension list`), naming neither the argument nor the
   missing semantic key. **Landed 2026-09-03:**
   `graph_ir.unresolved_element_type_diagnostics` emits a named
   `GRAPH_IR_UNRESOLVED_ELEMENT_TYPE` per offending argument / result / op (the
   string form is the test, so `index`, `i1`, and `!tessera.*` handles — valid
   MLIR with `dtype=None` — are never flagged), and the driver's value lane
   consults it *before* rendering, so the recorded reason is the #21a
   diagnostic. The renders are byte-untouched: `?` stays the legitimate "not
   yet specialized" placeholder of the symbolic module, which is never
   parser-bound. (The reason-*discarding* public wrapper
   `lower_apple_value_target_ir` has no callers; left for a follow-up.)

2. **Emit `loc` from the tracer** — and do it **before** `_OpExtractor` is
   deleted. `trace.py` has **zero** `source_span` sites; the AST `_OpExtractor`
   slated for deletion has **13** `source_span=_span_from_ast(node)` sites.
   Promoting the tracer to sole authority deletes the only frontend that records
   spans at all, regressing Decision #13. This is the first repayment of the
   lossy-projection debt: the rule to adopt is *any fact a Python pass reads must
   be an attribute on the IR the C++ passes see*, and `loc` is the most visible
   violator.

   **Landed 2026-09-03.** `trace._user_source_span` records the first stack
   frame *outside* the tessera package on every recorded op (`record_op` and
   the four control-flow recorders), so the wrapper-chain depth never matters
   and a call made from inside a tessera layer reports the caller's line, not
   the library's. The canonical render appends `loc("file":line:col)`; the
   paren (golden-text) render is byte-unchanged. The path is **repo-relative
   for in-repo files** — the canonical text is content-addressed downstream
   (`stable_hash` of what is fed to `tessera-opt`), so an absolute path would
   have made the same program hash differently on every checkout — and
   absolute otherwise. Verified end-to-end: `tessera-opt -mlir-print-debuginfo`
   preserves the location (`tests/unit/test_trace_loc.py`).

3. **Declare the symbolic→concrete elaboration boundary under Decision #32.**
   It is the single largest information-loss event in the stack (the entire
   parametric program is discarded), and #32's boundary verifier does not cover
   it. **Extend the existing mechanism, do not add a second** (Decision #29/#31):
   W1.3 already ships `--tessera-record-metadata` /
   `--tessera-verify-metadata-obligation`, which rides the snapshot as a module
   attribute so record→lower→verify is one `tessera-opt` invocation — the same
   pass that caught the canonical `numeric_policy`-lost-below-MMA scar. The
   elaboration drop, the region privileges (Decision #2), and the
   `ConstraintSolver` (Decision #4) should be declared *through that obligation*,
   with `not_yet_carried:<item>` keeping the debt attributable. The perf/rigor
   program in (b) is what the declaration should point at.

**Verdict.** 1/2/3 are necessary but they are the down-payment, not the reform.
The reform is to treat **the IR as the sole medium of record** (extend
E2E-REAL-6's "one authority" from *frontend* to *medium*), which turns (b) and
(c) from ideas into a program: a pre-elaboration parametric optimization tier on
the existing Presburger substrate, and a raising path that feeds user math into
the Decision #28 arbiter.

---

## Filing

- Queue ID: **FRONTEND-IR-MEDIUM-1** in
  [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md), bound from this
  assessment. Global ordering and promotion authority remain there.
- Decisions engaged: **#28** (arbiter comparison integrity depends on a shared
  medium and on raising to populate tiers), **#29** (consumer-must-read-from-the-
  medium; the `SymbolicDimEqualityPass` producerless consumer), **#32**
  (elaboration and privilege/constraint drops must be declared).
- Reproduction probes are inline in each section; all host-independent.
