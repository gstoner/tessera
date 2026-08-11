---
last_updated: 2026-08-02
audit_role: reference
scope: python/tessera/__init__.py, compiler/{jit,graph_ir,trace,schedule_ir,constraints}.py, src/compiler/programming_model (GraphToSchedulePass), Graph IR ODS
companions: COMPILER_ARCHITECTURE_SWEEP.md · AUTODIFF_ARCHITECTURE_REVIEW.md · ../domain/GA_EBM_ARCHITECTURE_REVIEW.md · AUTODIFF_UNIFICATION_PLAN.md
---

# Developer Interface → Graph IR → Schedule IR Review

The front half of the stack: what a developer writes, how it becomes Graph IR,
and how Graph IR becomes Schedule IR. This is the layer the three previous
reviews kept pointing back at without examining.

The headline: **several downstream findings turn out to have a single upstream
cause.** The effect lattice walks the Python AST because the *frontend* walks the
Python AST. Autodiff can't differentiate loops partly because the frontend can't
produce them. The shape system is weak partly because type inference is an
if-chain that defaults to the wrong answer.

Status truth stays with the generated dashboards (Decision #26).

**2026-08-10 status correction:** E3/U7 is complete. `_EffectVisitor` has been
removed; emitted/traced Graph operations carry registered effect, alias, and
stochastic-identity contracts consumed by Python and C++. The pipeline diagram
below is the review-time snapshot and is retained to explain the original
finding, not current status.

---

## 0. The pipeline as documented vs as built

`CLAUDE.md` describes:

```
Python API (@jit, Region[...], domain, index_launch)
  ▼
Graph IR (tessera dialect — mathematical ops, effects, shapes)
  ▼
Schedule IR (schedule.* — mesh regions, pipeline stages, optimizer sharding)
  ▼
Tile IR → Target IR
```

What is actually built:

```
Python source
  ├── ast.parse ──► _ConstraintExtractor    (jit.py:150)     — constraints
  ├── ast.parse ──► _EffectVisitor          (effects.py:173) — effects
  └── ast.parse ──► _OpExtractor            (graph_ir.py:993, ~680 lines)  ◄── DEFAULT
                       │
       OR (only when target == "apple_gpu" AND function_needs_tracer())
        └── Tracer/TraceBuilder             (trace.py, 881 lines)
                       │
                       ▼
                  GraphIRModule
                       │
       ├── Python: lower_graph_to_schedule_ir()      (schedule_ir.py:277)
       └── C++:    GraphToSchedulePass               (PassPipelinesPM11.cpp:200)
                       │
                       ▼
                  Schedule IR
```

Three independent AST walks over the same source, two frontends selected by
target string, and two Graph→Schedule lowerings.

---

## 1. Architectural findings

### G1 — Two frontends with **opposite failure policies**, and the one that fails open is the default

`jit.py:394` decides per function:

```python
self._needs_trace = function_needs_tracer(graph_ir, fn)
```

and `jit.py:664` dispatches:

```python
if self.target == "apple_gpu" and self._needs_trace:
```

So the tracer — the more faithful frontend — is reached **only on Apple GPU**.
Every other target falls to the AST extractor. The same Python function produces
a different Graph IR depending on the target string, which means the frontend has
target-conditional semantics. That is a category of bug no amount of backend
testing catches.

Worse, the two disagree on the most important policy in a frontend. On an op it
does not recognize:

```python
# trace.py:130 — FAILS CLOSED
raise TesseraTraceError(
    f"trace: no shape rule for op {name!r} ...; register one via "
    f"tessera.compiler.trace.register_shape_rule")
```

```python
# graph_ir.py:1730 — FAILS OPEN
return operand_types[0]
```

**The frontend that raises is the special case; the frontend that silently
returns the wrong type is the default.** This is defect shape L4 from the
[sweep](COMPILER_ARCHITECTURE_SWEEP.md), now found at the entry point to the
entire compiler.

### G2 — Type inference is a five-case if-chain over 315 ops, defaulting to "the first operand's type"

[`graph_ir.py:1702`](../../../python/tessera/compiler/graph_ir.py#L1702)
`_infer_result_type` handles `tessera.matmul`, `tessera.batched_gemm`, two EBM
ops, and `tessera.transpose`; its fallback returns `operand_types[0]`.
`_struct_result_type` subsequently repairs a bounded set of structural
reshape/view/squeeze/permute-style operations once keyword attributes are
bound.  The fail-open default remains for the rest of the catalog, but the
structural helper is an existing partial consumer and should be folded into the
canonical rule registry rather than described as absent.

That default is *silently wrong* for unhandled reductions, concat,
gather/scatter, dtype-changing casts, multi-output ops, and other families not
covered by `_struct_result_type`—still a large portion of the catalog.

The code knows. `_struct_result_type`'s own docstring:

> Result type for the structural view ops from their shape-determining attr.
> **Without this, `_infer_result_type`'s fallback returns the INPUT type**, …

So a second patch function was written to cover a subset of the cases the
fallback gets wrong, rather than the fallback being made an error.

And the contract already exists elsewhere: `primitive_coverage.py` carries a
`shape_rule` axis that `MASTER_AUDIT.md` reports closed. **This is the fourth
instance of the same pattern** — after the `manifold` attribute, GA grade
sparsity, and `vmap` vs the batching-rule registry: an audited contract exists and
the consumer that needs it doesn't read it.

### G3 — Three shape/type systems, none authoritative, none aware of the others

| System | Design | Failure policy | Reached from |
|---|---|---|---|
| `graph_ir._infer_result_type` | hand-written if-chain | **fails open** (`operand_types[0]`) | all targets except Apple GPU |
| `trace._SHAPE_RULES` + `register_shape_rule` | **rule registry, extensible** | **fails closed** (`TesseraTraceError`) | Apple GPU only |
| `shape.py` `matmul_shape` / `broadcast_shape` / `dims_compatible` | symbolic-ish, name equality | mixed | the public `Shape`/`Dim` API |
| C++ ODS verifiers | per-op `verify()` | fails closed | `tessera-opt` |

`op_catalog.OpSpec` — the "what we accept today" registry — carries arity
(`min_operands`, `max_operands`) and a `lowering` category, and **no shape rule
at all**, so the one registry that should be authoritative is not among the four.

`trace.py` has the right design. It should be the only one.

### G4 — SSA is hand-constructed on an AST with no CFG, which is a design that cannot converge

[`graph_ir.py:1050`](../../../python/tessera/compiler/graph_ir.py#L1050),
`_reserve_ssa_for_assign` plus `_name_alias`, minting `c` → `c__1` → `c__2` on
reassignment, with an explicit comment about the ordering hazard in `c = c + 1`.

This is re-implementing SSA renaming on Python source text. It works for
straight-line reassignment and **cannot** work for control-flow merges, because
merges need φ-nodes and φ-nodes need a CFG, which an AST visitor does not have.

That is exactly why the neighbouring visitors are restricted:

- `visit_If` (line 1161) — "Diagnostic is demoted from 'unsupported warning' to …"
- `visit_For` (line 1256) — "static trip count (`range(N)`) keeps the …"
- `visit_While` (line 1319) — `self._unsupported(...)` paths

These are not gaps to be filled in a later sprint. They are the boundary of what
the chosen representation can express. The standard algorithms (Braun et al.,
*Simple and Efficient Construction of SSA Form*; or MLIR's structural
region/`scf` model, which sidesteps φ entirely) both require what an AST walk does
not provide.

**A tracer produces SSA by construction** — every traced value is a fresh name,
no renaming, no aliasing, no `c__1`. The problem disappears rather than being
solved.

### G5 — `JitFn` contains ~1400 lines of per-target, per-op-family backward dispatch

`jit.py` is 3646 lines. Inside the `JitFn` class:

`_native_norm_backward`, `_native_sgd_backward`, `_native_momentum_backward`,
`_native_rocm_adam_backward`, `_native_rocm_lion_backward`,
`_native_regression_loss_backward`, `_native_rocm_distribution_loss_backward`,
`_native_binary_loss_backward`, `_native_class_loss_backward`,
`_native_nvidia_backward`, `_native_rocm_backward` — roughly lines 816–2256.

**The `@jit` decorator knows about ROCm Adam and NVIDIA layer-norm backward.**
That is a layering violation on its own, and it is also duplicated
responsibility: this is precisely what `emit/candidate.py`'s arbiter exists to do
(Decision #28) behind the paired `@f__bwd(inputs, cotangents) -> cotangents` ABI
that `AutodiffPairedPass` defines. The dispatch belongs in the candidate registry,
where it would be measured; instead it is an if-chain in the decorator, where it
is not.

### G6 — `__init__.py` is 5375 lines, ~5000 of them inside one function

`_make_ops_namespace()` (line 221) contains **315 nested `def`s** — the entire op
reference implementation surface, built inline, in the package root's module
body.

Consequences: import cost paid by every consumer, no op is independently
testable or lazily loadable, and the package root is the least navigable file in
the tree. The lazy-binding machinery already exists in the same file
(`__getattr__` at line 5353, PEP 562, used for `train`) — the pattern to follow is
present and unused for the bulk.

### G7 — Graph→Schedule is implemented twice, and the C++ pass lacks independent ownership

- Python: `lower_graph_to_schedule_ir` (schedule_ir.py:277) with per-op-family
  constructors — `_flash_attention_pipeline`, `_sequence_mixer_pipeline`,
  `_msa_kv_outer_sparse`, `_media_op`, `_jepa_op`.
- C++: `GraphToSchedulePass`, defined in
  `src/compiler/programming_model/tools/tessera-opt/PassPipelinesPM11.cpp:200`.
  Despite the path, CMake compiles that source into the `TesseraPM` library and
  the test dependency set links `TesseraPM`, so the pass is linkable. The real
  ownership defect is co-location with pipeline-driver code: it has no dedicated
  pass source/header or focused lit fixtures, which obscures its reusable API and
  lets the Python and C++ implementations drift without differential coverage.

Defect shape L2 (two disconnected compilers), at the Graph→Schedule seam.

### G8 — Schedule IR records schedules; it does not choose them

The Python lowering constructs pipeline stages and attributes per op family by
hand. `_schedule_raster_order` and `_schedule_raster_group` read values out of a
config dict. `_base_attrs` copies through. `_matmul_flops` / `_matmul_bytes`
compute cost *metadata* that nothing in the file consumes to make a decision.

So no scheduling **decision** happens at Schedule IR: tile sizes, stage counts,
raster order, and warp specialization all arrive pre-decided from elsewhere
(`@jit` kwargs, `autotune_v2`, per-target defaults). "Schedule IR" is today a
serialization format for choices made outside it, not a level at which
scheduling happens.

That is a missed layer, not a broken one — and it is the natural home for
exactly the machinery that already exists elsewhere: `fusion_core`'s cost models,
`autotune_v2`'s measured search, and `emit/candidate.py`'s arbiter.

---

## 2. The convergent finding: dynamic control flow is blocked at three independent layers

This is the most important structural fact in the four reviews:

| Layer | Blocker | Evidence |
|---|---|---|
| **Frontend** | `visit_For` needs a static `range(N)`; `visit_While` emits `_unsupported`; no CFG for φ-nodes | G4 |
| **Shape system** | no symbolic dims — cannot represent a trip count or a data-dependent extent | [sweep F2](COMPILER_ARCHITECTURE_SWEEP.md) |
| **Autodiff** | `AutodiffPass` hard-errors on any op with a nested region | [AD review A3](AUTODIFF_ARCHITECTURE_REVIEW.md) |

**Fixing any one of these alone changes nothing observable.** A frontend that can
emit `scf.for` produces IR that autodiff rejects; an autodiff pass that handles
loops has no loops to handle; symbolic dims with no producer are unused.

That matters for planning: these three items must be sequenced as one program with
a single end-to-end exit criterion — *a `@jit` function containing a
data-dependent loop compiles, differentiates, and executes* — or each will look
like it "landed" while the capability stays at zero. It also reframes their
individual cost: they are expensive separately and coherent together.

---

## 3. Algorithmic updates

### U1 — Make tracing the only frontend; delete `_OpExtractor`

The tracer already exists (881 lines), already has the extensible shape-rule
registry, already fails closed, already handles control flow structurally
(`_has_control_flow`, `_branch_dicts`, `_live_refs`, `_region_flat`), and already
produces SSA by construction. `control.py`'s `scan` / `while_loop` / `cond` /
`fori_loop` already hook `_active_trace_builder()`, which is the correct design
for the one thing tracing genuinely cannot do — Python control flow branching on
traced values.

This is not a new system (the `AUTODIFF_UNIFICATION_PLAN` §2a warning applies): it
is deleting one of two existing systems and promoting the better one. It resolves
G1, most of G2, all of G4, and unblocks the frontend third of §2.

The migration risk is real and should be stated: the AST path is the default for
every non-Apple target today, so this is a broad behavior change and needs the
op-by-op differential harness (trace vs AST on the same function, compare emitted
IR) before the switch, not after.

### U2 — One shape-rule registry, owned by `op_catalog`, auto-flipping into coverage

Move `_SHAPE_RULES` onto `OpSpec` so the catalog that already declares arity also
declares the shape rule. Then `primitive_coverage`'s `shape_rule` axis
**auto-flips from it** — exactly the mechanism `vjp`/`jvp` already use against
`_VJPS`/`_JVPS`, so the pattern is established and the drift gate comes free.

Unknown op ⇒ diagnostic, never `operand_types[0]`. Same rule for `dtype` and
`layout` propagation.

### U3 — Schedule IR becomes a level where decisions are made

Give Schedule IR a **cost-model-driven scheduling pass** rather than a per-op
transcription table:

- inputs: Graph IR + `fusion_core`'s cost models + target profile
- decisions: fusion-region boundaries, tile sizes, pipeline stage count, raster
  order, warp-specialization roles
- selection: the existing arbiter (`emit/candidate.py`) with the existing measured
  loop (`emit/autotune.py`), so the choice is measured, not defaulted

This also gives Decision #28's arbiter its natural insertion point in the *IR
pipeline* rather than only at kernel-emit time, and it is where the generic
fusion-region discovery from [sweep F3](COMPILER_ARCHITECTURE_SWEEP.md) should
live.

### U4 — Give `GraphToSchedulePass` dedicated ownership; delete the Python duplicate

`GraphToSchedulePass` is already compiled into the linkable `TesseraPM` library,
but its implementation and factory are buried in pipeline-driver source. Move
the implementation to a dedicated library-owned pass source/header, retain its
registration, add focused lit fixtures, and retire `lower_graph_to_schedule_ir`
once U3's decisions live in the pass.

### U5 — Decompose `JitFn`

`@jit` should own exactly: constraint check (Decision #4), effect inference
(Decision #5), trace, compile request, call. Everything per-target moves out:

- the eleven `_native_*_backward` methods → `emit/candidate.py` candidates behind
  the `@f__bwd` paired ABI, where the arbiter measures them
- the four `_*_fast_call` methods → the same candidate registry

Target: `jit.py` under ~800 lines with no target name appearing in it.

### U6 — Split `__init__.py`

Move `_make_ops_namespace`'s 315 op references into `tessera/ops/` modules by
family, and bind them through the PEP 562 `__getattr__` already present at line
5353. The package root becomes an export surface.

### U7 — One AST walk, or none

`_ConstraintExtractor`, `_EffectVisitor`, and `_OpExtractor` each independently
`ast.parse` the same source. After U1, constraints and effects should be derived
from the **traced IR**, not from source text — which is also the fix for
[sweep F1](COMPILER_ARCHITECTURE_SWEEP.md) (the effect lattice failing open on
aliased calls) and the correction Decision #5 needs.

Note the ordering constraint: Decision #4 requires the constraint solver to run at
*decoration* time, before any call, so constraints cannot come from a trace
(which needs example inputs). Constraint extraction from the AST is therefore
legitimate and should stay — but it should be the **only** AST walk, and it should
be explicitly scoped to declarations, not semantics.

---

## 4. Phasing

| Phase | Contents | Effort | Gate |
|---|---|---|---|
| **E1** | U2 — one shape-rule registry on `OpSpec`; fail closed; auto-flip coverage | 2w | no op reaches `operand_types[0]` |
| **E2** | U1 — differential harness (trace vs AST over the op catalog), then promote the tracer to default per target | 4w | byte-identical or explained IR for every catalog op |
| **E3** | **Completed 2026-08-10:** U7 — registered traced-IR effects, `_EffectVisitor` retired, indirect RNG proven | done | aliased/indirect RNG call is detected |
| **E4** | U5 + U6 — decompose `JitFn` and `__init__.py` | 3w | no target string in `jit.py` |
| **E5** | U4 — dedicated `GraphToSchedulePass` source/header; lit fixtures; retire the Python duplicate | 2w | one Graph→Schedule implementation |
| **E6** | U3 — cost-model-driven scheduling at Schedule IR | 5w | tile sizes chosen by measurement, not config |
| **E7** | Dynamic control flow **as one program** — frontend regions + symbolic dims + region adjoints | 10w | a data-dependent loop compiles, differentiates, and executes |

E1–E3 are ~8 weeks and remove three fail-open paths and one duplicated frontend.
E7 is the §2 convergence and subsumes [sweep F2](COMPILER_ARCHITECTURE_SWEEP.md)
item 10 and [AD](AUTODIFF_ARCHITECTURE_REVIEW.md) D4 — **budget it once, not
three times.**

---

## 5. How this changes the consolidated queue

Two revisions to
[`COMPILER_ARCHITECTURE_SWEEP.md §4`](COMPILER_ARCHITECTURE_SWEEP.md):

1. **E1 (one shape-rule registry) joins Tier 0.** It is two weeks, it closes the
   fourth instance of the audited-contract-with-no-consumer pattern, and it
   removes a fail-open path at the compiler's entry point.
2. **Sweep item 10 (symbolic shapes) and AD item 13 (control-flow adjoints)
   should merge into E7.** They were costed independently at 3w and 6w; as
   separate deliverables neither produces an observable capability. As one
   program with the end-to-end gate above, ~10w buys the capability all three
   were reaching for.

The rest of the queue stands. The new entries slot as:

- Tier 0 gains **E1** (2w) and the Decision #5 correction already listed.
- Tier 1 gains **E2** (4w) and **E3** (2w); E3 *replaces* sweep item 9, since
  deriving effects from traced IR is strictly better than re-homing an AST walker.
- Tier 2 gains **E4** (3w) and **E5** (2w) as cleanup that unblocks the arbiter.
- Tier 3 gains **E6** (5w), which is where sweep item 17 (generic fusion
  discovery) should actually live.
- **E7** (10w) becomes its own tier — the single largest capability gap in the
  compiler, and the only item where three previously-separate line items collapse
  into one.

---

## 6. What is working and should not be touched

- **`trace.py`'s design.** Rule registry, `register_shape_rule` extension point,
  explicit `TesseraTraceError`, structural control-flow handling. Promote it;
  don't rewrite it.
- **`control.py`'s transform hooks.** `scan`/`while_loop`/`cond`/`fori_loop`
  binding `_active_trace_builder()` is the correct answer to tracing's one real
  limitation, and it matches the design every mature tracing system converged on.
- **The Graph IR verifier and diagnostic plumbing.** `GraphIRVerifier`,
  `GraphIRDiagnostic`, `SourceSpan`, `_source_location`, and the
  `TesseraErrorCode` mapping give real source locations per Decision #13 — this is
  better than most compilers' frontends and it is what makes fail-closed
  affordable.
- **The `@jit` responsibility list itself** — constraints at decoration time,
  effects inferred, one compile request — is right. Only its implementation has
  accreted.
