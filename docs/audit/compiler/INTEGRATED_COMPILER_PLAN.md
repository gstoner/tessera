---
last_updated: 2026-08-02
audit_role: plan
plan_state: open
supersedes_queues_in:
  - COMPILER_ARCHITECTURE_SWEEP.md §4
  - FRONTEND_GRAPH_SCHEDULE_REVIEW.md §5
  - IR_STACK_INTEGRATION_REVIEW.md §5
  - AUTODIFF_ARCHITECTURE_REVIEW.md §5
  - ../domain/GA_EBM_ARCHITECTURE_REVIEW.md §4
  - RIEMANNIAN_OT_PLAN.md §4
---

# Integrated Compiler Plan

One plan across six reviews. Each review's own ranked queue stays as *evidence
and rationale*; **this document owns sequencing and de-duplication.** Where a
review's queue and this document disagree on ordering or cost, this document
wins — the reviews were written independently and double-counted overlapping
work.

Status truth remains `MASTER_AUDIT.md` and `docs/audit/generated/`
(Decision #26). Nothing here reclassifies a row. Effort figures are engineering
estimates for a single track with no hardware gates, not commitments.

**Source reviews:**
[GA/EBM](../domain/GA_EBM_ARCHITECTURE_REVIEW.md) ·
[Autodiff](AUTODIFF_ARCHITECTURE_REVIEW.md) ·
[Sweep](COMPILER_ARCHITECTURE_SWEEP.md) ·
[Frontend→Graph→Schedule](FRONTEND_GRAPH_SCHEDULE_REVIEW.md) ·
[IR Stack Integration](IR_STACK_INTEGRATION_REVIEW.md) ·
[Target IR](TARGET_IR_REVIEW.md) ·
[Riemannian OT](RIEMANNIAN_OT_PLAN.md)

---

## 1. The thesis

Across six independent reviews, roughly forty findings reduce to **two root
causes and one consequence.**

### Root cause A — Declared but not consumed

The compiler computes, validates, and attaches the information a pass needs, and
then no pass reads it.

| Declaration | Consumer that ignores it | Review |
|---|---|---|
| `manifold` attribute on `ebm.langevin_step` | every backend codegen (6 grep hits, all comments) | GA/EBM §1.1 |
| `MultivectorSpec.grades`, `IsRotor`, `Even`/`Odd` | `geometric_product` — iterates all `dim²` pairs | GA/EBM §2.1 |
| `batching_rule` axis, closed across 480 primitives | `vmap` — a Python `for` loop | Autodiff §B3 |
| `shape_rule` axis, reported closed | `_infer_result_type` — a five-case if-chain | Frontend §G2 |
| `!tile.fragment`, `!tile.buffer`, `!tile.tmem`, … (9 types) | every Tile op — `Variadic<AnyType>`, 70× | IR Stack §T1 |
| `numeric_policy` (Decision #15a) | no carrier below Graph IR at all | IR Stack §T6 |
| `TilingInterface` on Matmul/Conv/FlashAttn | `fusion_core.py` — 7 hand-enumerated regions | Sweep §F3 |

### Root cause B — Told, not derived

Passes are *given* facts syntactically instead of *computing* them, so they are
wrong at the edges and fail open.

| Missing analysis | What happens instead | Review |
|---|---|---|
| Effect/purity on the IR | `ast.NodeVisitor` name-matching; aliased RNG ⇒ inferred pure ⇒ `deterministic=True` passes | Sweep §F1 |
| Differentiation activity | `AutodiffPass` builds adjoints for everything | Autodiff §A5 |
| Gradient demand / trajectory liveness | `CheckpointInnerLoop` marks every step in every loop | GA/EBM §1.5 |
| Symbolic shape constraints | `dims_compatible` is `str(lhs) == str(rhs)` | Sweep §F2 |
| Fusion legality | region shapes enumerated by hand | Sweep §F3 |
| Sharding propagation | every layer annotated by hand; `validate()` contains `pass` | Sweep §F4 |
| Tile-level invariants | six `*LegalityPass` re-deriving what types would give free | IR Stack §T2 |

### The consequence — C: duplication

**A + B ⇒ C.** When the declared contract doesn't carry the information, and no
analysis derives it, the only way to ship is to write a second implementation
that does. Every duplication in the tree traces to this:

| Duplication | Why it exists |
|---|---|
| Two frontends (AST `_OpExtractor` / tracer), opposite failure policies | the AST path can't produce SSA through control flow |
| Two Graph→Schedule, two Schedule→Tile lowerings; Python canonical | the MLIR types don't carry what codegen needs |
| Python AD tape + `AutodiffPass` | the tape can't be a transform (global monkey-patching) |
| GA Python fast paths + `RotorSandwichFold` marker | `ExpandProductTable` rejects batched operands |
| Two `Queue.td`, two `Attn.td` defining the same dialect | accretion, uncaught |
| Two remat passes with opposite rigor | no shared liveness analysis |

**This is why the plan is ordered A → B → C.** Collapsing a duplication before
the surviving path can carry the information just deletes a working system. Every
attempt to start at C fails.

---

## 2. Governance — the rules that stop regrowth

A cleanup without rules regrows. Six standing decisions, each derived from a
specific finding, each drift-gateable:

| # | Rule | Prevents | From |
|---|---|---|---|
| **#21a** | **Semantic keys never default.** An attribute selecting *semantics* fails closed on absence; one selecting *performance* may fall back with a diagnostic. Semantic: `manifold`, `algebra`, `math_mode`, `rounding_mode`, `distribution`, `dtype`. Performance: tile sizes, stage depth, `auto_batch`, checkpoint budget. | silent Euclidean fallback; `operand_types[0]`; unvalidated `StrAttr` | OT §H1 |
| **#10a** | **An eligibility-marking pass ships a negative fixture.** Any pass annotating work as rematerializable / fusable / pipelineable gates on demand analysis and ships ≥1 fixture where the correct output is *no annotation*. | 2500 dead steps marked rematerializable | OT §H2 |
| **#29** | **A declaration must have a consumer.** If the compiler declares metadata — an ODS type, a coverage axis, an attribute — a named pass must consume it, or the declaration is deleted. Drift-gated: a test asserting every `primitive_coverage` axis names its consumer. | root cause A, all seven instances | this doc |
| **#30** | **Derive, don't ask.** A pass needing a program fact queries the analysis layer. New bespoke walkers are rejected in review. | root cause B; the eighth hand-rolled analysis | Sweep §5 |
| **#31** | **One implementation per boundary.** Each level boundary has exactly one production lowering. A second implementation is either a declared oracle with a differential test, or deleted. | root cause C | IR Stack §T3 |
| **#32** | **Information loss across a level boundary must be declared.** A lowering carries each Decision #15a attribute forward or records a named reason it dropped it; a boundary verifier fails on silent loss. | `numeric_policy` vanishing above the MMA | IR Stack §U5 |

Adopt these **before** Wave 1, not after. #29 and #31 in particular change what
gets accepted in review, and they are what make Waves 1–3 stick.

---

## 3. De-duplication ledger

The six reviews costed overlapping work independently. Corrections applied here:

| Double-counted work | Costed as | Merged into | Saved |
|---|---|---|---|
| Symbolic shapes (Sweep #10, 3w) + control-flow adjoints (AD D4, 6w) + frontend regions (E7) | 3 separate items | **W4** — one program, one gate | ~4w and 2 items that would each have "landed" with zero capability |
| Differential harness: trace-vs-AST (E2) + Python-spine-vs-MLIR (I5) | 2 harnesses | **W3.1** — one harness, two uses | ~2w |
| Schedule-decision work (Frontend U3/E6) + (IR Stack U6/I6) | 2 items | **W5.2** | ~5w |
| Effects re-homing (Sweep #9) vs derive-from-traced-IR (Frontend E3) | 2 approaches | **W2.2** — E3 strictly supersedes | ~2w |
| Implicit diff: OT R2 `custom_root` + AD "finish NewtonAutodiff" | 2 items | **W3.5** — same pass | ~2w |
| Legality collapse (IR Stack I2) as independent work | standalone | **W2.4** — client of the dataflow layer | ~1w |
| Remat unification: delete `EBMCheckpointInnerLoop` (GA/EBM) + AD D5 | 2 items | **W5.1** | ~1w |

**~17 weeks of double-counting removed.** The naive sum of the six queues is
~80 weeks; the integrated plan is ~63.

---

## 4. The waves

Each wave has **one observable exit criterion**. A wave is not done when its
items are merged; it is done when the criterion holds.

### W0 — Stop the bleeding *(4 weeks · no dependencies · start immediately)*

Live defects, fail-open paths, and false documentation. Every item is
independent; run them in parallel.

| # | Item | Source | Effort |
|---|---|---|---|
| W0.1 | `manifold` → required verified enum; delete the Euclidean default (copy `AnnotateAlgebra`'s `emitError`+interrupt) | GA/EBM §1.1 | 3d |
| W0.2 | Demand-gate `CheckpointInnerLoop`; `CHECK-NOT` fixtures; `steps_annotated` counter | GA/EBM §1.5 | 4d |
| W0.3 | Define traceable EBM energies; use `autodiff.tape` only when a supported cotangent path is recorded, with numerical differentiation retained for untraceable NumPy callbacks and regression coverage for both paths | GA/EBM §2.6 | 1w |
| W0.4 | Fix `jacrev`/`jacfwd` forward-pass-per-element; correct their docstrings | Autodiff §B1–B2 | 3d |
| W0.5 | **Correct Decision #5 in `CLAUDE.md`** — the effect lattice walks the AST, not the IR | Sweep §F1 | 1h |
| W0.6 | Delete duplicate `dialects/tessera_{queue,attn}/*.td`; split the already-linkable `GraphToSchedulePass` into a dedicated library-owned source/header with focused lit fixtures | IR Stack §T5, §T3 | 3d |
| W0.7 | `.td` summary drift: distinguish "stub" from "annotation-only"; remove `AnnotateAlgebra`'s false "GA8 lowering will refuse" | GA/EBM §1.4 | 1d |
| W0.8 | Adopt Decisions #21a, #10a, #29, #30, #31, #32 | §2 | 1d |
| W0.9 | Replace `test_target_ir_contract.py`'s substring assertions (`assert "tessera_rocm.mfma" in mm.target_ir`) with a real MLIR parse + dialect load + verifier run. Decision #19's named validation is currently `str.__contains__` | Target §X4 | 1w |
| W0.10 | **Decide x86's Decision #19 status** — build `tessera_x86` (AMX tile / AVX-512 vector / pack ops) or add an explicit carve-out. x86 has no `.td` anywhere; the decision reads as universal and the oldest, most-executable backend silently doesn't follow it | Target §X1 | 1h to decide |

**Exit:** no known silent-wrong-answer path remains open; `CLAUDE.md` Decision #5
is accurate; every dialect has exactly one ODS.

### W1 — Make declarations binding *(8 weeks · depends on W0.8)*

Root cause A. Nothing here designs a new concept — every item enforces a contract
that already exists.

| # | Item | Source | Effort |
|---|---|---|---|
| W1.1 | **Type the Tile dialect and the three Target IR dialects.** `Variadic<AnyType>` → the nine declared Tile types; parameterize `!tile.fragment`/`!tile.buffer` on element type, tile shape, layout, memory space, accumulator `numeric_policy`. Then `ROCM_{MFMA,WMMA}` and `NVIDIA_{MmaSync,Wgmma}` get the `vector<16xf16>`/`vector<8xf32>` types **their own ODS descriptions already specify in prose** | IR Stack §U1 + Target §X2 | 5w |
| W1.1b | `EnumAttr` for every semantic `StrAttr` in the target dialects (62 × `$name`, 4 × `$kind`, 1 × `$mode`; zero enums today) — Decision #21a enforcement | Target §X3 | 1w |
| W1.2 | **One shape-rule registry**, owned by `op_catalog.OpSpec`; `primitive_coverage.shape_rule` auto-flips from it (same mechanism as `_VJPS`/`_JVPS`); unknown op ⇒ diagnostic, never `operand_types[0]` | Frontend §U2 | 2w |
| W1.3 | Metadata lowering obligation (#32) + boundary verifier | IR Stack §U5 | 2w |
| W1.4 | Thread `MultivectorSpec.grades` into `geometric_product`; add `input_grades` to `GradeFusion` | GA/EBM §2.1 | 1w |

**Exit:** `AnyType` count in `TileOps.td` is 0; a mismatched `tile.mma` fails to
parse; no op reaches the `operand_types[0]` fallback; no Decision #15a attribute
drops across a boundary without a recorded reason.

> **W1.1 is the single highest-leverage item in the plan.** Three weeks, no new
> concepts, and it is the precondition for W2.4, W3.2, W3.3, and every backend
> having a real contract to lower against.

### W2 — Build the analysis layer *(8 weeks · depends on W0)*

Root cause B. One framework, then each analysis is a transfer function rather
than a subsystem.

| # | Item | Source | Effort |
|---|---|---|---|
| W2.1 | **Graph IR dataflow framework** on MLIR `DataFlowSolver`. Three required properties: **fail closed** (unprovable ⇒ ⊤, consumers treat ⊤ as unsafe), **recomputable/invalidated** (queries against current IR, not decoration-time facts), **queryable from C++ and Python** (or duplication regrows) | Sweep §3 | 3w |
| W2.2 | Effects derived from **traced IR**; retire `_EffectVisitor`; reconcile with `EffectAnnotationPass` | Frontend §U7 | 2w |
| W2.3 | Differentiation activity analysis (`ActivityInterface`) as a client | Autodiff D3 | 2w |
| W2.4 | Collapse six `*LegalityPass` → ODS constraints (post-W1.1) + one `TileDataflowLegalityPass` | IR Stack §U2 | 1w |

**Exit:** an aliased/indirect RNG call is detected by effect inference; an
inactive branch's adjoint is provably **not emitted** (`CHECK-NOT` fixture per
#10a); six legality passes are one.

> W2.2 depends on W3.1's tracer promotion for full value, but can land against
> the existing Apple-GPU tracer first and widen with W3.1.

### W3 — Collapse the duplications *(10 weeks · depends on W1, W2)*

Root cause C. Now possible, because the surviving path can carry what the deleted
path was carrying.

| # | Item | Source | Effort |
|---|---|---|---|
| W3.1 | **One differential harness**, then promote the tracer to the only frontend; delete `_OpExtractor` | Frontend §U1 + IR Stack §U3 | 4w |
| W3.2 | One lowering per boundary: converge Graph→Schedule→Tile on MLIR; Python spine demoted to oracle | IR Stack §U3 | 3w |
| W3.3 | Split the Tile dialect by level: primitives stay `tile.*`; whole-kernel ops → Graph IR / `tessera.kernel.*`; domain ops → `tessera_ebm`; `svd`/`qr`/`cholesky`/`lu` → linalg solver | IR Stack §U4 | 2w |
| W3.4 | Decompose `JitFn` (11 `_native_*_backward` → `emit/candidate.py` candidates behind `@f__bwd`); split `__init__.py`'s 315 nested defs into `tessera/ops/` | Frontend §U5–U6 | 3w |
| W3.5 | Finish `NewtonAutodiff`'s IFT body (`dF/dx = -(dR/dx)⁻¹dR/du`) — emits real `residual` + `linear_solve` ops | Autodiff §B8 + OT R2 | 2w |
| W3.6 | Batched operands in `ExpandProductTable`; connect `RotorSandwichFold`'s marker to a consumer | GA/EBM §1.3 | 2w |

**Exit:** one frontend, one lowering per boundary, no target string in `jit.py`,
every `tile.*` op is a tile primitive.

### W4 — The control-flow program *(10 weeks · depends on W1.2, W2.1, W3.1)*

**This is the plan's most important structural correction.** Dynamic control flow
is blocked at three independent layers — frontend (no CFG ⇒ no φ), shape system
(no symbolic dims), autodiff (`AutodiffPass` hard-errors on nested regions).
Fixing any one alone produces **zero observable capability**. They must ship as
one program with one gate.

| # | Item | Effort |
|---|---|---|
| W4.1 | Frontend emits structured regions (`scf.*`, `tessera.control_*`) via the tracer + `control.py` hooks | 2w |
| W4.2 | Symbolic dims: `Dim`/`DimProduct` → `AffineExpr` + `presburger`; converge the Python shape system onto the `tessera.dim_bindings` carrier `SymbolicDimEqualityPass` already checks | 3w |
| W4.3 | `RegionAdjointInterface` + reverse mode over `scf.for`/`if`/`while` and `tessera.control_*` | 5w |

**Exit (single gate):** *a `@jit` function containing a data-dependent loop
compiles, differentiates, and executes, with numerical agreement against the
Python oracle.*

### W5 — Decisions become measured *(9 weeks · depends on W1, W3)*

Root cause: L5, "a constant where a measured decision belongs." Every item routes
an existing hardcoded choice through the arbiter Decision #28 already built.

| # | Item | Source | Effort |
|---|---|---|---|
| W5.1 | Residual policy as an arbiter axis (SAVE/RECOMPUTE/HYBRID per `(op, bucket, dtype, target)`); Revolve/treeverse for counted loops; delete `EBMCheckpointInnerLoop` | Autodiff D5 + GA/EBM §1.5 | 4w |
| W5.2 | Scheduling decisions at Schedule IR — tile sizes, stage counts, raster order, warp roles chosen from `fusion_core` cost models via the measured arbiter, not from `--tile-q=64` | Frontend §U3 + IR Stack §U6 | 5w |
| W5.3 | Generic fusion region discovery over a legality oracle (a W2.1 client); keep the measured cost models | Sweep §F3 | *(folded into W5.2)* |
| W5.4 | Sharding **propagation** (GSPMD/Shardy-style) — annotate a few tensors, infer the rest | Sweep §F4 | 4w |
| W5.5 | Rule-table-driven canonicalization (PDL/PDLL). **Defer equality saturation** until the rule table is large enough that ordering demonstrably costs something | Sweep §F5 | 3w |
| W5.6 | **Consolidate the three ROCm codegen paths** — 67 `GenerateROCM*Kernel.cpp` passes, `emit/rocm_hip.py`, and `target_ir.py::_lower_rocm_op` — onto the `emit/` spine. **Must run on the Strix Halo gfx1151 box** (§6a) | Target §X6 | 6w |

**Exit:** no tile size, residual policy, or fusion boundary is chosen by a
constant; sharding a model requires O(few) annotations, not O(layers).

### W6 — Exceed the state of the art *(14 weeks · depends on W2, W4)*

| # | Item | Source | Effort |
|---|---|---|---|
| W6.1 | Forward mode in the compiler (`TangentInterface`) — cheapest large capability; unlocks exact HVP, `jacfwd`, and W6.3 | Autodiff D2 | 3w |
| W6.2 | Sparse AD — sparsity detection + coloring (client of W2.1/W4.2). PyTorch, TF, and JAX all lack this | Autodiff D7 | 5w |
| W6.3 | Taylor/jet mode over Weil algebras, **hosted on the GA multivector engine** — a Weil algebra is a graded algebra with a compile-time product table, which is exactly `ga/signature.py` | Autodiff D6 | 4w |
| W6.4 | Table-driven GA kernel synthesis via `emit/`; then PGA `Cl(3,0,1)` | GA/EBM §2.3–2.4 | 5w |

**Exit:** a defensible "exceeds SOTA" claim with a benchmark behind it — sparse
Jacobian scaling `O(colors)` not `O(rows)`; order-`k` derivatives sharing the
tuned GA kernels.

> W6.3 is much cheaper if W6.4 lands first — the graded-algebra kernel generator
> is the shared substrate. Order them 6.4 → 6.3 unless GA work is deprioritized.

### Riemannian OT — re-scoped as validation, not a track

The [OT plan](RIEMANNIAN_OT_PLAN.md) proposed R0–R5 at ~14 weeks. Integrated, most
of it is **already funded by W1–W5**:

| OT need | Provided by |
|---|---|
| manifold as a hard dispatch key (H1) | W0.1 |
| remat no-op proof (H2) | W0.2 |
| geometric primitive layer (R1) | new — 1.5w, first consumers are `ebm/geo_sampling.py` and `hyperbolic.py` |
| `stop_gradient` + implicit diff (R2) | W3.5 + W2.3 |
| `c_transform` fused loop (R3) | W4 (control flow) + W5.1 (residual policy) |
| backend lanes (R4) | W5.2 (schedule decisions) + existing EBM Langevin kernel precedent |
| oracles (R5) | W2.1 (fail-closed analyses) + the existing Evaluator |

**Net new OT scope: ~4 weeks** (the geometric primitive layer plus the RNOT
composite ops), down from 14. And RNOT becomes the plan's **acceptance test**: a
2500-step manifold inner loop with a stop-gradient boundary exercises W1 (typed
tiles), W2 (activity + liveness), W4 (control flow), and W5 (residual policy) in
one workload. If RNOT runs at wall-clock parity with RCPM, the plan worked.

---

## 5. Dependency graph

```
W0 ─────────────────────────────────────────────────────────────►  (start now)
 │
 ├─► W1 (declarations binding) ─────────┬──────────────────────────┐
 │      W1.1 type Tile IR ──────────────┤                          │
 │      W1.2 shape-rule registry ───────┼──► W4 (control flow) ────┤
 │                                      │      W4.2 needs W1.2     │
 └─► W2 (analysis layer) ───────────────┤                          │
        W2.1 dataflow framework ────────┼──► W4.3 needs W2.1       │
        W2.2 effects ───────────────────┤                          ├─► W6
        W2.3 activity ──────────────────┘                          │   (SOTA)
                │                                                  │
                └─► W3 (collapse duplication) ──► W5 (measured) ───┘
                       W3.1 one frontend ──► W4.1
```

Critical path: **W0 → W1.1 → W2.1 → W3.1 → W4 → W5** ≈ 40 weeks.
Full plan ≈ 63 weeks single-track; substantially parallelizable across W1/W2 and
within W0.

---

## 6. Three budget levels

**Minimum (5 weeks) — W0 only.** Closes three live silent-wrong-answer paths, one
false architecture decision, a duplicate-dialect trap, an `O(2^n)` default path,
and a substring-based "contract" test. Adopts the six governance rules so nothing
regrows. **Do this regardless of what else is decided.**

**Recommended (23 weeks) — W0 + W1 + W2 + W3.1.** Root causes A and B are fixed,
the frontend duplication is gone, and every subsequent piece of work becomes
cheaper rather than adding to the pile. This is the point at which the compiler
stops accumulating parallel systems. If one number is chosen, choose this.

**Full (72 weeks) — W0…W6**, of which ~6 weeks is hardware-routed to the ROCm box
(§6a). Adds the control-flow capability, measured decisions, and two defensible
exceeds-SOTA claims.

## 6a. Fleet routing — what must run on which box

Most of this plan is compile-time contract work and runs anywhere. Two items are
hardware-bound, and one of them is the highest-risk item in the plan.

| Work | Box | Why |
|---|---|---|
| W0, W1 (typing, enums, shape rules), W2 (analyses), W3.1–W3.4 | **Mac M1 Max** | ODS, `tessera-opt`, lit, unit tests. No device needed; tightening a type is a compile-time change. |
| W0.10 build branch — `tessera_x86` dialect | **Mac** for ODS/lit; **Zen5 box** for AMX/AVX-512 execution proof | The NR2 Pro's Core Ultra 7 has no AVX-512/AMX. |
| **W5.6 — ROCm codegen consolidation** | **Strix Halo gfx1151 — required** | It changes *generated kernels*. Each of the 67 passes moving onto the `emit/` spine needs an execute-and-compare against its current output on real silicon. Refactoring 67 code generators on a machine that cannot run them is refactoring without an oracle. |
| W4 (control flow) end-to-end gate | **any executing lane**; gfx1151 preferred | It has the broadest compiler-generated + hardware-verified op coverage. |
| ROCm arch breadth (gfx950 / gfx1201 / gfx1250) | deferred — no silicon | MASTER_AUDIT P2, unchanged. |

**Two sequencing consequences.** First, W5.6 must be incremental with a
per-pass differential harness — the *same harness design* W3.1 needs for
trace-vs-AST and W3.2 needs for Python-spine-vs-MLIR. One design, three uses;
build it in W3.1 and reuse it. Second, **W1.1's ROCm typing will produce compile
failures in exactly those 67 passes** — that is what typing is for — so W1.1's
ROCm half and W5.6 want the same person, on the ROCm box, in the same window.
Schedule them adjacent rather than at opposite ends of the plan.

**What to cut first if squeezed:** W6.4 (GA synthesis / PGA) and W5.5
(canonicalization rule tables) are the most deferrable — genuine value, no
downstream dependents. **What never to cut:** W0.5 and W0.8 (an hour and a day)
— a wrong architecture decision and absent governance are what let all of this
accumulate.

---

## 7. Risks

| Risk | Wave | Mitigation |
|---|---|---|
| **W3.1 is a broad behavior change** — the AST frontend is the default on every non-Apple target | W3 | The differential harness ships *before* the switch, not after; promote per-target with the harness green |
| **W4 will exceed its estimate.** Region adjoints are the hardest item here and structured reverse mode is genuinely difficult | W4 | Land W4.1+W4.2 with a forward-only gate first, so partial progress is observable before W4.3 |
| **W1.1 touches every Tile-consuming backend** | W1 | It is additive at the ODS level — types tighten, existing valid IR stays valid; the failures it produces are the point |
| **Waves 1–3 produce no user-visible feature.** Fifteen weeks of "the compiler now enforces what it already said" is hard to fund | all | RNOT (§4) is the visible acceptance workload; state the intermediate gates as capability claims, not cleanup |
| **The governance rules are ignored under delivery pressure** | all | #29 and #31 are drift-gateable; make them tests, not conventions |
| **Someone starts at W3** (deleting duplications first, because they are the most visible waste) | — | It fails: the surviving path cannot yet carry what the deleted one carried. Ordering A→B→C is the plan's core claim |

---

## 8. What this plan does not cover

Not examined across the seven reviews, and therefore not planned:

- ~~Target IR dialects~~ — **reviewed 2026-08-02**
  ([TARGET_IR_REVIEW.md](TARGET_IR_REVIEW.md)). The `AnyType` finding **does**
  repeat, in `ROCM_{MFMA,WMMA}` and the NVIDIA mma ops; W1.1 grew 3w → 5w
  accordingly, and three new items landed (W0.9, W0.10, W1.1b, W5.6).
- The bodies of the 67 `GenerateROCM*Kernel` passes — these need review on the
  ROCm box **before** W5.6 is scheduled, not after.
- `emit/nvidia_cuda.py` (4722 lines) and `emit/rocm_hip.py` internals.
- Spectral and TPP solver families; the collectives and neighbors dialects;
  the RubinCPX backend.
- Quantization numerics; the KV-cache and memory model.
- The Evaluator program (`EVALUATOR_PLAN.md` §9.5) — it is a consumer of this
  work, not a subject of it.
- `WarpSpecializationPass` and `AsyncCopyLoweringPass` bodies.

Absence from this plan is not a clean bill of health.
