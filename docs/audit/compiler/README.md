---
last_updated: 2026-09-11
audit_role: index
---

# Compiler Audit Map

Start here before using any document in this folder as a work queue. The files
serve different purposes; reading a review, historical design, or scoped plan
as current global priority is the main source of contradictory compiler
direction.

## Authority chain

1. **Generated status truth:**
   [`generated/compiler_progress.md`](../generated/compiler_progress.md),
   [`generated/autodiff_connection_ledger.md`](../generated/autodiff_connection_ledger.md),
   [`generated/dtype_flow.md`](../generated/dtype_flow.md),
   and the other generated dashboards report what is implemented and evidenced.
2. **Compiler narrative:** [`COMPILER_AUDIT.md`](COMPILER_AUDIT.md) explains the
   current architecture, important findings, and remaining gaps without owning
   cross-plan order.
3. **Global sequencing:**
   [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md) is the sole
   cross-domain compiler queue. Its live queue wins when a scoped
   plan proposes a different order.
4. **Scoped plans:** autodiff, evaluator, refactor, optimization, geometry, and
   sequence-mixer plans own their domain contracts and acceptance criteria.
   They do not independently reprioritize the compiler.
5. **Backend evidence:** the Apple, NVIDIA, ROCm, and x86
   [`todo.md`](../backend) queues own exact-device promotion and rejection.
   Evidence never transfers between architectures.

References and surveys explain *why*. Archived and historical-design documents
explain *how a decision was reached*. Neither is a live status or priority
surface.

## Re-indexed live tree

The live catalog below separates active plans, reference reviews and historical
routing files. Read current frontmatter rather than maintaining duplicate
file/plan counts here. The September 6 consolidation moves the three long AD
programs into the archive and carries their remaining tasks into one scoped plan.

The complete file-by-file catalog is below. Its practical summary is:

- [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md) now contains the
  one ordered **compiler + TSOL + tools** development queue.
- [`generated/compiler_progress.md`](../generated/compiler_progress.md) owns
  compiler-layer, proof, ABI, benchmark, and target-map status.
- [`generated/tsol_coverage.md`](../generated/tsol_coverage.md) shows that all
  canonical TSOL operations have semantic, shape, dtype/layout, lowering, and
  explicit AD dispositions. TSOL's live work is physical execution, sharding,
  policy breadth, performance, and architecture-owned proof—not another op
  registration sweep.
- [`generated/surface_status.md`](../generated/surface_status.md) owns the
  readiness state of compiler drivers, profiler tools, benchmarks, and test
  surfaces. `compile_only` and `scaffold` are not native-execution claims.
- The four backend `todo.md` files own promotion on Apple, NVIDIA, ROCm, and
  x86. A shared contract can land host-free; exact-device evidence cannot move
  between architectures.
- The [archived 2026-08-29 review](archive/CODE_REVIEW_2026-08-29.md)
  has reconciled source/device status and historical counts. Normal P3 device
  execution and dropout lit proof are recorded; allocation-failure injection
  and broader backend obligations remain in the live queues. See the
  [owning audit summary](COMPILER_AUDIT.md#archive-reconciliation--2026-09-04).
- [`MLIR_NATIVE_FOUNDATION_SURVEY.md`](MLIR_NATIVE_FOUNDATION_SURVEY.md)
  inventories every live compiler document and identifies historical Graph-owned
  packaging, source emitters and their canonical IR migration targets.

## Current plan and historical evidence

The [foundation map](INTEGRATED_COMPILER_PLAN.md#foundation-program) and
[grouped live queue](INTEGRATED_COMPILER_PLAN.md#live-queue) own sequencing.
[Start here](INTEGRATED_COMPILER_PLAN.md#start-here) selects the first host-free
entry point in each cut; prerequisites and device gates still apply.

`INTEGRATED_COMPILER_LOG.md` preserves dated implementation/validation records,
including their historical next-step lists. It is a reference, not a work queue.
`archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md` preserves older central/E2E
queues, wave tables and estimates. The plan's [routing index](INTEGRATED_COMPILER_PLAN.md#routing-index)
resolves old IDs to current owners, successors or archived dispositions.

## Route by question

| Question | Read first | Then use |
|---|---|---|
| What works now? | [`COMPILER_AUDIT.md`](COMPILER_AUDIT.md) | generated dashboards and the applicable backend plan |
| What should land next? | [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md) | the scoped plan named by the owning item |
| What core substrate do the capability papers share? | [`CORE_SUBSTRATE_VIEW.md`](CORE_SUBSTRATE_VIEW.md) | the seven source documents it maps, then the owning integrated-plan rows |
| How should Graph/Schedule/Tile/Target fit? | [`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md) | [`IR_STACK_INTEGRATION_REVIEW.md`](IR_STACK_INTEGRATION_REVIEW.md), [`TARGET_IR_REVIEW.md`](TARGET_IR_REVIEW.md) |
| How should the frontend and Graph IR change? | [`FRONTEND_GRAPH_SCHEDULE_REVIEW.md`](FRONTEND_GRAPH_SCHEDULE_REVIEW.md) | [active AD plan](AUTODIFF_EXECUTION_PLAN.md) |
| Should the IR be the single medium of record (vs the Python object graph)? | [`FRONT_END_LOWERING_ASSESSMENT.md`](FRONT_END_LOWERING_ASSESSMENT.md) | [`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md) for the arbiter, `FRONTEND-IR-MEDIUM-1` in [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md) |
| How should candidates be judged? | [`EVALUATOR_PLAN.md`](EVALUATOR_PLAN.md) | [`TILESIGHT_ASSESSMENT.md`](TILESIGHT_ASSESSMENT.md) |
| How should backend plugins and emitters converge? | [`COMPILER_REFACTOR_PLAN.md`](COMPILER_REFACTOR_PLAN.md) | [`OPTIMIZING_COMPILER_PLAN.md`](OPTIMIZING_COMPILER_PLAN.md) and the applicable backend plan |
| How should sequence/stateful programs lower? | [`SEQUENCE_MIXER_THEORY.md`](SEQUENCE_MIXER_THEORY.md) | [`SEQUENCE_MIXER_ENGINEERING_PLAN.md`](SEQUENCE_MIXER_ENGINEERING_PLAN.md) |
| How should solver/geometry differentiation land? | [`RIEMANNIAN_OT_PLAN.md`](RIEMANNIAN_OT_PLAN.md) | [active AD plan](AUTODIFF_EXECUTION_PLAN.md) |
| How should game-theoretic operators land? | [`GAME_THEORY_PLAN.md`](GAME_THEORY_PLAN.md) | [`EVALUATOR_PLAN.md`](EVALUATOR_PLAN.md) for the oracle rows |
| Where do higher-order derivatives, coordinate-aware calculus, and contraction algebra land? | [`MATH_SOURCE_WORKSTREAM.md`](MATH_SOURCE_WORKSTREAM.md) | [active AD plan](AUTODIFF_EXECUTION_PLAN.md) preserves the implemented algebra foundation and remaining native jets; [`PDE_STENCIL_CAPABILITY_PLAN.md`](PDE_STENCIL_CAPABILITY_PLAN.md) owns MSW-5 |
| What is the LSE identity contract? | [`LSE_CHECKPOINT_CONTRACT.md`](LSE_CHECKPOINT_CONTRACT.md) | architecture-owned attention plans |
| When may a consumer fuse into its producer's tiled epilogue? | [`FORGE_ASSESSMENT.md`](FORGE_ASSESSMENT.md) | [`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md) for the arbiter, [`TARGET_IR_REVIEW.md`](TARGET_IR_REVIEW.md) for the emitter seam |
| When may an effectful op enter a differentiated region? | [`W4_ADMISSIBLE_EFFECTS_PLAN.md`](W4_ADMISSIBLE_EFFECTS_PLAN.md) | [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md) W4-PRODUCT-1 for the ordering; `CONTROL_FLOW_CONTRACT.md` for the region carrier |
| How should layouts and index arithmetic be represented? | [`CUTE_IR_ASSESSMENT.md`](CUTE_IR_ASSESSMENT.md) | [`CORE_SUBSTRATE_VIEW.md`](CORE_SUBSTRATE_VIEW.md) S9 for the consumer, [`W1_1_TYPING_DESIGN.md`](W1_1_TYPING_DESIGN.md) for the typing precedent |

## Complete live-document catalog

### Status and sequencing

| Document | Role |
|---|---|
| [`COMPILER_AUDIT.md`](COMPILER_AUDIT.md) | Living compiler audit and narrative status. |
| [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md) | Sole cross-plan sequencing authority. |
| [`INTEGRATED_COMPILER_LOG.md`](INTEGRATED_COMPILER_LOG.md) | Revision-bound historical engineering reference; no current priorities or capability status. |
| [`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md) | Durable architecture and invariants; not a queue. |

### Scoped implementation plans

| Document | Owns |
|---|---|
| [`AUTODIFF_EXECUTION_PLAN.md`](AUTODIFF_EXECUTION_PLAN.md) | Remaining native AD, tapes, batching, higher-order, jets, solver/distributed integration and proof gates; original IDs preserved. |
| [`BLOCK_ATTNRES_ROCM_PLAN.md`](BLOCK_ATTNRES_ROCM_PLAN.md) | Block AttnRes mathematical contract, portable oracle, and ROCm-first physical acceptance criteria. |
| [`COMPILER_REFACTOR_PLAN.md`](COMPILER_REFACTOR_PLAN.md) | Shared spine, plugin, packaging, and backend reconciliation details. |
| [`CUTE_IR_ASSESSMENT.md`](CUTE_IR_ASSESSMENT.md) | CuTe IR (NVIDIA/cutlass#3426) review and verified layout algebra: the four-primitive scoping result, the mechanisms worth importing (partially-static value-in-type, fold-static, dynamic-leaf-only lowering, negative-scoped driver), and the `LAYOUT-ALG-1` sequence (integrated plan) serving the S9 `⊑` operator, SparDA's GQA-fold, TileSight's rasterization knob, and the G1b butterfly consolidation. Numeric contract in `tests/unit/test_layout_algebra_contracts.py`. Global order defers to `INTEGRATED_COMPILER_PLAN.md`. |
| [`EGGROLL_SUPPORT_PLAN.md`](EGGROLL_SUPPORT_PLAN.md) | Landing ES workload: native rank-1 fp32 foundation, with reverse/breadth, fusion and transport residuals mapped to integrated owners. |
| [`EVALUATOR_PLAN.md`](EVALUATOR_PLAN.md) | Correctness/evidence rung and promotion contract. |
| [`W4_ADMISSIBLE_EFFECTS_PLAN.md`](W4_ADMISSIBLE_EFFECTS_PLAN.md) | W4-EFFECTS-1: operation-owned recorded products that let keyed RNG, recorded-state mutation, and ordered collectives enter a differentiated region without weakening the fail-closed gate. States the admissibility criterion (reproducibility + confinement), the per-class verdicts including why I/O stays closed, and five delivery slices. Global order defers to `INTEGRATED_COMPILER_PLAN.md` queue order 2. |
| [`GAME_THEORY_PLAN.md`](GAME_THEORY_PLAN.md) | Coalition-lattice / equilibrium operator family: subset zeta/Möbius butterfly, semivalues, differentiable equilibria, regret/CFR dynamics, and the numerically verified oracle set (`research/game_theory/`). Global order defers to `INTEGRATED_COMPILER_PLAN.md`. |
| [`INTRA_KERNEL_FEEDBACK_PLAN.md`](INTRA_KERNEL_FEEDBACK_PLAN.md) | IKF-1: intra-kernel measurement as compiler training data (assessment of CUTLASS IKET, 2026-08-27). Per-instance indexed-slot records keyed by schedule coordinates, constant-clock contract with fail-closed validity rules, offline stall classification + realized critical path, cost-model coefficient fitting with roofline prior bands and paired-instance statistics, the explain-vs-decide (`instr_level`) arbiter guard, and delivery phases IKF-P0..P6 (ROCm gfx1151 first). Global order defers to `INTEGRATED_COMPILER_PLAN.md`; routed through TPROF-NATIVE-1 in the integrated plan. |
| [`OPTIMIZING_COMPILER_PLAN.md`](OPTIMIZING_COMPILER_PLAN.md) | Middle-end synthesis and backend-lift details. |
| [`PDE_STENCIL_CAPABILITY_PLAN.md`](PDE_STENCIL_CAPABILITY_PLAN.md) | PDE-operator semantics, symbol classification, discrete-stability certificates, and the stencil/halo contract queue. |
| [`RIEMANNIAN_OT_PLAN.md`](RIEMANNIAN_OT_PLAN.md) | Geometry/implicit-differentiation consumer and acceptance workload. |
| [`SCHEDULE_OBJECT_DESIGN.md`](SCHEDULE_OBJECT_DESIGN.md) | The one schedule representation (actions/edges/roles/residency + digest) unifying CAKE Phases 2–3, TileRT E5, the W5.2 action DAG, and FORGE W2 — contracts, IR carrier, and SO-1..SO-5 build order. |
| [`compiler_enhancement.md`](compiler_enhancement.md) | CAKE lesson review reconciled with current typed synchronization, tracer CFG, native status composition and measured admission; preserves the dated statistical assessment and routes remaining work to existing owners. Global order defers to `INTEGRATED_COMPILER_PLAN.md`. |
| [`SPARDA_REVIEW.md`](SPARDA_REVIEW.md) | SparDA source review, verified compressed-key/block-selection contracts, and the cross-layer prefetch + block-sparse iteration extraction queue. |
| [`SEQUENCE_MIXER_ENGINEERING_PLAN.md`](SEQUENCE_MIXER_ENGINEERING_PLAN.md) | Sequence-mixer family contracts and physical rollout. |

### Architecture reviews and focused references

| Document | Use |
|---|---|
| [`FORGE_ASSESSMENT.md`](FORGE_ASSESSMENT.md) | Historical proposal now archived; live ownership note. Original FORGE (arXiv:2606.22932) assessment and the residency-aware epilogue-fusion track it opens: locality lattice, static materialization proof, `matmul → optimizer` fusion, fail-closed clipping/routing keys, and the precision-realizability oracle. Numeric contract in `tests/unit/test_fused_wgrad_optimizer_contract.py`. Global order defers to `INTEGRATED_COMPILER_PLAN.md`. |
| [`MATH_SOURCE_WORKSTREAM.md`](MATH_SOURCE_WORKSTREAM.md) | Consolidated routing note (original proposal archived). MSW-1..MSW-9: a host-free reference-lane workstream derived from two tensor-calculus texts (Sochi) and a deep-learning theory book (Jentzen/Kuckuck/von Wurstemberger, arXiv:2310.20360v3). Higher-order autodiff fail-closed guard (landed) and exact jet path, optimizer breadth, a vector-identity law family, coordinate-aware field calculus, contraction normal form, two samples, and ANN-calculus fusion laws. Orthogonal to the integrated queue's physical-execution items rather than competing with them; global order still defers to `INTEGRATED_COMPILER_PLAN.md`. |
| Historical AD routing: [`AUTODIFF_NEXTGEN_PLAN.md`](AUTODIFF_NEXTGEN_PLAN.md), [`AUTODIFF_UNIFICATION_PLAN.md`](AUTODIFF_UNIFICATION_PLAN.md), [`AUTODIFF_ARCHITECTURE_REVIEW.md`](AUTODIFF_ARCHITECTURE_REVIEW.md) | Redirects to archived designs and the active AD plan; no independent queues. |
| [`DIFFERENTIABLE_PROGRAMMING_REVIEW.md`](DIFFERENTIABLE_PROGRAMMING_REVIEW.md) | Book-derived reference with one current status/owner table; remaining AD work is consolidated into AUTODIFF_EXECUTION_PLAN.md. |
| [`MATRIX_CALCULUS_REVIEW.md`](MATRIX_CALCULUS_REVIEW.md) | Book-derived delta against Edelman & Johnson's matrix-calculus notes (arXiv:2501.14787): the missing matrix-function/factorization derivative family, a verified `svd` VJP NaN at repeated singular values, the metric-parameterized gradient as the `manifold` key's consumer, and the Kronecker/`vec` cost identity. Companion runnable tutorial at `examples/matrix_calculus/`. |
| [`COMPILER_ARCHITECTURE_SWEEP.md`](COMPILER_ARCHITECTURE_SWEEP.md) | Cross-layer findings feeding the integrated plan. |
| [`CORE_SUBSTRATE_VIEW.md`](CORE_SUBSTRATE_VIEW.md) | Current nine-substrate ownership map across seven sources; historical status and build sequence archived. |
| [`FRONTEND_GRAPH_SCHEDULE_REVIEW.md`](FRONTEND_GRAPH_SCHEDULE_REVIEW.md) | Frontend, Graph, and Schedule ownership findings. |
| [`FRONT_END_LOWERING_ASSESSMENT.md`](FRONT_END_LOWERING_ASSESSMENT.md) | KGEN-as-yardstick assessment of the Python→MLIR front half: the two-memories finding (IR as lossy projection of the Python object graph), the pre-elaboration parametric-optimization gap on the existing Presburger substrate, raising as the arbiter on-ramp for user math, and the fail-closed-dtype / `loc` / elaboration-boundary down-payment. Global order defers to `INTEGRATED_COMPILER_PLAN.md` (`FRONTEND-IR-MEDIUM-1`). |
| [`IR_STACK_INTEGRATION_REVIEW.md`](IR_STACK_INTEGRATION_REVIEW.md) | IR adjacency and lowering-boundary findings. |
| [`TARGET_IR_REVIEW.md`](TARGET_IR_REVIEW.md) | Target-dialect typing and target-lowering review. |
| [`W1_1_TYPING_DESIGN.md`](W1_1_TYPING_DESIGN.md) | Current typed Tile design. |
| [`LSE_CHECKPOINT_CONTRACT.md`](LSE_CHECKPOINT_CONTRACT.md) | Shared saved/recomputed-LSE identity contract. |
| [`SEQUENCE_MIXER_THEORY.md`](SEQUENCE_MIXER_THEORY.md) | Sequence-mixer semantic model. |
| [`TILESIGHT_ASSESSMENT.md`](TILESIGHT_ASSESSMENT.md) | Analytical-model research and candidate-pruning guidance. |
| [`TILERT_ASSESSMENT.md`](TILERT_ASSESSMENT.md) | TileRT assessment; overlap-scheduling models and W6/T3/T4 composition-layer direction. |
| [`AMD_KERNEL_COMPILER_SURVEY.md`](AMD_KERNEL_COMPILER_SURVEY.md) | AMD compiler research survey; input to ROCm design, not ROCm evidence. |

The [old typing inventory](archive/W1_1_TYPING_INVENTORY.md) is archived;
the [current typing census](MLIR_NATIVE_FOUNDATION_SURVEY.md#typing-inventory-replacement--2026-09-04)
and `W1_1_TYPING_DESIGN.md` retain the NVIDIA producer obligation.

Documents under [`archive/`](archive) are point-in-time evidence only. This
includes the superseded
[`STAGE_A_EMIT_PLAN.md`](archive/STAGE_A_EMIT_PLAN.md) and the completed
[`WORKSTREAM_C_HANDOFF.md`](archive/WORKSTREAM_C_HANDOFF.md), plus the completed
[`STRIX_HALO_WORKLIST_2026-08-10.md`](archive/STRIX_HALO_WORKLIST_2026-08-10.md);
none is an active setup or execution guide.

## Maintenance rule

- Do not copy generated counts into plans when a dashboard link is sufficient.
- Add a new finding to a review, but bind executable work to an ID in the
  integrated plan before treating it as priority.
- Every live scoped compiler plan links back to the integrated plan and this
  index. A plan may own acceptance criteria, never a competing global queue.
- Regenerate derived dashboards with their owning generator; never hand-edit
  generated status.


MSW-9's bounded ANN-calculus prototype and integration decisions are recorded in
[`ANN_CALCULUS_DESIGN_SPIKE.md`](ANN_CALCULUS_DESIGN_SPIKE.md). The spike separates
reference laws from native proof. Bounded native evaluation and scoped admission now have device evidence;
broader composition, tuned schedules and promotion remain under MSW-9 in the live queue.


Functional-analysis FA-1–FA-7 tasks are consolidated in the
[integrated routing index](INTEGRATED_COMPILER_PLAN.md#routing-index),
with AD and recurrence follow-ups in their scoped plans. The former
[`FUNCTIONAL_ANALYSIS_TSOL_PLAN.md`](FUNCTIONAL_ANALYSIS_TSOL_PLAN.md) is a reference redirect to the
preserved mathematical design, not an independent execution queue.


## Math audit consolidation — 2026-09-07

The [foundation reconciliation](INTEGRATED_COMPILER_LOG.md#2026-09-07--math-audit--foundation-reconciliation)
assigns the math plans to native ownership, AD, numerical legality and measured
admission. FORGE and the original math-source proposal are archived references
with live routing notes; their remaining tasks are not closed. Matrix calculus
and sequence-mixer theory remain mathematical references. Geometry, PDE and
sequence-mixer implementation plans remain live scoped consumer plans.


### September capability-plan reconciliation

[The integrated mapping](INTEGRATED_COMPILER_LOG.md#2026-09-07--capability-plan-reconciliation)
consolidates the five reviewed documents under existing F0–F4, AD, NUMPOL,
layout and transport owners. All five live paths remain useful: three scoped
landing plans and two references. The August substrate snapshot and superseded
AD/workload status excerpts are archived; they are not additional queues.

- `HEAP_BARRIER_ARCHITECTURE_REVIEW.md` — [publication, reader epochs and reclamation design](HEAP_BARRIER_ARCHITECTURE_REVIEW.md); scoped exploration under W4-PRODUCT-1 / W2.4a / DISPATCH-BREAKER, not a status queue.
