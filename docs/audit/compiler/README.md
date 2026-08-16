---
last_updated: 2026-08-14
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
   cross-domain compiler queue. Its current-route section wins when a scoped
   plan proposes a different order.
4. **Scoped plans:** autodiff, evaluator, refactor, optimization, geometry, and
   sequence-mixer plans own their domain contracts and acceptance criteria.
   They do not independently reprioritize the compiler.
5. **Backend evidence:** the Apple, NVIDIA, ROCm, and x86
   [`todo.md`](../backend/) queues own exact-device promotion and rejection.
   Evidence never transfers between architectures.

References and surveys explain *why*. Archived and historical-design documents
explain *how a decision was reached*. Neither is a live status or priority
surface.

## Current execution path

The next compiler work should move down one semantic spine and leave executable
proof at every boundary:

| Order | Owning item | Outcome |
|---|---|---|
| 1 | **AD-CORE-LINEAR-1 — complete** | `LinearTransposeInterface` owns transpose/reshape, broadcast/expand, structural views, and operand-wise matmul; both compiler autodiff passes and paired CPU numerical proofs consume it. |
| 2 | **AD-TSOL-SPECTRAL-1 — bounded native slice landed** | Explicit Graph spectral identity, FFT/IFFT/RFFT/IRFFT/DCT transposes, and compound VJPs are implemented. Complex-f32 spectral-filter and unbroadcast full-f32 spectral-convolution adjoints now have content-addressed native AVX-512 and gfx1151 consumers with exact-host/device numerical proof. Native STFT/ISTFT backward, broader axes/dtypes/broadcasting, and performance packets remain architecture-owned and fail closed. |
| 3 | **GRAPH-VERIFY-SIGNED-1 — complete** | Graph and canonical-attention integer verifiers consume signed `IntegerAttr` values, with negative IR tests proving that MLIR 23 unsigned accessors cannot bypass legality. |
| 4 | **AD-CORE-EFFECT-CONTROL-1 — complete** | Canonical `stop_gradient`, SSA activity, Graph effect propagation, active-stochastic rejection, and fail-closed active-region/residual behavior are compiler-owned and directly tested. |
| 5 | **W4 / AD-SOLVER-IFT-1 — typed residual-program expansion landing** | The canonical tracer now recovers explicit multi-block CFG identity from nested `if`/counted-loop/bounded-while/scan regions without AST re-entry. Typed Presburger identity propagates to every recovered block. `RegionAdjointInterface` still generates the native single-block pullbacks, while the execution-derived evaluator now binds SAVE/HYBRID checkpoints to a typed, digest-validated region residual ABI. Native MLIR residual operands/results and general raw-Python source CFG remain open. Solver residual/JVP/VJP children and restarted GMRES execution remain proven on AVX-512 and gfx1151; clean selector-grade timing remains fail-closed. |
| 6 | **AD-RESIDUAL-EVAL-1 — executable policy evaluator landed** | The Evaluator measures complete backward samples and unique retained residual allocation, and rematerialization consumes only eligible exact-device rows. Counted-region cohorts now execute and label SAVE/RECOMPUTE/HYBRID plans with exact replay/backward counts. Connecting selected checkpoints to generated MLIR and collecting broader exact-device packets remain open. |
| 7 | **DIST-SHARD-ALIAS-1 — bounded portable slice landed** | The nine public names are classified by ownership: three placement/region contracts, five exact aliases of registered collective Target IR, and one distinct point-to-point `collective_permute` gap. The five aliases execute through the deterministic multi-rank runtime; frontend capture, point-to-point Target IR, and native transport remain open. |
| 8 | **AD-FWD-PRODUCT-2 / AD-FWD-NATIVE-1 / AD-HIGHER-1 — bounded products landed** | Public requests carry forward/JVP mode and stable `wrt_indices`; compiler products include compound spectral rules and the exact `tessera.istft_jvp` quotient carrier. The compiler now composes paired reverse with forward mode as `@f__bwd__jvp`, exposes it through `compiled_hvp_ir`, and numerically proves the emitted quadratic product; unsupported second-order operations fail closed. The eager Python `hvp` helper remains a separately labelled finite-difference compatibility path. Native NCCL/RCCL products remain hardware-gated. Broader HVP/ISTFT coverage, Apple/NVIDIA consumption, native multi-rank evidence, and clean performance packets remain open. |
| 9 | **TILE-SYNC-RECONCILE-2026-08-10 — compiler contract and gfx1151 correctness closed** | `tile.async_copy` and `tile.wait_async` share one declared dual-form contract; typed `!tile.async_token` SSA is production and legacy grouping keys are optional compatibility inputs. PR #544 closed required host-free compiler parity, and the follow-up gfx1151 global→LDS/LDS-WMMA/via-Tile cohort passed. Migration of remaining Python/stage-only carriers remains open; host-compiler timing makes no selector claim. |
| 10 | **COMP-SCHED-OVERLAP-1 R1–R4 + W2.1/W2.2/W5.2g — shared software boundary closed** | Explicit async lineage, registered Graph effects, and fail-closed shape/alias/liveness/memory-dependence/activity analysis are landed. Measured resource vectors feed a deterministic critical-path/list scheduler with an admissible resource/queue lower bound; exhaustive enumeration survives only as the ≤8-action oracle. Safe lower-bound losers may be pruned, but exact-device scalar latency remains selection authority. Native transport, clean calibration, and wiring inferred edges into remaining producers stay architecture/client work. |
| 11 | **E2E-REAL-6 — active family-migration cohort** | Native forward products use tracer-produced canonical Graph IR and explicit family plugins. Normalization reverse products now also have one declared Graph/Schedule/Tile/Target family owner; `JitFn` binds the call and records the result instead of constructing that package. ROCm scheduled attention backward consumes its Tile artifact without Graph reconstruction. General solver products generate digest-bound target children for pointwise, reduction, rank-2 matmul, bounded-dynamic/mixed-storage, counted-region, and pure scalar predicate-bearing residuals. AVX-512 and gfx1151 own expanded-family correctness packets; clean selector evidence remains open. Continue per family; `_OpExtractor`, effectful/unmigrated capture, and remaining native-backward helpers cannot be deleted until their lineage, correctness, and architecture-owned evidence are complete. |
| 12 | **PDE-STENCIL-FOUNDATION-1 — semantic correctness slice landed** | Neighbors owns explicit tap coefficients; TPP owns required scheme/order/per-axis spacing; only linkable Target implementations receive callable symbols. Typed PDE classification/stability analysis, stencil-stack unification, and architecture-owned gfx1151/x86 physical packets remain open. |
| 13 | **BLOCK-ATTNRES-1 — gfx1151 Phase 5 landed** | Shared statistics/merge/finalize semantics, typed Graph products, and the content-addressed Schedule→Tile boundary now feed a typed ROCm Target record and exact gfx1151 HSACO/runtime consumer. Three exact-device shapes pass; WSL operation-total timing is retained but selector-ineligible pending bare-metal device timing. AVX-512 Phase 6 remains independent. |
| 14 | **AMD-ISA-DTYPE-1 — cross-generation foundation landing** | A dtype-total selector now distinguishes RDNA3.5, RDNA4, and CDNA5 scalar/vector, dense, sparse, accumulator, scale, shape, and evidence states. CDNA5 gfx1250/MI455X and gfx1251/MI430X are wave32 XDL-WMMA targets with separate cost identities; only existing f16/bf16 K32 materialization is executable, and every newer dtype path remains explicitly gated. |
| 15 | **OP-DTYPE-FLOW-1 — generated end-to-end datatype audit landing** | A normalized generated matrix joins frontend/Graph/Schedule/Tile state, numeric-policy storage and accumulator identity, TSOL membership, per-operation physical manifests, target capability declarations, and exact AMD architecture legality. Derived dtype legality is reported as `legal_only` and cannot masquerade as an operator-specific kernel. |

Hardware packets and backend-specific tuning are synchronized follow-ups to
these slices, not blockers for landing shared contracts with honest fail-closed
states.

Generated evidence inventories are triage inputs, not a flat implementation
queue. Route direct-test and benchmark gaps through the owning family row above
and then the architecture queue. Prioritize correctness-sensitive value
semantics and selector-bearing native paths; structural carriers and host-only
metadata do not acquire artificial device benchmarks merely to reduce a count.

## Route by question

| Question | Read first | Then use |
|---|---|---|
| What works now? | [`COMPILER_AUDIT.md`](COMPILER_AUDIT.md) | generated dashboards and the applicable backend plan |
| What should land next? | [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md) | the scoped plan named by the owning item |
| What core substrate do the capability papers share? | [`CORE_SUBSTRATE_VIEW.md`](CORE_SUBSTRATE_VIEW.md) | the six source docs it maps, then the owning integrated-plan rows |
| How should Graph/Schedule/Tile/Target fit? | [`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md) | [`IR_STACK_INTEGRATION_REVIEW.md`](IR_STACK_INTEGRATION_REVIEW.md), [`TARGET_IR_REVIEW.md`](TARGET_IR_REVIEW.md) |
| How should the frontend and Graph IR change? | [`FRONTEND_GRAPH_SCHEDULE_REVIEW.md`](FRONTEND_GRAPH_SCHEDULE_REVIEW.md) | [`AUTODIFF_UNIFICATION_PLAN.md`](AUTODIFF_UNIFICATION_PLAN.md) |
| How should candidates be judged? | [`EVALUATOR_PLAN.md`](EVALUATOR_PLAN.md) | [`TILESIGHT_ASSESSMENT.md`](TILESIGHT_ASSESSMENT.md) |
| How should backend plugins and emitters converge? | [`COMPILER_REFACTOR_PLAN.md`](COMPILER_REFACTOR_PLAN.md) | [`OPTIMIZING_COMPILER_PLAN.md`](OPTIMIZING_COMPILER_PLAN.md) and the applicable backend plan |
| How should sequence/stateful programs lower? | [`SEQUENCE_MIXER_THEORY.md`](SEQUENCE_MIXER_THEORY.md) | [`SEQUENCE_MIXER_ENGINEERING_PLAN.md`](SEQUENCE_MIXER_ENGINEERING_PLAN.md) |
| How should solver/geometry differentiation land? | [`RIEMANNIAN_OT_PLAN.md`](RIEMANNIAN_OT_PLAN.md) | [`AUTODIFF_ARCHITECTURE_REVIEW.md`](AUTODIFF_ARCHITECTURE_REVIEW.md) |
| How should game-theoretic operators land? | [`GAME_THEORY_PLAN.md`](GAME_THEORY_PLAN.md) | [`EVALUATOR_PLAN.md`](EVALUATOR_PLAN.md) for the oracle rows |
| What is the LSE identity contract? | [`LSE_CHECKPOINT_CONTRACT.md`](LSE_CHECKPOINT_CONTRACT.md) | architecture-owned attention plans |
| When may a consumer fuse into its producer's tiled epilogue? | [`FORGE_ASSESSMENT.md`](FORGE_ASSESSMENT.md) | [`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md) for the arbiter, [`TARGET_IR_REVIEW.md`](TARGET_IR_REVIEW.md) for the emitter seam |
| How should layouts and index arithmetic be represented? | [`CUTE_IR_ASSESSMENT.md`](CUTE_IR_ASSESSMENT.md) | [`CORE_SUBSTRATE_VIEW.md`](CORE_SUBSTRATE_VIEW.md) S9 for the consumer, [`W1_1_TYPING_DESIGN.md`](W1_1_TYPING_DESIGN.md) for the typing precedent |

## Complete live-document catalog

### Status and sequencing

| Document | Role |
|---|---|
| [`COMPILER_AUDIT.md`](COMPILER_AUDIT.md) | Living compiler audit and narrative status. |
| [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md) | Sole cross-plan sequencing authority. |
| [`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md) | Durable architecture and invariants; not a queue. |

### Scoped implementation plans

| Document | Owns |
|---|---|
| [`AUTODIFF_UNIFICATION_PLAN.md`](AUTODIFF_UNIFICATION_PLAN.md) | Compiler-owned differentiation boundary and proof ledger. |
| [`BLOCK_ATTNRES_ROCM_PLAN.md`](BLOCK_ATTNRES_ROCM_PLAN.md) | Block AttnRes mathematical contract, portable oracle, and ROCm-first physical acceptance criteria. |
| [`COMPILER_REFACTOR_PLAN.md`](COMPILER_REFACTOR_PLAN.md) | Shared spine, plugin, packaging, and backend reconciliation details. |
| [`CUTE_IR_ASSESSMENT.md`](CUTE_IR_ASSESSMENT.md) | CuTe IR (NVIDIA/cutlass#3426) review and verified layout algebra: the four-primitive scoping result, the mechanisms worth importing (partially-static value-in-type, fold-static, dynamic-leaf-only lowering, negative-scoped driver), and the LAYOUT-ALG-0..5 sequence serving the S9 `⊑` operator, SparDA's GQA-fold, TileSight's rasterization knob, and the G1b butterfly consolidation. Numeric contract in `tests/unit/test_layout_algebra_contracts.py`. Global order defers to `INTEGRATED_COMPILER_PLAN.md`. |
| [`EGGROLL_SUPPORT_PLAN.md`](EGGROLL_SUPPORT_PLAN.md) | Gradient-free / Evolution-Strategies track: low-rank ES op contract, reference tier, and operator-improvement catalog. |
| [`EVALUATOR_PLAN.md`](EVALUATOR_PLAN.md) | Correctness/evidence rung and promotion contract. |
| [`FORGE_ASSESSMENT.md`](FORGE_ASSESSMENT.md) | FORGE (arXiv:2606.22932) assessment and the residency-aware epilogue-fusion track it opens: locality lattice, static materialization proof, `matmul → optimizer` fusion, fail-closed clipping/routing keys, and the precision-realizability oracle. Numeric contract in `tests/unit/test_fused_wgrad_optimizer_contract.py`. Global order defers to `INTEGRATED_COMPILER_PLAN.md`. |
| [`GAME_THEORY_PLAN.md`](GAME_THEORY_PLAN.md) | Coalition-lattice / equilibrium operator family: subset zeta/Möbius butterfly, semivalues, differentiable equilibria, regret/CFR dynamics, and the numerically verified oracle set (`research/game_theory/`). Global order defers to `INTEGRATED_COMPILER_PLAN.md`. |
| [`OPTIMIZING_COMPILER_PLAN.md`](OPTIMIZING_COMPILER_PLAN.md) | Middle-end synthesis and backend-lift details. |
| [`PDE_STENCIL_CAPABILITY_PLAN.md`](PDE_STENCIL_CAPABILITY_PLAN.md) | PDE-operator semantics, symbol classification, discrete-stability certificates, and the stencil/halo contract queue. |
| [`RIEMANNIAN_OT_PLAN.md`](RIEMANNIAN_OT_PLAN.md) | Geometry/implicit-differentiation consumer and acceptance workload. |
| [`SCHEDULE_OBJECT_DESIGN.md`](SCHEDULE_OBJECT_DESIGN.md) | The one schedule representation (actions/edges/roles/residency + digest) unifying CAKE Phases 2–3, TileRT E5, the W5.2 action DAG, and FORGE W2 — contracts, IR carrier, and SO-1..SO-5 build order. |
| [`compiler_enhancement.md`](compiler_enhancement.md) | CAKE compiler–agent co-design assessment: statistical audit of its clean-start A/B, the Tile sync/memory typing + type-blind-verifier findings, and the scoped Phase 1 / Phase 2 work they open. Global order defers to `INTEGRATED_COMPILER_PLAN.md`. |
| [`SPARDA_REVIEW.md`](SPARDA_REVIEW.md) | SparDA source review, verified compressed-key/block-selection contracts, and the cross-layer prefetch + block-sparse iteration extraction queue. |
| [`SEQUENCE_MIXER_ENGINEERING_PLAN.md`](SEQUENCE_MIXER_ENGINEERING_PLAN.md) | Sequence-mixer family contracts and physical rollout. |

### Architecture reviews and focused references

| Document | Use |
|---|---|
| [`AUTODIFF_ARCHITECTURE_REVIEW.md`](AUTODIFF_ARCHITECTURE_REVIEW.md) | Compiler autodiff gaps and algorithmic review. |
| [`DIFFERENTIABLE_PROGRAMMING_REVIEW.md`](DIFFERENTIABLE_PROGRAMMING_REVIEW.md) | Book-derived delta; distinguishes Python reference work from compiled support. |
| [`COMPILER_ARCHITECTURE_SWEEP.md`](COMPILER_ARCHITECTURE_SWEEP.md) | Cross-layer findings feeding the integrated plan. |
| [`CORE_SUBSTRATE_VIEW.md`](CORE_SUBSTRATE_VIEW.md) | Integrated read across SparDA/TileRT/TileSight/PDE/game-theory/CAKE: the eight shared core-compiler substrate investments, their consumers, and their owning rows (flags the unowned ones). |
| [`FRONTEND_GRAPH_SCHEDULE_REVIEW.md`](FRONTEND_GRAPH_SCHEDULE_REVIEW.md) | Frontend, Graph, and Schedule ownership findings. |
| [`IR_STACK_INTEGRATION_REVIEW.md`](IR_STACK_INTEGRATION_REVIEW.md) | IR adjacency and lowering-boundary findings. |
| [`TARGET_IR_REVIEW.md`](TARGET_IR_REVIEW.md) | Target-dialect typing and target-lowering review. |
| [`W1_1_TYPING_DESIGN.md`](W1_1_TYPING_DESIGN.md) | Current typed Tile design. |
| [`W1_1_TYPING_INVENTORY.md`](W1_1_TYPING_INVENTORY.md) | Source inventory behind the typing design. |
| [`LSE_CHECKPOINT_CONTRACT.md`](LSE_CHECKPOINT_CONTRACT.md) | Shared saved/recomputed-LSE identity contract. |
| [`SEQUENCE_MIXER_THEORY.md`](SEQUENCE_MIXER_THEORY.md) | Sequence-mixer semantic model. |
| [`TILESIGHT_ASSESSMENT.md`](TILESIGHT_ASSESSMENT.md) | Analytical-model research and candidate-pruning guidance. |
| [`TILERT_ASSESSMENT.md`](TILERT_ASSESSMENT.md) | TileRT assessment; overlap-scheduling models and W6/T3/T4 composition-layer direction. |
| [`AMD_KERNEL_COMPILER_SURVEY.md`](AMD_KERNEL_COMPILER_SURVEY.md) | AMD compiler research survey; input to ROCm design, not ROCm evidence. |

Documents under [`archive/`](archive/) are point-in-time evidence only. This
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
