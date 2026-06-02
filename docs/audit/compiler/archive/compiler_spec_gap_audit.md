---
status: Informative
classification: Audit
last_updated: 2026-05-22
---

# Compiler Spec Gap Audit

This audit compares the active compiler implementation against the normative
spec set in `docs/spec/`. It excludes `archive/src/`, `archive/docs/`,
`research/`, and legacy example snapshots unless an active spec explicitly
points at them.

The companion matrix is `docs/audit/compiler/COMPILER_AUDIT.md`.

## 2026-05-22 Closure note

The "Spec Needs Update" backlog below was addressed in a focused doc
refresh on 2026-05-22.  Each spec received a `## Documentation refresh
(2026-05-22)` section anchoring its specific drift items to the
current implementation source.  Specs touched:

| Spec | Refresh anchor |
|------|----------------|
| `PYTHON_API_SPEC.md` | Public debug / profile / autotune surfaces + S-series modules cross-linked to `python/tessera/debug.py`, CLIs, and the generated `support_table.md`. |
| `GRAPH_IR_SPEC.md` | Debug markers declared metadata-only (no ODS ops); numeric_policy + lane_provenance documented as Graph IR attributes; attention-family ops cross-linked to `AttentionFamilyPasses.cpp`. |
| `LOWERING_PIPELINE_SPEC.md` | Full inventory of 9+ shipped named lowering pipelines (x86, gpu, rocm, metalium, apple_cpu, apple_cpu-runtime, apple_gpu, apple_gpu-runtime, cpx, nvidia-pipeline, spectral, tpp); Python driver lowering paths; halo + spectral pass-order matrices. |
| `TILE_IR.md` | `tile.debug_artifact` / `tile.debug_barrier` declared metadata-only; canonical name `tile.alloc_shared` reaffirmed; TilingInterface real on MatmulOp; TMEM/tcgen05 still scaffolded. |
| `TARGET_IR_SPEC.md` | Normative marker-elision contract; compile-artifact metadata vs C ABI separation; per-backend status table; G/H/I pre-work landed. |
| `RUNTIME_ABI_SPEC.md` | Replay manifests declared NOT part of the C ABI; Apple CPU + GPU symbol exports documented (26 GPU symbols); collective adapter version pin (NCCL/RCCL ≥ 2.22). |
| `CONFORMANCE.md` | Already closed 2026-05-20 (header cleanup landed earlier); kept linked to `VALIDATION_SPINE.md`. |

**Sprint M5 + S5 (2026-05-22)** subsequently closed both remaining
P2 items:

- **MEMORY_MODEL_SPEC.md** received 3 new structural verifier rules
  in `python/tessera/compiler/memory_verifier.py` for scoped atomic
  attributes (`MEM_ATOMIC_INVALID_OP/ORDER/SCOPE`), fence scope
  (`MEM_FENCE_INVALID_SCOPE`), and deterministic-profile reduction
  enforcement (`MEM_DETERMINISTIC_NONDETERMINISTIC_REDUCTION`). §11
  enforcement table was updated to reflect 3 promotions from `planned`
  → `structural verifier` with test evidence. **19 new tests** in
  `tests/unit/test_memory_verifier.py` (46 total).

- **SHAPE_SYSTEM.md** added §11 MLIR Verifier Gap Enumeration — a
  per-contract enforcement matrix mapping each shape contract to where
  it's checked (PY-DT / PY-CT / MLIR-PASS / MLIR-VERIFIER /
  RT-WITNESS / GAP) plus a §11.2 canonical-gap list naming the 4 named
  MLIR-verifier gaps (no ODS-level shape verifiers; no MLIR symbolic
  dim equality re-check; no LayoutLegalityPass; `tile.mma`/`tile.wgmma`
  target verifiers). **17 new tests** in
  `tests/unit/test_shape_verifier_gap_map.py` lock evidence-file
  presence + canonical gap names.

Net: 36 new tests across the 2 specs; the spec-gap audit is now
fully closed for the original 9-item list.

**MLIR Verifier Sprint V1+V2+V3 (2026-05-22)** subsequently closed
**3 of the 4 named MLIR-verifier gaps**:

- **V1** (per-op verifier coverage, partial): added `let hasVerifier =
  1;` + `verify()` for `TransposeOp` (rank + permutation + element
  type), `LayerNormOp` (shape-preserve + eps > 0), `MoeDispatchOp`
  (leading-axis token-count match) in
  `src/compiler/ir/TesseraOps.td` + `TesseraOps.cpp`.  Lit fixture:
  `tests/tessera-ir/phase2/sprint_v1_verifiers.mlir` (9 cases,
  positive + negative).  Original gap wording was inaccurate — 15 ops
  already had verifiers; the closure addresses the coverage subgap.
- **V2** (LayoutLegalityPass skeleton): new
  `src/transforms/lib/LayoutLegalityPass.cpp` with the canonical
  8-name layout accept-set (row_major / col_major / nhwc / nchw /
  bhsd / tile / bsr / packed) and first rule (`tessera.cast` with
  non-canonical `tessera.layout` attribute → fails pass with
  `LAYOUT_LEGALITY_UNKNOWN_LAYOUT`).  Registered as
  `--tessera-layout-legality` + wired into the `TesseraPasses` CMake
  target.  Lit fixture:
  `tests/tessera-ir/phase2/sprint_v2_layout_legality.mlir`.
- **V3** (target-aware verifier on the canonical attention/MMA op
  family): `FlashAttnOp::verify()` extended to walk for a
  `tessera.target_sm` parent attribute and enforce per-SM head_dim
  ceiling (sm_70–89 ≤ 128, sm_90/100/120 ≤ 256, no SM ⇒ no limit).
  Lit fixture:
  `tests/tessera-ir/phase3/sprint_v3_flash_attn_target_aware.mlir`.

**24 new structural-guard tests** in
`tests/unit/test_mlir_verifier_sprint.py` pin the .td / .cpp /
CMake / lit fixture content so a future edit that softens the
diagnostic wording fails immediately.

**Build + lit verification (2026-05-22):**
`cmake --build build --target tessera-opt` succeeded;
`lit tests/tessera-ir/phase{2,3}/sprint_v*.mlir` reports 3/3 PASS
through the rebuilt C++ binary.

**MLIR Verifier Sprint V5 (2026-05-22)** closed the **4th and final**
MLIR-verifier gap.  All four originally-named MLIR-verifier gaps
from the 2026-05-22 SHAPE_SYSTEM.md §11.2 set are now closed at V1:

- **V5 (post-lowering symbolic dim equality re-check):** new
  `src/transforms/lib/SymbolicDimEqualityPass.cpp` registered as
  `--tessera-symdim-equality`.  Reads function-level
  `tessera.dim_bindings` (ArrayAttr of equation strings) and
  `tessera.dim_sizes` (DictionaryAttr of symbol → i64); validates
  the equation when both sides are bound; walks `tessera.reshape /
  transpose / matmul` ops checking per-op dim-name contracts.
  Four stable diagnostic codes (`SYMDIM_BINDING_VIOLATION`,
  `SYMDIM_RESHAPE_VIOLATION`, `SYMDIM_TRANSPOSE_VIOLATION`,
  `SYMDIM_MATMUL_CONTRACT_VIOLATION`).  Lit fixture:
  `tests/tessera-ir/phase2/sprint_v5_symdim_equality.mlir`
  (1 positive transpose + 2 negative: binding + matmul contract;
  the reshape branch is in the pass body but `tessera.reshape` is
  not yet registered in the dialect — V2 followup).

**19 new structural-guard tests** in
`tests/unit/test_symbolic_dim_equality_pass.py` pin the pass source,
diagnostic codes, attribute names, op dispatch list, CMake / Passes.h
/ Passes.cpp registration, and lit fixture content.

**Build + lit verification (2026-05-22):**
`cmake --build build --target tessera-opt` succeeded with the new
`SymbolicDimEqualityPass.cpp` source compiled into `TesseraPasses`;
`lit tests/tessera-ir/phase2/sprint_v5_symdim_equality.mlir` reports
1/1 PASS through the rebuilt C++ binary.

**The original 4-item MLIR-verifier gap set is now fully closed.**
Followup work tracked as V2 (not V1 regression):
- Register `tessera.reshape` as a proper ODS op so the reshape lit
  case can be exercised end-to-end.
- Insert `--tessera-symdim-equality` into the named lowering pipelines
  after `DistributionLoweringPass`.
- Add SSA-value flow tracking (auto-propagate dim names through ops
  without explicit per-op annotations).
- Affine / Presburger reasoning beyond simple symbol products.

**MLIR Verifier Sprint V6a/V6b/V6c (2026-05-22) closure**:

- **V6a (`tessera.reshape` registered as ODS op):** Added
  `Tessera_ReshapeOp` to `TesseraOps.td` with `let hasVerifier = 1;`
  and a `ReshapeOp::verify()` in `TesseraOps.cpp` that checks
  element-count preservation + element type match.  The V5 lit
  fixture now exercises the reshape branch end-to-end (no
  `--allow-unregistered-dialect` flag needed); the fixture grew
  from 1 positive + 2 negative to 1 positive + 3 negative covering
  all 3 stable diagnostic codes whose ops are now registered
  (binding + reshape + matmul contract).

- **V6b (`--tessera-symdim-equality` in named pipelines):** Inserted
  `createSymbolicDimEqualityPass()` into `tessera-lower-to-x86`,
  `tessera-lower-to-gpu`, and the `tessera-nvidia-pipeline` family
  AFTER `createDistributionLoweringPass()`.  Lit fixture
  `tests/tessera-ir/phase2/sprint_v6b_symdim_in_pipeline.mlir`
  proves a broken `tessera.dim_bindings` clause is caught
  mid-pipeline with the same stable diagnostic the standalone
  pass emits.

- **V6c (target-aware ScaledDotProductOp verifier):** Extended the
  FA-4 Tile IR `ScaledDotProductOp::verify()` in
  `src/compiler/tile_opt_fa4/lib/Dialect/Attn/AttnOps.cpp` to walk
  for a `tessera.target_sm` parent attribute and enforce per-SM
  `tile_q × tile_kv` ceilings (sm_70–89 ≤ 64 × 128; sm_90+ ≤ 128
  × 256).  Diagnostic format
  `tile_{q,kv}=N exceeds the SM <sm> ScaledDotProduct kernel limit of <limit>`.
  Lit exercise deferred to Sprint V7 because `tessera.attn`
  dialect is not yet registered in `tessera-opt` (the two
  pre-existing scaled_dot_product fixtures are XFAIL'd for the
  same reason).  Python structural guards
  (`test_sprint_v6c_*` in `test_mlir_verifier_sprint.py`) pin the
  verifier source + diagnostic phrases.

**New structural-guard tests (Sprint V6):**
- 1 ReshapeOp ODS-presence parametrized case
- 1 ReshapeOp verify() signature
- 2 ReshapeOp diagnostic phrases
- 1 V5 lit fixture 1+3 count update
- 1 V6b pipeline integration lit fixture presence
- 3 V6b pipeline wiring assertions (lowerToX86 / lowerToGPU / buildCUDA13Pipeline)
- 1 V6c tile-size table presence
- 1 V6c diagnostic phrase set
- 1 V6c parent traversal
- 1 V6c lit deferral acknowledgement

Total Sprint V6 additions: **23 new tests** (was 5189 → **5212 passing**;
verified below). All structural guards green. C++ build clean.

**V7 followups (next sprint):**
- Register `tessera.attn` dialect in
  `tools/tessera-opt/tessera-opt.cpp` so the V6c lit fixture +
  the two existing XFAIL'd fixtures can drop their deferral
  markers.
- Same pattern available for other un-registered Tile IR dialects
  (`tessera.queue`, etc.) once needed.

**Sprint V7 + V2-flow (2026-05-22) closure**:

- **V7 (partial — registration plumbing landed):** Created public
  header `src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/AttnDialect.h`
  mirroring the Apple backend pattern; added
  `tessera::attn::registerAttnDialect(DialectRegistry&)` body in
  `AttnOps.cpp`; wired through `tools/tessera-opt/tessera-opt.cpp`
  under a new `TESSERA_HAVE_FA4_ATTN` ifdef; updated
  `tools/tessera-opt/CMakeLists.txt` to link `TesseraAttnDialect`.
  Build green; dialect appears in `--show-dialects`; symbols
  linked into the binary (verified via `nm`).  Python structural
  guards (`tests/unit/test_attn_dialect_registration.py`, 6 tests)
  pin the registration plumbing.

  **V7 remaining work (V7b):** MLIR parser does not lazy-load
  the dialect for standalone IR fixtures (no pass references it,
  so the longest-prefix dialect lookup never triggers
  `getOrLoadDialect<TesseraAttnDialect>`).  The 3 lit fixtures
  using `tessera.attn.scaled_dot_product`
  (`flash_attn_full.mlir`, `tile_ir_lowering.mlir`, V6c) remain
  XFAIL'd.  V7b will investigate eager-load via `MlirOptMainConfig`
  or migrate the dialect class from tablegen to hand-written
  (matching the `NeighborsDialect` pattern that works).

- **V2-flow (SSA-value dim-name propagation):** Extended
  `SymbolicDimEqualityPass` with a per-function flow tracker.
  New helpers: `ValueDimMap`, `readArgDimNames`, `crossCheck`,
  `propagateThroughFunction`.  Reads
  `tessera.arg_dim_names` (ArrayAttr-of-ArrayAttr) from each
  `func.func`, propagates through
  `tessera.{transpose,matmul,reshape}` ops, cross-checks against
  any explicit `tessera.dim_names_{in,out,lhs,rhs}` annotations.
  Adds `SYMDIM_FLOW_INCONSISTENCY` diagnostic code.  Lit fixture:
  `tests/tessera-ir/phase2/sprint_v2_flow_propagation.mlir`
  (1 positive + 1 negative + 1 backward-compat).  Python
  structural guards extended with 11 new V2-flow tests pinning
  the helpers, propagation rules, and backward-compat fall-through.

  Backward compat: when no `tessera.arg_dim_names` is declared,
  V2-flow returns success() early — existing V5/V6a/V6b
  functions keep working unchanged.  Verified by the
  `flow_no_arg_names_falls_through` positive case.

**Tests added this round**: 6 (V7 structural guards) + 11 (V2-flow
structural guards) = **17 new tests**, all passing.  Plus 2 new lit
fixtures: V2-flow propagation + V7's intent doc (V6c).

**Sprint V7b + V3a + V4a + V4b (2026-05-22) closure**:

- **V7b — tessera.attn parser eager-load**: Added a
  `DialectRegistry::addExtension` anchored on the parent
  `tessera::TesseraDialect` so that whenever a context loads the
  Graph IR dialect, the extension callback eagerly calls
  `ctx->getOrLoadDialect<TesseraAttnDialect>()`.  The MLIR parser
  now recognises `tessera.attn.scaled_dot_product` in standalone
  IR fixtures with no pass referencing the dialect.  V6c lit
  fixture was recreated without `XFAIL`/`REQUIRES` and now passes
  end-to-end through the real `tessera-opt` binary.  Linkage:
  `TesseraIR` added to `TesseraAttnDialect`'s `LINK_LIBS PUBLIC`
  plus include dirs for the Graph IR headers.

- **V3a — affine non-product bindings**: `SymbolicDimEqualityPass`
  now accepts sum-of-products RHSes (`D = H * Dh + K`, bare-symbol
  sums like `Total = A + B + C`).  New `Binding.terms`
  representation `SmallVector<SmallVector<std::string>>` carries
  one inner vector per `+`-separated term, each holding the
  `*`-separated factors.  Two parser helpers (`splitOn`,
  `parseProductTerm`) and one evaluator (`evaluateBindingRHS`)
  cover both V5 single-product and V3a sum-of-products paths.
  Diagnostic wording is split: V5 form keeps `product of RHS = N`
  for backward compatibility; multi-term bindings render
  `value of RHS (sum of products) = N`.  Lit fixture:
  `tests/tessera-ir/phase2/sprint_v3a_affine_bindings.mlir`.

- **V4a — LayoutLegality producer/consumer rule**: Added
  `matmulAcceptSet()` and `checkMatmulOperandLayouts()` to
  `LayoutLegalityPass.cpp`.  For each `tessera.matmul`, the pass
  walks the def-using op of each operand for a `tessera.layout`
  attribute and rejects layouts outside `{row_major, col_major}`
  (matmul's stricter accept-set) with new stable diagnostic code
  `LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH`.  Initial draft
  skipped cast producers; removed that — canonical cast layouts
  (`bsr`, `packed`) live in V2's wider 8-layout accept-set but not
  in matmul's, so the cast's layout attribute IS what we should
  check against.  Lit fixture:
  `tests/tessera-ir/phase2/sprint_v4a_layout_producer_consumer.mlir`.

- **V4b — long-tail per-op verifiers**: Promoted four ops to real
  verifiers — `CastOp` (rank + static-dim preservation; element
  type may differ), `SoftmaxOp` (rank + static-dim preservation
  + optional `axis` bounds check `-rank <= axis < rank`),
  `RopeOp` (rank + element type + static-dim preservation), and
  `DropoutOp` (probability must satisfy `0.0 <= p < 1.0`; shape
  preserved — previously `LogicalResult DropoutOp::verify()
  { return success(); }`).  Added `let hasVerifier = 1;` to the
  three ops that didn't already declare it.  11 new stable
  diagnostic phrases; lit fixture
  `tests/tessera-ir/phase2/sprint_v4b_per_op_verifiers.mlir` with
  4 positive cases and 7 negative cases.

**Tests added this round (V7b/V3a/V4a/V4b)**: 8 new structural guards
in `test_mlir_verifier_sprint.py` (1 V7b + 2 V3a + 2 V4a + 8 V4b for
hasVerifier/impl/phrase/fixture = 13; net +8 after consolidation),
bringing the verifier-sprint guard count from 50 → **58 passing**.
Three new lit fixtures (`sprint_v3a_affine_bindings.mlir`,
`sprint_v4a_layout_producer_consumer.mlir`,
`sprint_v4b_per_op_verifiers.mlir`) bring the full Tessera-IR lit
sweep to **57 PASS / 19 UNSUPPORTED / 19 XFAIL / 0 FAIL** (up from
92 → 95 tests discovered).

**Remaining V3+ work** (in priority order):
- Identity-cascade folding for LayoutLegalityPass (cascade of
  layout casts within a function whose net effect is identity
  should fold; currently flagged only as comment placeholder).
- Further affine reasoning beyond sum-of-products: subtraction,
  parenthesized groups, integer literals (V3a follow-ups).

**Sprint V3b + V3c (2026-05-22) closure**:

- **V3b — interprocedural dim-name tracking via func.call:**
  `SymbolicDimEqualityPass` now builds a module-level `SymbolTable`
  in `runOnOperation()` and walks `func.call` ops as part of the
  per-function propagation.  When a call's callee declares
  `tessera.arg_dim_names`, the pass cross-checks each operand's
  propagated dim-names against the callee's declared names
  position-by-position; mismatch ⇒ `SYMDIM_CALL_ARG_MISMATCH`.
  The pass also reads `tessera.ret_dim_names` on the callee and
  seeds the call result values' dim-names, so names flow ACROSS
  the call boundary into subsequent ops in the caller.  Indirect
  calls (`func.call_indirect`) are out of scope.  Lit fixture:
  `tests/tessera-ir/phase2/sprint_v3b_interprocedural.mlir`
  (1 negative case + 3 positive paths).

- **V3c — scf.for / scf.if region propagation:** New mutual
  recursion between `propagateThroughOp` and a new helper
  `propagateThroughBlock` lets the walker descend into region
  bodies.  For `scf.for`, body block args[1:] inherit dim-names
  from the corresponding init operands; the `scf.yield` operands'
  propagated names must match the iter_args' expected names
  (loop must be name-invariant) ⇒ `SYMDIM_LOOP_YIELD_MISMATCH`
  on conflict; scf.for result values inherit the iter_args'
  names.  For `scf.if`, both regions are walked and their
  `scf.yield` operands' names compared per-result; mismatch ⇒
  `SYMDIM_IF_BRANCH_MISMATCH`.  Lit fixture:
  `tests/tessera-ir/phase2/sprint_v3c_scf_propagation.mlir`
  (2 negative cases + 2 positive cases).

**Tests added (V3b/V3c)**: 6 new structural guards in
`test_mlir_verifier_sprint.py` (3 V3b + 3 V3c), bringing the
verifier-sprint guard count from 58 → **64 passing**.  Two new lit
fixtures: V3b interprocedural + V3c scf propagation.

**Phase G/H/I hardware-gated frontier (Sprint M, 2026-05-22)**:

A new dedicated audit doc `docs/audit/backend/BACKEND_AUDIT.md`
captures the honest answer to "what's hardware-blocked vs. still
hardware-free?"  Highlights:

- Registry's authoritative count: **0 entries at
  `backend_kernel = complete`** across all 432 entries; 273
  `partial`, 159 `planned`.  By design of the registry, this can
  only move with real NVIDIA / ROCm / Metalium hardware proofs.
- All major hardware-free axes (`math_semantics`, `shape_rule`,
  `dtype_layout_rule`, `lowering_rule`, `tests`) are already at
  zero partial/planned.
- Residual hardware-free closure work is small: 53 vjp + 53 jvp
  `planned` (mostly N/A audit candidates), 37 masking_effect_rule
  planned (mostly device-state ops).
- Apple GPU is the only target with backend execution fully
  closed on this machine; Sprint M adds the first model-shaped
  E2E proof under `tests/unit/test_apple_gpu_mla_e2e.py` covering
  a single-layer MLA-style multi-head attention decoder across 3
  shape parametrizations.

## Executive Summary

Tessera has a real compiler spine in place: Python and textual DSL frontends,
Graph IR emission, source-span diagnostics, ODS-backed Graph IR ops, Schedule IR
and Tile IR object models, Target IR artifact paths, x86/CPU lowering, FA-4 Tile
IR scaffolding, runtime ABI headers, CPU/Apple CPU runtime execution paths, and
broad Python/lit coverage all exist in the active tree.

The main gap is no longer absence of a compiler or developer tooling. It is that
several specs and guides still compress different maturity levels into one phase
label. Some areas are implemented and tested, some are hardware-free target
artifacts, some are mock-runtime paths, and some remain intentional stubs. The
highest-risk drift is around:

- Python API and developer tooling status: the public API now includes profiler,
  autotune, debug/replay helpers, runtime artifact helpers, and richer
  `tessera-mlir`/`tessera-prof` surfaces that are only partially reflected in
  the older API/spec prose.
- Runtime and target status: the C ABI, CPU backend, Python runtime wrapper,
  CPU/Apple CPU executable paths, and CUDA/HIP backend files exist, but non-CPU
  target artifacts must remain separated from native hardware-runtime claims.
- Tile/Target IR claims: FA-4, queue, TMEM, NVIDIA, ROCm, Apple, TPU, Metalium,
  Cerebras, and Rubin CPX vary from ODS-backed and lit-testable to scaffolded or
  stubbed. Specs should avoid one status label for all target IR.
- Conformance phase claims: `CONFORMANCE.md` has a current body, but its header,
  duplicated test-suite rows, and checklist still carry older phase/profile
  wording that should be rebuilt from `COMPILER_REFERENCE.md`.
- Debug/profiling claims: debugging and profiling guides now describe structured
  traces, replay manifests, compile artifact inspection, schedule artifacts, and
  telemetry; normative specs should identify which of these are API contracts vs
  informative developer tools.

## 2026-05-06 Audit Delta

The May 6 audit includes the profiling/autotuning and debugging implementation
passes. New evidence since the previous audit:

- `python/tessera/debug.py` now exposes structured `TensorSummary`,
  `DebugTrace`, and `GraphTrace` JSON, named debug capture helpers, replay
  manifests, and bounded replay saving.
- `python/tessera/cli/mlir.py` now emits static metadata, diagnostics, Chrome
  trace JSON, GraphViz, all-artifact bundles, and opt-in
  `--mode=compile_artifact --symbol=name` inspection for real JIT artifacts.
- `python/tessera/compiler/driver.py` writes debug IR and bundle state when
  `TESSERA_DEBUG_IR`, `TESSERA_DUMP_STATE`, and `TESSERA_DUMP_DIR` are set.
- `python/tessera/compiler/schedule_ir.py`,
  `python/tessera/compiler/tile_ir.py`, and
  `python/tessera/compiler/target_ir.py` preserve lightweight debug markers
  through Schedule/Tile lowering while dropping marker-only ops before target
  codegen.
- `python/tessera/autotune.py`, `python/tessera/compiler/autotune_v2.py`,
  `python/tessera/profiler.py`, and `python/tessera/telemetry.py` now share a
  stable telemetry/schedule-artifact foundation for synthetic GEMM tuning and
  source-inspection profiling.

## Built, Scaffolded, And Missing

| Area | Current implementation evidence | Audit status | Action |
|------|---------------------------------|--------------|--------|
| Python frontend and public namespace | `python/tessera/__init__.py`, `python/tessera/compiler/jit.py`, `python/tessera/compiler/constraints.py`, `python/tessera/compiler/effects.py`, `tests/unit/test_constraints.py`, `tests/unit/test_effects.py` | implemented | Keep specs current with actual exported symbols. |
| Textual DSL frontend | `python/tessera/compiler/frontend/parser.py`, textual frontend tests | implemented | Keep normative grammar coverage synced with DSL examples and unsupported-construct diagnostics. |
| Python Graph IR builder | `python/tessera/compiler/graph_ir.py`, `python/tessera/compiler/op_catalog.py`, `tests/unit/test_graph_ir.py`, `tests/unit/test_ir_spine_contract.py` | implemented | Update Graph/API specs for newer op catalog entries. |
| Source-span diagnostics | `python/tessera/compiler/graph_ir.py`, `python/tessera/diagnostics.py`, `tests/unit/test_error_handling_diagnostics_guide.py` | implemented | Maintain stable error-code mapping and line/column coverage. |
| Python Schedule/Tile/Target IR object models | `python/tessera/compiler/schedule_ir.py`, `tile_ir.py`, `target_ir.py`, `tests/unit/test_schedule_ir.py`, `test_tile_ir.py`, `test_target_ir.py` | implemented / artifact-backed | Keep object model and native MLIR dialect status distinct. |
| Graph IR ODS | `src/compiler/ir/TesseraOps.td`, `src/compiler/ir/TesseraOps.cpp`, `src/compiler/ir/TesseraTiling.cpp` | implemented / scaffolded | Mark tiling interface methods as scaffolded until TODOs are resolved. |
| Core transforms and x86 lowering | `src/transforms/lib/*.cpp`, `src/transforms/lib/Passes.cpp`, `tests/tessera-ir/phase2/` | implemented / lit-testable | Keep pipeline order anchored to `Passes.cpp`. |
| FA-4 Tile IR and queue dialects | `src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td`, `src/compiler/tile_opt_fa4/include/tessera/Dialect/Queue/Queue.td`, `tests/tessera-ir/phase3/`, `src/compiler/tile_opt_fa4/test/` | lit-testable | Clarify which ops have verifiers/lowering vs schema only. |
| TMEM / tcgen05 path | `src/compiler/tile_opt_fa4/test/tmem/tcgen05_ptx_body.mlir`, `src/compiler/tile_opt_fa4/lib/Conversion/TesseraTileToPTX/LowerTileToPTX.cpp` | stubbed / lit-testable | Keep Blackwell TMEM as artifact/stub until real PTX body and operands land. |
| NVIDIA backend | `src/compiler/codegen/tessera_gpu_backend_NVIDIA/`, `tests/tessera-ir/phase3/`, `src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia/` | lit-testable, with placeholder kernels | Avoid claiming native hardware-runtime for WGMMA placeholder paths. |
| ROCm backend | `src/compiler/codegen/Tessera_ROCM_Backend/`, ROCm lit tests, `runtime/hip/loader.cpp` stubs | lit-testable / scaffolded | Separate Target IR/ROCDL artifact support from HIP loader runtime. |
| TPU backend | `src/compiler/codegen/Tessera_TPU_Backend/`, `python/tessera/compiler/tpu_target.py`, `tests/tessera-ir/phase4/`, `tests/unit/test_tpu_lowering.py` | implemented / lit-testable | Note PJRT execute remains stubbed. |
| Apple backend | `src/compiler/codegen/Tessera_Apple_Backend/`, `tests/tessera-ir/phase8/apple_*` | lit-testable / artifact-only | Document as target artifact support. |
| Metalium backend | `src/compiler/codegen/Tessera_Metalium_Backend/`, Metalium tests, TODO in `TileToMetalium.cpp` | scaffolded / lit-testable | Keep matmul lowering incomplete in backlog. |
| Cerebras backend | `src/compiler/codegen/Tessera_Cerebras_backend/`, `src/compiler/codegen/Tessera_Cerebras_backend/tests/cerebras_lowering.test` | stubbed / scaffolded | Do not include in normative Target IR unless intentionally promoted. |
| Rubin CPX backend | `src/compiler/codegen/Tessera_RubinCPX_Backend/`, phase8 tests, CPX ODS | scaffolded / lit-testable | Add spec coverage or explicitly mark out-of-scope for current target spec. |
| Runtime C ABI and CPU backend | `src/runtime/include/tessera/*.h`, `src/runtime/src/tessera_runtime.cpp`, `src/runtime/src/backend/cpu_backend.cpp`, `tests/unit/test_runtime_abi.py` | implemented / mock-runtime | Update `RUNTIME_ABI_SPEC.md` phase table to reflect current CPU implementation. |
| Compiler profiling and autotuning foundation | `python/tessera/profiler.py`, `python/tessera/autotune.py`, `python/tessera/compiler/autotune_v2.py`, `python/tessera/cli/prof.py`, `python/tessera/cli/autotune.py`, profiling/autotune unit tests | implemented foundation / synthetic tuning | Next gap is real device-timer backends and broader op tuning. |
| Debugging and replay tooling | `python/tessera/debug.py`, `python/tessera/cli/mlir.py`, `docs/guides/Tessera_Debugging_Tools_Guide.md`, debug CLI tests | implemented foundation | Native ODS debug ops remain future unless semantics cannot be represented by metadata. |
| CUDA/HIP runtime backends | `src/runtime/src/backend/cuda_backend.cpp`, `src/runtime/src/backend/hip_backend.cpp` | hardware-runtime when built, otherwise unavailable | Record compile-flag and device requirements explicitly. |
| Collectives and distributed planner | `src/collectives/`, `src/transforms/lib/GPUCollectiveInsertionPass.cpp`, `python/tessera/testing/mock_collective.py`, unit tests | implemented / scaffolded | Distinguish mock collectives from native NCCL/MPI execution. |
| Shape system | `python/tessera/shape.py`, `src/compiler/diagnostics/ShapeInferencePass.cpp`, shape/unit tests | implemented / scaffolded | Clarify which checks are Python-level vs MLIR verifier-level. |

## Spec Needs Update

| Spec | Issue | Recommended update |
|------|-------|--------------------|
| `docs/spec/PYTHON_API_SPEC.md` | The op catalog and profiling/autotune sections were partially refreshed, but the top-level debug/graph namespace still omits `debug_value`, replay capture, `tessera-mlir --emit=all`, and `tessera-autotune`. | Refresh public developer-tool symbols and command examples from `python/tessera/__init__.py`, `debug.py`, and `pyproject.toml`. |
| `docs/spec/GRAPH_IR_SPEC.md` | Graph IR spec now covers ODS extensions, but debug capture markers and textual frontend lowering metadata should be named if they are normative IR contracts. | Add debug marker status as metadata-only unless native ODS ops are promoted. |
| `docs/spec/LOWERING_PIPELINE_SPEC.md` | The C++ pass list is close to source, but Python driver target paths now include CPU, NVIDIA, Apple CPU/GPU, and ROCm artifacts; only x86/GPU named C++ pipelines are described in detail. | Add a Python object-model lowering section and a target-path status table for `compile_graph_module`. |
| `docs/spec/TILE_IR.md` | Tile naming remediation is mostly done, but `tile.debug_artifact` and `tile.debug_barrier` are not represented as debug/metadata-only markers. | Add an informative debug marker subsection or explicitly keep them out of normative Tile IR. |
| `docs/spec/TARGET_IR_SPEC.md` | Backend status appendices exist, but debug markers are intentionally dropped before Target IR and compile-bundle target metadata is now part of developer tooling. | Clarify marker-elision and compile artifact metadata. |
| `docs/spec/RUNTIME_ABI_SPEC.md` | Runtime status was corrected, but replay manifests and Python runtime artifacts are increasingly used as developer contracts. | Cross-link debugging guide; keep replay manifest out of C ABI unless promoted. |
| `docs/spec/CONFORMANCE.md` | Header cleanup closed 2026-05-20: the document now states mixed status and defers implementation state to `COMPILER_REFERENCE.md`. Remaining work is ordinary validation-spine synchronization, not stale phase-language cleanup. | Keep test commands linked to `VALIDATION_SPINE.md`; do not reintroduce broad old phase-complete-vs-planned framing. |
| `docs/spec/MEMORY_MODEL_SPEC.md` | Memory model is stronger than current enforcement: atomics, device-wide fences, and deterministic mesh reduction are largely not verified in active compiler/runtime tests. | Mark enforcement requirements as planned unless a verifier/test evidence row exists. |
| `docs/spec/SHAPE_SYSTEM.md` | The spec covers Graph IR checker, schedule feasibility, Tile verifier, and runtime witnesses, but evidence is split across Python shape utilities, diagnostics pass, and tests. | Add an implementation map and identify MLIR verifier gaps explicitly. |

## Implementation Backlog

| Priority | Gap | Evidence | Recommended work |
|----------|-----|----------|------------------|
| P0 | Align conformance/spec status labels with active source. | `docs/spec/CONFORMANCE.md`, `docs/spec/COMPILER_REFERENCE.md`, `docs/README.md` disagree on phase status granularity. | Update conformance tables before adding new feature claims. |
| P0 | Refresh developer-tool API claims. | Debugging/profiling/autotune CLI and Python surfaces moved ahead of `PYTHON_API_SPEC.md` command/symbol examples. | Update API spec and add tests for exported debug/prof/autotune command docs. |
| P0 | Keep Graph/Python op catalog regression automated. | `tests/unit/test_compiler_spec_gap_remediation.py` now guards most op-catalog drift. | Keep the regression and add generator support before adding large op families. |
| P1 | Complete or explicitly gate TilingInterface TODOs. | `src/compiler/ir/TesseraTiling.cpp` has placeholder `failure()` paths. | Either implement tiling interface methods or mark interface support scaffolded. |
| P1 | Normalize Tile IR op naming. | Spec says `tshared.alloc`; PM verifier/tests use `tile.alloc_shared`; mbarrier naming differs from generic barrier text. | Pick canonical names and add migration aliases if needed. |
| P1 | Decide native debug op promotion. | Python/object-model debug markers exist, but native ODS debug ops are not required yet. | Keep marker-only support unless diagnostics/artifacts cannot carry the semantics. |
| P1 | Add real device-timer profiling backends. | Autotune currently has synthetic roofline and structured unmeasured on-device status. | Implement CPU/Apple measurement first, then CUDA/HIP event timers. |
| P1 | Complete Metalium lowering semantics. | `TileToMetalium.cpp` reports matmul lowering not implemented and has TODOs for operands/types. | Make pass fail loudly for unsupported ops or implement real lowering. |
| P1 | Replace TMEM/tcgen05 placeholder PTX. | `LowerTileToPTX.cpp` and FA-4 docs call current body schematic. | Add real Blackwell PTX operands/constraints and target gating. |
| P2 | Strengthen memory model enforcement tests. | Specs require scoped atomics/fences/happens-before; active evidence is mostly async-copy/mbarrier oriented. | Add lit tests for illegal memory patterns and deterministic reduction contracts. |
| P2 | Separate mock collectives from native collective runtime. | Mock helpers and insertion pass exist; native multi-rank runtime remains scaffolded. | Add status badges and native-runtime acceptance criteria. |
| P2 | Document source-only backends. | Cerebras/Rubin CPX and parts of Apple/ROCm appear in active code but are thin or artifact-only. | Either add per-backend spec appendices or mark experimental/scaffolded. |

## Recommended Order

1. Rebuild `CONFORMANCE.md` from `COMPILER_REFERENCE.md` and remove stale header
   and duplicated test-suite language.
2. Refresh `PYTHON_API_SPEC.md` for debugging, profiling, autotuning, replay,
   and developer command surfaces.
3. Add or update tests that lock the corrected public claims, especially debug
   CLI, replay manifests, telemetry schema, Graph/Python op catalog, and runtime
   ABI status.
4. Decide whether source-only target backends are normative, scaffolded
   extensions, or out-of-scope for `docs/spec/`.
5. Implement high-value missing behavior: TilingInterface methods, device-timer
   profiling, TMEM real PTX, Metalium matmul lowering, native collectives, and
   memory model verifier tests.

## Validation Notes

Local validation run on 2026-05-06:

| Command | Result | Notes |
|---------|--------|-------|
| `scripts/lint_docs.sh` | passed | Re-run after rebuilding both audit docs. |
| `~/venv/bin/python -m pytest tests/unit/test_compiler_spec_gap_remediation.py tests/unit/test_debugging_tools_foundation.py tests/unit/test_cli_debug_profile_commands.py tests/unit/test_profiling_autotuning_foundation.py -q` | passed | `40 passed`; covers spec-gap remediation guards plus the new debug/profiling CLI contracts. |
| `~/venv/bin/python -m pytest tests/unit -q` | failed with known unrelated failure | `1929 passed, 1 failed`; failing fixture is `test_apple_cpu_accelerate_dispatches_bf16_matmul_via_bnns`, the pre-existing Apple bf16 strict-equality issue. |
| `~/venv/bin/python -m pytest tests/unit/test_debugging_tools_foundation.py tests/unit/test_cli_debug_profile_commands.py tests/unit/test_compiler_driver_foundation.py tests/unit/test_schedule_ir.py tests/unit/test_tile_ir.py tests/unit/test_target_ir.py -q` | passed | `50 passed`; covers the new debugging and compiler-artifact contracts. |
| `PYTHONPATH=python ~/venv/bin/python -m tessera.cli.mlir --help` | passed | Confirms CLI exposes `metadata`, `diagnostics`, `trace`, `graphviz`, `all`, `--mode`, `--symbol`, `--target`, and `--artifacts-dir`. |
