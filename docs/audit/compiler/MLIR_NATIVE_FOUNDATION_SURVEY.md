---
last_updated: 2026-09-05
audit_role: reference
---

# Compiler audit survey — MLIR/LLVM foundation to native backends

The next architectural milestone is a program that can be serialized as MLIR,
compiled without reconstructing its semantics from Python objects, and executed
through a backend-owned native artifact. Adding another executable kernel is
useful, but does not by itself close that milestone.

This survey supplies evidence and disposition to
[`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md#mlirllvm-native-foundation-program--2026-09-04).
Only that plan sequences work. [`README.md`](README.md) owns navigation; existing
registries and their generated projections retain status authority.

## Scope and confidence

The survey inventories all **42 pre-existing top-level Markdown documents**:
22 plans (14 open, eight landing), 17 references, one snapshot, one theme and
one index. Eleven archived documents are historical inputs, not active queues.
This reference initially made the live total 43. The subsequent archive
reconciliation moves two historical documents out: 41 live, 13 archived, with
all 22 plans still live. Counts do not measure implementation completeness.

Review depth: frontmatter, queue/status sections and dependencies across the
whole live folder; source tracing at the frontend, package driver, four native
packagers, emitter interface, CPU JIT, evaluator and parametric-recipe boundary.
This is not a fresh reproduction of every historical finding or every backend
packet. The source baseline is main `e2a30df1`, with the plan corrections and
Apple experiment in now-merged PR #724 (`1d35284a`). No new kernel proof or paper
claim is established here. Source links below name durable symbols; old line
numbers and device/toolchain assertions in historical reviews need rechecking.

## Source-verified architectural findings

| Finding | Evidence | Consequence for the plan |
|---|---|---|
| Native packaging still has two input authorities. | [`driver.py`](../../../python/tessera/compiler/driver.py), `_package_artifacts`, explicitly names Graph as the package fork's parent; `compile_graph_module` chooses scheduled matmul/attention/kernel paths for supported envelopes and retained Graph-owned paths otherwise. | Inventory exact family/target routes. Extend the existing scheduled artifact boundary; do not make all packages appear adjacent by changing hashes. |
| A typed carrier and native bytes do not necessarily mean the user's kernel body was compiled from that IR. | [`x86_native.py`](../../../python/tessera/compiler/x86_native.py), `_lower`, checks the emitted call symbol and packages a prebuilt shared image. In contrast, [`tools/tessera-jit/tessera_jit.cpp`](../../../tools/tessera-jit/tessera_jit.cpp) lowers through linalg/vector/SCF/LLVM and creates an MLIR `ExecutionEngine`. | Track compiled-body and explicitly delegated-library routes independently. Reuse the existing JIT path for a general CPU body; preserve tuned library calls as declared candidates. |
| The source-emitter subsystem is a second semantic compilation surface. | [`emit/kernel_emitter.py`](../../../python/tessera/compiler/emit/kernel_emitter.py), `KernelEmitter.emit`, accepts a Python region plus dtype/dims/options. [`emit/__init__.py`](../../../python/tessera/compiler/emit/__init__.py) exposes separate emitter/compiler/runner registries. [`emit/x86_llvm.py`](../../../python/tessera/compiler/emit/x86_llvm.py) is explicitly a compatibility alias for C emission. | Adapt candidate registration to canonical IR-derived artifacts, then retire redundant semantic generation per family. Do not invent a fourth plugin registry or count a rename as LLVM migration. |
| NVIDIA and ROCm already have native lowering infrastructure to extend. | [`nvidia_native.py`](../../../python/tessera/compiler/nvidia_native.py) owns Tile→NVIDIA→NVVM→PTX packaging and storage-specific ABIs. [`rocm_native.py`](../../../python/tessera/compiler/rocm_native.py), `_compile_native_tile_ir`, invokes separate target and binary pipelines and verifies Tile consumption. | Consolidate input ownership and transformation authority; do not restart either backend. Keep architecture, device libraries and numeric policy in the identity chain. |
| Apple requires an explicit terminal code-generation contract. | [`apple_native.py`](../../../python/tessera/compiler/apple_native.py), `package_scheduled_matmul`, `package_scheduled_kernel` and `_synthesized_reduce_source`, mixes scheduled consumers, native delegated routines, and synthesized MSL. | MLIR owns semantics, schedule and ABI. A compiler-owned MSL→Metal/metallib endpoint is a valid terminal path; a direct LLVM-IR→Metal backend is not assumed. Separate delegated MPS/library calls from generated kernel bodies. |
| Parametric optimization is real but stops before executable instantiation. | [`parametric_recipe.py`](../../../python/tessera/compiler/parametric_recipe.py), `prepare_recipe` invokes native passes; `rank_buckets` returns `BucketRank` with promotion disabled. | Native witness-checked instantiation and artifact admission precede broader idiom recognition. Preserve one optimized recipe and compiler identity across buckets. |
| The ANN evaluator extension is not a fusion integration. | [`evaluator.py`](../../../python/tessera/compiler/evaluator.py), `program_pair_equivalence`, checks native provenance on both outputs. Its current production adapter is `horizontal_equivalence`; [`autodiff/ann_laws.py`](../../../python/tessera/autodiff/ann_laws.py) remains a reference-law producer. | Add a bounded IR fragment/rewriter consumer, then evaluate the original and transformed native programs. Reference identities alone cannot authorize a rewrite. |

These are migration gaps, not allegations that all current native routes are
incorrect. In particular, an explicit library call lowered from IR is legitimate
native execution. It is different evidence from compiling a general kernel body.

## Live Graph-owned package migration inventory

These are **exit-bound historical paths**, not approved permanent parallel
architectures. They are selected by current driver branches, not just unused
helper definitions. The listed families are the package classifier's envelope;
a family with an earlier scheduled fast path reaches the old branch only when
that scheduled envelope does not apply. This source census does not claim every
shape/dtype variant was executed. The first implementation slice adds executable
route fixtures and an inventory derived from those existing routes.

| Route and current entry | What still comes from Python Graph state | Existing replacement / next migration |
|---|---|---|
| NVIDIA `nvidia_native.package_scheduled_matmul(artifact)` | F1 removed the Graph argument and base-package compilation. The consumer validates artifact shape/ABI facts and compiles Tile once; the driver retains adjacent boundary lineage. | Keep the one-compile, no-Graph and tampering gates. Broader matmul storage and dynamic envelopes remain separately scoped. |
| NVIDIA `package_native` | Softmax and reduction now enter native Schedule/Tile lowering, including narrow storage, min, keepdims and cooperative reduction. Their Graph constructors are deleted from production. Norm and forward attention now also consume native Schedule/Tile; their former constructors are test-only baselines. Recompute backward attention, fused paged attention and scaled/quantized matmul still reconstruct Tile. Saved-LSE pairs, signed INT4 and bounded paged reads now have native producers and replayed Schedule consumers. | F2-U1–U10 close the bounded unary migration with RTX differential proof. Aligned masks and paired-LSE validation landed in the next cut; native paired, signed-INT4 and bounded paged producers are implemented; reuse NVVM/PTX compilation and retain per-storage ABI gates. |
| ROCm `rocm_native.package_native` | `package_softmax`, `package_reduction`, `package_paged_kv_read`, `package_attention`, `package_moe_dispatch` reconstruct Tile/Graph text. | Matmul, attention, depth-attention and generic semantic-kernel scheduled consumers already exist. Verify old/new envelope differences, extend missing paged-KV/MoE movement carriers, migrate one family, and delete its Graph constructor after equivalence. Reuse Target→ROCDL→HSACO. |
| x86 `x86_native.package_native` | Matmul/softmax/reduction/attention, cohort2, elementwise and delegated `x86_breadth.package_graph_breadth` classify Graph and construct lower-level calls/carriers. | Use existing scheduled matmul/attention/kernel consumers for covered shapes; move remaining pointwise/cohort/breadth contracts through canonical IR. For general compiled bodies, reuse `tessera-jit` LLVM lowering; keep native library calls explicit rather than relabeling them as generated bodies. |
| Apple GPU `apple_native.package_native` | Batched GEMM, static/dynamic softmax and GELU, transpose, popcount/count-nonzero/topk, SVD and the value-ABI fallback read Graph operands/kwargs and construct descriptors. | Extend existing scheduled matmul/kernel/attention consumers. Migrate one value family to a verified native Target call with IR-derived ABI, then MSL body generation where appropriate. Keep MPS delegation explicitly represented in IR. |
| Apple CPU `apple_cpu_native.package_native` | Reads the single Graph op, shapes, kwargs and dtype to synthesize a runtime-call record and package the prebuilt Accelerate/runtime image. | Add the typed call/ABI lowering from canonical IR; use the CPU LLVM path for general program bodies. This fifth package entry point must not disappear from a “four backend” inventory. |

Source anchors are the corresponding `native_package_kind`, `package_native`
and `package_scheduled_*` symbols in `python/tessera/compiler/`, plus
`driver.compile_graph_module`. Calls from family plugins and direct package
clients also need migration fixtures; removing the driver fallback is not proof
that the historical constructor has no remaining callers.

A `GraphIRModule` at the **frontend capture** boundary is not itself the defect.
It must stop being a downstream source of semantics once canonical MLIR exists.
Likewise, the current scheduled artifacts are useful transport records, but their
shape/layout/ABI fields must become verified projections of the serialized IR;
replacing Graph with another independently authored Python object is not closure.

## Audit reconciliation

The folder has useful contracts but too many time horizons in its summaries.
The following discrepancies should not become new implementation work:

- The index has been recounted after archiving the superseded review snapshot
  and typing inventory: 41 live documents, 13 archived. No active plan is closed.
- The archived snapshot now separates 103 listed severity rows from its original
  102-confirmed aggregate and resolves stale P3/dropout statements against later
  backend records. Allocation-failure injection and broader P2 backend proof
  remain live; see the owning audit's archive reconciliation.
- `SCHEDULE_OBJECT_DESIGN.md` still proposes SO-3/SO-4 work whose bounded shared
  objectives the integrated plan records as landed. `CORE_SUBSTRATE_VIEW.md`
  still calls FORGE locality/residency unowned, while the integrated layout
  section explicitly absorbs that work through LAYOUT-ALG-1. Reuse that authority.
- Frontend location and pre-elaboration summaries were corrected in #724.
  Remaining native instantiation and broader raising are separate deliverables.
- `COMPILER_THEORY_OF_OPERATION.md` describes the current source-emitter plugin
  contract. It is evidence of the starting point, not the desired final
  architecture. Older fleet tables also conflict with the WSL workflow and the
  now-available NVIDIA lane.
- Open/landing frontmatter is a lifecycle label, not a count of missing code.
  No plan is closed or archived by this survey. Closure needs the residual
  acceptance tests and the owning audit's summary.

## Disposition of the original live-document cohort

“Residual” below is a scoped planning disposition, not newly executed proof.
Where only documentary evidence was inspected, the next route inventory must
verify reachability and exact tests before scheduling a deletion.

| Document | Role/state | Disposition under the foundation program |
|---|---|---|
| `README.md` | index | Repair membership/counts and route readers to the current foundation sequence. |
| `COMPILER_AUDIT.md` | theme | Summarize mixed IR authority; keep dated evidence distinct from current sequencing. |
| `INTEGRATED_COMPILER_PLAN.md` | plan/open | Own the foundation milestones, dependencies and backend exits. |
| `COMPILER_REFACTOR_PLAN.md` | plan/landing | Reuse A–E interfaces and differential gates; migrate semantic emitter inputs to IR without another plugin system. |
| `OPTIMIZING_COMPILER_PLAN.md` | plan/landing | General fusion discovery and native synthesis become IR consumers; source synthesis remains a measured baseline during migration. |
| `AUTODIFF_UNIFICATION_PLAN.md` | plan/landing | Finish exact family/target products and remove Python runtime differentiation from native-required paths only after replacement proof. |
| `AUTODIFF_NEXTGEN_PLAN.md` | plan/open | Keep law/Weil work distinct from AD-JET-IR-1. Native jet descent depends on actual IR batching, policy and residual consumers. |
| `W4_ADMISSIBLE_EFFECTS_PLAN.md` | plan/open | E1/E3/E4/E5 have bounded landings; keyed-RNG adjoint and deterministic transport proof remain scoped extensions. Never admit arbitrary I/O. |
| `EVALUATOR_PLAN.md` | plan/landing | Reuse native provenance and verdicts; prioritize independent optimization/IR-pipeline comparisons and exact candidate admission over search breadth. |
| `SCHEDULE_OBJECT_DESIGN.md` | plan/open | Reconcile bounded SO-3/SO-4 landings; retain NVIDIA producer/barrier and stated-entry residuals. Do not create a second scheduler. |
| `CUTE_IR_ASSESSMENT.md` | plan/open | Reuse C++ layout algebra and carried proofs. Unsupported dynamic nonseparable layouts remain explicit, not a reason to restart L0–L5. |
| `compiler_enhancement.md` | plan/open | CAKE sync/role ownership extends existing Tile legality. NVIDIA barrier-at-birth and later cost/role consumers remain architecture-gated. |
| `FORGE_ASSESSMENT.md` | plan/open | Locality/residency requirements map to layout and numeric-policy authority. Optimizer fusion and distributed reduce-into-state are consumer workloads. |
| `INTRA_KERNEL_FEEDBACK_PLAN.md` | plan/open | P1 schema/math independent; P0 owning-clock validation gates P2/P3 IR instrumentation. No L2/L3 dispatch admission. |
| `FUNCTIONAL_ANALYSIS_TSOL_PLAN.md` | plan/open | Error/adjoint/stability contracts feed IR legality and evaluator admission. Do not expand a parallel semantic registry or count reference analysis as native lowering. |
| `PDE_STENCIL_CAPABILITY_PLAN.md` | plan/open | Python classification/FTCS certificates are bounded; C++ pass/carrier consumption and broader native stencils remain the descent work. |
| `MATH_SOURCE_WORKSTREAM.md` | plan/landing | Keep landed MSW reference/calculus/examples; bind remaining native ANN work to MSW-9, not another math inventory. |
| `ANN_CALCULUS_DESIGN_SPIKE.md` | plan/landing | Spike and native pair adapter exist; Graph fragment inventory and fusion consumer remain. |
| `SEQUENCE_MIXER_ENGINEERING_PLAN.md` | plan/landing | Transition/state semantics, backward and algorithm candidates must consume the common IR boundary; per-target performance stays local. |
| `BLOCK_ATTNRES_ROCM_PLAN.md` | plan/landing | Preserve typed statistics/merge/adjoint and gfx1151 slice; sibling packages and hoisting/liveness are separately gated consumers. |
| `EGGROLL_SUPPORT_PLAN.md` | plan/open | Preserve rank-1 gfx1151/x86 and fixed-key JVP landings; reverse transpose, rank/dtype breadth and native distribution remain consumers. |
| `GAME_THEORY_PLAN.md` | plan/open | Route transform/precision/collective needs through shared infrastructure; defer unrelated domain breadth from the compiler critical path. |
| `RIEMANNIAN_OT_PLAN.md` | plan/open | Use geometric inner loops as control/AD/residual acceptance workloads; retain semantic-key and replay restrictions. |
| `SPARDA_REVIEW.md` | plan/open | Native block-sparse iteration and prefetch edges consume effects, memory and schedule authority; cache helpers alone do not close them. |
| `COMPILER_ARCHITECTURE_SWEEP.md` | reference | Historical duplicate-authority/analysis rationale; map residuals to existing owners. |
| `FRONTEND_GRAPH_SCHEDULE_REVIEW.md` | reference | Historical capture/control-flow seams; verify today's bounded CFG routes before reopening findings. |
| `FRONT_END_LOWERING_ASSESSMENT.md` | reference | Main input for IR-as-record, parametric instantiation and raising; early diagnostic gaps have partial fixes. |
| `IR_STACK_INTEGRATION_REVIEW.md` | reference | Boundary/legality rationale; use current driver lineage and native consumers to determine residuals. |
| `TARGET_IR_REVIEW.md` | reference | Hardware-free target contracts and lowering inventory; old target counts are not current proof. |
| `COMPILER_THEORY_OF_OPERATION.md` | reference | Preserve tier/performance safeguards; distinguish today's source plugin model from the IR-owned destination. |
| `CORE_SUBSTRATE_VIEW.md` | reference | Group overlapping demands; reconcile old “unowned” rows with layout, policy and Schedule Object landings. |
| `AUTODIFF_ARCHITECTURE_REVIEW.md` | reference | AD authority and algorithm rationale; shared rules and family plugins are the migration baseline. |
| `DIFFERENTIABLE_PROGRAMMING_REVIEW.md` | reference | Rule/transform/evaluator requirements; do not reopen documented implementation fixes without reproduction. |
| `MATRIX_CALCULUS_REVIEW.md` | reference | Degeneracy, structural and higher-order oracles; separate existing remediation from native transform needs. |
| `W1_1_TYPING_DESIGN.md` | reference | Binding typed Tile/layout/numeric-policy semantics; preserve unknown-case refusal. |
| `archive/W1_1_TYPING_INVENTORY.md` | archived reference | Superseded census; current source census below and typing design retain the NVIDIA residual. |
| `LSE_CHECKPOINT_CONTRACT.md` | reference | Reuse saved/recomputed state and backward ABI as a paired-program witness; per-target evidence remains independent. |
| `SEQUENCE_MIXER_THEORY.md` | reference | Semantic transitions/state/reassociation contracts, not a second lowering implementation. |
| `TILESIGHT_ASSESSMENT.md` | reference | Calibrate and constrain analytical pruning; measured target latency remains promotion authority. |
| `TILERT_ASSESSMENT.md` | reference | Composition/overlap constraints feed the existing schedule; no new queue/runtime hierarchy. |
| `AMD_KERNEL_COMPILER_SURVEY.md` | reference | Algorithm options for native ROCm generators, not a mandate to import an alternative compiler stack. |
| `archive/CODE_REVIEW_2026-08-29.md` | archived snapshot | Reconciled historical source/device dispositions; unresolved fault-injection and P2 obligations remain owned in live queues. |

## Desired execution boundary

Python remains the user API, tracing interface, experiment orchestration,
validation and independent oracle. It may invoke compilers, load/cache artifacts,
and describe deployment context. It must not be the only owner of a semantic
fact needed to lower or execute a program in the native-required envelope.

A serialized IR module carries types, effects, shape constraints, layout,
numeric policy, derivative/residual identity and the selected schedule. Native
passes verify and transform those facts. A backend consumes the resulting IR
and produces an image plus an ABI descriptor. The runtime loads and dispatches
that artifact; it does not rediscover a graph or silently execute a Python body.
Device identities, live memory capacity and measurements remain explicit external
inputs bound to the artifact, rather than pretending all deployment facts are IR.

A useful acceptance test is independent replay: save canonical IR and declared
options, release the originating Python graph/region objects, compile in a fresh
process using the native toolchain, then launch through the same descriptor ABI.
Compare numerics and semantics against the original oracle; mutate a shape,
policy or schedule carrier and require rejection or a changed artifact identity.
Hashes prove identity, not correctness: independent outputs and negative tests
remain necessary.

The integrated plan specifies the ordered cuts, backend endpoints and stop
conditions. The initial deliverable is a refreshed *route* inventory and one
native replay slice, not a new registry, another set of Python lowerers, or a
blanket rewrite of all working generators.

## Typing inventory replacement — 2026-09-04

Source checks superseding the archived August census:

| Current source fact | Migration consequence / owner |
|---|---|
| `src/compiler/ir/TileOps.cpp::MMAOp::verify` filters data operands, rejects unknown fragment types and calls `verifyMMAFromTypes`. Its separate tensor-value lane checks rank, types, arity and accumulator compatibility. | No permissive bare-fragment branch remains to delete. W1.1 owns eventual tensor-value lane retirement after its actual callers migrate. |
| `GenerateWMMAGemmKernel.cpp` constructs parameterized A/B/accumulator fragment types; its typed view/pack/zero/MMA chain is a real producer. | The inventory's “zero typed C++ producers” premise is obsolete. Preserve the existing typed ROCm path and its regression gates. |
| `src/transforms/lib/TileIRLoweringPass.cpp` still has two `OperationState(..., "tile.mma")` tensor-valued construction sites. | W1.1 / IR-NATIVE-FOUNDATION-1 F2: migrate both to the logical/typed boundary and prove NVIDIA lowering before removing the value lane. Constructor-site presence was checked; this is not a new device execution result. |
| The typing design's completion ledger retains NVIDIA producer/Target proof and the separately checked tensor lane as its residuals. | Keep `W1_1_TYPING_DESIGN.md` live. The old inventory is historical; no plan closure or native capability promotion follows. |


### F1 follow-through — 2026-09-04

The Graph re-entry described in the original census has been removed from
NVIDIA scheduled matmul. `package_scheduled_matmul` now accepts only the scheduled
artifact, checks its descriptor contract against Schedule/Tile IR, and compiles
once. Driver ancestry follows the scheduled boundaries. The census above records
the pre-migration finding; remaining Graph-owned package roots stay in F2.

## Unary caller closure — 2026-09-05

`driver.compile_graph_module` consumes the scheduled artifact directly.
`package_native` dispatches unary requests to `package_softmax` / `package_reduction`,
which enter `lower_scheduled_kernel`; `package_f32_softmax` delegates to the same
entry. None invokes a Graph-to-Tile constructor after scheduling. The consumer
accepts only `ScheduledKernelArtifact` and compiles its Tile IR once. Serialized
Schedule IR replays through `tessera-opt` after discarding the Python Graph object.

The removed constructors survive only in `tests/_support/nvidia_unary_baseline.py`
for differential evidence. Public `emit_softmax_tile_ir`, `emit_reduce_tile_ir`
and typed convenience emitters remain low-level compatibility/artifact utilities;
production unary package routes no longer call them. Their continued existence
is not a claim that every Python text emitter has been retired.

Mixed-storage and keepdims reductions are legal Graph IR forward contracts.
The generic Linalg reduction matcher still declines these envelopes, and reverse
AD explicitly fails before producing an unsupported cotangent. NVIDIA forward
proof does not confer AD or sibling-backend execution support.

## Norm/attention migration and remaining constructor census

The NVIDIA driver and direct `package_norm` / `package_attention` calls now use
native scheduled artifacts. `schedule.norm` owns kind, storage, axis, f32 epsilon
and workgroup size. Forward `schedule.attention` owns NVIDIA's explicit
`sm120_recompute` policy. The emitted LLVM wrappers feed the existing NVVM/PTX
pipeline without Python Graph-to-Tile emission. Their historical constructors
are retained only in `tests/_support/nvidia_{norm,attention}_baseline.py`.

Review findings fixed in this cut:

- Forward attention hashes used six-decimal float strings, allowing distinct f32
  scale/softcap/dropout policies to share an identity. Hashes now encode exact
  f32 bits on every backend. Old serialized forward Schedule artifacts must be
  regenerated; stale hashes fail validation.
- The SM120 mask uses local query positions, while canonical ragged attention
  aligns a shorter query to the key sequence's end. F2-A2 now aligns both physical
  kernels and admits short-query masks after exact-device comparisons.
- Norm admission now rejects epsilon that is nonpositive, nonfinite or not
  representable as positive finite f32, and refuses unsupported numeric policy
  overrides. This matches the native verifier and immutable runtime constant.

The remaining NVIDIA Graph-owned constructors are explicit next work:

| Entry/family | Native contract still required |
|---|---|
| `package_attention_backward` | dQ/dK/dV result roles, mask alignment, workspace/reduction policy |
| `package_matmul` and NVFP4/MX variants | Remaining dtype/layout/packing envelopes outside the scheduled consumer |
| `package_paged_attention` | Page table, logical positions, ownership and bounds |
| `package_replay_ssm_kernels` | Paired decode/flush state and ordered effects |
| `package_moe_kernels` | Multi-entry routing, movement, capacities and workspace |

ROCm, x86, Apple GPU and Apple CPU direct package paths remain as listed in the
main census. Existing scheduled consumers do not establish direct-client
retirement. The next cuts must migrate those callers with backend-owned evidence,
not transplant NVIDIA's 128-thread schedule. F2-A2 also fixes backward float
identity with exact f32 bits. F2-A3 now has native two-result forward and three-result backward checkpoint
producers. Signed INT4 and bounded paged reads also consume native Schedule/Tile
artifacts, with their Python Tile constructors retired. Scaled packing, fused
paged attention, replay-SSM and MoE remain in this census. A checkpoint descriptor
still does not prove runtime buffer provenance or complete AD policy integration.
