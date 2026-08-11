---
last_updated: 2026-08-10
audit_role: theme
---

# Compiler Audit

This document consolidates the compiler audit material that previously lived in
multiple root audit documents and compiler archive files.

## TileRT assessment — composition scheduling direction (2026-08-10)

[`TILERT_ASSESSMENT.md`](TILERT_ASSESSMENT.md) assesses TileRT (tile-ai's
closed low-latency inference runtime) as external validation of the W6 /
TileSight-T3/T4 overlap-scheduling direction, with four analytic results: the
bubble decomposition proving MoE is the max-bubble target, a hard ≤3× (batch-1
≤2×) overlap-speedup ceiling that shows TileRT's own 3–4× claim is composite,
a counterexample proving scalar-latency arbitration mis-selects kernels once
any composition layer exists (→ record resource vectors in autotune records
now, via the open `hot_path_metadata` slot — zero schema change), and a
static-first/dynamic-only-under-variance scheduling rule with an explicit
determinism constraint. The trace behind it found the connective tissue absent
but the parts built: `comm_overlap.py` (contract, zero production consumers),
the 2026-08-09 typed futures (awaits still adjacent to dispatch — window
zero), `pipeline_planner.ScheduleStep` (discarded at the IR boundary), and the
threaded MegaMoE pipeline. Negative findings: `tessera.queue` is dead
unparseable vocabulary whose claimed producer never emits it;
`CollectiveScheduler`/`ChunkPlanner` exist only as names in comments and docs.
Direction only — no status rows changed. W2.2 is a named hard prerequisite for
any scheduler beyond await-sinking.

## `tessera.queue` MLIR dialect deleted (2026-08-10)

Decision #29/#31 disposition, per the TileRT assessment's §2.1 negative
finding: the Sprint V8 `tessera.queue` MLIR dialect (3 ops, 2 types, 6
diagnostic codes) had **zero producers and zero consumers** — no C++ code ever
constructed a queue op, plugin registration was commented out, and the
dotted-name type syntax (`!tessera.queue.tile_queue`) was unparseable in
standalone lit IR, so its one fixture could never pass. The
warp-specialization boundary's production synchronization mechanism is
`!tile.pipeline_state` + `!tile.async_token` SSA chains; `WarpSpecializationPass`
comments claiming queue-triple emission were corrected at the same time.

Deleted: the dialect (`Queue.td`, `QueueOps.cpp`, `QueueVerifiers.cpp`,
headers, CMake targets, `tessera-opt` registration/feature `fa4-queue`), the
orphaned Python twin `compiler/queue_dialect.py`, the unpassable
`queue_show.mlir` fixture, the six `QUEUE_*` diagnostic codes, and the
`dialects_manifest.py` row. **Kept, untouched:** the live Python tile IR
queue vocabulary — `lower_schedule_to_tile_ir` emits
`tessera.queue.{create,push,pop,barrier}` strings, `tile_ir.py` +
`memory_verifier.py` verify them (happens-before), and they feed the
`queue_depth` resource records. A stays-deleted gate lives in
`tests/unit/test_mlir_verifier_sprint.py::test_queue_mlir_dialect_stays_deleted`;
any revival must ship a parseable single-segment name, a real producer, and a
passing fixture.

## Dead verifier/plugin surfaces deleted (2026-08-10)

Two C++ surfaces compiled by no CMake target were resolved per Decision #31
(delete, not consume) after checking provenance: both landed together in the
April 2026 scaffold commit `8fbc4eb` ("Compiler backend update") as a planned
central `registerTesseraAll` registration entry that `tessera-opt` never
adopted — the real drivers register through
`src/compiler/ir/TesseraDialect.cpp::registerTesseraDialects`, a live function
the dead plugin *also* declared under the same name (a latent symbol collision
had it ever been linked).

**Deleted:**
- `src/compiler/programming_model/ir/ScheduleOps.cpp` — whole file. Its
  `verifyProgrammingModelOp` dispatcher had zero callers;
  `PMPasses.cpp::PMV11VerifierPass` (built into `TesseraPM`) is the single
  production implementation of the `schedule.`/`cache.`/`tile.` structural
  checks. Because the file was never compiled, none of its extra checks
  (schedule.prefetch/async_copy/artifact, cache.*, tile.alloc_shared/reduce)
  were ever enforced, so deletion loses nothing that was live — and several of
  its contracts had drifted (no mbarrier arch gate; laxer mbarrier semantics
  than PMV11's release/acq_rel/seq_cst). Note: this tree (main @ `af27ed8`)
  still carried the `tile.async_copy`/`tile.wait_async` entries; the
  whole-file deletion subsumes the async-contract reconciliation's partial
  removal.
- `src/compiler/mlir/` scaffold: `TesseraMLIRPlugin.cpp`,
  `include/Tessera/TesseraMLIRPlugin.h`, `lib/Graph/TesseraGraphIR.cpp`,
  `lib/Schedule/TesseraScheduleIR.cpp`, `lib/Target/TesseraTargetIR.cpp`
  (unbuilt `emitAsyncCopy`/`verifyTargetOp` with zero callers — distinct from
  the live `emitAsyncCopy` in `TileIRLoweringPass.cpp`).

**Retained — the directory is NOT wholly dead:**
`src/compiler/mlir/include/Tessera/Common/Lowering.h` is the Workstream A1
shared Tile→Target lowering helper (`tessera::common::extractPtr`/
`ensureExternalDecl`/fusion-call skeleton), consumed by
`src/transforms/lib/TileToX86Pass.cpp` and the Apple backend via the root
`CMakeLists.txt` `include_directories(src/compiler/mlir/include)`. That is its
named consumer; it stays.

**Drift-gate fallout fixed in `tests/unit/test_pipeline_registry.py`:** the
`_PASS_REGISTRATION_FILES` scan had been treating the never-compiled plugin as
C++ pipeline truth — `tessera-neighbors-pipeline` and `tessera-full-pipeline`
existed *only* there and are now removed from `_KNOWN_UNTRACKED_PIPELINES`;
the stale path to the long-deleted `PassPipelinesPM11.cpp` is replaced by the
live `programming_model/lib/PMPasses.cpp`, which actually registers
`tessera-pm-{verify,legalize}-pipeline`. Stale doc pointers updated in
`docs/spec/COMPILER_REFERENCE.md`, `PROJECT_STRUCTURE.md`, and
`docs/context/knowledge_map.yaml` (+ regenerated context outputs).

## tile.async_copy / tile.wait_async — one declared contract (2026-08-10)

Found during the TileRT assessment trace: the two sync ops shipped with three
simultaneous contracts. The ODS (`TileOps.td`) declared the W1.1 dual form
(typed `!tile.async_token` SSA edge, legacy `tile.barrier_id`/`tile.depends_on`
attrs); a name-dispatched verifier in the **unbuilt** `ScheduleOps.cpp` — and
its live mirror in `PMV11VerifierPass` (`tessera-pm-verify`) — REQUIRED a
`stage` attribute and memref operands; the Python spine
(`tile_ir.py::_verify_async_copy`) required `stage >= 0` and `vector >= 1`.
The production emitter (`TileIRLoweringPass::emitAsyncCopy`: tensor operand,
tile + token results, no stage) satisfied only the first — running
`tessera-pm-verify` over production Tile IR failed with `'stage' must be >= 0`.

Resolution (Decisions #29/#31/#21a): the **ODS dual form is the single
declared contract**, now stated in full in `TileOps.td` — typed token form is
production; the legacy form is the declared compatibility envelope whose
grouping keys (`barrier_id`/`depends_on` on ROCm, integer `stage` on the
Python spine and `TileBufferReusePass`) are **optional and conservative on
absence** (a key-less wait retires everything). `stage` is well-formedness-
checked when present (`TILE_ASYNC_STAGE_NEGATIVE`, both ops, TileOps.cpp).
The required-stage model was deleted from `ScheduleOps.cpp` (dead code — that
file is not in any CMake target) and relaxed to when-present in
`PMV11VerifierPass` and `tile_ir.py`. The dead stage-model *emitter*
(`src/compiler/mlir/lib/Target/TesseraTargetIR.cpp`, also unbuilt) is now
legal under the envelope and was left for the dead-code audit.

Gates: `phase2/pm_verify_async_token.mlir` (red at baseline, green after —
verified by rebuilding HEAD), negative-stage cases in
`phase2/tile_async_token_invalid.mlir`, a legacy-stage positive in
`phase2/tile_async_token.mlir`, and a no-key-verifies-clean unit test in
`test_tile_ir.py`. Full `lit tests/tessera-ir/` failure set is unchanged vs.
baseline on the Mac config (29 pre-existing ROCm/x86-lane fixtures that need
the primary box's build). Remaining primary-box gates and follow-ups:
[`STRIX_HALO_WORKLIST_2026-08-10.md`](STRIX_HALO_WORKLIST_2026-08-10.md).

## Collective async unification (2026-08-09)

Cross-backend sync `COLLECTIVE-ASYNC-UNIFY-2026-08-09` removes the final active
unregistered `tessera.collective.*` producers. `GPUCollectiveInsertionPass` and
`AdjointCollectiveInsertionPass` now depend on the registered collective
dialect, produce typed futures, insert explicit awaits, and route forward or
cotangent SSA consumers through the awaited values. Their lit lanes no longer
disable per-pass verification to tolerate marker strings.

The second-pass functional hardening proves equal all-to-all partitioning,
positive QoS/chunk contracts, future/payload identity, and explicit mesh-axis
topology. The runtime rejects unknown axes and subgroup execution that its
full-communicator adapter cannot honor; dtype and chunk metadata survive the
content-addressed artifact; native v1 rejects implicit conversion to fp32.
Native adapter replacement and singleton shutdown are also completion-safe:
in-flight callbacks retain strong runtime and adapter owners until tracing is
closed and the limiter token is returned.
This closes the portable software contract, not architecture evidence. Native
NCCL/RCCL/Metal/x86 process transport packets and performance remain open.

## Effect/control and collective boundary closeout (2026-08-08)

`AD-CORE-EFFECT-CONTROL-1` is complete at the shared compiler boundary.
`tessera.stop_gradient` is registered in the public catalog and Graph dialect;
both autodiff passes compute backward SSA activity, propagate registered Graph
effects, reject active stochastic operations, ignore inactive regions, and
fail closed on active regions or stopped values whose residuals cannot be
safely replayed. Direct lit negatives and the paired CPU oracle cover these
contracts.

The four collectives now lower as exact `tile.all_reduce`,
`tile.reduce_scatter`, `tile.all_gather`, and `tile.all_to_all` ODS operations
with shared mesh/tensor-axis, reduction, rank, dtype, and static extent checks.
`tessera-lower-tile-collectives` carries those operations into the registered
asynchronous `tessera_collective` Target dialect, inserts explicit await
dependencies, and stamps the portable runtime-adapter ABI. A content-addressed
Python package executes the same ordered Target records through the existing
collective adapter; deterministic two-rank tests prove all four operations and
SSA input/output lineage. This is the functional software transport boundary.
Native NCCL/RCCL/Metal/x86 process transport, exact multi-rank evidence, and
device performance remain architecture-owned.

The same Target dialect now registers a typed window resource and rank-local
`window.register`, `put_signal`, `signal`, `wait_signal`, and
`window.deregister` operations. The content-addressed GIN/RMA package verifies
window lifetime and dispatches the ordered records through an explicit
rank-local adapter. A launcher-neutral native harness binds
Tessera/OpenMPI/PMI/Slurm ranks to one RCCL rank and emits exact readback plus
dual-clock timing. This closes the software and launcher boundary, not the
still-hardware-gated multi-node gfx1151 evidence packet.

The seven formerly actionable thin-test rows are directly proven: five
differentiable relaxations use their public forward surface plus defining/VJP
oracles, while `training.loss_sgd` and `training.loss_adamw` use exact x86 and
gfx1151 fused-versus-unfused tests. The remaining thin-reference rows are
classified aliases, structural contracts, hardware gates, or internal rows;
the raw scanner count is not presented as untriaged numerical debt.

## Cost-model foundations: target perf + rasterization knob (2026-07-28)

An assessment of TileSight (arXiv:2607.22432) surfaced that Tessera's
hardware-free analytical cost model **was a mock at discovery time** —
`schedule_planner._estimate_latency_ms` had no memory term (latency was FLOPs over
a fudge-factored peak), `autotune_v2._mock_latency` was a hand-drawn bowl with
`tile_m=128, tile_n=128` placed at its minimum by hand, and the target profiles
carried capability data but **zero** performance data. Theory §4 step 3's
"without silicon, score by the Tier 2 cost model" therefore ranked nothing. Full
finding, verdict, and reference survey:
[`TILESIGHT_ASSESSMENT.md`](TILESIGHT_ASSESSMENT.md).

The two prerequisites and the first T1 model slice have now landed. The quoted
function names above describe the retired implementation, not current behavior.

**1. `compiler/target_perf.py` — calibrated performance parameters.**
Per-device (not per-arch: `nvidia_sm120` covers a 5070 Ti and an RTX PRO 6000,
>2x apart) peaks, DRAM bandwidth, LLC size, SMEM/CU, keyed by canonical
`normalize_target()` ids. Two honesty rules are gated by
`tests/unit/test_target_perf.py`: **provenance is per field** — literally, via
`field_provenance` overrides on a row default, because a real row mixes kinds
(the RTX 5070 Ti's FLOP identity is `DERIVED`, its core count `SPEC`, its SMEM
capacity `MEASURED` on-silicon) and a row-level label would misreport at least
one of them — and an absent value stays absent: accessors return `None`,
`require()` raises `TargetPerfError` naming the gap. A `measured` overlay
from a calibration sweep always beats spec and reports itself as measured, so a
spec-based roofline cannot masquerade as a silicon-based one. All three fleet
boxes are registered; `DERIVED` rows are checked against their stated identity by
test. Deliberately unpopulated: bf16/fp8 **matrix** peaks on every fleet box, and
all Zen5 peaks — no trustworthy public figure exists, so they wait for a sweep.
`SchedulePlanner.for_target()` consumes it and **refuses** rather than falling
back to the A100-shaped default. This is the missing input for **W7**.

**2. `compiler/tile_rasterization.py` — block rasterization as a real knob.**
Previously the only threadblock swizzle in the tree was Apple's MLX-inherited
`swizzle_log = 0 if tm <= 3 else 1` hardcode; ROCm's "swizzle" is an unrelated
LDS bank-conflict XOR; NVIDIA and x86 had none. TileSight reports this lever
moving measured L2 hit rate 35% → 72%. Now: `ROW_MAJOR` (the identity — existing
behaviour, byte-for-byte), `COLUMN_MAJOR`, `GROUPED_M`, `GROUPED_N`, wired as
`schedule.knob` ops, `schedule.tile` attrs, and `TuningConfig.raster_order` /
`raster_group` fields. Morton/Z-order is deliberately absent (clean bijection
only on power-of-two square grids; no closed-form emission).

**The axes are carried, not swept** (corrected 2026-07-30 — an earlier draft of
this entry claimed "autotuner axes", which overstated it). Neither
`LegalGEMMCandidateGenerator` nor the Optuna objective enumerates them, because
the new T1 reuse-distance scorer is a pruning model, not promotion evidence.
It can distinguish raster orders symbolically, but `ROCM-CALIB-1` rejected the
older step-distance locality score on its home architecture; allowing another
uncalibrated locality score to promote a raster would repeat that failure.
The axes become search dimensions only after exact-device correlation and a
backend-owned retain verdict. A test
(`test_raster_axes_are_carried_but_not_swept`) pins this evidence boundary.

Because a rasterization order is a *permutation of block ids*, it is
semantics-preserving by construction and has a **total, hardware-free oracle**:
`is_bijection()` enumerates the grid and proves every tile is hit exactly once
(the shape of proof `matmul_opt_ladder.verify_split_k_equivalence` uses).
`tests/unit/test_tile_rasterization.py` additionally **compiles the emitted C
snippet with the host clang and runs it against the Python reference for every
block id** across ragged grids — so the device kernel and the model of it cannot
drift, proven with no GPU (the snippet is plain integer arithmetic, valid
identically under CUDA and HIP).

**Explicitly not done.** No emitter consumes the knob yet: the NVIDIA `mma.sync`
GEMM (`emit/nvidia_cuda.py`, the 2-D `dim3 grid((M+15)/16,(N+7)/8)` launches) and
the ROCm GEMM path still compute `blockIdx` directly. That wiring needs a
measurement on the NR2 Pro / Strix Halo boxes to be worth anything, and neither
was available here. Default is the identity, so nothing changed until it is.
T1 is now implemented in `compiler/reuse_distance_cost.py`: it materializes the
symbolic GEMM tile order, counts A/B tile reuse through a capacity-bounded LRU,
uses real dtype storage widths, and combines cache-derived DRAM bytes with
`TargetPerf` compute/bandwidth inputs. It replaces both `_estimate_latency_ms`
and `_mock_latency`. It deliberately has no fitted warp, stage, or preferred-tile
coefficients. T3 action-DAG ordering and T4 occupancy/prologue/steady-state
overlap remain open.

**Cross-backend synchronization key `RASTER-CONTRACT-2026-07-28`.** The
rasterization knob is a shared Schedule IR contract, so all four architecture
queues record a state (AGENTS.md, "active architecture queues"):

| Queue | State | Owning item |
|---|---|---|
| [NVIDIA](../backend/nvidia/todo.md) | follow-up required — NR2 Pro (sm_120) | `NVIDIA-RASTER-1` |
| [ROCm](../backend/rocm/todo.md) | follow-up required — Strix Halo (gfx1151) | `ROCM-RASTER-1` |
| [Apple](../backend/apple/todo.md) | follow-up required — *reconciliation*, not implementation: Apple already carries an MLX-inherited `swizzle_log` hardcode, a second and incompatible spelling of the same lever | `APPLE-RASTER-1` |
| [x86](../backend/x86/todo.md) | not applicable — the AMX/AVX-512 lane emits OpenMP loop nests with no launch grid, so there is no block id to permute; the *idea* still ports as cache blocking under T1, the *contract* does not | — |

Validation performed is host-free and covers every queue: the permutation oracle
plus a compile-and-run check of the emitted C against the Python reference for
every block id. Missing exact-device evidence is per-queue and named there.

## Persistent attention LSE checkpoint (2026-07-27)

Cross-backend sync `LSE-CHECKPOINT-CONTRACT-2026-07-27` closes the broken
destination-less `lse.save/load` model. The ops now carry explicit memref
source/destination, SSA row offset, identity, global-memory space, lifetime
scope, optional cache policy, and `MemRead`/`MemWrite`; inference lowering
emits neither op without a real destination. ROCm consumes the contract through
selectable saved/recompute five-entry packages. Exact gfx1151 FP16/BF16
host-wall sweeps select saved LSE at 128+ rows, where both tested long shapes
win cross-dtype; shorter results are mixed and retain recompute. NVIDIA and Apple
remain architecture-owned follow-ups. Full contract and evidence:
[`LSE_CHECKPOINT_CONTRACT.md`](LSE_CHECKPOINT_CONTRACT.md).

> **Reconciliation note (updated 2026-07-25).** The per-IR scorecard and phased plan below
> are a 2026-06-15 point-in-time snapshot; several of their "dispatcher / stub"
> cells have since moved. Two in particular are now stale and are superseded by the
> **Finished — middle-end / arbiter wave (2026-07-08)** subsection and the backend
> audits: the **Autotuner** row (measured autotune + fleet corpus is now wired onto
> ROCm gfx1151, #308 — no longer `_mock_latency`-only) and the **Target IR /
> runtime** row (the "~87% of ops execute via the numpy reference interpreter"
> figure predates the ROCm/x86 native-lane wave and the 2026-07-09 Apple GPU
> op-family parity closure — read the generated `runtime_execution_matrix.md` for
> the live split, never the prose figure). The scorecard is kept for its strategic
> framing; treat the generated dashboards as status truth (Decision #26).
> PR #457 also supersedes the old C1–C3 claims that WarpSpec had no mbarriers,
> that structured layouts were unattached, and that pipeline state was
> annotation-only. The reconciled C1–C3 records below distinguish the landed
> SSA/typed structure from the remaining sibling-consumer and exact-device work.

> **Latest deep pass:** [DEEP_COMPILER_AUDIT_2026_06_10.md](archive/DEEP_COMPILER_AUDIT_2026_06_10.md)
> — source-backed audit of frontend/IR/manifest/runtime-ABI/Apple-envelope/
> benchmark coverage. Records the "generated drift clean vs semantic gap open"
> split, fixes the bench-axis staleness + the grouped_gemm/moe_swiglu_block
> manifest blind spot, and carries a prioritized gap table for the rest.
>
> **Code-level companion:** [CODE_AUDIT_2026_06_10.md](archive/CODE_AUDIT_2026_06_10.md)
> — refactoring / per-IR-level optimization correctness / glass jaws. Headline:
> a verified `TransposeIntoMatmul` flag-composition miscompile (fixed, commit
> `acb5c6f`), missing fusion use-guards (fixed), NSA gating-semantics hazard
> (guarded), silent autodiff chain breaks (diagnosed), no upstream
> canonicalizer/CSE in named pipelines (fixed), `TESSERA_STRICT_DISPATCH`
> against silent numpy fallbacks, and runtime consumption of `fusion_groups`.
> Two earlier agent claims refuted.
>
> **Evaluator program — substantially shipped (2026-06-12):** [EVALUATOR_PLAN.md](EVALUATOR_PLAN.md)
> (see its §9.5 "what has landed"). A generative, execution-derived,
> **backend-rung-aware** Evaluator that *derives* conformance/benchmark/autotune
> surfaces from one honest scoring engine (closing the "registry models reality"
> gap). **Landed:** the 8-rung verdict engine + provenance gate
> (`evaluator.py`); four oracles — vertical, horizontal/PolyJuice, metamorphic,
> and DESIL cross-path (`cross_path_equivalence`); conformance corroboration
> (`conformance_evaluator.py`); the autotuning flywheel + per-chip calibration +
> autotune_v2 bridge (`flywheel.py`, `flywheel_autotune.py`); and the scored
> environment — a TensorBench-style grader (`compiler_grader.py`), LongCA
> structure-keyed attention (`attention_tasks.py`), and Magellan/AlphaEvolve
> gated search (`magellan.py`, `alphaevolve.py`, with reward-hack rejection
> proven). **Open (hardware-gated):** NVIDIA/ROCm sit truthfully at rung 1–2.5
> (WGMMA PTX *emitted* via `ptx_emit.py`; rung-3 `ptxas` + complete kernel +
> silicon need a Linux/CUDA runner). Research-backed
> (DESIL/PolyJuice/Mirage/TensorBench/Magellan/AlphaEvolve/BaCO/TLP).

## Library → Optimizing Compiler: front-to-back closure plan (2026-06-15)

A front-to-back audit of every IR level framed by one question: *where is Tessera
still a library/dispatcher, and what does each layer need to become an optimizing
compiler?* This section is the strategic spine; the per-item status lives in
**Still Open** / **Next Work** below (cross-referenced, not duplicated).

### The central finding — two disconnected worlds, one half-closed seam

The executed path and the C++ MLIR optimizer are largely **disconnected**:

- `tessera-opt`'s optimized IR is run for **validation only** — its stdout is
  hashed for provenance and discarded (`driver.py` `_try_validate_with_tessera_opt`,
  ~`:955/:970`). Execution dispatches off the **in-memory Python `GraphIRModule`**
  (`jit.py` `recognize(self.graph_ir)`), so the C++ fusion/canonicalize/CSE passes
  do not reach runtime.
- Consequence: fusion logic exists **twice** — real C++ rewrites the executor
  ignores (`SwigluFusion`/`MLAFusion`) *and* the Python path's own derivation.
  The seam is **half-closed already**: `canonical_compile._derive_fusion_groups`
  carries `fusion_groups`, the executor reads them (`runtime.py` ~`:2343`), and
  `stamp_fusion_intents` stamps `tessera.fusion.intent` for 4 chains (see the
  "Fusion intent is too late" item). But it is **advisory** — every dispatch
  branch is still `if fused_kernel=="X" OR _structural_rematcher(ops)`. Closing
  the seam = promote advisory → **authoritative** and delete the re-matchers.

Two facts make the transition cheaper than it looks:

1. **The seam mechanism half-exists** (above) — Phase 0 finishes it, it doesn't
   invent it.
2. **The canonical Tile GEMM K-reduction is now landing.** The
   `tessera_jit`/linalg lane already executes a scalarized K loop
   (`tessera.matmul → linalg.matmul → linalg-to-loops`), but it cannot represent
   target-neutral fragment tiling, asynchronous dependencies, or pipeline state.
   The former M/N-only `TilingPass` is therefore being replaced by an explicit
   M/N/K `scf.for` contract with a loop-carried FP32/INT32 accumulator, zero-pad
   tail guards, structured layouts, and threaded `!tile.pipeline_state`.
   Linalg remains the CPU scalar/vector route; GPU backends consume the shared
   Tile reduction rather than independently rediscovering its semantics.

### Per-IR scorecard (what's real vs. dispatcher)

| IR level | Real today | Dispatcher / stub | Primary gap to close |
|---|---|---|---|
| **Python `@jit`** | Decoration-time constraint + effect analysis; honest fallback gating (won't let eager Python masquerade as compiled). | Effect/constraint analysis is single-function, AST-only. A general IR-optimization step (folders/effects) between emission and execution is still thin. | Component-aware multi-op metadata **landed** (carried to the `@jit` artifact); fusion dispatch is **authoritative** (Phase 0 seam closed). Remaining: effect interfaces + broader folding. |
| **Graph IR** | 132 ops, 107 real verifiers; 5 canon patterns; real fusion passes (SwiGLU/MLA/NSA). 101/109 ops are `[Pure]` (CSE/DCE-eligible *today*). | **Folders/canonicalizers landed (2026-06-22):** `add`/`sub`/`mul`/`div`/`cast`/`reshape` folders + `matmul`/`transpose`/`reshape` canonicalizers (8 ops — `reshape` carries an identity fold + a `reshape(reshape(x))` chain-collapse, both guarding the optional `dim_names` symbolic-dim annotations), wired into the `tessera_jit` CPU `canonicalize→cse` pipeline (`graph_ir_folders.mlir`); `LayoutAssignmentPass` landed (seed→propagate→insert `cast{layout}`, `test_layout_assignment.py`). **Per-op effect interfaces landed (2026-06-22):** all 23 non-pure ops carry an explicit `MemoryEffectsOpInterface` — deterministic value ops (`adam`/`adamw`/`momentum`/`adafactor`/`lion`, `arch.ste_one_hot`/`weighted_sum`/`switch`/`mixed`) are `[Pure]`; random (`dropout`/`arch.gumbel_softmax`/`arch.hard_concrete`), stateful (`kv_cache.*`/`ring.create`/`arch.parameter`), collective (`all_reduce`/`reduce_scatter`/`all_gather`) and MoE-transport ops carry `MemWrite`/`MemRead`, so generic CSE/DCE is sound and precise across them (`graph_ir_op_effects.mlir`). `LayoutAssignmentPass` is wired into the named x86/GPU/CUDA-13 pipelines. x86 and NVIDIA default it on because their Graph-cast materializers consume the markers immediately; Apple and ROCm retain architecture-owned opt-in boundaries until their complete physical consumer envelopes are proven. **Phase 1 closed (2026-06-22)** — effect interfaces, target-scoped LayoutAssignment wiring, and reshape folder coverage all landed. ~5 passes are attribute-stamp-only. | Add folders opportunistically as new algebraic identities surface; ~5 attribute-stamp-only passes could gain real bodies. |
| **Schedule IR** | DistributionLowering performs structural wiring and collective insertion. `PipelineStagePartition` emits `tessera.pp_stage`; insertion rewires send/recv SSA; `PipelineScheduleLegality` proves the contract and materializes unique-clock `tessera.pipeline_steps` across warmup, steady, and cooldown. The shared runtime executes those steps in compiler order, overlaps selected backward collectives on an independent transport executor, joins them before completion, and binds typed collective descriptors to explicit OptimizerShard replicated/rank-local ownership transitions. All four descriptors also survive Tile into the registered asynchronous `tessera_collective` Target queue and its content-addressed runtime package. | The portable Target/runtime path is proven with deterministic adapters, but no backend has supplied a real multi-rank CUDA, ROCm, Metal, or x86 process packet. Collective placement and overlap remain runtime machinery rather than a complete middle-end optimization pass. | Land architecture-owned multi-rank NCCL/RCCL/Metal/x86 process execution and exact-device evidence; then promote placement/overlap policy into the middle end. |
| **Tile IR (FA-4)** | `#tile.layout` is attached to real views/copies/fragments; SSA buffer, pipeline, TMA, mbarrier, TMEM, and TCGen05 vocabulary is registered. Canonical GEMM has explicit M/N/K loops; its ROCm consumer now requires planned buffer/token/pipeline-state SSA ownership and has exact gfx1151 register/LDS comparison proof. FlashAttention has explicit rank-4 batch/query-head distribution into a KV `scf.for` carrying `(acc,m,l,producer,consumer,boundary)` with typed slice coordinates and ragged extents. ROCm consumes that shared forward recurrence and its deterministic tensor-valued backward workspace/reduction loops directly. NVIDIA, shared legality fixtures, and ROCm LDS consumers are free of name-based buffer identity. | The direct NVIDIA execution consumer of the shared attention loop remains open. Deprecated `#tile.buffer_ref` is parser-visible only for migration diagnostics and archived IR. SM100 TCGen05/TMEM has structural proof only. | Finish the direct NVIDIA distributed-attention consumer; exact SM100 TCGen05/TMEM. |
| **Autotuner** | `BayesianAutotuner` tunes `{tile_m/n/k, num_warps, num_stages}`. Target/evidence/latency-valid measured schedule records change the actual Schedule IR and Tile IR M/N/K, warp-count, and pipeline-depth attributes. Hardware-free pruning uses a symbolic tile-reuse/capacity model with explicit compute, DRAM-bandwidth, cache, dtype-width, and raster-order inputs; measured rows always outrank analytical rows and cache reuse is exact to shape/dtype/target/layout/movement. | T1 is GEMM-only and intentionally cannot distinguish warp/stage choices; T3 action-DAG ordering, T4 overlap structure, production target-cache semantics (especially Apple SLC and x86 hierarchy), and per-target correlation/selector packets remain incomplete. | Correlate T1 against each architecture's committed corpus and retain or reject it per backend. Keep exact-device latency authoritative; do not coefficient-tune a failed model. |
| **Target IR / runtime** | x86 AMX/AVX-512, Apple GPU, NVIDIA `sm_120`, and ROCm `gfx1151` all have checked-in native execution rows. The generated runtime matrix currently records 24 NVIDIA and 69 ROCm rows; the E2E fleet additionally seals release packets for NVIDIA softmax/reduction and ROCm softmax/reduction/paged-KV/MoE. `fusion.py` remains real runtime MSL codegen for matmul-epilogue regions, and `tessera_jit` is a real MLIR→LLVM CPU JIT. | Native breadth is architecture-specific, not backend-wide: NVIDIA `sm_80`/`sm_90`/`sm_100` and the ROCm target-map tail remain artifact-only or exact-device gated. Apple still has a name→lane→ctypes dispatcher seam, and reference execution remains explicit for unsupported rows. | Close the dispatcher seam and promote the remaining target-map tails only with exact-device execute-and-compare evidence. |

### The phased plan

`HF` = hardware-free (lands on this Mac); `HG` = hardware-gated. Phase 0 is the
keystone; once it lands, Phases 1/2/4 largely parallelize. Everything through
Phase 4 is HF; only GPU launch + silicon-perf is gated.

- **Phase 0 — Close the seam (keystone, HF).** Finish the half-built carry-intent
  mechanism. **(a) Landed (2026-06-15)** — each `known_chain` fusion group now
  carries a `dispatch` roles sub-dict (`a`/`b`/`c`/`x`/`wg`/`wu`/`wd`/`out` +
  scalar `eps`), resolved once from Graph-IR operand order in
  `canonical_compile._chain_dispatch_roles` — killing the value-shape guessing the
  re-matchers do inline. Strictly additive (a group carries no `dispatch` when
  roles don't resolve, so the executor path is unchanged); JSON round-trips into
  `fn.runtime_artifact().metadata`. Guard: `tests/unit/test_fusion_dispatch_roles.py`
  (5). **(b) Landed (2026-06-15)** — `_execute_apple_gpu_mps_metadata` now resolves
  a whole-program authoritative plan (`_apple_gpu_resolve_authoritative_plan`,
  reading both `fusion_groups` and the `canonical_fusion_groups` the `@jit`
  artifact actually stamps) and dispatches off the carried roles via
  `_APPLE_GPU_FUSION_DISPATCH` — no per-invoke re-matching, no value-shape
  guessing. Falls through to the structural cascade only when roles don't resolve
  (legacy safety). This surfaced a latent gap: the executor read a bare
  `fusion_groups` key the real artifact never set (it sets
  `canonical_fusion_groups`), so the re-matchers — not the carried intent — were
  what actually fired in production; the authoritative path closes that.
  **(c) Landed (2026-06-15)** — proved authoritative ≡ re-matcher (horizontal
  oracle, `tests/unit/test_fusion_authoritative_dispatch.py`, 12) then **deleted**
  the four `_apple_gpu_metadata_is_*_chain` re-matchers. Closed the one subsumption
  gap first (`matmul→rmsnorm_safe` is now a known_chain so authoritative dispatch
  covers it). The `fused_kernel == "X"` branches remain only for bare-`fusion_groups`
  metadata (hand-built / pre-`dispatch` legacy); truly-legacy no-metadata artifacts
  now run correctly per-op instead of via a re-discovered fuse. Full apple_gpu +
  canonical + fusion sweep: 2255 passed / 0 failed. **Seam closed — one fusion
  recognizer (the compiler), carried across to the executor.** Extends **Still
  Open → "Fusion intent is too late"** and **Next Work #3**.
- **Phase 1 — Make carried IR worth carrying (Graph-IR quality, HF, parallel).**
  **First increment landed (2026-06-15), observable end-to-end on the executed CPU
  JIT lane.** The tessera_jit pipeline had **no canonicalizer**; added
  `createCanonicalizerPass()` + `createCSEPass()` to `pm1` *before* `TesseraToLinalg`
  (`tools/tessera-jit/tessera_jit.cpp`), so Tessera per-op folders now bite on the
  executed path. Shipped the first two folders: `TransposeOp::getCanonicalizationPatterns`
  (`transpose(transpose(x)) → x`, a no-perm transpose is its own inverse) and
  `CastOp::fold` (identity `cast(x): T→T → x`, only when no `numeric_policy`), via
  `hasCanonicalizer`/`hasFolder` in `TesseraOps.td` + bodies in `TesseraOps.cpp`.
  Proven end-to-end: `@jit(target="cpu")` `transpose(transpose(x))` folds to
  `return %arg0` in the JIT trace and returns `x` exactly. Also registered the
  upstream `canonicalize`/`cse` passes in `tessera-opt` (`registerTransformsPasses`)
  so folders are lit-inspectable. Guards: lit `tests/tessera-ir/phase2/graph_ir_folders.mlir`
  (folders + negative cases + **DCE** of a dead pure op + **CSE** of duplicate
  matmuls — the shared-QKV-projection pattern) + `tests/unit/test_native_cpu_jit.py`.
  **CSE + DCE verified firing end-to-end on the executed CPU JIT path** (duplicate
  `matmul` → 1; dead `gelu` → eliminated; confirmed in the JIT trace) — these, not
  the rare algebraic folds, are the high-value Phase 1 wins, and they are now live.
  **Identity folders landed (2026-06-16):** `AddOp`/`SubOp`/`MulOp`/`DivOp` now
  have `hasFolder` + `fold()` bodies in `TesseraOps.cpp` — `x+0`/`0+x`/`x-0`/`x*1`/
  `1*x`/`x/1` collapse to the surviving operand when the other is a constant splat
  of the scalar identity (type-equality-guarded; no-signed-zeros, matching the
  fast-math GEMM model). Guard: 7 new cases (folds + negatives) in
  `tests/tessera-ir/phase2/graph_ir_folders.mlir` (18 total FileCheck'd). `matmul·I`
  deferred (needs identity-matrix recognition; never appears in real graphs).
  **Effect-interface item assessed + closed:** the genuinely non-pure ops
  (`dropout`=random, the `all_reduce`/`reduce_scatter`/`all_gather` collectives,
  `kv_cache_*` writes, the `adam`/`adamw`/`momentum`/… optimizer in-place updates)
  are **already conservatively sound** under MLIR's unknown-effects model (no `Pure`
  ⇒ never CSE'd, never DCE'd); the FFT/Clifford families that *look* non-pure
  actually inherit `[Pure]` from their base classes. Adding explicit
  `MemoryEffectsOpInterface` yields no practical CSE/DCE win (writes neither CSE
  nor DCE in MLIR's model) and risks subtle reordering bugs — so the current
  treatment is the right one. **Graph-IR folder tail closed (2026-06-17):** of the
  5 `CanonicalizeTesseraIR` patterns, only 2 were CPU-JIT-lowerable. `TransposeIntoMatmul`
  is now also a per-op hook — `MatmulOp::getCanonicalizationPatterns` (the exact
  proven XOR flag-composition: `transpose(Aᵀ)=A`) — so the transpose→flag fold fires
  under the generic `--canonicalize` the tessera_jit CPU lane runs, reaching the
  executed path (proven by `tests/tessera-ir/phase2/graph_ir_folders.mlir` +
  `test_native_cpu_jit.py::test_transpose_into_matmul_folds_on_executed_path`). The
  original stays in `CanonicalizeTesseraIR` for the custom-pass pipelines
  (zero-regression). `EraseIdentityCast` was already covered by `CastOp::fold`. The
  remaining 3 (`FuseMatmulBiasGELU`/`FuseConvRelu`/`DropoutZeroSimplify`) are
  deliberately NOT migrated — they emit `fused_epilogue`/`conv2d_nhwc`/`flash_attn`
  the rank-2 CPU JIT can't lower. **Layout-cast guard landed (2026-06-17):** the
  latent finding that `EraseIdentityCast` (in tessera-canonicalize) and
  `CastOp::fold` (generic --canonicalize) erased a same-type `cast{layout}` before
  the legality check / codegen saw it is **fixed** — both now skip a same-type
  cast carrying a `tessera.layout` attribute (a layout-change marker, not dead
  weight), while plain identity casts still fold. This is the prerequisite for
  `LayoutAssignmentPass` (which inserts same-type `cast{layout}` markers). Guard:
  the `@layout_cast_survives` case in `graph_ir_folders.mlir`.
  **LayoutAssignmentPass v1 landed (2026-06-17):** the assignment half of the
  layout contract (`src/transforms/lib/LayoutAssignmentPass.cpp`,
  `--tessera-layout-assignment`) — (1) seed kernel-producer layouts
  (matmul/batched_gemm→row_major, flash_attn→bhsd, conv2d_nhwc→nhwc), (2) propagate
  through single-result pointwise ops to a fixpoint, (3) insert
  `tessera.cast{tessera.layout=…}` markers at consumer accept-set boundaries (the
  same-type markers the 2026-06-17 cast-fold guard preserves). Paired with
  LayoutLegalityPass as its verifier — `tests/tessera-ir/phase2/layout_assignment.mlir`
  proves the assignment output runs clean through `--tessera-layout-legality`
  (assign + verify). Guards: that lit fixture + `tests/unit/test_layout_assignment.py`.
  *Executable-layout slice (CORE-COMPILER-2, 2026-07-22):* the generic x86
  emitter now publishes an operand binding/order/rank contract and the runner
  materializes A/B into that physical order before launch. The contract is part
  of the kernel-cache identity, so two physical layouts cannot alias one
  executable. ROCm also consumes structured `#tile.layout` in its owned Tile
  lowering. The Graph-level `tessera.cast{layout}` insertion remains opt-in:
  Apple and NVIDIA still need architecture-owned reorder/materialization routes
  before that shared marker may become default everywhere.
  *Flash-attn streaming is NOT a CPU-lane item* — the CPU JIT is a rank-2 simple-op
  lane; flash-attn (rank-4, batched) belongs to the Apple GPU work, where the
  streaming online-softmax kernel already exists as hand-written MSL
  (`kFlashAttnF32Source`). Extends **Still Open → "Layout and binding contracts"**.
- **Phase 2 — Real codegen in the executed path (linalg spine, HF).** **GEMM-lane
  convergence achieved + proven on the executed CPU JIT lane (2026-06-15).** Phase 4
  built the CPU JIT matmul on `linalg.matmul`, so the executed GEMM already lowers
  `tessera.matmul → linalg.fill + linalg.matmul → ConvertLinalgToLoops` into a real
  **M×N×K** loop nest with the K-reduction inner loop (`scf.for` over K + `mulf`/`addf`
  accumulate) — verified in the JIT trace and guarded by an exactly-representable
  GEMM equivalence test. **Canonical Tile K-loop continuation (2026-07-25):**
  `TilingPass` now independently represents the GPU-relevant M/N/K reduction
  with a loop-carried FP32/INT32 accumulator, ragged zero padding, structured
  layouts, async dependencies, and SSA pipeline state. This intentionally does
  not collapse into the scalar linalg loop because Tile must retain fragment
  and pipeline semantics; the two routes share mathematical/numerical policy
  rather than physical loop form. *Remaining Phase 2:*
  flash-attn streaming (wrap the attn ops in `scf.for` over KV with `(m,l,acc)`
  iter_args — `OnlineSoftmaxOp` ODS is already iter_args-shaped; only the loop
  wrapper + `kv_offset` threading is missing — **but note the executed Apple GPU
  path already streams** via the hand-written MSL `flash_attn_f32` online-softmax
  kernel, so this gap is the C++ Tile-IR validation lane + the NVIDIA emitter, not
  Apple execution).
  **Synthesizer generalization — A→B→C→D landed (2026-06-17).** **(A, keystone)**
  `fusion.py` gained `verify_synthesized_pointwise` — the F4 codegen oracle the
  pointwise-DAG path was *missing* (it was the only synthesizer region kind with
  no correctness gate; region/gated/attention all had one). The apple_gpu executor
  now gates the pointwise dispatch branch on it, so a divergent synthesizer falls
  back to the per-op MPSGraph lane instead of being trusted. **(B, measure —
  corrected the plan)** new `compiler/apple_gpu_coverage.py` + guard classifies
  every catalog op against the authoritative lane table: of **302 ops, 177 have a
  GPU lane, 125 are numpy-only, and 0 of those are elementwise/pointwise** — i.e.
  single-op elementwise displacement is *already complete*; the numpy tail is
  layout/indexing/quantize/linalg/spectral/complex. This refuted the original
  Phase-C assumption ("displace elementwise single-ops"). **(C, guided by B)** the
  real lever is enlarging fusable *DAGs*: added `sqrt`/`rsqrt`/`log`/`log1p`/
  `expm1`/`reciprocal`/`softplus` to `POINTWISE_OPS` (they already had single-op
  lanes, so DAGs containing them used to bail at those nodes — now they fuse into
  one kernel, a dispatch-count win), each auto-gated by the (A) oracle
  (`equal_nan`-aware for the domain-restricted ops). **(D, lock)** fused-DAG cases
  added to the differential harness (`_diff_lane.numeric_cases`).
  **Close-out follow-ups landed (2026-06-17):** **(C1 tail)** `maximum`/`minimum`/
  `sign` added to `POINTWISE_OPS`. **(C2)** closed by decision — `EPILOGUE_OPS` is
  deliberately *not* grown beyond the hot matmul-epilogue activations
  (bias/relu/gelu/silu/sigmoid/tanh); rarer activations ride the general
  pointwise-DAG path as a separate on-GPU dispatch, so further in-matmul-epilogue
  entries would be speculative (rationale in the `EPILOGUE_OPS` docstring).
  **(B1 runtime half)** `apple_gpu_coverage.fallback_histogram(run_fn)` runs a
  model under `@jit(apple_gpu)` and reports the failure-class fallbacks
  (shape/dtype/Metal-failure reasons + frequency) from
  `runtime.dispatch_fallback_log` — the runtime complement to the static no-lane
  worklist. **(D2)** the real no-silent-rot regression lock landed: a
  representative pre-norm decoder-MLP block (rmsnorm→matmul→silu→matmul→residual)
  runs on apple_gpu and asserts an **empty** fallback histogram (Darwin-gated);
  a kernel that quietly degrades to numpy trips it. **Parameterized-unary
  follow-up landed (2026-06-17):** `softcap` (the Gemma logit soft-cap
  `cap*tanh(x/cap)`) was the one genuinely numpy-only *real-valued* elementwise
  op. It carries a scalar `cap`, so it rides a GPU **compose** lane (div-by-scalar
  → tanh unary → mul-by-scalar — the clamp/where pattern, no dedicated kernel, no
  `.mm` change) rather than a pointwise-vocab entry. Made a first-class runtime op
  (`_APPLE_GPU_SOFTCAP_OPS` in the master envelope set + `"softcap"` lane +
  handler), which required regenerating the `apple_runtime_ops.inc` X-macro the
  C++ Tile→Apple pass `#include`s and rebuilding `tessera-opt` (the C++/Python
  single-source enforcer + `.inc` drift gate both pass). `cap` is a config literal
  in the jitted source in practice; closure-captured scalars are an unresolved SSA
  ref the apple_gpu metadata path doesn't fold (a known frontend limit, not
  softcap-specific) and the handler fails loudly rather than silently wrong.
  `clamp`/`clip`/`where` were already on GPU compose lanes. Guards:
  `tests/unit/test_apple_gpu_softcap.py`. Remaining: no parameterized-elementwise
  numpy-lane ops left — the displacement worklist's real-valued elementwise tail
  is closed. Guards (prior phases):
  `tests/unit/test_fusion_pointwise_oracle.py`, `test_apple_gpu_coverage.py`,
  `test_fusion_pointwise_vocab_phase_c.py`,
  `test_apple_gpu_displacement_regression.py`.
  **Non-elementwise tail — investigated + categorized (2026-06-17).** The naive
  "displace the 124 numpy-only ops" framing is mostly wrong (the same lesson as
  the elementwise + P2 findings). Investigation: `optim.adam` runs host-side on
  pytrees of numpy (a training-loop utility, never emitted as a single `@jit`
  graph op), and `matmul→transpose→gelu` *demotes to `artifact_only`* because a
  structural op mid-program isn't a recognized chain. So
  `apple_gpu_coverage.disposition_for` now classifies the numpy-only tail:
  **51 `real_gap_structural`** (layout/indexing/state/dropout/position-encoding —
  the genuine target: ops that demote an otherwise-GPU program off
  `metal_runtime`), **50 `hard_kernel`** (quantize packed-FP4/6/8, sparse,
  spectral, stencil, linalg, complex-elementwise, sort/einsum), **8 `host_utility`**
  (optimizers + RNG — no GPU gap), **6 `distributed`** (collectives + MoE
  transport), **9 `unclassified`** (per-op judgment). Guard:
  `test_apple_gpu_coverage.py::test_displacement_disposition_classifies_the_real_gap`.
  *The real displacement target is the 51 structural ops, not 124.*
  **First structural displacement landed — transpose (2026-06-17).**
  `tessera.transpose` now runs on a real MPSGraph kernel
  (`transposeTensor:permutation:`, SDK-header-grounded per Decision #27): N-D
  permute, value-preserving, f32 native + f16/bf16 on the 2-byte raw path, host
  fallback for non-Darwin / GPU-miss. New `.mm` `mpsg_run_transpose` +
  `tessera_apple_gpu_mpsgraph_transpose_{f32,f16}` symbols + stub parity;
  first-class runtime op (`_APPLE_GPU_TRANSPOSE_OPS` → `"transpose"` lane +
  `_apple_gpu_dispatch_transpose`); `.inc` regenerated + `tessera-opt` rebuilt
  (C++ enforcer + drift gate pass). A single-op `@jit(apple_gpu)` transpose now
  reports `execution_kind="native_gpu"` / driver `execution_mode="metal_runtime"`
  (was `fallback_eager`). Guards:
  `tests/unit/test_apple_gpu_transpose.py` (7: 2D/3D/4D + explicit permute, f16,
  jit, no-fallback-on-Metal).
  **General residency gate landed — `per_op_metal` (2026-06-17).**
  `_apple_gpu_chain_kind` now returns `"per_op_metal"` for any multi-op program
  where *every* op has a GPU lane (`lane_for(op) is not None`), checked LAST so the
  named fused chains still win. This closes the transpose-mid-program caveat:
  `matmul→transpose→gelu` (and `matmul→add→transpose→silu`) now run `native_gpu` /
  `metal_runtime` per-op (each op on its lane; the fusion prepass still fuses
  sub-chains) instead of demoting the whole program to `artifact_only`. Conservative
  by construction — a program with any non-lane op returns `None` (stays
  `artifact_only`); per-op handlers still fall back individually (recorded), so the
  program claim stays honest. Guards: `tests/unit/test_apple_gpu_per_op_metal.py`
  (recognizer accepts all-GPU-capable; named fusion still wins; non-GPU op stays
  conservative; mixed program runs `native_gpu` + no-fallback-on-Metal). Updated
  the two Phase-8.4 "multi-op = artifact_only" roadmap gate tests to the new
  contract (all-GPU-capable → `metal_runtime` + numpy-proven; non-lane op →
  artifact_only). *Representation gap (tracked):* the runtime *contract* is
  correctly `metal_runtime` (metadata + verified execution), but the `.target_ir`
  artifact-projection string still uses the per-op-contract / `metal_artifact`
  format for multi-op programs — routing per_op_metal through the runtime-pipeline
  target-IR text is a cosmetic follow-on, orthogonal to the (correct) residency
  claim. **Gather landed (2026-06-17) — second data-mover.** `tessera.gather` now
  runs on a real MPSGraph kernel (`gatherWithUpdatesTensor:axis:0`, header-grounded):
  embedding / attention-index row gather of a 2D table by int32 indices (v1
  envelope: axis-0 + 2D table; other axes / N-D tables fall back to `np.take`).
  Negative indices are normalized before the GPU call so the Metal path matches
  numpy. New `.mm` `mpsg_run_gather` + `tessera_apple_gpu_mpsgraph_gather_{f32,f16}`
  + host fallback + stub parity; first-class runtime op (`_APPLE_GPU_GATHER_OPS` →
  `"gather"` lane). It immediately compounds on the residency gate — an embedding
  lookup mid-program (`gather→matmul`) now runs `native_gpu` instead of demoting.
  Guards: `tests/unit/test_apple_gpu_gather.py` (handler vs numpy over 1D/N-D
  indices, negative indices, f16, jit, no-fallback-on-Metal).
  **Concat landed (2026-06-17) — third data-mover + a frontend-gap fix.**
  `tessera.cat` now runs on a real MPSGraph kernel (`concatTensors:dimension:`,
  header-grounded): the KV-cache-append data-mover — two operands stacked along
  one axis, flattened to an `(outer, axis, inner)` view so *any* rank/axis is one
  GPU concat along dim 1; value-preserving, f32 native + f16/bf16 on the 2-byte
  raw path. >2 operands or mixed dtypes fall back to `np.concatenate` inside the
  dispatcher. New `.mm` `mpsg_run_concat` + `tessera_apple_gpu_mpsgraph_concat_{f32,f16}`
  + host fallback + stub parity; first-class runtime op (`_APPLE_GPU_CONCAT_OPS` →
  `"concat"` lane + `_apple_gpu_dispatch_concat`). Unlike transpose/gather, cat
  was blocked in **both** frontend builders before it could reach a kernel: the
  AST `GraphIRBuilder` and the abstract-interp tracer each rejected a *list* of
  tensor operands (`cat([a, b], axis)` → empty body → `_trace_deferred` /
  "non-Tracer positional operand"), and the op-catalog declared cat/stack as
  fixed arity-1. Fixed generally (also unblocks `stack`): both builders now expand
  a list/tuple of defined tensor values into flat operands, cat/stack arity widened
  to variadic (1–64), and `_execute_op` re-packs the flattened operands for the CPU
  plan (`np.concatenate`/`np.stack`). A single-op `@jit(apple_gpu)` cat now reports
  `execution_kind="native_gpu"`; `matmul→cat` compounds on the per_op_metal gate
  (a KV append mid-program stays GPU-resident). Guards:
  `tests/unit/test_apple_gpu_concat.py` (handler vs numpy over axis 0/1/-1 + rank-3
  seq-axis + f16 + >2-operand fallback, jit native_gpu, matmul→cat per_op_metal,
  no-fallback-on-Metal).
  **Slice landed (2026-06-17) — fourth data-mover + the mirror frontend fix.**
  `tessera.slice` now runs on a real MPSGraph kernel (`sliceTensor:starts:ends:strides:`,
  header-grounded per Decision #27): the StableHLO dynamic-slice / KV-window data-
  mover — a static per-axis window `x[starts[i] : starts[i]+sizes[i]]` (stride 1)
  over an N-D input; `ends[i] = starts[i]+sizes[i]`, value-preserving, f32 native +
  f16/bf16 on the 2-byte raw path. Rank mismatch or out-of-bounds window falls back
  to numpy. New `.mm` `mpsg_run_slice` + `tessera_apple_gpu_mpsgraph_slice_{f32,f16}`
  + host fallback + stub parity; first-class runtime op (`_APPLE_GPU_SLICE_OPS` →
  `"slice"` lane + `_apple_gpu_dispatch_slice`). The frontend fix is the **mirror**
  of cat's: slice's two trailing positional args are index/size *lists of ints*
  (not tensors), so the AST `GraphIRBuilder` must bind them as **attributes**
  (`_POSITIONAL_ATTR_PARAMS["tessera.slice"] = ("start_indices","slice_sizes")`)
  rather than flatten them into operands — otherwise they dropped as `"%?"`
  operands and the op never reached a kernel (cat flattened a list-of-tensors *into*
  operands; slice binds a list-of-ints *out* of operands). A single-op
  `@jit(apple_gpu)` slice now reports `execution_kind="native_gpu"`; `matmul→slice`
  compounds on the per_op_metal gate (windowing a matmul output stays GPU-resident).
  Guards: `tests/unit/test_apple_gpu_slice.py` (handler vs numpy over 2D windows +
  rank-3 + f16 + out-of-bounds fallback, jit native_gpu, matmul→slice per_op_metal,
  no-fallback-on-Metal). *Still open:* the `norm_chain` broadening (bare norms
  already run on the MPSGraph rowop lane — no numpy there to displace, so
  deliberately deferred) — all Evaluator-gated, never displacing a working MPSGraph
  call. **The four structural data-movers (transpose, gather, concat, slice) are now
  all GPU-resident**, so the common KV-cache / embedding / reshape-window glue
  between matmuls no longer demotes a program off Metal.
- **Phase 3 — Close the optimizing loop (HF on Apple/CPU). ✅ landed (2026-06-16).**
  The synthesizer had a measured-latency, correctness-gated variant autotuner
  (`autotune_matmul_epilogue` — times each MSL variant on-device, gates each
  against the numpy reference, populates `_AUTOTUNE_CORPUS`) that was never
  auto-invoked, so `best_variant_for` always returned the static default. Closed
  the loop in `fusion.py`: `autotune_enabled()` (reads `TESSERA_AUTOTUNE`) +
  `select_variant(region, M, N, K, *, autotune=None)` — on a corpus miss with
  autotune on it measures + caches the measured-best variant, else it's an O(1)
  lookup. Wired into `runtime.py::_apple_gpu_try_synthesized_fusion` (replacing
  `best_variant_for`), so the executed Apple GPU lane runs the measured-best
  kernel. Latency is real (synthesizer dispatch timing); the roofline mock stays
  the honest NVIDIA/ROCm fallback. Guard: `tests/unit/test_autotune_loop.py` (5).
  *Superseded 2026-07-27:* the Schedule/Tile write path now validates measured
  records and stamps physical M/N/K, warp-count, and pipeline-depth attributes.
  Target-owned measurements and selector promotion remain hardware-gated.
- **Phase 4 — Grow `tessera_jit` toward default CPU (HF), then GPU spine.**
  **Brought forward (2026-06-15) — the keystone landed: the tessera_jit MLIR→LLVM
  lane is now the executed CPU path** for the covered f32 op set, so the C++ IR
  optimizations finally reach execution (closing the remaining seam for the CPU
  lane). `@jit(target="cpu")` now translates the executed `GraphIRModule` op-list
  into a whole-graph `GraphFn` (`_jit_boundary.run_graph_ops`) and runs it through
  `tessera_jit` (`tessera-to-linalg → one-shot-bufferize → linalg-to-loops → LLVM`,
  optLevel=2) **before** the numpy reference interpreter (`JitFn._try_tessera_jit_call`,
  tried in the CPU `__call__` branch). Covered set = `_JIT_GRAPH_OPS` (matmul,
  add/sub/mul/div, relu/sigmoid/tanh/silu/gelu, softmax, rmsnorm, layer_norm,
  transpose, select, masked_fill); anything else / non-f32 / unsupported rank falls
  back to numpy (correctness preserved — a fallback handles "couldn't run", never
  "ran wrong"). `TESSERA_DISABLE_CPU_JIT` is the kill-switch. Proof-of-execution via
  `_jit_boundary.invocation_count` (a silent numpy fallback can't masquerade).
  Guard: `tests/unit/test_native_cpu_jit.py` (per-op numpy equivalence + counter +
  fallbacks); 1929-test CPU/jit/ops sweep green. **Dtype breadth landed
  (2026-06-15), grounded in Apple M1 Max hardware:** the lane now routes **f32**,
  **f16** (native NEON, ARMv8.2-A FP16), and **bf16** (correct but emulated via f32
  in-kernel — M1 predates ARMv8.6 BFloat16), per-arg dtype detection in
  `_try_tessera_jit_call` (mixed dtypes → numpy fallback). matmul/reductions
  accumulate in f32 then truncate to storage (`TesseraToLinalgPass`, ABI §12.5 —
  already in the C++). Required adding `f16` to the `_jit_boundary` C-ABI dtype
  table (raw 16-bit at the boundary, like bf16) and making `_jit_unary` elem-aware
  (was f32-only while `_jit_binary` already used `_resolve_elem`). **f64 wired into
  the lane (2026-06-16)** — three contained table entries (`_elem_for` in `jit.py`,
  `_DTYPE_TABLE` + `_ELEM_TO_NP` in `_jit_boundary.py`; the C++ `TesseraToLinalgPass`
  and the whole tessera_jit LLVM pipeline were already f64-clean — `isa<FloatType>`
  includes f64 and the low-precision-→f32 accumulate rule never fires, so f64
  accumulates in f64 throughout). This is the **exact-precision lane** for
  gradient-checking / numerical validation (verified ~1.8e-15 GEMM error vs f32's
  ~1e-6). A lone rank-2 f64 GEMM still takes the numpy reference (the Accelerate
  `native_cpu` fast path is f32-only and numpy f64 matmul is already exact f64);
  multi-op f64 programs route through real f64 codegen. Guards:
  `test_f64_runs_through_jit_at_exact_precision` + `test_f64_gemm_is_exact_over_k`.
  **matmul perf — measured + diagnosed (2026-06-16).** The tessera_jit
  `linalg→loops→LLVM` GEMM runs at **~2.2 GFLOP/s** (256³/512³), **~50–110× off**
  numpy/Accelerate's 100–240 GFLOP/s — the `ConvertLinalgToLoops` body is naive
  scalar, un-tiled. Two cheap optimizer levers were tried and **measured
  insufficient**: (a) a host-detected `TargetMachine` into the transformer (was
  `nullptr` → no NEON cost model) + `-O3`; (b) stamping `fastmath<fast>` on the
  float arith ops after linalg→loops (a float reduction won't auto-vectorize
  without `reassoc`). Neither moved the GEMM (LLVM's loop vectorizer won't crack
  the reduction from this IR shape). **Both changes are kept** — they're correct
  (target-aware codegen; `fast` matches Tessera's documented fast-math GEMM
  contract) and prerequisites for vectorization — but the **real lever is an MLIR
  `linalg→vector` tiling+vectorization pipeline** (register-tile the matmul →
  `linalg::vectorize` → `vector→LLVM`), a focused multi-step effort.
  **`linalg→vector` GEMM lane ✅ LANDED (gated, 2026-06-16) — ~13× over scalar.**
  After two direct-`scf::tileUsingSCF` attempts null-derefed, the **transform
  interpreter** is the working path (it tiles the identical op cleanly under
  `mlir-opt`). The lane (`tools/tessera-jit/tessera_jit.cpp`, opt-in via
  `TESSERA_JIT_VECTORIZE`): run a parsed `transform.named_sequence`
  (`tile_using_for [8,16,16]` → `vectorize_children_and_apply_patterns`) via
  `transform::applyTransformNamedSequence` on the tensor-level IR before
  bufferization (so the K-reduction accumulates in a **register** iter_arg, not
  the memref accumulator that blocked LLVM's vectorizer); then post-bufferize lower
  the vectors (`reduction_to_contract` → contract `OuterProduct` → broadcast /
  shape_cast; **NOT** transfer→`vector.load`, which strands the strided-subview
  load) + `ExpandStridedMetadata` FIRST + `ConvertVectorToSCF` + `ConvertVectorToLLVM`
  + `UBToLLVM` (vectorize emits `ub.poison`); load MLIR's `libmlir_c_runner_utils`
  via `ExecutionEngineOptions.sharedLibPaths` so the DPS-copy `memrefCopy` symbol
  resolves. **Required registrations** (the hard-won set): `TilingInterface` on
  linalg+tensor, the linalg transform-dialect extension, the `vector`/`ub` dialects
  + vector bufferization models. **Result:** matmul programs with all dims ≤ 2048
  (`TESSERA_JIT_VECTORIZE_MAXDIM`, default raised 256→2048 on 2026-06-16) vectorize
  at **~40-46 GFLOP/s** (512³–1024³, ~30 at 128³) — ~13-20× the 2.3 GFLOP/s scalar
  — correct vs numpy; larger programs stay on the scalar JIT lane.
  **Large-N hardened (2026-06-16):** the earlier large-N failure was a *compile-time*
  explosion, not a crash — `vectorize_children` over-vectorized the **untiled**
  elementwise epilogue into a giant `vector<MxN>` that LLVM unrolled into M·N scalar
  ops. The transform now also tiles the 2D elementwise/fill/generic ops (`[8,16]`)
  before vectorizing the func, bounding every vector by the tile sizes; `MAXDIM` is
  now a compile-time safety valve (many tiles ⇒ long-but-finite compile), not a
  crash clamp. Default path (lane off) byte-identical; 25 CPU-JIT tests green incl.
  the gated-lane guard (now pins `MAXDIM=128` to exercise the scalar fallback).
  *Follow-ons:* tune tile sizes. Scope honesty: won't match hand-tuned Accelerate
  BLAS, and the
  **single-GEMM hot path already routes to Accelerate** (`_native_cpu_fast_call`);
  this lane targets multi-op programs that contain a small/medium GEMM.
  **GPU-emission spine landed (2026-06-17, HF).** `tessera-opt` now lowers a
  tessera kernel through `linalg → empty-tensor-to-alloc → one-shot-bufferize →
  convert-linalg-to-parallel-loops → gpu-map-parallel-loops →
  convert-parallel-loops-to-gpu → gpu-kernel-outlining →
  gpu.module(lower-affine, convert-gpu-to-nvvm)`, exposed as the
  `--tessera-emit-nvvm` pipeline. A `tessera.add` emits real NVVM — an outlined
  `gpu.module` with an `llvm.func` kernel (`nvvm.kernel`) reading
  `nvvm.read.ptx.sreg.ctaid.x` etc. Required registering the GPU dialect + the
  bufferization external models + the conversion passes in `tessera-opt` and
  linking the MLIR GPU/NVVM libs (Homebrew LLVM 23 ships `nvptx64` + the libs).
  Guards: `tests/tessera-ir/phase8/gpu_emit_nvvm.mlir` +
  `tests/unit/test_gpu_emit_nvvm.py`. **EMISSION ONLY** — the kernel is produced
  for inspection/codegen; the host `gpu.launch_func` stub remains and GPU launch
  (`tsrRegisterGpuLauncher` → `cuLaunchKernel`/`hipLaunchKernel`) is hardware-gated.
  **ROCDL emission landed (2026-06-17):** `--tessera-emit-rocdl` is the AMD twin
  of the NVVM lane (identical spine, `gpu.module(convert-gpu-to-rocdl)`); a
  `tessera.add` emits real ROCDL (`rocdl.kernel` + `rocdl.workgroup.id.x` + AMD
  data layout). Guard: the ROCDL RUN line in `gpu_emit_nvvm.mlir` +
  `test_gpu_emit_nvvm.py`. **PTX attempted + deferred (2026-06-17):** wired
  `nvvm-attach-target{chip=sm_90}` + `gpu-module-to-binary{format=isa}` (with
  NVPTX target init + the LLVM-IR translation interfaces) as `--tessera-emit-ptx`,
  but it **segfaults inside `mlir::gpu::transformGpuModulesToBinaries`** (the NVVM
  target serialization) on this macOS / Homebrew LLVM 23 build — likely a
  libdevice/toolkit lookup or an LLVM-23 serialization quirk even for `format=isa`.
  Reverted (won't ship a crashing pipeline); the NVVM/ROCDL emission is the proven
  layer. *Next on this thread:* debug the `gpu-module-to-binary` serialization
  (target options / toolkit path) — or chain `tessera-emit-nvvm` → isolate the
  `gpu.module` → `tessera-translate-mlir --mlir-to-llvmir` → `llc -mtriple=nvptx64`
  for PTX text; plus matmul/reduction GPU kernels (beyond elementwise) and the
  gated launch wiring.
- **Phase 5 — Schedule + pipelining (mixed).** Double-buffering structure (HF) /
  async overlap (HG); real 1F1B ordering (HF); collective↔compute overlap via the
  unused `ChunkPlanner`/`CollectiveScheduler` (plan HF, measurement HG); GPU MMA
  register accumulator (HG).

**Dependency spine:** Phase 0 is the keystone and is small because the mechanism
half-exists. Phases 1, 2, 4 parallelize after it. Through Phase 4 is entirely
hardware-free on this Mac; only the GPU launch + silicon-perf items are gated.
Detailed per-layer evidence (file:line) was captured in the 2026-06-15 deep-dive
agents and feeds the Still Open / Next Work items below.

**Status (2026-06-15):** Phase 0 (seam) **closed**; C++ Target IR consume-side
**reviewed + parity-guarded**; **Phase 4's keystone brought forward and landed** —
the tessera_jit MLIR→LLVM lane is the executed CPU path for the covered f32 op set.
This re-prioritizes Phases 1–2: with the C++ codegen lane now *executing*, the C++
Graph-IR folders/canonicalizers + effect interfaces (Phase 1) and the linalg
GEMM-convergence + flash-attn streaming (Phase 2) now **reach execution** through
the CPU JIT lane rather than only the discarded validation pipeline — so they are
worth building next, with measurable end-to-end impact. The immediate Phase-4
follow-ons (dtype breadth, wider `TesseraToLinalgPass` coverage, tiling before
LLVM) compound directly on this lane.

## Autodiff v1 tape — gaps closed (2026-06-13)

Surfaced while building the agent-native MoE training stack (`tessera.train`,
GRPO post-training). The `CODE_AUDIT_2026_06_10.md` already *diagnosed* "silent
autodiff chain breaks"; this pass found the **root cause and fixed it**, plus two
adjacent ergonomic gaps. All additive; full `tests/unit -m "not slow"` green.

- **Scalar/0-d tape-link break (root cause, fixed).** `autodiff/tape.py::_describe`
  keyed `np.generic` (scalar) inputs on `id(np.asarray(arg))` — a *fresh* array —
  while producers record outputs by `id(output)`. Any reduction-to-scalar feeding
  a later op (i.e. essentially every loss expression: `mul(reduce(...), k)`,
  `exp(reduce(...))`) silently severed the gradient chain (grad came back `None`).
  Fix: key on `id(arg)`. This is the concrete mechanism behind the previously
  "diagnosed" silent breaks.
- **`reduce(op=...)` was sum-only (fixed).** The op advertised an `op=` parameter
  but raised for anything but `"sum"` in both forward and VJP. Added `"mean"`
  (forward + `vjp_reduce`, axis/keepdims-correct). Max/min still route to
  `ops.amax`/`ops.amin` by design.
- **`clip` bound aliases (fixed).** `ops.clip` accepted only `min_val`/`max_val`;
  added `min`/`max` aliases coalesced in **both** the forward and `vjp_clip`, so
  PPO-style clipping is one tape-safe call (its bounds ride in kwargs, avoiding
  the scalar-operand detach below).

### Follow-on closures (2026-06-14): F1, F2, G1, G2

- **F1 + F2 (fixed — shared root cause).** `_make_wrapper`/`_describe` dropped
  *python-scalar* positional operands from the tape, so `ops.minimum(t, 1.2)`
  raised in backward (VJP missing `y`) and `ops.mul(scalar_tensor, -3.0)`
  returned grad as if the factor were `1`. Fix: `_describe` records python
  `int`/`float` (not `bool`) as **non-differentiable literal inputs**
  (`InputDesc.is_literal=True`), and `Tape.backward` tolerates a VJP that omits
  cotangents for *trailing literal* operands (pads `None`) — preserving the
  strict per-array count check that catches genuine VJP-author bugs. Verified
  safe across the full `tests/unit -m "not slow"` suite.
- **G1 (clarified + closed).** The earlier "no tape-traceable gather" claim was
  wrong: `ops.gather` already exists, is tape-wrapped, and scatter-adds
  correctly. The real gap was that `nn.Embedding`/models used raw numpy
  indexing. Added `ops.embedding(table, ids)` (gather + scatter-add VJP) and a
  fully-traceable LM proving the embedding table trains.
- **G2 (new op).** `ops.top_k` had no VJP for routing. Added
  `ops.top_k_routing(logits, *, k)` → full-width sparse-softmax gate (zero off
  the top-k) with a VJP that routes gradient to the selected logits via the
  sparse softmax jacobian (numerically verified vs central difference). This is
  the missing primitive for a **differentiable hard top-k MoE** — proven
  end-to-end in `tessera.train.models.TracedHardMoELM` (embedding + router +
  experts + head all train via `adamw_step`).

Guards: `tests/unit/test_autodiff_tape_fixes.py` (E1/E2/E3/F1/F2),
`tests/unit/test_train_hard_moe.py` (G1/G2). New ops registered as numpy
references (no OP_SPECS requirement — 11 registry-only refs already exist).

### Compute-sparse MoE dispatch (2026-06-14)

The differentiable hard-MoE above used a *dense soft-combine* (every expert
evaluated on every token, off-top-k contributions zeroed). Closed the deferred
follow-on: `tessera.train.engine.moe.sparse_moe_dispatch` does **real per-expert
routing** — each expert runs only on its routed tokens via `ops.gather` →
expert FFN → `ops.scatter_add`. Expert work drops from O(N·E) to O(N·k) while
the result is *numerically identical* to the dense combine (proven, atol=1e-5),
and the whole path stays tape-traceable (gradients reach embedding, router, and
experts; the data-dependent token-index sets are the non-differentiable runtime
part). Exposed as `TracedHardMoELM.logits(ids, dispatch="sparse")`. Guards in
`tests/unit/test_train_hard_moe.py` (parity, grad-flow, end-to-end training).

## Finished

### Driver + audit-generator hardening (2026-07-27)

A code review of `tools/tessera-opt/` and the TSOL coverage generator found two
classes of defect where a *build or generator* silently produced something that
looked correct. Both are now structurally prevented, not just patched.

**tessera-opt — the lean artifact driver could silently swallow whole backends.**
`tessera-opt.cpp` selected its stripped registration path by re-deriving intent
from `(ROCM || NVIDIA) && !CORE_TESSERA_IR`. CMake clears
`TESSERA_HAVE_CORE_TESSERA_IR` for any NVIDIA build without CUDA, so configuring
NVIDIA together with the Apple backend produced a binary that linked
`TesseraApple`, defined `TESSERA_HAVE_APPLE_BACKEND`, and compiled out every one
of its registration blocks — with no diagnostic. Same for Solvers, Neighbors,
TPP, scaling-resilience, and both FA-4 dialects.

- Leanness is now one named CMake intent (`TESSERA_OPT_LEAN_ARTIFACT_DRIVER`),
  and every optional capability registers itself in a feature ledger. Combining
  a lean driver with any feature outside
  `{core-tessera-ir, nvidia-backend, rocm-backend}` is a **configure error**
  naming the conflict and both ways to resolve it.
- `--tessera-build-info` (and the tool banner) report the build profile and
  feature list, so "which tessera-opt is this?" no longer requires diffing
  `--help` — the failure mode when a stale build directory shadows a current one.
- The Apple value-lane Tile IR envelope moved out of the driver into
  `tessera::apple::isValueLaneTileOp`, beside the lowering that consumes it. The
  driver had its own 21-name copy, so adding a value op to the backend meant
  editing the tool, and the rejection pointed at the wrong file.
- The `tessera-emit-{nvvm,rocdl}` aliases share one spine string, register only
  when the core IR is linked, and report a build failure through the registry's
  error handler instead of `report_fatal_error`. The convenience
  `PassPipelineRegistration<>` wrapper takes a `void` builder, which would have
  installed a silently **empty** pipeline on failure — a worse outcome than the
  abort it replaced (Decision #21).

**TSOL coverage — three stale claims the drift gate could not see.** The gate
compares the generated dashboard against `render_dashboard()`, so any constant
baked into the renderer is self-consistent and invisible to it. A "432-entry
registry" (actually 482), a `primitive_coverage.py line 351-352` citation
pointing at an unrelated table, and a hardcoded "zero" all survived that way.
All three are now derived from the registry, and `tests/unit/test_tsol_coverage.py`
gates the *class*: no hardcoded registry size, no source line-number citations,
and the backend-kernel aggregate must match `coverage_summary()`. The dashboard's
own regeneration instruction now names `generated_docs` (which writes both the
`.md` and the `.csv`) instead of a `python -c` snippet that left the CSV stale,
and `write_dashboard` pins UTF-8 for its ✅/◐/◯ glyphs.

**Test architecture — build-capability selection.** A full local sweep surfaced
**249 pre-existing failures** that say nothing about the code under test: 52 test
files resolve `build/tools/tessera-opt/tessera-opt` themselves and skip only when
the binary is *missing*. When it exists but was configured without the backend
under test — `TESSERA_BUILD_ROCM_BACKEND=OFF`, the default on this Mac — they
fail with `Unknown command line argument '--generate-rocm-…'`, which reads as a
broken test rather than a build-selection problem. (Confirmed pre-existing: that
`build/` tree is configured ROCm/NVIDIA `OFF`, registers zero `generate-rocm`
passes, and its binary predates this work.)

`tests/_support/compiler_tool.py` is now the shared capability-aware resolver —
it honours `TESSERA_OPT`, reads the binary's registered passes once, and skips
naming both the missing pass and the build profile. Nine files with an identical
helper body are migrated (32 failures → 0). The remaining ~43 have variant
bodies and need individual attention; tracked separately. `CompilerToolchain`
should ultimately delegate to this module so the tree has one resolver.

**Test architecture.** `CompilerToolchain.require_tessera_opt()` now takes the
pass names a test drives and skips with the binary's build profile when they are
absent. Previously a test needing an Apple pass picked up whichever binary was
found first and failed with `Unknown command line argument`, which reads as a
broken test rather than a build-selection problem. The pipeline drift gate also
learned the `registerPassPipeline` spelling; recognizing only the wrapper made it
report a live pipeline as stale.


### Middle-end / arbiter wave (2026-07-08, PRs #307–#314)

Landed against the COMPILER_REFACTOR_PLAN world-class dimensions (W-series) and
the A–E kernel spine — each PR test-gated:

- **A4 — shared cost-aware MMA selector (#309).** ROCm's MMA-shape cost model was
  lifted to a target-shared selector so all backends pick MMA shapes from one
  cost-aware model instead of per-backend heuristics.
- **G — dynamic-shape emitter for the generic lanes (#312).** One compiled kernel
  now serves all shapes (the `static | bucket | dynamic` policy from Decision #28),
  so dynamic shapes no longer force a per-shape recompile or an API break.
  **First guarded execution route (CORE-COMPILER-2, 2026-07-22):** the x86
  runner now requests `SpecPolicy.DYNAMIC`, passes M/N/K as runtime dimensions,
  and rejects non-rank-2, non-positive, contraction-mismatched, side-buffer-
  mismatched, or signed-i32-overflow shapes before native launch. Ragged valid
  shapes and non-contiguous inputs execute through the materialized row-major
  ABI; unsupported shape-specialized GPU routes remain bucketed.
- **H — Tile-buffer reuse + arena (W3, #311 + #314).** `TileBufferReusePass` does
  global shared-memory buffer assignment/reuse; `TileBufferArenaPass` realizes the
  reuse plan into a concrete SMEM/TMEM arena. Plus the **MatmulRegion transpose
  contract** (#310) — the backend now consumes the orientation flags.
- **J — roofline attainment (W7, #313).** % of peak is now the hot-path bar (the
  attainment metric the perf ratchet gates against), not raw latency alone.
- **D2 — measured autotune on real silicon (#308).** The measured-autotune lane +
  fleet corpus is wired onto ROCm gfx1151, so autotuning scores off measured device
  latency, not only the `_mock_latency` roofline model (this supersedes the
  Autotuner scorecard cell — see the reconciliation note at the top).
- **Solvers on the arbiter (#307).** The spectral FFT and TPP space-time passes are
  now real and retargeted onto the D1 measured arbiter.

- **Canonical driver:** `canonical_compile` and `CompileResult` are the common
  contract for compilation results.
- **Runtime handoff:** `@jit` and `runtime.launch()` consume canonical compile
  metadata rather than inventing a second path.
- **Capability gates:** legality, codegen, toolchain, link, runtime ABI,
  hardware smoke, and numerical gates report named failure axes.
- **Conformance matrix:** op-target proof is rendered in
  `../op_target_conformance.md` and drift-gated.
- **Schedule to Tile metadata:** mesh, layout, placement, artifacts, and related
  metadata survive lowering.
- **C++ pass honesty:** `LowerScheduleToTargetPass` stopped pretending to be an
  implemented lowering pass.
- **Tile to Apple parity:** C++ Apple status tags match the Python/runtime Apple
  envelope.
- **Dynamic control flow:** unsupported dynamic control flow now gets explicit
  diagnostics and fallback behavior.
- **Frontend bugs:** AugAssign sub/div lowering, ROCm sub-arch gates, and
  Darwin arm64 platform checks were fixed.
- **Compiler correctness tests:** pass-order and oracle fixtures cover string
  parsing, MLIR pass order, halo execution, CorrDiff IR visibility, spectral,
  linear attention, and Apple runtime pipeline order.
- **CSV-canonical generated dashboards + sprint regen (2026-06-04).**
  `runtime_abi` and `verifier_coverage` now emit a machine-readable CSV
  (`docs/audit/generated/*.csv`, stable-sorted, byte-diffable) as the
  drift-gated artifact, with the human Markdown demoted to a non-byte-gated
  companion. Both are wired into `scripts/check_generated_docs.sh`, which gained
  a `--write` mode so `scripts/check_generated_docs.sh --write` regenerates
  every registered doc at sprint end. This retired the byte-exact-markdown
  drift gates that had been reddening CI (`runtime_abi.md` was stale 234 vs 241
  symbols). The four Apple CPU+GPU state docs were also consolidated into the
  single reference `docs/backends/apple/`.

## Still Open

- **Program identity — component-op vectors + gating landed (2026-06-02);
  component-aware metadata landed (2026-06-07).** `CompileResult` carries
  ``component_ops`` (the whole-program distinct op vocabulary),
  ``program_executable`` (gated component-by-component, not just the primary
  op), and ``component_blockers`` ((op, failing-gate) pairs). **`effects` /
  `shape_envelope` / `layout_contracts` / `fusion_groups` now reach the
  user-facing `fn.runtime_artifact().metadata`** — derived in
  `canonical_compile._derive_*`, factored into
  `CompileResult.descriptive_metadata()`, and merged additively through
  `JitFn._build_runtime_artifact` (previously discarded on the `@jit` path —
  every key was absent for real jitted functions). `fusion_groups` recognizes
  the cross-family chains the Apple GPU runtime actually fuses
  (`matmul→softmax[→matmul]`, `matmul→gelu`, `matmul→rmsnorm`), not just
  same-family adjacency. Locked by `tests/unit/test_canonical_component_ops.py`
  + `tests/unit/test_canonical_metadata_jit.py`. **Graph outputs landed
  2026-06-11** (`canonical_outputs` + populated `return_values`/`result_types`;
  see Next Work #1). **Runtime consumption of `fusion_groups` landed
  2026-06-10** (see next item).
- **Fusion intent is too late — runtime half closed (2026-06-10).** The
  apple_gpu executor now consults `fusion_groups` known_chain metadata before
  the structural re-matchers (which remain as legacy-artifact fallback);
  locked by `tests/unit/test_strict_dispatch.py` (short-circuit + legacy-path
  tests). **SwiGLU is now derived too** (`_match_swiglu_at` handles the DAG —
  gate/up share %x — inside the known-chain scan) and consumed by the
  executor. **Target IR descriptor consume/emit — landed 2026-06-11.** All 7
  Apple fusion passes now *emit* a first-class fusion descriptor on the fused
  call (`tessera.fusion.kernel` + `tessera.fusion.source`): the 4 chain passes
  (matmul→softmax→matmul / matmul→softmax / matmul→gelu / matmul→rmsnorm) also
  *consume* an upstream `tessera.fusion.intent` (source `"descriptor"` vs
  `"rediscovered"`, with a Decision-#21 warning on descriptor/IR disagreement);
  the 3 composite passes (swiglu / mla_decode / native_sparse_attn) emit
  `source = "composite_op"` (the pre-fused op *is* the descriptor). The Python
  emit-half `canonical_compile.stamp_fusion_intents(module)` stamps the intent on
  each recognized chain's terminal op from the canonical `_KNOWN_FUSION_CHAINS`,
  so the frontend produces descriptor-annotated IR. Lit:
  `tests/tessera-ir/phase8/apple_gpu_fusion_descriptor.mlir`; Python:
  `tests/unit/test_apple_fusion_descriptor.py` + `test_fusion_intent_emitter.py`
  (incl. an emit↔consume contract guard). **Auto-wired 2026-06-11:**
  `driver.compile_graph_module` calls `stamp_fusion_intents(module)` before
  rendering the Graph IR for Apple targets (gated to `apple_gpu`/`apple_cpu`;
  the descriptor is backend-agnostic so it extends when other backends consume
  it), so every Apple compile now produces descriptor-annotated Graph IR that
  the Target IR passes consume. The intent is stamped into the op's MLIR `attrs`
  (not `kwargs`, which are the op's real call arguments in the reference/runtime
  path). Loop closed end-to-end.
- **Layout and binding contracts are uneven.** Graph/Schedule/Tile/Target IR
  need stronger dtype, layout, aliasing, and buffer-binding contracts.
  **Layout slice extended 2026-06-11:** `LayoutLegalityPass` was matmul-only; its
  producer/consumer accept-set rule now also covers `tessera.conv2d_nhwc` (nhwc
  on the data operand #0; the filter is a separate weight layout) and
  `tessera.flash_attn` (bhsd on Q/K/V #0..2), per-operand-scoped so it only
  checks the operands that carry each contract. matmul stays verbatim (the V4a
  diagnostic + `matmulAcceptSet()` are pinned by existing tests). Lit:
  `tests/tessera-ir/phase2/layout_conv_flashattn_accept_set.mlir`; Python:
  `tests/unit/test_layout_legality_extended.py`. **Pipeline wiring landed
  (2026-06-17):** `LayoutLegalityPass` now runs inside `tessera-lower-to-x86`,
  `tessera-lower-to-gpu`, and `addCUDA13PipelineForSM` (the exact-SM NVIDIA pipelines)
  — early, after distribution lowering and before `SymbolicDimEqualityPass`, so
  unknown-layout / producer-consumer-mismatch / scale-without-layout violations
  surface with the other structural diagnostics during real lowering (was
  standalone `--tessera-layout-legality`). Proven firing end-to-end by
  `tests/tessera-ir/phase2/layout_legality_in_pipeline.mlir` (x86) +
  `tests/unit/test_layout_legality_pipeline_wiring.py` (all three builders,
  before-symdim ordering). **dtype / aliasing / buffer-binding contracts landed
  2026-06-19:** `IRContractLegalityPass` (`--tessera-ir-contracts`,
  `src/transforms/lib/IRContractLegalityPass.cpp`) is LayoutLegalityPass's sibling
  — one `ModuleOp` walk, 7 stable-coded rules across three families: **dtype**
  (numeric_policy storage/accum coupling, `DTYPE_LEGALITY_TF32_AS_STORAGE`,
  `DTYPE_LEGALITY_LOWP_WITHOUT_WIDE_ACCUM`, `DTYPE_LEGALITY_UNKNOWN_STORAGE` —
  enforces Decision #15a: storage≠accum, TF32 is a math_mode not a storage dtype);
  **aliasing** (`tessera.inplace` requires an in-range `tessera.aliases` —
  `ALIAS_LEGALITY_MISSING_ALIASES` / `_OPERAND_OOB`); **buffer-binding**
  (`tessera.buffer_role` accept-set + no conflicting role per `tessera.binding` —
  `BUFFER_BINDING_UNKNOWN_ROLE` / `_CONFLICT`). Lit:
  `tests/tessera-ir/phase2/ir_contract_legality.mlir` (13 cases); Python:
  `tests/unit/test_ir_contract_legality.py` (12). **Wired into all three named
  lowering pipelines** (`tessera-lower-to-x86`, `tessera-lower-to-gpu`,
  `addCUDA13PipelineForSM`) right after `LayoutLegalityPass`, so the contracts fire
  during real lowering — full tessera-ir lit sweep 148 PASS / 19 UNSUPPORTED /
  0 FAIL confirms no existing fixture violates them. The earlier-open
  **Phase 1** of the closure plan
  added the missing *assignment* half — `LayoutAssignmentPass` (seed kernel layouts
  → propagate through pointwise → insert `cast{layout}`), with the legality pass
  reused as its verifier. **Landed 2026-06-22** (`test_layout_assignment.py` +
  `layout_assignment.mlir`); still **not wired into the named x86/GPU pipelines**
  (it mutates IR, so wiring is gated on a layout-sensitive backend consuming the
  attrs). The Graph-IR `hasFolder`/`hasCanonicalizer` gap is closing —
  8 ops now carry folders/canonicalizers (the arithmetic/cast set plus
  `reshape`: identity fold + `reshape(reshape(x))` chain-collapse) wired into
  the `tessera_jit` CPU `canonicalize→cse` pipeline (`graph_ir_folders.mlir`);
  **per-op effect
  interfaces landed 2026-06-22** — all 23 non-pure ops carry an explicit
  `MemoryEffectsOpInterface` (`[Pure]` for the deterministic optimizer/arch
  value ops, `MemWrite`/`MemRead` for random/stateful/collective/MoE-transport
  ops), so generic CSE merges/removes the pure ones and preserves the effectful
  ones (`graph_ir_op_effects.mlir`). `LayoutAssignmentPass` is now **wired into
  the named x86/GPU/CUDA-13 pipelines behind the `assign-layouts` option**
  (2026-06-22). x86 now defaults assignment on and immediately consumes legal
  row-major/BHSD/NHWC markers through its executable C-order binding contract;
  NVIDIA remains opt-in
  (2026-07-23;
  `layout_assignment_pipeline.mlir` + `test_layout_legality_pipeline_wiring.py`).
  Folder coverage was broadened to `reshape` the same day, so **Phase 1 is
  closed**; further folders land opportunistically as new algebraic identities
  surface.
- **Complete claims need fixtures.** A completed backend claim should resolve to
  an explicit compare fixture, `device_verified_abi` row, or packaged validation.
- **Compiler specs can still drift.** Generated dashboards must remain the
  source of counts; prose docs should link, not duplicate snapshots.
- **Generated-doc regeneration + drift gating — registry landed (2026-06-04),
  family-collapse consolidation still open.** The fragmentation finding (two
  parallel gate scripts + piecemeal unit gates + inconsistent generator CLIs)
  has been mostly addressed: `python/tessera/compiler/generated_docs.py` is now
  the single registry of all 21 fully-generated dashboards; `check_generated_docs.sh`
  and `release_gate.py` both delegate to it (the second entry point's per-doc
  drift gates were folded into one fleet-wide `generated_docs_drift`); a unified
  `--write` regenerates the whole fleet; and the fleet drift test
  `tests/unit/test_generated_docs_registry.py` includes an orphan guard so a new
  dashboard must register. The registry immediately caught 3 silently-stale
  dashboards (`test_coverage_by_op`, `test_coverage_classification`,
  `effect_lattice_audit`). **CSV-canonical data-shaped tail closed 2026-06-11 —
  12 dashboards now CSV-canonical:** `runtime_abi`, `verifier_coverage`,
  `support_table`, `op_target_conformance`, `runtime_execution_matrix`,
  `test_coverage`, `tsol_coverage`, `effect_lattice_audit`, `surface_status`, and
  the **3 target maps** (`apple_target_map` + `nvidia_sm90_target_map` +
  `rocm_target_map`, added via `apple_target_map.render_csv` /
  `gpu_target_map.render_csv(target)`, wired into the registry so the CSV is the
  drift-gated artifact). The remaining markdown-only docs are narrative rollups
  (`e2e_op_coverage`, `s_series_status`, `s_series_accelerator_proof`,
  `docs_freshness`), not row tables. *Still open (deliberately deferred):* the
  **aggressive content consolidation** (collapse the 3 target maps → 1
  multi-target doc; the `e2e_op_coverage` + `s_series_status` rollups into their
  primaries) — Next Work #6 reassessed these as low-value churn (per-platform
  maps are cross-referenced by the per-platform audit docs; the rollups are
  distinct MASTER_AUDIT truth views).
- **Code-level audit closeout (2026-06-10).** The
  [CODE_AUDIT_2026_06_10.md](archive/CODE_AUDIT_2026_06_10.md) "Closeout status" section
  drove every remaining code-level finding to done / refuted / accepted /
  tracked-deferred. Done: 1e zero-`TRACE_DEFERRED` corpus guard, `_APPLE_GPU_*`
  table-creep enforcer, binary/rowop strict-dispatch funnel coverage, the
  `LoweringUtils.h` dedup across 18 Apple passes, bf16-probe (already cached).
  Explicitly tracked-deferred (with rationale): C-ABI int return code, Target-IR
  C++ fusion-descriptor consumption, Schedule/Tile IR autotuner/LICM (hardware-
  gated), `forbid-ops` pipeline wiring, and the `jit.py` decorator extraction.

### External input — TIRx / "Modern GPU Programming for MLSys" review (2026-06-23)

Reviewed the CMU/mlc.ai book *Modern GPU Programming for MLSys*
(https://mlc.ai/modern-gpu-programming-for-mlsys/, TIRx DSL — a TVM-TIR-derived
Blackwell-gen Tile-IR/FA-4 stack). It is a parallel-universe analog of our Tile
IR + FA-4 dialects (TMA, tcgen05, TMEM, mbarriers, warp specialization,
clusters) and commits to several design choices we have not. Candidate work,
**not yet started** — captured here so the Tile-IR/FA-4 thread (Per-IR scorecard
row "Tile IR (FA-4)") can pull from it. Cross-refs noted; reference memory
`reference_mlsys_gpu_book`.

- **C1 — Layout algebra vs. our flat `tessera.layout` string (HF, foundational).**
  Graph `tessera.layout` remains a deliberately coarse **string enum**
  (`row_major`/`col_major`/`bhsd`/
  `nhwc`/`nchw` — `LayoutAssignmentPass.cpp` `producerLayout`/`consumerAcceptSet`)
  and Tile IR carries only a coarse "optional swizzle" flag on `smem.alloc`
  (`TileMemoryOps.td`). TIRx models layout as a compositional object:
  `S[(shape):(strides)]` shape–stride pairs whose strides carry **named hardware
  axes** (`@laneid/@reg/@warpid/@TLane/@TCol/@m/@gpuid`), with **replication**
  `R[n:stride]` and **swizzle** as a *separate* non-affine `ComposeLayout(swizzle,
  tile)` — never folded into the stride map. This is the abstraction the FA-4
  warp-spec lowering (`WarpSpecializationPass`, `WGMMA`/`TMA`/`TileToX86`) is
  missing — it would unify per-backend layout logic and give the autotuner a real
  object to sweep (tile/lane/swizzle) instead of hardcoded `m64n64k16`. The
  `@gpuid` axis means the *same* algebra spans intra-warp placement and our
  mesh-level `ShardSpec` (Decision #3) at a different scope. **Increment that
  fits today:** add a structured `TileLayoutAttr` (shard/replica/offset triples
  + an explicit `SwizzleAttr` composition) to Tile IR ODS, keep the Graph-IR
  string enum as the coarse producer/consumer contract, and lower the string →
  structured attr at the Schedule→Tile boundary. Extends **Still Open → "Layout
  and binding contracts are uneven"**.
  **v1 LANDED (2026-06-23).** Added first-class `#tile.layout` / `#tile.swizzle`
  attributes to the canonical Tile dialect (`src/compiler/ir/.../TileOps.td` +
  `TileDialect.cpp`): `#tile.layout<shard = [extents] : [strides] on [axes],
  replica = [..] : [..] on [..], offset = N (, swizzle = #tile.swizzle<..>)>` —
  the book's `S ⊕ R ⊕ O` with swizzle held as a *separate* attribute (never
  folded into the affine map). Hand-written parser/printer (the default
  ArrayRefParameter parser rejects the empty `replica = []` common case);
  `genVerifyDecl` enforces parallel-array lengths, positive extents, and a known
  hardware-axis accept-set (`m/tlane/tcol/laneid/warpid/reg/…/gpuid_x/y`) with
  stable codes `TILE_LAYOUT_{RANK_MISMATCH,NONPOSITIVE_EXTENT,UNKNOWN_AXIS}`.
  Lit: `tests/tessera-ir/phase2/tile_layout_attr.mlir` (round-trip incl. a
  TMEM replicated-scale `R[..]` + swizzle case + 3 verifier negatives).
  **Consumer continuation LANDED (PR #457, 2026-07-25; streaming continuation
  2026-07-26):** structured
  `#tile.layout` is attached to real buffer/view/copy/fragment/load/store
  operations; WarpSpecialization, NVIDIA TMA/fragment lowering, and the ROCm
  structured-pack reader consume it. Schedule→Tile retains the Graph string
  only as a coarse compatibility contract; NVIDIA's cast materializer attaches
  an operand-indexed structured physical attr before the Tile boundary, and
  Tile async copies consume it without an NVIDIA layout string. *Still open:*
  remove remaining string-layout compatibility metadata after sibling paths
  migrate, and add a general
  `.apply()` forward mapping for transformation tooling.

- **C2 — Barriers as a layout-reuse correctness property, not scheduling (HF→HG).**
  TIRx's central inversion: in FA-4 one `128×512` TMEM allocation is aliased as
  an fp32 view (S/O) *and* an fp16 view (P at 2× column density); the barriers
  exist because each region is **reused** strictly after its prior consumer
  finishes. So barrier requirements should be *generated* by an aliasing/reuse
  analysis over TMEM/SMEM buffers, not emitted alongside `tessera.schedule.warp`
  boundaries. Reinforces Decision #8 (warp roles structural) by making barrier
  slots a function of buffer-reuse decisions. Targets `WarpSpecializationPass` +
  the Queue dialect. The scorecard's former "WarpSpec emits no mbarriers"
  statement is superseded by PR #457.
  **v1 LANDED (2026-06-23), SSA identity continuation 2026-07-25.**
  `TileBarrierReuseLegalityPass`
  (`--tessera-tile-barrier-reuse-legality`, `src/transforms/lib/`, sibling to
  `LayoutLegalityPass`): for a buffer (keyed exclusively by the root
  `!tile.buffer` result of `tile.alloc`), two write ops whose `#tile.layout` storage-axis
  (`m/tlane/tcol`) footprints
  *overlap* with no intervening barrier op (name contains `mbarrier`/`wait_async`
  /`barrier`, or a `tile.barrier` attr) emit `TILE_BARRIER_REUSE_MISSING_BARRIER`
  + a note at the prior write. Footprint = `[offset, offset + Σ(extent-1)|stride|]`
  over storage-axis shard dims; a pure register/lane fragment has no storage
  footprint and never aliases. Lit: `tile_barrier_reuse_legality.mlir` — the
  canonical FA-4 fp32/fp16 TMEM-aliasing race (flagged), the same pair with a
  barrier between (clean), disjoint double-buffer offsets (clean), and a
  register-only fragment (clean). This is the **acceptance gate** for C3: once
  WarpSpec emits real typed barriers + buffer reuse, this pass going green on the
  FA-4 fixture is the correctness check. `tile.alloc`/`tile.dealloc` now provide
  typed def-use lifetime identity and the legality pass follows SSA alias roots.
  WarpSpecialization allocates parent-region SMEM/TMEM handles, threads them
  into staged copies and consumers, and deallocates them only after CTA sync;
  the real WarpSpec→TMA output passes this legality gate. NVIDIA, the
  Apple/shared fixture surface, and ROCm no longer emit or consume
  `#tile.buffer_ref`; compatibility readers were retired on 2026-07-26.

  **Compatibility classification (2026-07-26).** Active readers: **zero**.
  `TileBufferRefAttr` remains defined and verifier-checked only so archived IR
  fails with stable migration diagnostics rather than an unknown-attribute
  parser error; `tile_layout_attr.mlir` is its dedicated parser/diagnostic
  fixture. Mentions in architecture/audit prose are historical records.
  `CHECK-NOT` assertions and the ROCm compiler benchmark are negative ratchets,
  not producers. `test_ssa_buffer_ref_retirement.py` scans active shared/ROCm
  C++ and ROCm MLIR fixtures and fails if a compatibility reader or producer is
  restored.

- **C3 — Typed barrier domains + a `PipelineState` SSA value (HF→HG).** Three
  barrier primitives with distinct completion semantics — `TMABar`
  (byte-count/engine-signaled), `TCGen05Bar` (MMA-completion), `MBarrier`
  (thread-arrived) — and a `PipelineState` that auto-tracks `(stage, phase-bit)`
  with producer initialized `phase=1` / consumer `phase=0` (the packaged fix for
  the classic off-by-one ring deadlock). Typed SSA mbarrier/TMA dependencies and
  pipeline state are now canonical. NVIDIA no longer emits annotation-only
  pipeline state, and `TilePipelineLegality` rejects it with
  `TILE_PIPELINE_LEGACY_METADATA`.
  Targets the `AsyncCopy`/pipeline lowering + Queue dialect; pairs with C5.
  **v1 LANDED (2026-06-23).** Two Tile-dialect attributes —
  `#tile.barrier<kind = tma|tcgen05|mbarrier, expect = N>` (the three completion
  semantics) and `#tile.pipeline_state<depth, stage, phase, role>` — with
  `genVerifyDecl` bounds (`TILE_BARRIER_{UNKNOWN_KIND,NEGATIVE_EXPECT}`,
  `TILE_PIPELINE_{BAD_DEPTH,STAGE_OOB,BAD_PHASE,BAD_ROLE}`). Plus the cross-op
  `TilePipelineLegalityPass` (`--tessera-tile-pipeline-legality`): the initial
  producer-role op of a pipeline (keyed by `tile.pipeline`) must carry `phase=1`
  and the initial consumer `phase=0` (`TILE_PIPELINE_PHASE_ASYMMETRY` — the
  off-by-one deadlock fix), and all ops on one `tile.barrier_id` must agree on
  `kind` (`TILE_PIPELINE_BARRIER_KIND_MISMATCH` + note). Lit:
  `tile_pipeline_attrs.mlir` (round-trip + 6 verifier negatives),
  `tile_pipeline_legality.mlir` (well-formed pipeline clean; producer-phase-0 and
  mixed-kind flagged). The `!tile.pipeline_state`, `tile.pipeline_init`, and
  `tile.pipeline_advance` SSA vocabulary is now registered with initial
  producer-phase=1 / consumer-phase=0 verification. WarpSpecialization creates
  the two initial states and threads `tile.pipeline_advance` through
  producer/consumer operation results. The 2026-07-26 attention continuation
  carries both states through the KV loop; TMA copies carry typed slice
  coordinates and logical source extents so descriptor hoisting preserves
  ragged-tail zero fill. The 2026-07-25 continuation registers
  `!tile.tma_descriptor`, `!tile.mbarrier`, `!tile.mbarrier_token`, and
  `!tile.tmem` plus typed TMA copy, mbarrier arrive/wait, TMEM load/store, and
  TCGen05 operations. AsyncCopy formation, descriptor deduplication, and the
  FlashAttention barrier sequence now build SSA descriptor/barrier/token
  chains rather than unregistered strings. *Still open:* migrate sibling
  pipeline consumers and obtain exact SM100 TCGen05/TMEM execution; SM120
  supplies structural rejection proof, not substitute device evidence.

- **C4 — Separate *compute*-legalize from *storage*-legalize (HF).** TIRx runs
  `BF16/FP8 ComputeLegalize` (rewrite math to f32-upcast form) early and
  `…StorageLegalize` (packing) terminally — two passes. This is exactly our
  storage-dtype-vs-accumulator split (Decision #15a, enforced statically by
  `IRContractLegalityPass`) operationalized as *pass ordering*: `numeric_policy.
  accum=fp32` becomes a compute-legalize rewrite, low-precision storage packing a
  terminal pass. Gives the `numeric_policy` contract a concrete lowering home on
  the executed lane. Lowest-risk item; closest to landing.
  **v1 LANDED (2026-06-23).** Two ordered rewrite passes (`DtypeLegalizePass.cpp`):
  `--tessera-compute-legalize` (early) stamps `numeric_policy.accum` on any op
  whose `storage` is reduced-precision and lacks an accumulator — `fp32` for
  float storages, `int32` for `int4`/`int8`; `--tessera-storage-legalize`
  (terminal) stamps `tessera.storage_packed` + `tessera.storage_container` on
  sub-byte / block-scaled storage (`fp4`/`nvfp4`/`fp6`/`int4`). Both idempotent,
  additive, and reusing `IRContractLegalityPass`'s dtype sets. Lit:
  `dtype_legalize_split.mlir` — bf16→accum=fp32, int8→accum=int32, fp4→accum
  +packed-int8-container, fp32 untouched, already-has-accum idempotent; the
  3rd RUN composes `--tessera-ir-contracts` after the split to prove the
  legalized IR is contract-legal (the assign-then-verify pairing).
  **Part 1 — real consumer LANDED (2026-06-23).** `StoragePackConsume`
  (`--tessera-storage-pack-consume`) is the first real consumer of the packing
  markers (previously inert): it reads `tessera.storage_packed` /
  `storage_container` + `numeric_policy.storage` and emits a concrete
  structured `#tile.packed_format<logical, container, logical_bits,
  elements_per_container, signedness, encoding, lane_order>` descriptor.
  Concrete `#tile.packed_view` binds that format to a packing axis, physical
  strides, alignment/offset, and explicit `#tile.scale_layout`; fp6 retains
  logical_bits=6 even though its int8 factor is one. Generic
  `tile.packed_load`/`tile.packed_store` operations carry the value-level
  contract; scale-bearing stores fail closed. Bad widths emit
  `DTYPE_PACK_BAD_WIDTHS`. HF Target-IR step (Decision #19). Lit:
  `storage_pack_consume.mlir`. AMD's `GenerateWMMAGemmKernel` decodes
  physically packed signed int4 memory into `vector<2xi32>` (IU4 ABI) and
  bitcasts int8 to `vector<4xi32>` (IU8).
  **Reconciliation LANDED (2026-06-23).** `GenerateWMMAGemmKernel` now *consumes*
  `tessera.storage_pack`: the descriptor's `logical` selects the WMMA dtype and
  its `factor` and signedness are checked against the WMMA integer pack mode
  (int4 → 2 signed nibbles, int8 → 1), `DTYPE_PACK_FACTOR_MISMATCH` or
  `DTYPE_PACK_SIGNEDNESS_MISMATCH` on drift — so the abstract C4
  descriptor drives the real, shipping AMD int4/int8 codegen (one packing
  contract for both backends). Additive: falls back to the legacy `dtype` attr
  when no descriptor, so the existing ROCm tests are unchanged. Verified on a
  ROCm-backend build (`-DTESSERA_BUILD_ROCM_BACKEND=ON -DTESSERA_ENABLE_HIP=OFF`
  builds the MLIR passes against Homebrew LLVM/ROCDL — no HIP/`/opt/rocm`
  needed): ROCm lit 13/13 incl. `wmma_gemm_storage_pack.mlir` (descriptor drives
  signed int4 → packed-memory `vector<2xi32>` IU4 ABI; factor/signedness
  mismatches caught). **Per-target defaults landed (CORE-COMPILER-2,
  2026-07-22; NVIDIA consumer follow-up 2026-07-25):** x86 and NVIDIA named
  pipelines run compute legalization by default. NVIDIA's scale-bearing
  NVFP4/MXFP4/FP6 launch materializers and the signed-INT4 correctness
  materializer now require and consume
  `tessera.storage_pack`, validating logical dtype, int8 container, factor,
  and format-specific signedness before physical byte/nibble loads. The INT4
  consumer additionally rejects scale/fused operands and owns the
  two's-complement low-nibble-first contract. Terminal legalization is now
  capability-filtered by target + operation + descriptor + available consumer.
  SM120 generic packed loads consume explicit scale operands and origin-aware
  block indexing for NVFP4/FP4/FP6 plus signed INT4 decode and unscaled round
  trips. **Operation-specific expansion (2026-07-26):** the named SM120 gate
  now follows the actual def-use chain and enables only packed load to ordinary
  store (explicit unpack/format conversion), matching unscaled packed
  load/store round trips, and packed matmul whose A/B MMA descriptor agrees
  with the logical storage format. Orphan/mixed-use values, descriptor drift,
  arbitrary operations, and Graph quantize/dequantize stay logical. The empty
  target remains an explicit inspection transform. This does not imply that
  arbitrary FP4/FP6 Tile operations have a physical consumer. The ROCm
  backend pipeline runs the complete compute →
  storage → consume chain by default before its WMMA generator. The legacy
  `legalize-dtypes` option remains as an explicit force-on compatibility
  switch.

- **C5 — Independent per-stream pipeline depths (HF plan / HG perf).** FA-4 runs
  three *independent* rings (Q depth 2, KV depth 3, TMEM depth 2), not one global
  `pipeline_stages`. Our FA-4 config exposes a single `pipeline_stages=2` knob
  (`attn_lower.py`); attention wants per-ring depths the autotuner sweeps
  separately. Also: persistent kernel + **L2-aware tile scheduler** ordering and
  **cluster cross-CTA SMEM views** (`map_shared_rank`/`remote_view`/`cta_mask`) —
  both GPU-only-tier, model when SM90/SM100 execution ungates (Phase G/H).
  **HF scaffold LANDED (2026-06-23).** The hardware-free half — the IR vocabulary
  + the autotuner sweep surface: (1) `#tile.pipeline_depths<q, kv, tmem>`
  Tile-dialect attribute (verifier `TILE_PIPELINE_DEPTHS_NONPOSITIVE`, each ring
  >= 1; lit round-trip + negative in `tile_pipeline_attrs.mlir`); (2)
  `FlashAttnLoweringConfig` gains `q_depth`/`kv_depth`/`tmem_depth` (book defaults
  2/3/2, validated), emitted as `tessera.q_depth/kv_depth/tmem_depth` i32 attrs
  alongside the legacy `pipeline_stages` (which still drives `lds_bytes`, so the
  executing path is byte-identical), plus a `ring_depth_search_space()` that
  enumerates the per-ring sweep candidates (default first). Guard:
  `tests/unit/test_attn_ring_depths.py` (8). **Execution stays gated:** *scoring*
  a candidate needs on-device SM_90/SM_100 latency (Phase G/H) — the surface
  enumerates, it does not measure; persistent/L2/cluster scheduling are likewise
  HG. *Still open (HG):* the measured per-ring sweep, WarpSpec stamping
  `#tile.pipeline_depths` from the config, and the kernel consuming per-ring depths.

- **C6 — A warp-spec diagnostics pass (HF, tooling).** The book's "Debugging
  Warp-Specialized Kernels" appendix is a ready-made spec for a `tessera-opt`
  verification pass: a roles/storage/handoff/lifetime worksheet with checkable
  invariants — *arrival-count == init-count*, *producer/consumer initial phases
  differ*, *no `cta_sync()`/`next_tile()` inside a divergent warpgroup branch*,
  *`fence.proxy_async` before TMA store*, *`commit_group()`+`wait_group(0)` before
  storage reuse*, *`cta_sync()` before writeback dealloc*. These are statically
  checkable on warp-specialized Tile IR and would catch deadlocks/races at
  compile time instead of as device hangs. Natural sibling to
  `IRContractLegalityPass`/`LayoutLegalityPass`. Depends on C2/C3 landing the
  typed barriers + reuse model first. Detailed mapping in the 2026-06-23 review
  notes (this session).
  **v1 LANDED (2026-06-23).** `WarpSpecLegalityPass` (`--tessera-warpspec-legality`,
  `src/transforms/lib/`) checks the four *structural* invariants that complement
  C3's phase asymmetry: `WARPSPEC_INIT_UNDER_GUARD` (a barrier init must run at
  CTA top level, not inside a `tile.warp_role` region), `WARPSPEC_COLLECTIVE_IN_
  DIVERGENT_BRANCH` (cta_sync / cluster_sync / next_tile not inside a warp-role
  region), `WARPSPEC_LOOP_COUNT_DISAGREE` (ops sharing a `tile.pipeline` must
  agree on `tile.trip_count` — the "MMA does K_TILES-1" signature, + note), and
  `WARPSPEC_MISSING_VISIBILITY_FENCE` (a TMA store needs a prior
  fence.proxy_async / commit_group in its block). Convention-driven (warp-role
  region = any ancestor carrying `tile.warp_role`/`tile.warp_guard`/`tile.wg_id`;
  op classes by marker attr or name substring), so it runs on the value lane and
  unregistered husks alike. Lit: `tile_warpspec_legality.mlir` (well-formed
  kernel clean + one negative per invariant). *Still open* (need lifetime
  modeling — the C2↔C6 join): `arrival-count == init-count` and
  cta_sync-before-writeback-dealloc (use-after-free).

**Suggested order:** C4 (cheapest, validates #15a) → C1 (`TileLayoutAttr`,
foundational) → C2+C3 (reuse model + typed barriers/`PipelineState`, mutually
enabling) → C6 (diagnostics, needs C2/C3) → C5 (HG perf). Not to port: TIRx's
TVM plumbing passes (`FlattenBuffer`/`MakePackedAPI`/`LowerWarpMemory`) — MLIR
handles those differently.

**Status (2026-06-23): C1–C4 + C6 v1 LANDED** — the structured `#tile.layout`/
`#tile.swizzle` algebra (C1), the `TileBarrierReuseLegalityPass` reuse-as-
correctness rule (C2), the typed `#tile.barrier` + `#tile.pipeline_state`
attributes and `TilePipelineLegalityPass` (C3), the compute/storage legalize
split (C4), and the `WarpSpecLegalityPass` structural diagnostics (C6) all build
into `tessera-opt` and are lit-green (full `tests/tessera-ir/` sweep 160 passed /
19 unsupported / 0 failed). All five are hardware-free and attribute/convention-
driven (and now wired into the named GPU pipelines + fed by real WarpSpec
markers — see the "Join + pipeline wiring" block below). Together C2 (reuse),
C3 (typed barriers + phase asymmetry), and C6
(structural invariants) are the **deadlock-freedom gate** for the FA-4 warp-spec
lowering.

**Join + pipeline wiring LANDED (2026-06-23).** Two follow-ons closed the gap
between "standalone convention-checkers" and "live lowering gates":
1. **WarpSpec emits the markers.** `WarpSpecializationPass`
   (`src/compiler/tile_opt_fa4/lib/`) now stamps `tile.warp_role` +
   `tile.pipeline` + the typed `#tile.pipeline_state` (producer `phase=1`,
   consumer `phase=0`, `depth=2`) on the producer/consumer `schedule.warp` ops
   it creates — one `warpspec.N` pipeline id per region. So C3/C6 verify *real
   lowering output*, not a hand-written convention. Guard:
   `tests/tessera-ir/phase3/warpspec_emits_markers.mlir` (markers emitted +
   output flows clean through C3+C6).
2. **Wired into the named pipelines.** `tessera-lower-to-gpu` and the four
   `tessera-nvidia-pipeline*` aliases now run `TilePipelineLegality` (C3) +
   `WarpSpecLegality` (C6) + `TileBarrierReuseLegality` (C2) **always-on**
   immediately after `WarpSpecialization` (verified by `--mlir-print-ir-after-all`
   showing the four passes in sequence; full `tests/tessera-ir/` sweep
   **158 passed / 19 unsupported / 0 failed** — the gates pass on every existing
   GPU-pipeline fixture incl. `flash_attn_full`). C4's compute-legalize (before
   `IRContractLegality`) + storage-legalize (terminal) are wired into all three
  pipelines (x86/gpu/CUDA13) behind a `legalize-dtypes` force-on option.
  CORE-COMPILER-2 makes compute legalization default for x86/NVIDIA and the
  full compute/storage/consumer chain default in the ROCm-owned pipeline;
  terminal packed storage remains opt-in on targets without a consumer.

**Buffer-marker emission LANDED (2026-06-23) — C1/C2 markers now on real output.**
`WarpSpecializationPass` also stamps the staged-buffer writes it moves into the
warp regions: each `tile.async_copy` gets `tile.access="write"` +
`tile.buffer="warpspec.N.smem.K"` + a row-major `#tile.layout` on the linear `m`
axis (distinct buffer per copy), and each `tile.mma` gets a TMEM accumulator
buffer (`warpspec.N.tmem.acc.K`) with a `#tile.layout` on the `tlane`/`tcol`
axes. So **C2 (`TileBarrierReuseLegality`) now runs live on real lowering output**
— clean on well-formed lowering (distinct buffers don't alias), and it still
fires `TILE_BARRIER_REUSE_MISSING_BARRIER` on a genuine same-buffer overlap.
Guard: `tests/tessera-ir/phase3/warpspec_buffer_markers.mlir` (markers on
async_copy + mma; C2 clean) + `flash_attn_full` lowers clean through the gate.
*Robustness fix surfaced here:* `TileLayoutAttr::get` runs the `genVerifyDecl`
verifier and **fatal-errors** on an invalid layout, so the stamper skips the
`#tile.layout` (buffer identity only) when a tile has dynamic / placeholder
(`kDynamic`/-1) extents — caught via the flash-attn dynamic-shape path.

**`#tile.barrier` emission + C6 arrival-count LANDED (2026-06-23).**
`NVTMADescriptorPass` now stamps a typed `#tile.barrier<kind="tma", expect=
expect_tx>` + a per-slot `tile.barrier_id="mbar.N"` on **both** the
`tile.tma.setup_descriptor` (init site — declares the expected transaction byte
count) and the `tile.tma.copy_async` (arrive site) for each mbarrier slot, so the
init and arrive of one slot carry the same `(kind, expect, id)`. New C6 rule
`WARPSPEC_ARRIVAL_COUNT_MISMATCH` (`WarpSpecLegalityPass`): per `tile.barrier_id`,
all `#tile.barrier` `expect` values must agree (init count == arrival count) —
else the wait never releases. C3's existing per-id kind-consistency check now
also runs live on these. **The barrier checks need to run *after*
NVTMADescriptor**, so the GPU + CUDA13 pipelines run a *second*
`TilePipelineLegality (C3) + WarpSpecLegality (C6)` placement right after
NVTMADescriptor (the first placement, after WarpSpecialization, still gates the
warp-structure + buffer markers). Verified end-to-end: `flash_attn_full` lowers
through the full `tessera-lower-to-gpu` reaching **both** gate placements, emits
6 consistent `#tile.barrier` markers, exits clean. Guards:
`tests/tessera-ir/phase3/nvtma_barrier_emission.mlir` (emission on setup +
copy_async; output passes C3+C6) + the `arrival_count_mismatch` negative in
`tile_warpspec_legality.mlir`.

**C6 use-after-free LANDED (2026-06-23) — C6 now fully closed (all 7 invariants).**
`WarpSpecializationPass` emits a **writeback-dealloc epilogue** before each
specialized region's terminator: a `tile.cta_sync` followed by a
`tile.buffer_free {tile.buffer=…, tile.access="free"}` for every buffer the
region allocated (the `smem.K` + `tmem.acc.K` it stamped). New C6 rule
`WARPSPEC_USE_AFTER_FREE` (`WarpSpecLegalityPass`, block-local like the fence
check): a buffer free needs a prior `cta_sync` in its block, else a warp may
still be reading the buffer during writeback. Correct lowering is clean (the
epilogue's `cta_sync` precedes the frees); the negative fires on a free with no
preceding sync. Verified: `flash_attn_full` still lowers clean through the full
`tessera-lower-to-gpu` with the epilogue ops flowing downstream. Guards: the
dealloc-epilogue CHECK in `warpspec_buffer_markers.mlir` + the `use_after_free`
negative in `tile_warpspec_legality.mlir`. **All seven appendix invariants
(init-placement, collective-in-branch, loop-count, visibility-fence,
phase-asymmetry [C3], arrival-count, use-after-free) are now checked, and the
full C1–C3/C6 marker vocabulary — `#tile.layout`, `tile.buffer`/`access`,
`#tile.pipeline_state`, `#tile.barrier`, and buffer-free lifetimes — is emitted
by real lowering passes and gated in-pipeline.**

**C5 HF scaffold LANDED (2026-06-23).** The initial hardware-free half of C5 — the
`#tile.pipeline_depths<q, kv, tmem>` IR attribute + `FlashAttnLoweringConfig`'s
per-ring depths/emission/`ring_depth_search_space()` sweep surface — is in
(`test_attn_ring_depths.py`). **Every TIRx-review item (C1–C6) now has its
hardware-free portion landed, lit/unit-green, and (for C1–C4/C6) wired into the
named GPU pipelines fed by real lowering markers.** What remains is strictly
**hardware-gated** (Phase G/H, SM90/SM100 silicon): the measured per-ring depth
sweep, persistent/L2-aware tile scheduling + cluster cross-CTA SMEM views (C5),
WarpSpec stamping `#tile.pipeline_depths`, and the kernels that consume the
per-ring depths. That statement closed the original C1–C6 review inventory,
not all later structural work. PR #457 subsequently replaced the C1–C3
compatibility scaffolding with real SSA consumers; the canonical GEMM K-loop,
  sibling-backend SSA migration and removal of fallback metadata remained
  legitimate hardware-free compiler closure work. The autotuner write path
  subsequently landed under `CORE-COMPILER-RUNTIME-CLOSEOUT-2026-07-27`;
  analytical scoring and target-owned selector evidence remain open.

**Registry sync + typed-contract hardening LANDED (2026-06-23).** Two follow-ups
after the C1–C6 feature: (1) the Python meta-registries now reflect the new
surface — `dialects_manifest` registers the `tile` dialect; `diagnostic_codes`
registers the 23 new MLIR codes (TILE_LAYOUT_* / TILE_BARRIER_* / TILE_PIPELINE_*
/ TILE_PIPELINE_DEPTHS_NONPOSITIVE / TILE_BUFFER_REF_* / WARPSPEC_*);
`pass_metadata` adds the 5 new passes (compute/storage-legalize + the C2/C3/C6
gates) with their codes/dialects/required-attrs; `pipeline_registry` reflects the
two gate placements + the `legalize-dtypes` option in the GPU/nvidia pipelines.
All drift gates green (108 registry tests, 17 generated docs in sync). (2) The
first **typed contract** strengthening of the marker conventions: the loose
`tile.buffer` + `tile.access` string pair is replaced by a typed
`#tile.buffer_ref<name, space, access>` attribute whose `space` (smem/tmem/gmem/
reg) and `access` (read/write/free) are closed, verifier-checked sets
(`TILE_BUFFER_REF_{EMPTY_NAME,BAD_SPACE,BAD_ACCESS}`). WarpSpec emits it on staged
writes + the dealloc epilogue; C2/C6 read the typed handle; flash_attn lowers
clean. *Next (the SSA half):* promote buffer/barrier *identity* from a string
name to an SSA `!tile.buffer` handle produced by a `tile.alloc` op and consumed
by `tile.dealloc` (def-use lifetimes instead of name matching) — a TypeDef +
op-pair refactor that lets C2/C6 track real values, scoped as a focused follow-on.

**Backend parity — ROCm is first-class, not second-class to CUDA (2026-06-23).**
The C1–C6 IR contracts were initially NVIDIA-shaped (the typed vocabularies only
spoke CUDA). Corrected: the contracts now name AMD hardware natively, neither
backend privileged — `#tile.layout` axes add `lds` (AMD shared) + `waveid` (AMD
wave) alongside `m`/`warpid`; `#tile.barrier` kinds add `s_barrier` (workgroup
arrival) + `waitcnt` (async vmcnt/lgkmcnt) alongside tma/tcgen05/mbarrier;
`#tile.buffer_ref` space adds `lds`; and C2's storage-aliasing treats `lds` as a
memory axis, so **LDS reuse-without-barrier is caught exactly like SMEM/TMEM**
(`tile_{layout_attr,pipeline_attrs,barrier_reuse_legality}.mlir` carry the AMD
cases). Reality check that drove this: the ROCm WMMA lane is the *more active
execution path* (the #87–90 commits ship real hsaco + int4/int8 WMMA + flash-attn
fwd/bwd on gfx1151), so it earns first-class treatment in the shared contracts.
*Deliberately NOT done:* bolting the NVIDIA pass-chain / legality gates onto
`tessera-lower-to-rocm` — that lane is a different (direct WMMA kernel-gen)
architecture the backend team actively owns; the gates apply there only once/if
ROCm grows a warp-specialized Tile-IR path, and wiring them is a coordinated
change, not a unilateral one.

**ROCm Tile-IR convergence — barrier-id rewrite LANDED (2026-06-23).** The ROCm
backend now consumes the shared Tile contracts (`#tile.layout`,
`#tile.buffer_ref<space="lds">`, `numeric_policy`, `tessera.storage_pack`,
`#tile.pipeline_depths`) via `rocm-wave-lds-pipeline` (planner/stamper) +
`rocm-wave-lds-legality`, wired before `lower-tile-to-rocm`; RDNA→`tessera_rocm.wmma`,
CDNA→`tessera_rocm.mfma` preserved. A review found the first slice modeled async
deps as scalar global state; **rewritten (not patched) onto the typed barrier-id
contract**: (1) sync discrimination is typed — `tile.mbarrier.*`/`tile.tma.*`/
`tile.tmem.*` are rejected (`ROCM_WAVE_LDS_UNSUPPORTED_NV_CONSTRUCT`), no
`name.contains("barrier")` sniff; (2) each `tile.async_copy` carries a
`tile.barrier_id` + `#tile.barrier<kind="waitcnt">`, each `tile.wait_async`
retires the oldest id with a `tile.waitcnt_threshold` (vmcnt watermark, op-count
not byte-count — hardware-correct), and lowering keys a per-id FIFO so each wait
gates the right copy (no "last token"); (3) legality tracks outstanding ids per
id (not one bool) — an mma runs while *unrelated* prefetch ids are outstanding.
Dependency resolution is **precise, never count-based**: an mma consumes the
stage named by explicit `tile.depends_on`, else an SSA value link to its
`tile.async_copy`, else the most-recently-*retired* stage (the
prefetch→wait→compute idiom) — a live prefetch is never mistaken for the mma's
dependency, so software-pipelined double buffering is accepted even without an
explicit `tile.depends_on`. This removed the prior `AMBIGUOUS_DEPENDENCY`
over-rejection that blocked valid multi-stage ROCm kernels.

**Op-layer convergence — SSA token edge LANDED (Phase A0→C-ROCm, 2026-06-23).**
The four Tile sync ops (`tile.async_copy`/`wait_async`/`mma`/`s_barrier`) are now
registered ODS ops (non-`Pure`, so a wait/barrier is never DCE'd nor an mma
CSE-merged or hoisted past its wait), and the Tile dialect owns a payload-free
`!tile.async_token` type. The ROCm planner mints a token on each copy and threads
it into the operands of the wait that retires it and the mmas that consume it,
turning the copy→consumer dependency into an SSA def-use edge. Legality is now a
pure token def-use check — every token an mma consumes must already be retired —
with the program-order/`retiredCtx` re-derivation **deleted**: the over-rejection
the count-based guess produced is structurally impossible because the planner
encodes the dependency as SSA, not order. Lowering retires by SSA Value (the
wait's token operand), falling back to `tile.barrier_id`/oldest only for
token-less IR. Backend-neutral: the token carries no count (NV expect-bytes stays
in `#tile.barrier`; ROCm vmcnt stays in legality arithmetic). Verified on both
build trees: ROCm lit 22/22 (incl. token def-use legality + planner
token-threading pins), tessera-ir phase2/3/6/8 + apple-value + registry/tiling
pytest green.

**Op-layer convergence — NV token edge threads + survives lowering (Phase C-NV,
2026-06-23).** The same `!tile.async_token` converges on the NVIDIA path. The
generic producer→consumer SSA threading in `WarpSpecialization` carries the token
across the `schedule.warp` boundary with no pass change: a producer
`tile.async_copy` mints it, and a consumer `tile.mma` consuming it has its operand
rewired to the producer warp's token *result* (`schedule.warp`/`schedule.yield`
tolerate the type). `AsyncCopyLowering` now carries the token through SM≥90 TMA
lowering (`tile.tma.copy_async -> (tile, !tile.async_token)`) so the consuming
wait/mma/yield operands stay valid SSA after the copy is lowered; the SM<90
cp.async fallback, which has no completion-token path, refuses a token result with
`ASYNC_COPY_TOKEN_NO_CP_ASYNC_PATH` rather than silently dropping the edge.
`NVWGMMALowering` ignores the token operand (reads operands 0/1), and
`NVTMADescriptor` keeps `expect=bytes` on `#tile.barrier`.

`WarpSpecialization` now **auto-mints** the edge by default: from a token-less
input it reads each consumer `tile.mma`'s data operands, mints a
`!tile.async_token` on every producer `tile.async_copy` the mma consumes, and
threads it as an explicit mma operand — the dependency is derived straight from
the dataflow (no program-order guess), and the existing producer→consumer SSA
threading carries it across the boundary. C6 legality **consumes** the edge: the
new `WARPSPEC_MMA_NOT_TOKEN_SYNCED` invariant is the SSA *ordering* half of
`arrival==init` — a consumer mma reading a producer's async-staged tile must also
read a completion token from that producer (gated on copy completion by SSA), per
producer (precise per-copy pre-warpspec, per-warp post-warpspec); `arrival==init`
keeps the mbarrier *byte-count* half on `#tile.barrier` (the token is countless by
design). So the NV path now both carries and verifies the edge — convergence with
the ROCm token model is complete (NV via warpspec + mbarrier, ROCm via the
planner + waitcnt). Verified: tessera-ir phase2/3/6/8 lit (incl.
`warpspec_async_token.mlir` auto-mint+crossing+TMA, `warpspec_token_sync_legality.mlir`
synced/unsynced/no-async cases), ROCm 22/22 unaffected, diagnostic registry (two
new codes) + apple-value + rocm-tiling pytest (198) green, generated-doc drift
clean.

**Op-layer convergence — Phase D hygiene (2026-06-23).** The ROCm planner no
longer stamps the redundant `tile.depends_on` string on a `tile.mma`: the threaded
`!tile.async_token` operand is the sole dependency representation it emits. A
frontend may still *provide* `tile.depends_on` as an explicit input (the planner's
`inferMmaDeps` consults it to decide the token), and the legality pass keeps a
`depends_on` fallback for hand-written token-less IR — so the input contract and
the token-less path are intact while the duplicated output marker is gone.
`tile.barrier_id` is **kept** (it names the vmcnt counter identity for ROCDL
emission and is the lowering's FIFO fallback key — not redundant with the
"completed" token). Two Phase-D items were assessed and deliberately *not* done:
(1) **token non-optional in ODS** — the four sync ops are `Variadic<AnyType>` by
design (the value lane / Apple use them with no token), and the token edge is
already *enforced* at the right layer by the backend legality passes
(`ROCM_WAVE_LDS_MISSING_WAITCNT`, `WARPSPEC_MMA_NOT_TOKEN_SYNCED`); an ODS-level
required operand would break the token-less lanes for no added safety. (2) **bulk
fixture pretty-form / strict migration** — the Tile dialect's
`allowUnknownOperations(true)` makes `--allow-unregistered-dialect` functionally
redundant on these paths already, so the migration is cosmetic flag-removal across
~29 fixtures with real FileCheck-rewrite risk and no functional gain; left as
opt-in. Verified: ROCm lit 22/22 (double buffer still accepted via the token edge
with no `depends_on`), tessera-ir phase2/3 + registry/tiling pytest green.

## Sequence Mixer unification — linear/hybrid attention (2026-07-17)

**Direction, not status** (extends Decision #28; MASTER_AUDIT + generated
dashboards stay status truth). Design pair authored:

- [`SEQUENCE_MIXER_THEORY.md`](SEQUENCE_MIXER_THEORY.md) — the paper.
- [`SEQUENCE_MIXER_ENGINEERING_PLAN.md`](SEQUENCE_MIXER_ENGINEERING_PLAN.md) — 8
  workstreams (W1–W8).

**Thesis.** Fold the scattered linear/hybrid sequence mixers (KDA, Gated
DeltaNet, GLA/RetNet, Mamba-2 SSD, sliding-window attention, short causal conv,
MLA) into **one Graph-IR `tessera.linear_recurrence` op × four orthogonal
facets** — (A) transition-structure tag, (B) carried-state/cache type, (C) the
`(QKᵀ)V→Q(KᵀV)` reassociation normal form, (D) numeric policy incl.
NVFP4/MXFP8 — so "add a mixer" becomes "register a tag," each a small,
oracle-gated increment. Grounded in a teardown of Kimi Linear (KDA linear
recurrence) and Inkling (local/global + `sconv` + GQA-8 + NVFP4), which bracket
the design space.

**Why now.** The pieces already exist but scattered: `op_catalog` registers
`kimi_delta_attention`/`gated_deltanet`/`selective_ssm`; `stdlib/delta_rule.py`
has the scalar-gated delta rule (recurrent + chunked UT-transform);
`stdlib/hybrid.py` has stringly-typed span mixers + the dual-cache
streaming≡recompute oracle; `lsa.py`/`attn_sliding_window` do windowed attention;
`DeltaNetStateHandle`/`SSMStateHandle` are the carried-state handles. The #1
gap: `kimi_delta_attention`'s reference is scalar/additive, not faithful
channel-wise `dplr_bound`. So this is **unification + faithful completion**, not
greenfield.

**Backend binding (all three leads).** W5/W6/W7/W8 open no new backend items on
any target — they thread mixer candidates + state types into the live per-target
queues and inherit each queue's evidence contract. **Apple** (fastest oracle
loop): `apple/todo.md` items 8–14. **NVIDIA** (perf ceiling; sm_120 verified):
NVIDIA-TEST-3/-5 attention / KV-ReplaySSM / GEMM-Tile families, NVFP4 already in
the TEST-4 numerical policy (the executing FP4 lane). **ROCm** (perf ceiling;
gfx1151 verified): extends the **already-complete** ROCM-REPLAY-1 (decode) and
ROCM-9 (paged-KV), with ROCM-6 G6-B/G6-C as the attention fwd/bwd origin
methodology; gfx1151 has no FP8/FP4 WMMA so low precision is CDNA4/RDNA4
access-gated. ROCm and CUDA set the ceiling and are never capped by the shared
framework (Decision #28). Backward (W8) closes the one gap the first draft
missed; it composes with APPLE-ATTN-BWD-1 / ROCM-6 G6-C.

**First slice** (host-free, opening PR): `SequenceMixer` protocol in `stdlib/`
+ channel-wise KDA in `stdlib/delta_rule.py` + `chunk≡recurrent` /
`scalar-reduction` oracles; route `tessera.ops.kimi_delta_attention` to it.
Everything after is "register another tag," each its own oracle-gated PR.

## Next Work

### Canonical streaming attention synchronization
(`CORE-STREAMING-ATTN-2026-07-26`)

The shared rank-2 forward contract now follows the canonical GEMM reduction
foundation rather than lowering attention as one whole-tensor sequence:

1. Graph/Schedule→Tile emits a KV-block `scf.for` carrying the FP32 output
   accumulator, running maximum, running normalization sum, producer and
   consumer `!tile.pipeline_state` values, and absolute boundary offset.
   Structured K/V slices carry typed coordinates and logical source extents,
   preserving ragged-tail zero fill through descriptor hoisting.
2. `tessera_attn.boundary_mask` owns causal and sliding-window state;
   `tessera_attn.block_dropout` owns offset-keyed dropout; and
   `tessera_attn.streaming_update` consumes both scores and V, closing the
   former value-accumulation hole.
3. NVIDIA WarpSpec now relies on `!tile.buffer` SSA identity and real pipeline
   values. It neither emits `#tile.buffer_ref` names nor annotation-only
   `#tile.pipeline_state`; the pipeline legality pass rejects the latter.
   NVIDIA's Schedule→Tile path consumes structured per-operand layouts directly.
4. Static rank-4 attention now distributes through explicit batch/query-head
   loops, maps GQA heads before rank reduction, and reuses the one canonical
   KV-block recurrence. ROCm consumes this loop directly through gfx1151
   Target IR/runtime with SSA-owned LDS planning and exact-device proof. A
   launch carrier verifies split-workspace ownership, block-loop
   metadata, and ascending reduction order. Tensor-valued shared backward
   `scf.for` bodies now carry dQ, split dK/dV partials, and fixed-order
   reduction. ROCm directly consumes them on gfx1151, and x86 now structurally
   consumes the same forward and backward loop contracts before selecting its
   Zen 5 AVX-512 package. Apple and NVIDIA remain architecture-owned
   follow-ups.
5. Exact-device evidence is not transferable. NVIDIA SM120, Apple, and ROCm
   must each lower the shared contract into architecture-owned schedules and
   retain numerical, resource, cache, device-event, and end-to-end proof before
   changing selectors.

ROCm consumes the canonical rank-4 KV-block loop directly for MHA/GQA, causal
left-window, ragged, and deterministic-dropout forms. Sync
`CORE-ATTENTION-TENSOR-LOOPS-MODIFIERS-2026-07-26` adds registered score-bias
and softcap operations directly to that recurrence, including per-head rank-4
bias, and the gfx1151 adapter consumes the combined form. The same sync lowers
the verified backward contract to tensor-valued dQ, split dK/dV partial, and
ascending-reduction `scf.for` bodies. Exact gfx1151 combined
bias+softcap+dropout forward execution has max error `0.000271678` and resident
median `0.098631 ms` against the `0.097763 ms` baseline. Direct physical
consumption now lands on gfx1151 under sync
`ROCM-ATTENTION-SHARED-BACKWARD-CONSUMER-2026-07-26`: ROCm validates the
tensor-valued phase topology and emits the compiler-owned five-entry package
without rebuilding semantics from the launch carrier. Apple and NVIDIA remain
architecture-owned follow-ups.

Cross-backend sync `ROCM-E2E-ATTENTION-BACKWARD-2026-07-26` additionally maps
the canonical launch-level backward carrier to one gfx1151 five-entry HSACO and
a ROCm-owned deterministic split/reduced program workspace. The carrier now
states launch ownership, split count, block-loop order, and fixed reduction
order instead of claiming zero workspace. Exact-device dropout replay with
combined bias+softcap gradients passes. Tensor-valued shared backward
`scf.for` bodies had landed while direct target consumption of their phase
operations remained open at that synchronization point; the follow-up below
closes ROCm only, and no AMD schedule or evidence transfers.

Follow-up sync `ROCM-ATTENTION-SHARED-BACKWARD-CONSUMER-2026-07-26` closes
that gfx1151 gap. The native package source is the shared rank-4 forward plus
`tessera_attn.backward` program; lowering exposes dQ, launch-owned two-split
dK/dV partial, and ascending-reduction loops, and ROCm fails closed on any
incomplete or reordered phase topology. Exact combined-feature gradients pass
at maximum absolute errors dQ `0.000024833`, dK `0.000035211`, and dV
`0.000329971`; the resident five-launch median is `0.367367 ms` versus the
`0.368203 ms` baseline and `0.405023 ms` cap. Apple/NVIDIA physical
consumption remains open and inherits no AMD artifact or timing evidence.

`CORE-ATTENTION-TRAINING-X86-2026-07-30` closes the x86 sibling consumption
without transferring AMD schedules. X86 canonical forward packaging now starts
from the shared rank-4 Graph recurrence; its backward package requires the
canonical dQ, split-dK/dV, and fixed-order reduction phase topology. Exact Zen
5 AVX-512 numerics preserve f32 MHA/GQA, bias, causal/window, and softcap
behavior. `X86-LSE-1` selects saved row LSE from a 21-sample 32/64/128 sequence
packet (1.45x/1.23x/1.06x over end-to-end recomputation). The x86 Lion VJP and
factored/full Adafactor execution plus physical adjoints are also complete.
ROCm retains its complete attention/Lion/Adafactor implementations, while its
128+ saved-LSE threshold remains provisional pending bare-metal gfx1151 events.

> **Open items: #4 (fixture-backed numerical proof before conformance cells go
> complete) and #5 (point specs at dashboards/this audit, not old root audits).**
> Items #1, #2, #3, and #6 have **landed** — they are kept below (struck through)
> for provenance, not as pending work.

1. ~~Add `component_ops`, `fusion_groups`, `shape_envelope`, `effects`, and
   `layout_contracts` to canonical compile metadata.~~ **Landed** —
   `component_ops` (2026-06-02) + `effects` / `shape_envelope` /
   `layout_contracts` / `fusion_groups` (2026-06-07), all reaching the
   user-facing `fn.runtime_artifact().metadata`. **Graph outputs landed
   2026-06-11** — `CompileResult.outputs` / `canonical_outputs`
   (`tessera.compile.outputs.v1`: each returned value + producer op + type /
   shape / dtype / layout), backed by populating `GraphIRFunction.return_values`
   + `result_types` from the jit AST `return` (the AST path previously emitted a
   value-less `return`, so outputs/`shape_envelope.returns` were empty). Locked
   by `tests/unit/test_canonical_outputs.py`; full IR/lit/canonical sweep green.
   Remaining: runtime *consumption* of `fusion_groups` (Next Work #3 / "fusion
   intent too late").
2. ~~Gate whole programs and component ops separately.~~ **Landed 2026-06-02**
   — `program_executable` + `component_blockers` gate the whole program
   component-by-component alongside the primary-op `executable` answer.
3. ~~Make Target IR emit backend descriptors rather than embedding/rediscovering
   large Apple-specific fusion/runtime decisions.~~ **Landed as Phase 0
   (2026-06-15)** — the apple_gpu executor is now authoritative over carried
   fusion roles (`dispatch` on each `known_chain` group) and the four structural
   re-matchers are deleted. Fusion is recognized once (the compiler) and carried
   across the seam to the executor; the executor no longer re-discovers it. See
   the Phase 0a/0b/0c entries in the front-to-back closure plan above.
   **C++ Target IR consume-side reviewed + parity-guarded (2026-06-15).** Unlike
   the Python executor (whose re-matchers were pure duplication and were deleted),
   the C++ Apple fusion passes *must* walk the def-use graph to collect operand
   `Value`s for codegen — that walk is intrinsic, not deletable. The chain passes
   already consume `tessera.fusion.intent` (source `"descriptor"` vs
   `"rediscovered"`, with a Decision-#21 mismatch warning) and the 2-op chains in
   fact lower through the *generic* `synth_matmul_epilogue` synthesizer (F2b), not
   per-pattern hand-kernels. Added a **producer-covers-consumer parity guard**
   (`tests/unit/test_apple_fusion_parity.py`, the C++ analogue of the 0c oracle):
   every producer-stamped chain lowers `source="descriptor"`, never
   `rediscovered`. This caught + fixed a real cross-representation drift Phase 0c
   introduced — `matmul→rmsnorm_safe` is a distinct producer kernel but the C++
   reads a single `"matmul_rmsnorm"` intent for both variants, so
   `stamp_fusion_intents` now maps it via `_FUSION_INTENT_NAME`. Note the C++ Apple
   passes run in lit/validation, **not on the execution path** (that is the Python
   runtime, closed in 0a–0c); their value today is IR auditability + keeping the
   two fusion recognizers in sync for when real codegen eventually routes through
   compiled IR (Phase 4). *Next on this thread:* extend descriptors to NVIDIA/ROCm
   when those backends light up.
4. Require fixture-backed numerical proof before conformance cells become
   complete.
5. Update specs to point at dashboards and this audit instead of old root audit
   documents. **Verified 2026-06-19:** the only specs that link an audit
   (`TARGET_IR_SPEC.md`, `AUTODIFF_SPEC.md`) already use the current theme-audit
   path `docs/audit/coverage/COVERAGE_AUDIT.md` — no stale root-audit references
   remain. Adding generated-dashboard pointers to the remaining specs is optional
   polish, not a correctness gap.
6. **Unify generated-doc regeneration + drift into one contract — landed
   2026-06-04.** `tessera.compiler.generated_docs` is the single registry
   consumed by both `check_generated_docs.sh` and `release_gate.py` (the latter's
   per-doc drift gates folded into one fleet-wide `generated_docs_drift`), with a
   fleet `--write`/`--check`, an orphan-guard test
   (`tests/unit/test_generated_docs_registry.py`), and a `--list` view.
   - **9 dashboards CSV-canonical:** `runtime_abi`, `verifier_coverage`,
     `support_table`, `op_target_conformance`, `runtime_execution_matrix`,
     `tsol_coverage`, `effect_lattice_audit`, plus the merged `test_coverage` and
     consolidated `surface_status`.
   - **Content consolidation done (genuinely-duplicative docs):** the 5
     surface-status docs + `operator_benchmarks_coverage` → one `surface_status`
     (6→1); `test_coverage_by_op` + `test_coverage_classification` → one
     `test_coverage` (2→1). Registry count 24 → 15.
   - **Deliberately not consolidated (reassessed):** the 3 target maps stay
     per-platform — they are *not* duplicative (per-target capability matrices),
     have heterogeneous schemas, and are cross-referenced by 8 per-platform audit
     docs (`backend/{apple,nvidia,rocm}/`); collapsing them would fight the
   per-platform audit structure for a 3→1 saving. The `e2e_op_coverage` /
   `s_series_status` rollups likewise stay standalone — they are distinct
   MASTER_AUDIT truth views, and the registry already removed the duplication
   that mattered (one regen/drift contract). Folding them is available if
   desired but is low-value churn now.

## 2026-07-27 runtime and scheduling closeout synchronization

`CORE-COMPILER-RUNTIME-CLOSEOUT-2026-07-27` advances six previously disconnected
compiler/runtime seams:

- ROCm Adafactor now has an explicit five-entry physical state program covering
  factored and lower-rank full-moment updates, exact multi-step gfx1151 proof,
  and operation-total benchmark evidence.
- ROCm injects total/free bytes from the retained HIP context into the shared
  model-derived rematerialization budget with bounded dynamic-parameter
  validation.
- compiler-emitted unique-clock 1F1B steps have a runtime consumer; selected
  backward collectives can overlap later compute on an independent transport
  executor and are joined before completion.
- measured autotune records, when target/evidence/latency-valid, change actual
  Schedule IR and Tile IR M/N/K, warp, and pipeline-depth attributes.
- NVIDIA layout assignment defaults on now that its named lowering pipelines
  immediately consume Graph layout casts; x86 remains default-on, Apple and
  ROCm retain their architecture-owned layout boundaries.
- DeltaNet/Kimi/modified-delta reverse mode is an analytic O(S) carried-state
  recurrence rather than finite differences, with explicit FP32 state and
  reverse-token scheduling metadata.

The remaining gates are intentionally narrower: full-model measured
rematerialization selection; real
multi-rank collective/OptimizerShard transport on each GPU runtime; per-target
measured selector packets; and physical sequence-mixer backward packaging and
chunk-parallel scheduling on CUDA and Apple. The following
`CORE-PRODUCTION-EVIDENCE-2026-07-27` record closes the gfx1151 physical
Adafactor adjoint that was still open at the start of this synchronization.

`CORE-PRODUCTION-EVIDENCE-2026-07-27` advances each seam without overstating
exact-device scope. Typed collective descriptors are now attached to emitted
steps, and runtime OptimizerShard execution has explicit replicated/rank-local
ownership, normalization, rank order, overlap, and completion joins. NCCL and
RCCL select CUDA versus HIP device runtimes behind the same collective ABI.
Deterministic two-rank integration is fixture-backed; real multi-rank packets
remain architecture-owned gates.

The shared DeltaNet reverse recurrence now has direct ROCm and AVX-512
consumers. `CORE-SEQUENCE-MIXER-PHYSICAL-BACKWARD-2026-07-28` extends the AMD
package to five compiler-owned entries: checkpoint, affine chunk summary,
deterministic prefix, parallel chunk fill, and reverse. The AVX-512 ABI uses
caller-owned resident workspaces and the same affine composition law. Both
physical paths implement the exact modified-Delta normalization VJP and match
the analytic oracle below `4e-7`.

Consequently, both ROCm gfx1151 and x86 AVX-512 have physical sequence-mixer
backward packages; CUDA and Apple remain the architecture-owned packaging
follow-ups.

Affine chunk composition is deliberately limited to `erase=false`, where a
chunk is exactly `state_out = scale * state_in + update`. Erase-dependent
targets read incoming state, so their checkpoint dependency remains serial
while independent `(batch,head)` work stays parallel. Resident two-cohort
packets choose chunk 16 on both hosts. The AVX-512 packet is not selector
eligible because cohort medians differ by 12.0%; the stable gfx1151 packet is
not selector eligible because WSL wall-clock timing is not bare-metal
production evidence. CUDA/Apple physical backward packaging and bare-metal
selector refreshes remain architecture-owned; no result transfers across
targets.

## Source Material Consolidated

- `archive/compiler_apple_backend_end_to_end_audit_2026_06_02.md`
- `archive/compiler_correctness_testing_audit.md`
- `archive/compiler_improvement_milestone_plan_2026_05_18.md`
- `archive/compiler_layer_gap_remediation.md`
- `archive/compiler_spec_gap_audit.md`
- `archive/docs/audit/compiler/COMPILER_AUDIT.md`
