---
last_updated: 2026-09-25
audit_role: root
---

# Tessera Audit Master

This is the root routing document for compiler status and open engineering
work. It deliberately owns no copied totals. Generated dashboards own counts
and row-level states; plans own sequencing; backend audits own exact-device
claims.

## Start here

1. Read [`generated/compiler_progress.md`](generated/compiler_progress.md) for
   the live phase rollup.
2. Use [`generated/support_table.md`](generated/support_table.md) to locate the
   first incomplete compiler layer for an operation.
3. Use [`generated/s_series_status.md`](generated/s_series_status.md) for
   primitive transform, sharding, and backend-contract state.
4. Use [`generated/runtime_execution_matrix.md`](generated/runtime_execution_matrix.md)
   and the applicable backend plan before claiming native execution.
5. Use [`compiler/INTEGRATED_COMPILER_PLAN.md`](compiler/INTEGRATED_COMPILER_PLAN.md#live-queue)
   for compiler sequencing, and [`roadmap/ROADMAP_AUDIT.md`](roadmap/ROADMAP_AUDIT.md)
   for wider project routing and archived-plan provenance.

The curated matrix in [`op_target_conformance.md`](op_target_conformance.md) is
an exact-target representative suite. It is not the all-up compiler denominator.

## Current interpretation

The public API, frontend-capture inventory, Graph registration, Schedule IR,
runtime-readiness, verifier, batching, transpose, and lowering contract axes are
closed in the generated rollup. Tile IR still has partial rows. These are
inventory/contract gates, not proof of general native lowering or composition.
Read the route census and execution evidence alongside them.

The compiler is not finished. Backend-kernel counts represent conservative
contract promotion states, not a count of missing implementations or a workload
priority order. Device access blocks particular target proofs; semantic-authority
migration, general AD, specialization and runtime safety also require engineering.
The active AD scope is [AUTODIFF_EXECUTION_PLAN.md](compiler/AUTODIFF_EXECUTION_PLAN.md).

Its active work is concentrated in the following
programs. The numbering identifies themes, not priority or delivery order;
[the integrated plan](compiler/INTEGRATED_COMPILER_PLAN.md#live-queue) alone owns sequencing.

### 1. E2E-REAL-6 — one compiler authority

Promote the tracer to the sole general frontend, move family selection and
package construction out of `JitFn`, and require native packages to consume
their exact content-addressed Schedule→Tile parent. Delete the AST
`_OpExtractor` and Graph-to-backend reconstruction only after differential
execution and architecture-owned proof cover each migrated family.

Owner: [`compiler/INTEGRATED_COMPILER_PLAN.md`](compiler/INTEGRATED_COMPILER_PLAN.md).

### 2. General structured programs

Bounded `if`, counted `for`, canonical bounded `while`, and forward
`control_scan` exist. Remaining work is general source-CFG recovery, multi-block
regions, broader typed affine/Presburger constraints and scan JVP/VJP.
Bounded checkpoint execution and persistent products exist; general heterogeneous
products, effectful composition and automatic frontend integration remain open.

Owners: [`compiler/INTEGRATED_COMPILER_PLAN.md`](compiler/INTEGRATED_COMPILER_PLAN.md)
and [`../spec/CONTROL_FLOW_CONTRACT.md`](../spec/CONTROL_FLOW_CONTRACT.md).

### 3. Measured scheduling

The shared dataflow analysis, bounded dependence producers, native recipe
instantiation and prune-only action-DAG ranker are real. Extend the remaining
value, alias, effect, memory-dependence and ordered-collective consumers.
Ranked candidates still need
clean target calibration and selector-grade packets before a schedule can be
promoted. Analytical or WSL-only timing remains candidate-pruning evidence.

Owners: [native recipes](compiler/INTEGRATED_COMPILER_PLAN.md#frontend-ir-medium-1),
[ANN admission](compiler/INTEGRATED_COMPILER_PLAN.md#msw-9) and
[measured scheduling](compiler/INTEGRATED_COMPILER_PLAN.md#w52).

### 4. Native distributed execution and sharding

The core collective Schedule/Tile contracts and a bounded two-rank MPI slice
exist. Broader rank/subgroup participation, native NCCL/RCCL and other transports,
reshard insertion, and wider multi-rank correctness/performance packets remain open. Sharding propagation
must use a typed, fail-closed placement lattice and explicit incompatibilities;
it must not infer safety across unknown effects, aliases, or regions.

Owners: [DIST-NATIVE-1](compiler/INTEGRATED_COMPILER_PLAN.md#dist-native-1),
the generated sharding queue, and the four backend plans.

### 5. Tiled SSD

The internal `schedule.ssd` family now lowers through Schedule→Tile to native
CPU and replay-bound serial/cooperative CUDA and HIP packages. Forward carry,
chunk checkpoints and checkpoint VJP have owning-device correctness evidence.
Resident program owners support bounded public VJP and connected acyclic SSD
compositions with scoped reader-aware retirement.

These compositions remain host orchestration of verified packages. Canonical
whole-program composition, broader frontend and alias/effect integration,
uncertain-unload recovery, further schedule tuning and selector-grade performance
admission remain open. ReplaySSM remains a separate oracle/candidate; its ABI
is not the shared compiler authority.

Owner: [W5.2f](compiler/INTEGRATED_COMPILER_PLAN.md#w52f); detailed evidence and
remaining boundaries live in that queue and its linked engineering log.

### 6. Model-level physical closure

The frontier-model graph vocabulary and scaled reference execution exist. The
active physical boundary is packed INT4/FP8 weight ingestion without full-weight
materialization, architecture-owned DeepSeek and MiniMax fused paths, and
full-scale distributed execution with real routing and transport. Model
configuration, artifact compilation, scaled correctness, fused execution, and
full-scale performance are distinct proof rungs.

Owner: [`roadmap/MODEL_CLASS_ROADMAP.md`](roadmap/MODEL_CLASS_ROADMAP.md) plus
the applicable backend plan.

### 7. Architecture promotion

RX 9070 XT gfx1201 correctness commissioning now has an owning host, Tajasarus.
R9700-specific performance and native-Linux counter evidence are future
features, not compiler gates (see the scope decision in the consolidated
action list below).
Bounded scheduled unary and standalone backward execute on gfx1201; paired AD,
general matrix packaging and performance promotion remain open.
See the [ROCm queue](backend/rocm/todo.md#gfx1201-scheduled-integration--2026-09-13).

Exact-device evidence never transfers between architectures. x86/AVX-512,
Apple, gfx1151, gfx1200/gfx1250, and individual NVIDIA SM generations retain
separate correctness and performance gates. A fused or packaged implementation
is not selector authority without valid target timing provenance.

The assertions-enabled LLVM/MLIR compiler and installed-driver smoke are now
validated infrastructure, not missing prerequisites. Preserve their regression
gates; they do not replace owning-device execution or timing.

PR #745 closes the bounded mixed u8×s8 x86 matmul migration and byte-backed
FP8 scalar/vector conversion failures on SM120 and gfx1151. Generic packed
loads and bounded bool/complex probes do not establish matrix acceleration or
general dtype support. Apple's legacy unspecified arithmetic retains the recorded subnormal failures.
The explicit Apple arena gradual/FTZ policies now have integer-significand
f32 add/sub/mul/div consumers and owning-M1-Max boundary/random proof; broader
policy consumers and performance admission remain open. See the
[dtype follow-through evidence](../../benchmarks/baselines/dtype_followthrough_20260911/README.md)
and [numerical-policy owner](compiler/INTEGRATED_COMPILER_PLAN.md#numpol-carrier-1).

Owners:
[`backend/apple/todo.md`](backend/apple/todo.md),
[`backend/nvidia/todo.md`](backend/nvidia/todo.md),
[`backend/rocm/todo.md`](backend/rocm/todo.md), and
[`backend/x86/todo.md`](backend/x86/todo.md).

## Consolidated action list (2026-09-25)

A cross-read of every non-archived theme audit, backend queue and generated
dashboard, reduced to what is still open. It **routes**; it does not
sequence (the [integrated plan](compiler/INTEGRATED_COMPILER_PLAN.md#live-queue)
owns order) and it copies no totals (follow the dashboard links). Items are
grouped by what unblocks them, and each names its owning ID.

**Direction (owner decision, recorded here 2026-09-25):** every backend —
NVIDIA, AMD, **Apple**, and **x86 / AVX-512** — builds on the same MLIR/LLVM
compiler foundation (Graph → Schedule → Tile → contract-carrying Target IR →
compiler-produced device code). Apple and x86 are not separate compilers
with their own endgame; their items below are gaps on that shared path.
The Apple queue's AIR analysis (APPLE-AOT-2) still frames joining the spine
as an open risk-appetite call, and is superseded on that point. On x86, ACE
is the matrix path but is **deferred until a shipping processor supports
it**; it is not open work.

**Scope (owner decision, 2026-09-25): a fully functional end-to-end compiler on
the targets that run today.** Those are gfx1151, gfx1201, sm_120, Apple7 GPU +
Apple CPU, and x86 / AVX-512. Every GPU the fleet cannot run — sm_80/90/100,
CDNA gfx90a/942/950, gfx1250, gfx1200, R9700-specific parts, a second Apple
device, multi-GPU transports — is a **future feature**: it keeps its backlog
entry and its own proof obligation when it arrives, but it is **not a gate on
the compiler** and not a priority now. Evidence still never transfers
between architectures; this changes what gates, not what may be claimed.

**Performance evidence does not require a profiler or KFD.** The fleet has a
proven non-profiler method: the in-kernel `wall_clock64` device clock, cross-
checked against banded HIP/CUDA events and the host wall clock
([DEVICE-CLOCK-DISCIPLINE-2026-08-31](backend/rocm/todo.md#cross-backend-sync-device-clock-discipline-2026-08-31):
all three agree to four significant figures on gfx1151), plus paired,
interleaved A/B runs and ISA / resource census (`hipFuncGetAttribute`,
instruction counts). Missing `/dev/kfd` or a bare-metal host does not stop the
compiler from being completed or its performance from being checked;
hardware counters are diagnostic extras.

### Functional-complete alpha: definition and guard rails

Owner statement (2026-09-25). These are the **release criteria for
functional-complete alpha** — a software definition, not a direction. Alpha
ships when every in-scope program runs through one pipeline, on every fleet
lane, with each stage produced by MLIR passes and optimized at that stage:

```
Python / textual frontend
  → typed semantic Graph IR
  → structured differentiation and optimization
  → Schedule IR → Tile IR
  → backend Target IR and native lowering (MLIR → LLVM / NVVM / ROCDL / Apple)
  → native image + checked runtime ABI → execution
```

**Eight lanes on four machines — all required, none optional:**

| Machine | CPU lane | GPU lane | Lane state today ([spine](generated/compilation_spine_inventory.md)) |
|---|---|---|---|
| Mac M1 Max | `apple_cpu` — arm64 (LLVM AArch64) | `apple_gpu` — Apple7, Metal 4 | CPU and GPU Level C `partial` |
| Princess-Luna, Ryzen AI MAX+ 395 | `x86` — Zen 5, AVX-512 | `rocm_gfx1151` — RDNA 3.5 | x86 C `partial` (prebuilt kernel image); gfx1151 C `partial` |
| The-Super-Bear, Threadripper 3970X | `x86` — **Zen 2, AVX2, no AVX-512** | `nvidia_sm120` — RTX 5070 | **no Zen 2 lane** beyond `x86_64_base` softmax/reduction; sm_120 C **`absent`** |
| Tajasarus, Ryzen 7 9800X3D | `x86` — Zen 5, AVX-512 | `rocm_gfx1201` — RDNA4 | x86 as Princess-Luna; gfx1201 C **`absent`** |

**What "native, not a bypass" means at each stage** — each is a checkable
exit criterion, and a family counts toward 100% only when all of them hold on
the lane:

| Stage | Done when | Known bypass today |
|---|---|---|
| Frontend | The tracer is the only frontend; no AST `_OpExtractor`, no family selection in `JitFn` | E2E-REAL-6 |
| Graph IR + AD + optimization | Typed Graph IR; AD is a Graph-IR transformation; canonicalize / fold / CSE / fusion / layout run as MLIR passes | Python-side fusion recognizer and synthesizer; attribute-stamp-only passes |
| Schedule IR | Tiling, staging, pipeline and distribution decisions are Schedule IR produced and consumed by passes | Schedule→Tile is op-name macro expansion with tile sizes from pass options (IR_STACK U6); Python/C++ dual boundaries (U3) |
| Tile IR | Typed Tile IR; buffer reuse / arena / async-copy / legality passes run | Untyped `tile.mma` sites (W1.1); whole-kernel and domain ops in Tile (W3.3) |
| Target IR + native lowering | Contract-carrying Target IR lowered by MLIR to LLVM / NVVM / ROCDL / Apple device code | Python `package_*` constructors ([`bootstrap_prune_gap`](generated/bootstrap_prune_gap.md)); `emit/*` source emitters; x86 packages a prebuilt image; Apple `matmul2d` / hand-written MSL reached by symbol |
| Image + ABI + execution | Native image with a checked runtime ABI, launched and compared on the lane's own device | [`runtime_abi`](generated/runtime_abi.md) stubs |

Hand-tuned or library kernels remain allowed only as **arbiter candidates
behind a declared Target IR op** (Decisions #28/#31) — never as the only way
a family reaches the device.

**Guard rails.** Every change is judged against these until alpha ships:

1. **No new bypass.** A change may not add a Python `package_*` constructor,
   an `emit/*` source emitter, a prebuilt-kernel lowering, or an
   optional-only Target IR contract. The counts in
   [`bootstrap_prune_gap`](generated/bootstrap_prune_gap.md) and
   [`target_ir_membership`](generated/target_ir_membership.md) only go down.
2. **Every stage is real.** A family counts toward alpha only when each
   stage above is MLIR-produced on the lane and the stage's exit criterion
   holds; a later stage cannot compensate for a skipped one.
3. **All eight lanes.** A family is alpha-complete only when all eight lanes
   pass; a lane that cannot run a family records a named, stable refusal
   (Decision #21), never a silent reference fallback.
4. **Execute-and-compare on the lane's own device.** Evidence never transfers
   between lanes. Performance is checked with the accepted non-profiler
   method (device clock cross-checked against events and host wall, paired
   interleaved runs); counters are optional.
5. **Future features are out of alpha.** Non-fleet GPUs, multi-GPU
   transports and ACE neither gate alpha nor count toward it.
6. **Delete only after absorption.** A bypass is removed only once its
   family passes all stages on the lanes it served (Decision #31 ordering).

**Enforcement.** 4 and 6 are enforced by the existing per-device proof rules
and Decision #31 gates. 1–3 are enforced by
`tests/unit/test_alpha_scoreboard.py` against
[`alpha_scoreboard`](generated/alpha_scoreboard.md): the bypass counts
(guard rail 1) may only fall and the per-stage / per-lane native counts
(2–3) may only rise, both pinned exactly in
`tests/unit/alpha_ratchet_baseline.json` so a gain must be locked in. Not
yet measured: the Graph optimization/AD stage per family (shown
`unmeasured`), and guard rail 3's "named refusal, never silent fallback".

**Scoreboard.** [`generated/alpha_scoreboard.md`](generated/alpha_scoreboard.md)
measures this definition: eight lanes × the alpha family set × five stages,
derived from the frontend source, the bootstrap route map, the compilation
spine and the E2E fleet packets. The lanes and the family set (with
per-backend family names) are declared; a declared name that no longer
matches its module fails generation. Read the counts there.

### Three blockers most other items wait on

1. **One compiler authority.** Most families still reach a backend through a
   Python `package_*` constructor rather than Graph→Schedule→Tile→Target:
   see [`bootstrap_prune_gap`](generated/bootstrap_prune_gap.md) (now
   separating family-named routes, generic-route wrappers and true gaps),
   [`primitive_route_map`](generated/primitive_route_map.md) and
   [`compilation_spine_inventory`](generated/compilation_spine_inventory.md)
   (Level C is absent for `nvidia_sm120` and `gfx1201`). Owner E2E-REAL-6,
   after E2E-REAL-6F's census review.
2. **Contract carriage across levels.** Most Target IR ops declare their
   contract as optional and verify with it unset
   ([`target_ir_membership`](generated/target_ir_membership.md)); no boundary
   verifier checks that `numeric_policy` / `layout` / `distribution` survive a
   lowering (IR_STACK U5); and [`verifier_coverage`](generated/verifier_coverage.md)
   scans only `TesseraOps.td` and `Attn.td`, so its closed status does not
   cover Tile, Schedule or Target ODS (it now fails closed on a missing mapped
   `.td`; a dead entry for the deleted Queue dialect had been skipped
   silently).
3. **Timing the compiler can act on.** The code admits the method since
   2026-09-26 (sync `WSL-TIMING-ADMISSION-2026-09-26`, recorded in all four
   backend plans): WSL timing promotes when a kernel-side clock of the
   sample's own target has at least one valid admissible witness in the same
   sample (never host wall) and every such witness agrees within 5%; WSL
   corpora must carry such samples per device; the ROCm profiler packet and
   gfx1151 SSD admission gain a re-derived `device_clock_witness` route.
   **Explicit exception:** the CUDA activity-window calibration no longer
   refuses WSL2 — its witness is the Nsight activity window, which is
   profiler-derived and unverified on WSL2. **Kept by owner decision
   (2026-09-26);** the 5% agreement gate is what rejects bad WSL2 windows. What remains: no fleet packet recorded on the new routes; the
   x86 packet has a derived `tsc_witness` route since 2026-09-26 (per-row
   TSC vs CLOCK_MONOTONIC_RAW around the real trial region, frequency from
   separate pinned calibration intervals, the per-launch samples bound to the
   witnessed region, everything re-derived from stored integers). **Under
   WSL2 the raw clock is itself derived from the TSC, so the route shows a
   stable TSC scale bracketing the samples, not agreement with an independent
   oscillator**; its Zen 5 packet is recorded on Princess-Luna; NVIDIA has no non-profiler witness (`%globaltimer`); event-only
   recorders stay ineligible. Separately: NVIDIA timers run on the
   default stream (DEVICE-CLOCK-DISCIPLINE); Apple `kernelStartTime` does not
   measure work; and runtime libraries in empty-build-type trees compile at
   `-O0` (RUNTIME-LIB-OPT-1, **applied 2026-09-26** as `-O2` for the runtime
   libraries only, with a `runtime_library_build.json` stamp), which biased
   every comparison packet recorded from them; those packets stay stale until
   re-recorded.

### Foundation still to build, by IR level

| Level | Open | Owners |
|---|---|---|
| Frontend | Tracer as sole frontend; family selection out of `JitFn`; tracer-emitted `loc` before `_OpExtractor` deletion; masks/padding as operands | E2E-REAL-6, E2E-REAL-6F, FRONTEND-IR-MEDIUM-1 |
| Graph IR | Opportunistic folders; attribute-stamp-only passes; one `tessera.source_kind` admission for `NativeTapeToGPUPass` | [COMPILER_AUDIT](compiler/COMPILER_AUDIT.md) scorecard |
| Schedule IR | Schedule→Tile is op-name macro expansion with tile sizes from pass options (U6); every boundary exists in Python and C++ with no differential test and `LowerScheduleToTarget` is a scaffold (U3); collective placement/overlap is runtime code, not a pass | [IR_STACK review](compiler/IR_STACK_INTEGRATION_REVIEW.md) U3/U6 (unrouted), DIST-NATIVE-1 |
| Tile IR | Partial Tile rows in [`compiler_progress`](generated/compiler_progress.md) (linalg, optimizers, `game_*`, `depth_attn`); two NVIDIA tensor-valued `tile.mma` sites; `Tile_MMAOp` still `Variadic<AnyType>`; split whole-kernel/domain ops out of Tile; collapse the legality passes | W1.1 (NVIDIA closure), W3.3, IR_STACK U2 |
| Target IR | Required (not optional) contracts per op; an ODS-op→consumer gate; layout/packing/scale-layout witnesses | GOV-ODS-CONSUMER-1, LAYOUT-ALG-1, IR_STACK U5 |
| MLIR→LLVM/NVVM/ROCDL | x86 packages a prebuilt image instead of compiling the body (reuse `tessera-jit`); retire the `emit/*` source emitters family by family; compiler-owned Apple MSL endpoint; NVIDIA AOT beyond the f16 GEMM | [foundation program](compiler/INTEGRATED_COMPILER_PLAN.md#foundation-program) F2/F3 |
| Runtime | Dispatch-bridge waits and checked-output ownership; allocation- and control-flow-scoped release; ROCm host-transfer coalescing; JIT `scf.while` forward crash | DISPATCH-BREAKER, IR-NATIVE-FOUNDATION-1, ROCM-TRANSFER-RESIDENCY-1, COMPILER-DEVEX-1 |
| AD | General nested persistent tapes; native SAVE/RECOMPUTE plans; real batching; sparse coloring; native jets; KKT/IFT | [AUTODIFF plan](compiler/AUTODIFF_EXECUTION_PLAN.md): W4-PRODUCT-1 → AD-RESIDUAL-EVAL-1 → W2.4a / AD-HIGHER-1 |
| Control flow | Source-CFG recovery, multi-block regions, Presburger constraints, scan JVP/VJP | §2 above |
| Distributed | Typed fail-closed placement lattice, reshard insertion, native NCCL/RCCL and MPI beyond two ranks | DIST-NATIVE-1, TSOL-SHARD-1 |
| Numeric policy | Typed carrier below Graph IR; nvfp4 vs mxfp4 distinguishable below Graph; FP8 block-scale as a schedule change | NUMPOL-CARRIER-1, ROCM-FP8-BLOCKSCALE-1 → MXFP4-W4A8 → NVFP4-INGEST; [`dtype_flow`](generated/dtype_flow.md) |

### Optimization foundation, by layer

| Layer | Exists | Open |
|---|---|---|
| Analysis | W2.1 dataflow substrate, per-op memory effects, symbolic-dim equality | Value, alias, effect, memory-dependence and ordered-collective **consumers** (§3 above) |
| Fusion | One authoritative recognizer; Apple synthesizer F0–F5 | Synthesizer not portable through IR; consumer-driven canonicalization (W5.5); ANN admission of measured candidates (MSW-9) |
| Arbiter / autotune | D1 registry, D2 `measured_arbitrate`, D3 fallback log | **Decision #11 is not enforced in production**: neither cache key carries toolchain or delegate-ABI identity, and while `emit/autotune.py` can fail closed on compiler/resource fingerprints as *evidence* fields, only a benchmark script and a unit test ever pass `required_evidence` — the default warm-start loads rows unchecked; Decision #12 `route` is stamped per recorder, not a schema field; `measured_arbitrate` defaults to `device_repeats=3`, too few to separate candidates (AUTOTUNE-SEPARATION-NVIDIA); Apple registers no arbiter candidates; W5.2 waits on EVIDENCE-PACKET-1 + TPROF-NATIVE-1 |
| Tiling / layout | M/N/K K-loop; LayoutAssignment default-on for x86 and NVIDIA; ROCm split-K predicate keyed on occupancy (2026-09-20) | Apple/ROCm layout opt-in; the split-K predicate has no production consumer (ROCM-SPLIT-K-1) |
| Memory | `TileBufferReusePass`, `TileBufferArenaPass` on ROCm/NVIDIA | Control-flow path-max sizing; multiple dynamic arenas; measured full-model remat |
| Cost models | `target_perf.py`, T1 GEMM model, `FusionCost` | T1 failed ranking — replace, do not coefficient-tune; per-arch correlation (NVIDIA-CALIB-1, ROCM-COSTMODEL-T1, X86-CALIB-1, APPLE-CALIB-1); sm_120 and Apple roofline peaks |

### Per backend

- **NVIDIA** ([queue](backend/nvidia/todo.md)). Lane-B-pattern sweep
  (`NVIDIA-LANE-B-1`, 2026-09-26): the Graph→Tile `tessera-lower-to-gpu` /
  `nvidia-pipeline-sm*` validation route, `@jit` matmul on the hand NVRTC
  `nvidia_mma` kernel, and the Python-Tile `package_matmul` fallback all sit
  beside the scheduled route. Absorb the remaining
  Graph-input families (most already build Tile IR, so this is absorption,
  not rewrite); settle the `package_matmul` fallback as oracle or retire it
  (Decision #31, coverage comparison first); the eight delegate-contract gaps
  in NVIDIA-DELEGATE-CONTRACT-2026-08-30, starting with non-composing
  accuracy budgets; `NVWGMMALoweringPass` cannot thread the accumulator (it
  refuses with `NVWGMMA_ACCUMULATOR_DROPPED`; W1.1 step 2b);
  DEVICE-CLOCK-DISCIPLINE before any promotion; NVIDIA-CALIB-1 corpus
  descriptors; one block-index convention. Future features (not gates):
  sm_90 WGMMA, sm_100 tcgen05/TMEM.
- **ROCm** ([queue](backend/rocm/todo.md), [lane map](backend/rocm/ROCM_LANE_MAP.md)).
  The broad production lane still skips Graph/Schedule/Tile; the 78
  `generate-*` expanders are decided **(a): every family is entered from Tile IR**
  (owner, 2026-09-26; the expander stays as the Tile→Target generator, the
  Python-built directive entry goes); Lane B (the Graph→Tile GEMM shortcut
  that skipped Schedule IR) is **retired 2026-09-26** — see the lane map;
  ROCM-SPLIT-K-1 needs a production consumer (the predicate is already keyed
  on occupancy); the LDS body's remaining lever is the
  **VGPR** ceiling (the K1-blocked layout was refuted 2026-09-20 and the pad
  default corrected to 1 — see ROCM-LDS-BANKPAD-1 / ROCM-LDS-STAGE-VECTOR-1);
  `ROCM_WaitTokenOp` names a counter class but carries no count immediate, so
  it can only drain fully (gfx1250 async overlap); gfx1201 native JVP beyond
  `spectral_compound`. The fleet parts are RDNA (WMMA): WMMA shape selection
  is table-driven (`_WMMA_VARIANTS`) and `math_mode="tf32"` is correctly
  refused there. MFMA is CDNA-only — its `mfma_table.inc` has no consumer and
  Decision #14's `MFMAFullCoveragePass` was never built, which matters only
  when a CDNA part arrives. ROCM-6, RASTER-1B and COSTMODEL-T1 proceed on
  `wall_clock64`-validated paired timing rather than waiting for KFD
  counters. Future features (not gates): gfx950/942/1250/1200 (ROCM-1/3/4).
- **Apple** ([queue](backend/apple/todo.md)), on the shared MLIR/LLVM path.
  Lane-B-pattern sweep (`APPLE-LANE-B-1`, 2026-09-26): the `-full` value
  pipelines skip Schedule IR while described as Graph→Schedule→Tile→Target,
  the matmul2d corpus measured a tiling-only route, `@jit` matmul dispatches
  MPS/MTL4 from metadata, and the canonical route's Tile→Target step is Python. `gpu.matmul2d` still lowers to
  runtime symbols, not compiler-owned MSL (APPLE-MATMUL2D-1); simdgroup
  lowering stages per tile but lacks the cooperative K-slab copy
  (APPLE-SIMDGROUP-IR-1); F2 Schedule
  consumers (norm, attention, unary); `FlashAttnToAppleGPU` skips Tile; AOT-2
  B/C/D; Apple into the arbiter with a device-latency witness and dual-clock
  capture (APPLE-TIMER-WITNESS, row 32); block-scaled FP8/FP4 and GPU
  packing (APPLE-DTYPE-1, row 17).
- **x86 / AVX-512** ([queue](backend/x86/todo.md)), on the shared MLIR/LLVM
  path. `x86vector` AVX-512 lowering to replace the C-shim `func.call`
  (unlocks a `tile.mma` consumer and compiled microkernels), with the kernel
  body compiled through the `tessera-jit` linalg/vector/LLVM lane instead of
  packaging a prebuilt image; the `TileToX86Pass` assertions-build rerun
  (the dependent-dialect fix is in; Tajasarus confirms it); packed-byte
  INT4/FP8 VNNI consumer (MODEL-WEIGHT-PHYS-1); seal the AVX-512 E2E
  packets with the non-profiler timing discipline (no PMU needed). **Add a
  Zen 2 / AVX2 lane for Super-Bear** — today only `x86_64_base`
  (softmax/reduction) and `zen5-avx512` exist; compiling bodies through LLVM
  with per-host target features (`znver2` / `znver5`) is how one pipeline
  covers both CPUs. ACE: deferred until shipping hardware.

### Grouped by what unblocks it

**Software, on existing boxes**
- Make the [`alpha_scoreboard`](generated/alpha_scoreboard.md) complete:
  derive the Graph optimization/AD stage per family; key fleet packets by
  host so the two Zen 5 lanes are distinguished; register gfx1201 and
  Zen 2 fleet packets (neither lane has one).
0. Record evidence on the new timing routes (`WSL-TIMING-ADMISSION-2026-09-26`).
   **Done 2026-09-26:** the gfx1151 SSD calibrated-pairs packet — the
   production selector admits the cooperative candidate on compiler-built
   device-clock markers, no KFD ([packet](../../benchmarks/baselines/gfx1151_ssd_calibrated_pairs_20260926/README.md),
   sync `DEVICE-CLOCK-MARKER-2026-09-26`). Open: validate the NVIDIA
   `%globaltimer` marker on Super-Bear and record its SSD packet; an SSD Nsight
   activity-window packet on Super-Bear's WSL2; marker timing in
   `calibrate_gfx1151.py`; gfx1201 and Zen 2 packet adapters. **Done
   2026-09-26:** the Zen 5 x86 profiler packet on the `tsc_witness` route
   (Princess-Luna, `-O2` library; environment tags are diagnostic gaps). Its
   verdict is E2E-REAL-4's **non-regression check between the production and
   scheduled images, which are byte-identical in both rows** — a parity check
   under WSL2, not a performance promotion. Also host-keyed AVX-512 E2E packets
   on both Zen 5 boxes (`AVX512-E2E-PACKETS-2026-09-26`).
1. RUNTIME-LIB-OPT-1: applied 2026-09-26 (`-O2` runtime libraries + build
   record); open: re-measure the affected packets on their own boxes.
2. Native timing: DEVICE-CLOCK-DISCIPLINE (NVIDIA), TPROF-ROCM-TIME-1 (ROCm),
   dual-clock + MPSGraph timer (Apple) → EVIDENCE-PACKET-1 → W5.2.
3. Decision #11 versioned cache key and a Decision #12 `route` schema field.
4. E2E-REAL-6F census review, then bootstrap absorption (NVIDIA gap families,
   ROCm softmax/reduction/paged-KV first).
5. Required Target IR contracts + GOV-ODS-CONSUMER-1; extend
   `verifier_coverage` to Tile/Schedule/Target ODS.
6. NVIDIA `tile.mma` sites (W1.1), then W3.3.
7. ROCM-SPLIT-K-1; the LDS-body VGPR lever.
8. `x86vector` lowering via the `tessera-jit` path.
9. Apple `matmul2d` → compiler-owned MSL; Apple arbiter candidacy.

**Needs an owner decision**
- Live-queue IDs for IR_STACK U2, U3, U5 and U6 (only U4 is routed, as W3.3).

**Future features — tracked, not compiler gates**
- sm_80/90/100 (WGMMA, tcgen05/TMEM); CDNA gfx90a/942/950 (MFMA);
  gfx1250; gfx1200; R9700-specific parts; a second Apple device.
- Multi-GPU transports and multi-rank device packets
  ([`single_gpu_closeout`](generated/single_gpu_closeout.md)
  `multi_gpu_deferred`); the CPU / mock-rank distributed contracts stay in scope.
- ACE on x86, once a shipping processor supports it.
- Native-Linux KFD counters and bare-metal hosts: useful diagnostics, not a
  prerequisite for completing or measuring the compiler.

## Proof vocabulary

| Term | Meaning |
|---|---|
| `complete` | Every required rung in the stated scope is proven. |
| `reference` | Correct execution exists without a native target implementation. |
| `device_verified_jit` | A compiler-generated binary launched and matched its oracle on the exact target. |
| `device_verified_abi` | A shipped stable ABI launched and matched its oracle on the exact target. |
| `fused` / `packaged` | An owned implementation exists; execution or promotion proof may still be absent. |
| `artifact_only` | Compilation evidence exists without link/launch proof. |
| `partial` / `planned` | An explicit contract or evidence obligation remains. |
| explicit terminal status | The axis is closed by design with a specific reason. |

## Dashboard map

| Question | Authority |
|---|---|
| What phase is open? | [`generated/compiler_progress.md`](generated/compiler_progress.md) |
| Which operation is affected? | [`generated/support_table.md`](generated/support_table.md) |
| Which primitive contracts remain? | [`generated/s_series_status.md`](generated/s_series_status.md) |
| Which target paths launch? | [`generated/runtime_execution_matrix.md`](generated/runtime_execution_matrix.md) |
| How close is functional-complete alpha? | [`generated/alpha_scoreboard.md`](generated/alpha_scoreboard.md) |
| Which ABI symbols are real? | [`generated/runtime_abi.md`](generated/runtime_abi.md) |
| Which tests are direct or structural? | [revision-bound coverage evidence](coverage/COVERAGE_AUDIT.md#test-coverage-evidence) |
| Which verifiers are registered? | [`generated/verifier_coverage.md`](generated/verifier_coverage.md) |
| Which target rows are exact, packaged, or reference? | Generated target maps plus the backend plan |
| What work is software-actionable? | [`generated/single_gpu_closeout.md`](generated/single_gpu_closeout.md) and [`stub_surface.md`](stub_surface.md) |

## Lifecycle rule

Generated dashboards own status. The integrated compiler plan and active
backend/model plans own work. Completed sprint plans and point-in-time enablement
maps live under `archive/` and may be cited only for provenance. A historical
plan must not remain an active owner merely because source comments still use
its old work-item labels.
