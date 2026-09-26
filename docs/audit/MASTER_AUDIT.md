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

RX 9070 XT gfx1201 correctness commissioning now has an owning host, Tajasarus;
R9700-specific performance and native-Linux counter evidence remain open.
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
   cover Tile, Schedule or Target ODS.
3. **Timing that can promote.** WSL2 wall clock does not promote; NVIDIA
   timers run on the default stream; ROCm has no `/dev/kfd` counters under
   WSL2; Apple `kernelStartTime` does not measure work; and runtime libraries
   in empty-build-type trees compile at `-O0` (RUNTIME-LIB-OPT-1, proposed,
   not applied), which biases every comparison packet recorded from them.

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
| Arbiter / autotune | D1 registry, D2 `measured_arbitrate`, D3 fallback log | **Decision #11's versioned cache key is not implemented** (neither `autotune_v2.cache_key` nor `emit/autotune.py` carries toolchain or delegate-ABI identity); Decision #12 `route` is stamped per recorder, not a schema field; repeat counts too low to separate candidates (AUTOTUNE-SEPARATION-NVIDIA); Apple registers no arbiter candidates; W5.2 waits on EVIDENCE-PACKET-1 + TPROF-NATIVE-1 |
| Tiling / layout | M/N/K K-loop; LayoutAssignment default-on for x86 and NVIDIA | Apple/ROCm layout opt-in; ROCm split-K unwired and keyed on `k > 4096` instead of occupancy (ROCM-SPLIT-K-1) |
| Memory | `TileBufferReusePass`, `TileBufferArenaPass` on ROCm/NVIDIA | Control-flow path-max sizing; multiple dynamic arenas; measured full-model remat |
| Cost models | `target_perf.py`, T1 GEMM model, `FusionCost` | T1 failed ranking — replace, do not coefficient-tune; per-arch correlation (NVIDIA-CALIB-1, ROCM-COSTMODEL-T1, X86-CALIB-1, APPLE-CALIB-1); sm_120 and Apple roofline peaks |

### Per backend

- **NVIDIA** ([queue](backend/nvidia/todo.md)). Absorb the remaining
  Graph-input families (most already build Tile IR, so this is absorption,
  not rewrite); settle the `package_matmul` fallback as oracle or retire it
  (Decision #31, coverage comparison first); the eight delegate-contract gaps
  in NVIDIA-DELEGATE-CONTRACT-2026-08-30, starting with non-composing
  accuracy budgets; `NVWGMMALoweringPass` still drops the accumulator;
  DEVICE-CLOCK-DISCIPLINE before any promotion; NVIDIA-CALIB-1 corpus
  descriptors; one block-index convention. Hardware-gated: sm_90 WGMMA,
  sm_100 tcgen05/TMEM, bare-metal calibration.
- **ROCm** ([queue](backend/rocm/todo.md), [lane map](backend/rocm/ROCM_LANE_MAP.md)).
  The broad production lane still skips Graph/Schedule/Tile; the ~58
  `generate-*` expander adoption policy (a/b/c) and Lane B are undecided;
  ROCM-SPLIT-K-1 re-keyed on occupancy; the LDS body's remaining lever is the
  **VGPR** ceiling (the K1-blocked layout was refuted 2026-09-20 and the pad
  default corrected to 1 — see ROCM-LDS-BANKPAD-1 / ROCM-LDS-STAGE-VECTOR-1);
  no MFMA descriptor table and no ROCm `math_mode` consumer; `ROCM_WaitTokenOp`
  lacks a wait immediate (gfx1250 async overlap); gfx1201 native JVP beyond
  `spectral_compound`. Hardware-gated: a native-Linux KFD host (unblocks
  ROCM-6, RASTER-1B, COSTMODEL-T1, TPROF-ROCM-TIME-1), gfx950/942/1250/1200.
- **Apple** ([queue](backend/apple/todo.md)). `gpu.matmul2d` still lowers to
  runtime symbols, not compiler-owned MSL (APPLE-MATMUL2D-1); simdgroup
  lowering lacks threadgroup staging (APPLE-SIMDGROUP-IR-1); F2 Schedule
  consumers (norm, attention, unary); `FlashAttnToAppleGPU` skips Tile; AOT-2
  B/C/D; Apple into the arbiter with a device-latency witness and dual-clock
  capture (APPLE-TIMER-WITNESS, row 32); block-scaled FP8/FP4 and GPU
  packing (APPLE-DTYPE-1, row 17).
- **x86** ([queue](backend/x86/todo.md)). `x86vector` AVX-512 lowering to
  replace the C-shim `func.call` (unlocks a `tile.mma` consumer and compiled
  microkernels); the `TileToX86Pass` P0 assertions-build rerun (Tajasarus);
  packed-byte INT4/FP8 VNNI consumer (MODEL-WEIGHT-PHYS-1); no ACE/AVX10
  capability plan exists; AVX-512 E2E packets need bare-metal Zen 5.

### Grouped by what unblocks it

**Software, on existing boxes**
1. RUNTIME-LIB-OPT-1 on all four backends, then re-measure affected packets.
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
- ROCm expander adoption (a/b/c) and Lane B's disposition.
- Apple MLIR→AIR versus a supported MSL emitter (AOT-2).
- Whether Apple fp32-only accumulation is permanent (gates DIAG-PY-BACKLOG-1).
- An ACE/AVX10 capability-as-attribute plan for x86.
- Live-queue IDs for IR_STACK U2, U3, U5 and U6 (only U4 is routed, as W3.3).

**Hardware-gated**
- Native-Linux KFD ROCm host; bare-metal Zen 5 and NVIDIA hosts — the single
  gap behind every performance promotion.
- sm_90, sm_100, gfx950, gfx942, gfx1250, gfx1200; second Apple device.
- Multi-GPU / multi-rank ([`single_gpu_closeout`](generated/single_gpu_closeout.md) `multi_gpu_deferred`).

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
