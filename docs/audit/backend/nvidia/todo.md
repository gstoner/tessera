---
audit_role: plan
plan_state: landing
owner: NVIDIA backend
target: nvidia_sm120
last_updated: 2026-08-29
---

# NVIDIA compiler test-suite evaluation and rearchitecture

## Cross-backend sync `DEVICE-CLOCK-DISCIPLINE-2026-08-31`

A **shared runtime timing contract** now decides which clock a device latency
may be read from, so all four backends are assessed here per AGENTS.md.

`runtime._select_rocm_latency_ms` ranks up to three clocks for one timed loop:

1. **`wall_clock64` (in-kernel)** — a device-side counter at a constant,
   queryable rate (`hipDeviceAttributeWallClockRate`; 100 MHz / 10 ns ticks on
   gfx1151). The only one that is both kernel-only *and* independent of the
   host event API. Unlike `clock()`, its rate does not move with DVFS.
2. **HIP events**, accepted only inside a two-sided band against the host wall
   clock.
3. **The host wall clock**, which includes launch overhead and can therefore
   only make a kernel look slower. A benchmark must not be able to flatter
   itself.

**Measured on gfx1151, 20 launches of the generic fused kernel — all three
agree to four significant figures:**

| shape | wall | event | `wall_clock64` | device/event |
|---|---|---|---|---|
| 256³ | 82.6946 ms | 82.5909 ms | 82.5600 ms | 1.000 |
| 512³ | 498.0912 ms | 497.9570 ms | 497.8904 ms | 1.000 |

The ordering `wall > event > device` is exactly right: wall includes launch
overhead, the event brackets the stream, `wall_clock64` measures the kernel
span. This is a mutual validation with an **independent witness**, not the
weaker "the event agrees with the host clock".

**Two rules that came out of this and generalize beyond ROCm.**

* **`hipEventSynchronize` is mandatory; `hipDeviceSynchronize` is not the way
  to get it.** Launches are async, so without an event (or stream) sync the
  wall clock times the *enqueue*, producing a catastrophically small number
  that then drags the acceptance band down with it. A device-wide barrier does
  work, but halts every stream — it is now kept strictly as the fallback for a
  host whose event API is unusable.
* **Never time on the default stream.** Stream 0 implicitly serialises against
  every other stream, so a measurement taken while other GPU work is in flight
  is distorted by it. The generated bench entry creates a dedicated
  `hipStreamNonBlocking` stream and synchronises *that*.

**NVIDIA outcome: follow-up required — the same discipline is not yet applied
here, and one rule is measurably violated.**

The NVIDIA timers landed before this contract existed and use CUDA events
without an independent witness:

* `_nvidia_mma_gemm_device_latency` records events, synchronises, and divides —
  no wall-clock cross-check and no band. It has not been observed lying, but
  neither had HIP events before they were measured doing so.
* Both NVIDIA timers record on the **default stream** (`cuEventRecord(ev, 0)`
  and the launch bridge's `cuLaunchKernel(..., 0, ...)`), which implicitly
  serialises against every other stream.

CUDA has the direct analogue of the in-kernel clock (`%%globaltimer` /
`clock64()`), so the three-clock model transfers in shape. **It does not
transfer as evidence:** whether sm_120's event clock is trustworthy is an open
question here, and the gfx1151 agreement says nothing about it. Owed on
Super-Bear.

## Cross-backend sync `AUTOTUNE-RACED-FIELD-SYNC-2026-08-30`

PR (this branch) changes a **shared measurement contract**: an autotune
`MeasureRecord` must now declare which applicable candidates it did *not* race
(`unmeasured`), and `corpus_winner` refuses a verdict whose race was smaller
than the one the live registry would hold. All four backends read this corpus,
so all four are assessed here per AGENTS.md.

**The defect, measured in the committed corpus.** Every device-timed row was
missing exactly the candidates that had no `measure_device_latency`: matmul
raced 2 of 4, attention 5 of 6, fused_region 6 of 10, gated_matmul 6 of 7.
`_measure` scored an untimeable candidate `float("inf")`, so it lost silently,
and the record stored a `winner` with nothing to say the field had been
reduced. The verdicts read as "the compiled kernel is faster"; they meant "the
compiled kernel was the only one that could be timed". End-to-end rows are
unaffected — `measure_latency` just calls `run()`, so they raced the full field.

**Why it matters more than bookkeeping (sm_120, f16, device-resident):** with
all four NVIDIA matmul candidates raced for the first time, the
**compiler-emitted PTX lane wins at every shape** — 0.0095 / 0.0291 / 0.1930 /
1.4719 ms at 256/512/1024/2048³ against the hand-tuned delegate's 0.0155 /
0.0431 / 0.3202 / 2.4509 ms, i.e. **1.5–1.7× faster**. That candidate had been
excluded from every device measurement ever recorded. A biased corpus did not
merely mis-rank; it hid the fastest kernel in the registry.

**NVIDIA outcome: parity validated, on device (RTX 5070 / sm_120).** This
backend owns both the defect and the fix. `tileLaunchConfig` now carries the
block-index convention as a flag (`columnMajorGrid`, the same one
`invokeMmaGemm16` already took), so `tessera_mma_gemm_f16` no longer returns
rc=5 and the emitted lane has a device timer. The grid was verified
empirically, not by construction: at 2048×128×256 and 128×2048×256 the two
orientations time identically (0.0162 / 0.0161 ms, ~8.3 TFLOP/s), whereas a
transposed grid would put most blocks fully out of bounds and finish much
faster.

**Consequence for `NVIDIA-TIER-PRIORITY-IS-WRONG-AT-SCALE-2026-08-30`, which
understated the case.** That entry compared the delegate against the Tile lanes
only and concluded the compiled route wins at 1024³+ by 2.3–16%. With the
emitted lane in the race the margin is 1.5–1.7× at *every* shape including
256³, where the entry had the delegate winning. Read the table above, not that
one, for the ranking.

**Follow-up owned here — CLOSED 2026-09-01, and this paragraph's own
assessment was wrong twice.** See
`APPLIES-TO-SHAPE-BLIND-2026-09-01` below. What it got right: `applies_to(region)`
is shape-blind, and the emitted lane is aligned-only. What it got wrong:

* *"applicable-but-unmeasurable … every ragged device verdict is refused …
  safe and honest."* True of the **device** path only, where
  `measure_device_latency` already returned `None` for a ragged shape. On the
  **end-to-end** path `_measure` timed `run`, and `run` on a ragged shape
  returns `region.reference(...)` — so the row recorded numpy's latency under
  the kernel's name. Not unmeasurable: **mis-measured**.
* *"selection falls back to tier priority"* — stated as the safe outcome, and
  it is the unsafe one. Tier priority picks the aligned-only lane, which then
  declines to numpy while a lower-tier lane that could serve the shape goes
  untried.

Fixing it also did **not** require the shared-signature change predicted here.
An additive `applies_to_inputs(region, *inputs)` with a `True` default left all
17 existing `applies_to` implementations untouched.

## Cross-backend sync `DELTA-OPERAND-ABI-SYNC-2026-08-30`

PR #653 changes a **shared Graph IR ABI**: the delta-rule family
(`gated_deltanet`, `kimi_delta_attention`, `modified_delta_attention`) now
declares its optional tensor operands in `graph_ir._KEYWORD_OPERANDS` as
`(gate, beta, decay)` and emits `has_gate`/`has_beta`/`has_decay` presence
flags from both frontends. All four backends consume this ABI, so all four are
assessed here per AGENTS.md.

**What was wrong.** Undeclared, the AST emitter appended keyword operands
*sorted by name*, so `gated_deltanet(q, k, v, gate=g, beta=b, decay=d)` emitted
them as `(beta, decay, gate)`. Order alone would not have been enough either:
with `[Q, K, V, %x]` the lone optional sits at index 3 whichever slot it fills.

**Load-bearing fact for every backend: no producer had ever set these flags.**
`has_gate`/`has_beta`/`has_decay` were read by four executors and written by
none. The compiled ROCm, NVIDIA and x86 deltanet lanes all compute
`need = 3 + has_gate + has_beta + has_decay` and raise when that disagrees with
the operand count — so with the flags absent they accepted **only** the
three-operand form and raised on any traced call carrying `beta`/`decay`. This
PR is therefore what makes those lanes reachable with optionals at all. That is
a behaviour change on three backends and each owes its own exact-device proof;
the Apple result does not transfer to any of them.

**NVIDIA outcome: follow-up required — exact-device proof owed on sm_120.**

Two NVIDIA consumers change behaviour and neither has been run on hardware for
this PR:

* `_execute_nvidia_deltanet_compiled` now receives the presence flags it always
  parsed for, so the compiled sm_120 lane becomes reachable with
  `gate`/`beta`/`decay` for the first time.
* `native_vjp_plugins._nvidia_sm120_deltanet_backward` **stops guessing.** It
  previously derived presence from operand *variable names* and, when no name
  matched, fell back to "trailing operands fill the slots in declaration
  order" — which mis-binds whenever a caller names its locals anything but
  `beta`/`decay`. `gated_deltanet(q, k, v, beta=b, decay=d)` presents `b` and
  `d`, so the fallback fired and bound beta into the gate slot: the VJP
  differentiated a different recurrence and returned gradients that were wrong
  but finite. It now raises rather than infer.

Owed on Super-Bear: forward parity for the compiled lane with each optional
subset, and a gradient check for the backward plugin. Neither transfers from
the Apple result.

## `NVIDIA-DELEGATE-CONTRACT-2026-08-30` — the fast-path boundary is real; NVIDIA goes first

**Enabling step for the bootstrap prune, and it had to land before any
deletion.**

*Corrected 2026-08-30, by measuring rather than assuming.* This section first
said "the 19 NVIDIA bootstrap packagers contain legitimate fast paths — vendor
library entries, hand-tuned kernels, inline PTX". **They contain none.**
`nvidia_native.py` has zero references to NVRTC, cuBLAS/cuDNN/CUTLASS, any
`.so`, or raw device source; 13 of its 19 bootstrap packagers construct Tile
IR and compile it through `tessera-opt`. NVIDIA's real delegation surface is
`ptx_emit.py`, `emit/nvidia_cuda.py` and `runtime.py` — different files.

Across all four backends the same holds: **24 of 34 bootstrap packagers are
IR-constructing, 1 delegates** (`bootstrap_prune_gap.md`). So the prune is
overwhelmingly an *absorption* job — moving Graph → Schedule → Tile into the
compiled route — not a delegation-migration job. The boundary below was still
the right thing to land first, but for the delegation surface that actually
exists, not for these packagers.

NVIDIA was
chosen over ROCm because it has both the largest gap (19 of 34 bootstrap
packagers) and **working profiling tools**, which matters more than gap size:
Decision #28's arbiter is *measured*, so a delegation boundary on a target
that cannot be profiled is bookkeeping rather than a candidate.

**What `tessera_nvidia.kernel_call` was.** A summary line and nothing else.
It inherited `TesseraNVIDIA_Op`'s shared `attr-dict`, so `callee` — the single
fact naming *what is delegated to* — rode as an unvalidated discardable
attribute. An emitter could name any symbol, or none, and still verify. The
dialect header says why it existed: Python emitters "may add
`tessera_nvidia.kernel_call`", and registering it "keeps the emitted surface
parseable". It was a parse-compatibility stub **for the bootstrap packagers
being pruned** — Decision #29's anti-pattern exactly.

**Both pathways are now declared, as two ops rather than one with a mode.**

| Op | Delegate is | Required contract |
|---|---|---|
| `kernel_call` | a named CUDA kernel or host C-ABI symbol | `callee`, `arch`, `binding` ∈ {`cuda_kernel`,`c_abi`}, `provenance` ∈ {`vendor_library`,`handwritten_kernel`}, `accuracy` |
| `inline_ptx` | PTX text embedded in the artifact | `ptx`, `constraints`, `arch`, `accuracy`, optional `has_side_effects` |

They are separate ops because the delegate differs in kind: one is a binding
resolved at link/launch time, the other is text carried in the artifact. An
empty `callee` is an unresolved-symbol error; an empty `ptx` body is a
*silently successful no-op*. One op with a mode attribute would need a
verifier that decides which half of its own attributes to trust — the shape
that lets a malformed candidate through.

**The attributes are the arbiter's inputs, which is what "real" means here.**
`accuracy` is the budget half of "fastest *in-budget* candidate": a delegate
claiming `tolerance_bounded` must state `tolerance`, and `reference_exact`
must not carry one, because two contradictory claims leave a reader unable to
tell which is honoured. It is a semantic key and never defaults (#21a).
`provenance` is what lets the arbiter tell delegated from compiler-generated
work when scoring; `binding` separates two pathways whose launch costs and
failure modes differ.

*Evidence (The-Super-Bear, full driver):* `tessera-nvidia-opt` builds clean;
the positive fixture parses both ops with full attributes; the new negative
fixture `nvidia_delegate_contract_invalid.mlir` rejects **7 cases** —
empty callee, bounded-without-a-bound, exact-carrying-a-tolerance,
non-positive tolerance, unknown `binding`, empty constraints, empty ptx.
NVIDIA lit suite **60/60**.

*Arbiter integration landed:* `DelegatedCandidate`
(`emit/delegate_contract.py`) derives tier from `provenance` and the F4 budget
from `accuracy`/`tolerance`/`tolerance_rel`, so a delegate cannot claim in
Python a budget it did not declare in IR. The ROCm equivalent is still owed.

### Gaps found by stress-testing this design (2026-08-30)

Two were live defects in the contract as first shipped and are **fixed**:

* **Determinism was undeclarable.** Tessera guarantees
  `@jit(deterministic=True)`, and a split-K delegate accumulating with atomics
  is not reproducible run to run — the arbiter could have selected one inside
  a deterministic region. `determinism` is now a required enum. Same shape as
  the Decision #5 scar: a guarantee defeated through a path nobody checked.
* **The accuracy claim was absolute-only** while `Candidate` already carried
  both atol *and* rtol. An absolute bound is meaningless without the result's
  magnitude — 1e-6 is vacuous at 1e6 and unsatisfiable at 1e-9 — so a delegate
  whose real claim is relative had to overclaim. `tolerance_rel` added;
  either or both now satisfy a bounded claim.

Open, ordered by whether the design is *wrong* versus merely incomplete:

1. **Per-op accuracy budgets do not compose (mathematical, unsound as stated).**
   Five delegates each within 1e-3 do not give an end-to-end result within
   1e-3; propagation depends on conditioning. A graph can be assembled
   entirely from in-budget candidates and land outside any budget with nothing
   detecting it. Needs a graph-level check, or the composition claim must be
   withdrawn.
2. ~~Fusion foreclosure is not costed.~~ **Closed 2026-08-30 — and the bias
   was worse than first described.** `arbitrate()` picks by **tier** by
   default, and `Tier.HAND_TUNED` is the highest, so a delegate won
   *outright, before anything was measured*; on the measured path it won
   because the latency excluded the work it displaced. Both paths preferred
   delegates on exactly the graphs where fusion is the win.

   Fixed structurally rather than with a penalty. A delegate now declares
   `covers` (`root_only` | `whole_region`), and `DelegatedCandidate.applies_to`
   **declines a region it implements only part of**. A penalty would have been
   a guess at foregone DRAM traffic that then had to outweigh a tier bonus;
   "this candidate does not serve this region" is a fact the delegate
   declared. If the delegate-plus-separate-epilogue plan really is faster,
   that is a comparison of *plans* and does not belong as a peer candidate.
   Whole-region hand-tuned kernels still compete, so the #28 governing rule
   (never cap the leads) is preserved.
3. **`kernel_call` does not verify operands against the callee ABI.** The op
   requires `constraints` for inline PTX on the argument that unstated
   constraints become silent miscompiles — and then leaves the symbol path
   unchecked. Inconsistent; `tessera_x86.abi_call` has the same hole.
4. **No delegate versioning.** cuBLAS 12 and 13 differ numerically and in
   performance; with no version or ABI hash, a cached measurement from one
   applies to the other. That is the stale-baseline failure this queue already
   recorded once for the Krylov ratchet.
5. **Accuracy uses a vocabulary parallel to `numeric_policy`.** Decision #15a
   puts accumulator contracts there; `reference_exact` cannot even be honoured
   for float reductions, where result depends on accumulation order. This is a
   Decision #32 information-loss issue inside the op meant to prevent them.
6. **`has_side_effects` is one bit.** Reads, writes and barriers have
   different legality; one bit forces treating any side-effecting asm as a
   full barrier, which costs real performance. MLIR has `MemoryEffects`.
7. **Shape-bucket boundaries are undefined.** The sm_120 macro-CTA threshold
   is one number (67,108,864 FLOPs) measured once under WSL, and this file
   already says it is not global selector authority. Coarse buckets apply an
   M=4096 measurement at M=17.
8. **No measurement statistic or hysteresis.** "Fastest" by mean, median or
   min, over how many reps? Without a minimum effect size the arbiter thrashes
   inside noise.
9. **The arbiter is on the wrong side of the prune (architectural).** It lives
   in Python. Pruning the Python backend path while keeping a Python arbiter
   keeps the seam. Probable resolution: arbitration is legitimately *outside*
   the IR pipeline because it requires execution, like PGO — but then the
   contract must be stated: IR declares candidates, an orchestrator measures
   and selects, selection is recorded back as an attribute.
10. **`tessera_rocm.mfma` vs `rocdl.mfma`** — measure whether the Target IR op
    carries a contract ROCDL cannot, per Decision #19's amended membership
    test. If it mirrors, it is Decision #31 duplication.

**Sequencing rule for the Apple/x86 operator expansion.** Add each op only
when its producer and its consumer land with it. Apple needs ~12 ops and x86
~8; landing the families ahead of the passes that produce them manufactures
exactly the unconsumed-declaration anti-pattern (Decision #29) this contract
work exists to remove. One op proven end-to-end beats twelve declared.

**And when x86's `avx512_gemm_microkernel` is decomposed into primitives, the
microkernel must survive as a Tier-3 candidate rather than being replaced.**
If it is hand-scheduled, decomposing it and hoping LLVM re-schedules is the
"generic IR caps the ceiling" trap applied within x86 — the arbiter should
decide, not the refactor.

Cross-backend sync `AVX512-MARKER-AND-AMX-CONSUMER-2026-08-30` — **shared
marker vocabulary and conftest boundary changed; per-backend outcome below.**
`hardware_avx512` joins `policy.MARKERS`, the PR marker expression and its
four verbatim copies, `pyproject.toml`, and the device-accounting families.
`conftest` now consumes `hardware_avx512` and `hardware_amx` centrally,
matching the existing `hardware_nvidia` / `hardware_apple_gpu` boundaries.

*NVIDIA outcome: parity validated — no behaviour change, and the reason
matters.* The `hardware_nvidia` arm of `pytest_runtest_setup` is checked
**before** the two new arms and returns early, so no NVIDIA lane can be
diverted by them; no test under `tests/device/nvidia/` carries
`hardware_avx512` or `hardware_amx`. The new PR-expression term
(`not hardware_avx512`) deselects nothing here for the same reason.

One fleet fact worth recording, because it is the opposite of the intuition:
**The-Super-Bear has no AVX-512.** Its Threadripper 3970X (Zen 2) reports no
`avx512f`, so despite building the x86 backend (`TESSERA_BUILD_X86_BACKEND=ON`)
that box probes `avx512=False` and its x86 lanes skip honestly. Princess-Luna
(Zen 5) is the only AVX-512 host in the fleet, which is also why
`hardware_amx` must never be used to mean "x86 hardware" — see the standing
section in `docs/audit/backend/x86/todo.md`.

## Cross-backend sync `DELEGATE-CONTRACT-SYNC-2026-08-30`

PR #652 changed two **shared** runtime contracts, so all four backends are
assessed here per AGENTS.md:

1. `Candidate.accuracy_budget(region)` — a new hook on the shared arbiter
   base class. `candidate._as_runner()` now resolves the F4 oracle's budget
   through it instead of reading `accuracy_atol` off the class.
2. `DelegatedCandidate` gained a `name` override and a per-dtype contract
   *family* (`variants`), so one delegate may bind a different callee per
   storage dtype and still derive tier and budget from declared IR.

**Measured blast radius (37 registered candidates: nvidia 32, rocm 3, x86 2,
apple 0): exactly one overrides `accuracy_budget`** — `nvidia_mma_gemm_shipped`.
Every other candidate inherits the base implementation, which returns
`(self.accuracy_atol, self.accuracy_rtol)`: the same two values the arbiter
previously read directly, at the same call site. That equivalence is static,
not a measurement, so no sibling backend owes a device re-proof for change (1).

**NVIDIA outcome: parity validated, on device.** The delegate is this PR's
subject. `tests/device/nvidia/test_shipped_gemm_delegate.py` — 14 passed on
sm_120 (RTX 5070): declared contract, both dtype callees executing, the
declared budget holding across K=32..4096, and device-resident latency for the
delegate and both compiled Tile lanes.

**Follow-up owned here:** `nvidia_mma_gemm_emitted` still has no device timer
(two block-index conventions, see
`NVIDIA-TIER-PRIORITY-IS-WRONG-AT-SCALE-2026-08-30`), and shape-bucketed
measured selection is not yet wired into the `OP_MATMUL` path.

## `NVIDIA-TIER-PRIORITY-IS-WRONG-AT-SCALE-2026-08-30` — measured, not argued

**The first thing the delegate's device timer produced, and it contradicts the
arbiter's default.** Decision #28 displaces a hand-tuned kernel when a compiled
one measures **faster and in accuracy budget**. On sm_120 (RTX 5070), f16,
square, device-resident CUDA-event timing, spreads of 0.000–0.008 ms across
repeats:

| shape | `nvidia_mma_gemm_shipped` (T3) | `nvidia_tile_matmul_shared` (T2) | faster | max\|err\| |
|---|---|---|---|---|
| 512³ | **0.043 ms** | 0.059 ms | delegate, by 37% | both 2.48e-05 |
| 1024³ | 0.320 ms | **0.312 ms** | compiled, by 2.3% | both 6.10e-05 |
| 2048³ | 2.448 ms | **2.051 ms** | compiled, by 16.2% | both 1.54e-04 |

The error columns are **equal at every shape**, so the in-budget half is
satisfied outright. The displacement condition therefore holds at 1024³ and
above — and `arbitrate()` still returns the delegate, because tier priority is
the default and D2's measured loop is not wired into this path.

Two things follow, and neither was visible before:

* **The compiled Tessera kernel beats the hand-tuned one at scale.** That is a
  result about the compiler, not about the arbiter.
* **The crossover is shape-dependent**, which is the concrete argument for
  shape-bucketed measured selection rather than a single global winner. A
  flat "measurement beats tier" switch would regress 512³ by 37%.

**Do not read this as "delete the delegate."** It wins by 37% at 512³, and
Decision #28's lead-safety exists precisely so a crown-jewel lane is displaced
per shape by evidence rather than wholesale by policy.

**Why it was invisible until now.** End-to-end wall time ranks the two the
*other* way — 9.4 ms vs 33.1 ms at 2048³ — because it is host-dominated: the
Tile lane spends 2.99 ms on device inside 34.0 ms of wall time, and the two
lanes do not share a host path, so e2e compares numpy conversions. The Tier-3
lane had no device timer at all, so the honest comparison could not be made.
Pinned by `tests/device/nvidia/test_shipped_gemm_delegate.py`.

**Open follow-ups.**
1. Wire shape-bucketed measured selection into the `OP_MATMUL` NVIDIA path so
   the 1024³+ crossover is acted on. The `measure` hook and the autotune
   corpus already exist; nothing calls them for this bucket.
2. `nvidia_mma_gemm_emitted` still has no device timer. The NVIDIA backend
   carries **two block-index conventions**: `ptx_emit` and the shipped AOT
   kernel map x→M, y→N, while `NVIDIALowering.cpp` and the launch bridge's
   `benchmarkTileGemm16` map x→N, y→M. Driving the emitted kernel through the
   harness returns rc=5, and registering its geometry would launch a
   transposed grid (at 512×512: rows to 1024, columns only to 256 — half the
   output unwritten, with a plausible-looking latency). Unify the convention,
   or give the harness an explicit axis-order field. A unit test pins the
   current mapping so "fixing" one side fails loudly.

## `SM120-BUILD-CONFIG-RESOLVED-2026-08-30` — there was no trade; use CUDA=ON

**Superseded the "fleet-config decision" framing below: configuring the NVIDIA
backend on Super-Bear costs nothing, because that box has CUDA.** The lean
driver that carves out core/x86/Apple registration is gated on

```cmake
(TESSERA_BUILD_NVIDIA_BACKEND AND NOT TESSERA_ENABLE_CUDA)
```

— it only fires for a **CUDA-less** NVIDIA build. `tools/tessera-opt/CMakeLists.txt`
says so directly: "A backend built against a real toolchain (CUDA or HIP) IS a
full build and gets the core dialect." Measured: configuring
`build-nvidia-cuda/` with `-DTESSERA_ENABLE_CUDA=ON
-DTESSERA_BUILD_NVIDIA_BACKEND=ON -DTESSERA_BUILD_X86_BACKEND=ON` yields
`tessera-opt --tessera-build-info` → **`build profile: full`, features
`… core-tessera-ir … nvidia-backend … x86-target-ir`** — all three at once.
The existing `build/` was left untouched; point tests at the new tree with
`TESSERA_BUILD_DIR=build-nvidia-cuda`.

## `SM120-STAGING-ROUTING-DIAGNOSED-2026-08-30` — the filed description was wrong on every count

With the full driver the four tests get far enough to say what is actually
happening, and **it is not "4 stale shared-staging assertions".** They fail at
**three different assertions**, and only one of them is about staging:

| storage | shape | `tile.matmul_kernel` in Tile IR | `ab_stage` in Target IR | PTX entry | fails at |
|---|---|---|---|---|---|
| f16 | (16,8,16) | ✗ | ✗ | `nvidia_sm120_scheduled_matmul_…_kernel` | L495 `tile.matmul_kernel` |
| f16 | (37,29,23) | ✓ | ✗ | `nvidia_sm120_scheduled_matmul_…_kernel` | L500 entry name |
| bf16 | (16,8,16) | ✗ | ✗ | `nvidia_sm120_scheduled_matmul_…_kernel` | L560 `tile.matmul_kernel` |
| bf16 | (37,29,23) | ✓ | ✗ | `nvidia_sm120_scheduled_matmul_…_kernel` | L564 `ab_stage_bf16` |

**Root cause: `nvidia_schedule="shared"` is silently dropped on the
scheduled-matmul route.** All four requests take
`package_scheduled_matmul`, whose entry is named by
`_SM120_SCHEDULED_MATMUL_PREFIX`; `package_native`'s `kind` dispatch forwards
`nvidia_schedule` **only** on the fall-through `package_matmul` branch. The
tests request `shared` and then assert `package_matmul`'s artifacts (entry
`tessera_tile_matmul_shared_*`, the `__tessera_sm120_ab_stage_*` global,
`tile.matmul_kernel`), so they are describing a route the compiler no longer
sends them down.

Two things this settles, and one it does not:

* The earlier source-read holds: `emit_matmul_tile_ir(schedule="shared")` does
  emit `warps = 4 : i64, staging = "shared"`, and `NVIDIALowering.cpp:1228`
  consumes it with no shape guard and no silent fallback. The Python and C++
  halves of the *shared* route are fine. Nothing is quietly re-routing small
  shapes away from staging.
* Within the scheduled route there **is** a shape split — (16,8,16) has no
  `tile.matmul_kernel` at all while (37,29,23) does — matching the
  `sm120_scheduled_typed_16x8_mn` vs `macro_cta_32x32_mn` policy selection.
* **Resolved 2026-08-30 (project direction): the compiled route is
  authoritative and `nvidia_schedule` does NOT select it.** The Tessera
  foundation is core MLIR/LLVM → Tile IR → codegen; hand-written NVIDIA/CUDA
  kernels are not what should fall out of a compile. `driver.py:526` already
  encodes this — the scheduled route is taken whenever `tessera-opt` is
  available, and `package_matmul` is the fallback for when it is not. So
  `nvidia_schedule` steers only that fallback.

  What was wrong was the silence, and that is fixed. `driver.py` now emits
  **`SCHEDULE_KEY_NOT_HONORED_ON_COMPILED_ROUTE`** (registered in
  `diagnostic_codes.py`, severity `warning`) when a fallback-only key is
  supplied on the compiled route. `"auto"` is deliberately not reported: it
  means "you choose", which is what the compiled route does, and warning on it
  would train people to ignore the diagnostic.

  The four tests now assert the **contract** rather than a spelling — entry
  carries the `nvidia_sm120_scheduled_matmul_` prefix, the launch policy is one
  of the two scheduled policies, and the k contract goes through the existing
  producer-aware `_assert_canonical_k_loop` rather than pinning
  `canonical_k_loop` (which the typed-16x8 route legitimately does not emit).
  Two new tests cover the diagnostic itself, in both directions. Measured on
  sm_120 with the full driver: `test_e2e_spine_native.py` **304 passed, 0
  failed** (was 4 failed at three different assertions).

  The diagnostic paid for itself immediately: it surfaced 16 warnings per run
  from `test_canonical_sm120_k_loop_shape_matrix`, which was still supplying
  `nvidia_schedule="shared"` on the compiled route. That key is now dropped
  there. One site remains by design —
  `benchmarks/e2e_spine/record_sm120_packet.py:256` passes `"direct"`, which is
  inert on any host with `tessera-opt` but **does** change the fallback's
  choice on a host without it (`auto` resolves to `shared` for fp16/bf16), so
  removing it would be a silent behaviour change on that path. Left alone; the
  diagnostic will tell whoever runs it.

* **Follow-on, not done here: Decision #31 on this boundary.** Two packagers
  now serve one IR-level boundary — the compiled scheduled route and the
  templated `package_matmul`. Decision #31 allows exactly one production
  lowering per boundary; a second must be either a **declared oracle with a
  differential test** or deleted. Given the direction above, the fallback is
  the one to declare or retire. Decision #31's own ordering caveat applies —
  do not collapse it before the compiled route demonstrably carries what it
  carried — so this wants a scoped plan with a coverage comparison
  (which `native_package_kind` families reach which packager, and what happens
  on a host with no `tessera-opt`), not a drive-by deletion.

## `SM120-BASELINE-IS-BUILD-DEPENDENT-2026-08-30` — read before trusting any sm_120 suite count

**The Super-Bear device-suite baseline of "5 failed / 844 passed" is only
valid for a build with `TESSERA_BUILD_NVIDIA_BACKEND=ON`. That box is
currently configured OFF, and the identical commit then reports 34 failed /
815 passed.** Measured 2026-08-30, both numbers on `main`, same box, same
GPU, differing only in that cmake flag.

The mechanism is not subtle once seen: without the NVIDIA backend the
`tessera-opt` in `build/` never registers the NVIDIA Target IR dialect, so
every scheduled-matmul lane dies at the `--tessera-schedule-to-tile`
boundary with `SM120 scheduled matmul requires the registered NVIDIA Target
IR dialect`. Thirteen tests in `test_scheduled_matmul_consumers.py` fail in
2.5 s having touched no GPU at all. Nothing in the suite output says "your
build cannot evaluate these lanes" — it reads as a code regression, and a
control run on `main` is the only thing that tells the two apart.

**Three configurations, and do not collapse them** (a README edit did, and
review caught it). `TESSERA_BUILD_NVIDIA_BACKEND=ON` **always** builds the
hardware-free Target IR spine — `src/compiler/codegen/tessera_gpu_backend_NVIDIA/CMakeLists.txt`
says so in its header, and `tools/tessera-opt/CMakeLists.txt` links
`TesseraNVIDIAIR`/`TesseraNVIDIAConversion` under `if(TARGET
TesseraNVIDIAConversion)`, which is not gated on leanness.

| Config | NVIDIA Target IR | Core spine | Scheduled lanes |
|---|---|---|---|
| backend ON + `ENABLE_CUDA=ON` | registered | linked + registered | run |
| backend ON + CUDA off (**lean**) | **registered** | linked, **not registered** | unavailable — missing core *registration*, not the dialect |
| backend OFF (what this box has) | not built | linked + registered | fail with `requires the registered NVIDIA Target IR dialect` |

The middle row is the **supported host-free artifact configuration** that
Decision #19's hardware-free Target IR exists to enable; `_tessera_opt_lean_permitted`
lists `nvidia-backend` explicitly. Saying a CUDA-less NVIDIA build "never
registers the dialect" erases it and contradicts that contract — the symptom
above belongs to the third row only.

**Consequence for the open staging item.** The four
`__tessera_sm120_ab_stage_bf16` assertions were recorded as "stale
shared-staging assertions — a routing question". **That characterisation
does not currently reproduce and should not be acted on until it is
re-measured.** On this box the four tests
(`test_canonical_sm120_{bf16,request}_*`, two shapes each) never reach the
staging assertion: they fail 1.4 s in, at the same missing-dialect boundary.
Whatever was seen last session was seen on a differently configured build.

What *is* established about that item, from source rather than from a run:
the Python side is correct end to end. `emit_matmul_tile_ir(schedule="shared")`
emits `warps = 4 : i64, staging = "shared"` into the Tile IR, and
`NVIDIALowering.cpp:1228` reads that attribute with **no shape guard and no
silent fallback** — if `staging == "shared"` arrives, the buffer is
materialised or the pass hard-errors. So the routing question, if one
survives re-measurement, is about which op the attribute reaches, not about
small shapes being quietly re-routed. The scheduled path is the suspect:
`package_native`'s `kind` dispatch forwards `nvidia_schedule` only on the
fall-through `package_matmul` branch, and `package_scheduled_matmul` selects
`sm120_scheduled_macro_cta_32x32_mn` vs `sm120_scheduled_typed_16x8_mn` from
the entry name without consulting the requested schedule. If a caller's
`nvidia_schedule="shared"` is being dropped there, Decision #21a applies —
a performance key may fall back, but not silently.

**Corrected 2026-08-30 — see `SM120-BUILD-CONFIG-RESOLVED` above.** This
section originally said turning the NVIDIA backend on would carve out
core/x86/Apple registration, making it a fleet-configuration trade. That is
true only for a **CUDA-less** NVIDIA build; Super-Bear has CUDA, so
`-DTESSERA_ENABLE_CUDA=ON -DTESSERA_BUILD_NVIDIA_BACKEND=ON` gives a full
driver with all three registered and there is no trade to make. The
build-dependence of the suite count, which is the point of this section,
still stands.

Cross-backend sync `HOLLOW-GREEN-GATES-2026-08-30` — **shared test infra
changed; per-backend outcome below.**
A pytest session ledger (`tests/_support/device_accounting.py`) now tallies
executed-vs-skipped per hardware family and **fails the session** when a
family skipped everything on a host that plausibly has the device. It exists
because `pytest tests/device/nvidia/` on this box once reported 454 passed /
395 skipped / exit 0 while running zero GPU work.

*NVIDIA outcome: parity validated, with follow-up.* Eight files under
`tests/device/nvidia/` carried no hardware marker
(`test_{fft_workspace,optimizer_reverse,philox_jvp,plugin,rng_compiled,spectral_autodiff,spectral_jvp,spectral_policy}.py`);
they were invisible to both the PR-lane deselection and the new ledger, and
they *failed* rather than skipped on a non-CUDA host. All eight are marked,
and `conftest` now consumes `hardware_nvidia` centrally, matching the
existing `hardware_apple_gpu` boundary — verified on the Mac, where those
four optimizer tests changed from hard failures to honest skips. Follow-up:
the device suite count above cannot be re-baselined until the build-config
question in this section is settled.

Cross-backend sync `ADAFACTOR-BIAS-CORRECTION-2026-08-30` — **shared numerical
policy changed; per-backend outcome below.**
`optim.adafactor_decay` makes the Adafactor second-moment decay step-dependent
(`b2_t = b2*(1 - b2^(t-1))/(1 - b2^t)`), removing an early-step update
inflation of 1/sqrt(1 - b2^t) — 31.6x at step 1, 10.0x at step 10, 1.26x at
step 1000 for the default beta2. The correction is applied HOST-SIDE as a
scalar decay, so **no kernel ABI moves**: every physical kernel already takes
`beta2` as a scalar and receives the effective value instead of the nominal
one. The flat op gained an optional `step` kwarg matching the `adam`/`adamw`
ABI beside it.

Two contract details a backend owner needs to know. `state["v"]` now carries
the DEBIASED estimate rather than the raw EMA, so the state dict grew a
`v_representation` marker and a state without one is migrated on load rather
than misread. And an absent `step` is NOT treated as step 1 — `decay(b2, 1)`
is exactly 0, so defaulting would have made a stateful caller that never
passes one discard its own moments; such a caller keeps the legacy
uncorrected decay.

*NVIDIA outcome: parity validated 2026-08-30 (was follow-up required).*
`sm120_adafactor_*` receives the effective decay through the existing scalar;
`tests/device/nvidia/test_optimizer_reverse.py` was migrated to pass `step`.
**The owed exact-device run is done.** On The-Super-Bear (RTX 5070, sm_120,
CUDA 13.3, `scripts/_nvidia_env.sh` sourced) that file is **4 passed, 0
skipped**, including
`test_sm120_adafactor_full_and_factored_exact_certificates`, which compares
the device reverse package against `get_vjp("adafactor")` for both the full
and factored topologies. Zero skips is the load-bearing half of that
sentence: the same file reported a clean *skip* on this host for as long as
the driver shim was off `PATH`.

Cross-backend sync `P3-DEVICE-VERIFIED-2026-08-30` — **the two NVIDIA rows
owed by `P3-SOURCE-ONLY` are now measured, and one of them was a regression.**

* `emit/nvidia_solver_krylov.py` `tsr_matvec` — the warp-per-row rewrite was
  shipped on a reasoned access-pattern claim. Measured on an RTX 5070
  (sm_120), medians of 9 reps, device_event: **dense_cg 0.44-0.63x (a
  REGRESSION of up to 2.3x)** and **dense_gmres 1.22-1.56x (a win)**. The
  coalescing argument was correct and still lost, because a COOPERATIVE
  launch caps the grid at what stays resident, so warp-per-row also buys 32x
  fewer rows in flight. The solvers no longer share one matvec:
  `tsr_matvec_scalar` for CG, `tsr_matvec_warp` for GMRES, with the table in
  the source. Re-measured after the split: CG back to 1.00-1.06x of scalar,
  GMRES keeps 1.24-1.56x. `benchmarks/baselines/nvidia_sm120_solver_krylov_performance.json`
  was recorded with the OLD matvec and **passed throughout the regression** —
  re-recorded at 15 reps / 5 warmup, and the ratchet now measures reality.
* `emit/nvidia_cuda.py` flash-backward cleanup — the 20 Krylov/solver device
  tests and the flash-backward route tests pass on sm_120. An induced
  allocation failure is still not exercised; that remains the honest gap.

Also closed here: the `rc=5` invoke failure (the runtime dispatches scheduled
sm_120 matmuls by NAME PREFIX while the compiler named the kernel after the
caller's Graph function) and the sm_120 packager reading matmul epilogue
edges from `op.kwargs` when the verifier requires operands. Device suite:
**81 failed -> 5 failed / 844 passed.** The 5 remaining are 4 stale
shared-staging assertions (`__tessera_sm120_ab_stage_bf16`, pre-existing and
a routing question, not a test-editing one) and NCCL not being installed.

Cross-backend sync `P3-SOURCE-ONLY-2026-08-30` — **two rows are fixed in
source and have never run on a GPU; they are this queue's to close.**
The P3 batch changed two NVIDIA emitters with no CUDA host available:

* `emit/nvidia_cuda.py` flash-backward — `TSR_ATOMIC_ENTRY` and the f16
  wrapper now free through a `goto fail` block, and the atomic entry CHECKS
  its H2D copies, which it previously fired unchecked (a failed upload
  yielded a confidently wrong gradient). Needs a real run, and ideally an
  induced allocation failure confirming the cleanup frees exactly what was
  allocated.
* `emit/nvidia_solver_krylov.py` `tsr_matvec` — now a warp per row with
  lane-strided columns and a `__shfl_down_sync` butterfly. The claim made is
  structural only: at a fixed inner iteration a load's 32 lanes touch 32
  consecutive elements of one row rather than 32 rows `n*sizeof(T)` apart,
  so transactions per load drop from 32 to 4 for f32. **No speedup was
  claimed and none is known.** Two consequences to check on device: the
  per-row summation order changed, so CG/GMRES convergence needs
  re-confirming and results are no longer bit-identical to a sequential sum;
  and `tests/performance/nvidia/test_solver_krylov_ratchet.py` compares
  against `benchmarks/baselines/nvidia_sm120_solver_krylov_performance.json`,
  recorded with the OLD matvec — that baseline will report a false result
  until re-recorded. Whether to enlarge the launch geometry now that a warp
  owns a row (`useful = ceil(n/256)` was sized for one thread per row) is an
  open measured question, deliberately left alone.

Evidence that does exist: the generated text was asserted on, and both
sources parse clean under `clang++ -std=c++17 -fsyntax-only -Wall` with CUDA
stubs — a harness confirmed to reject a `goto`-crosses-initialization, so the
clean parse means something. It is not device evidence.

Cross-backend sync `P2-REVIEW-SHARED-PASSES-2026-08-29` — **15 shared MLIR
passes changed; only the Mac's fixture set could be run.**
The P2 code-review batch touched passes every backend lowers through:
`TesseraToLinalgPass` (rejection checks moved before IR creation),
`SymbolicDimEqualityPass` (transposeA/B in the contract + flow rules, and
malformed `dim_bindings`/`dim_sizes` now fail closed),
`AdjointCollectiveInsertionPass` (cotangent-array bounds),
`AutodiffPairedPass` (dynamic while state refused; erase re-checks use_empty),
`RegionAdjointInterface` (O(1) dense-checkpoint slot),
`ActivationRematerializationPass` (difference-array peak),
`WarpSpecLegalityPass` (transitive staged-data provenance),
`TileBufferArenaPass` (non-scalar element types),
`IRContractLegalityPass` (narrowing-accum restricted to same-domain pairs),
`MaterializeControlPayloadPass` (shared body-stub conflict),
`InsertRecomputePass` (real live-set), `LegalizeSpaceTime` + the CPU stencil
hook (orders 6 and 8 implemented; unimplemented orders refused), and
`AsyncPrefetch` (memory-write dependence).
Evidence produced: `lit tests/tessera-ir/` **437 discovered, 396 passed, 41
unsupported, 0 failed** on the Mac (M1 Max, brew LLVM/MLIR 23.1.0, assertions
OFF), plus per-finding reproductions with controls. **Not evidence for this
backend's own fixtures.**

*What this queue must run on sm_120.* The split-route flash backward gained a
`tsr_flash_bwd_stats` kernel so per-`(b,qh,m)` softmax statistics are computed
once instead of once per KV split; `tsr_flash_bwd_dq` was restructured n-outer
into a `aq[D]` accumulator. Host-executed through a launch-emulation shim the
split route now matches the untouched atomic route to <=1.9e-7 across causal /
sliding-window / bias / logit-cap, with a negative control at 2.8e-2 — but no
line of it has run on a GPU. **Re-measure
`measure_flash_attention_backward_device`**: the route arbiter was choosing
between these two routes using the inflated split number. Also owed: the
decayed `run_linear_attention_variant{,_backward}` F4 tolerance under the new
Horner recurrence (forward folds decay into the accumulator, keeping the
summation ascending; backward uses a descending running product because the
factor feeds per-key atomics), and one `run_optimizer_f32` call per valid kind
to confirm the new `kind<0||kind>5` rc-2 guard left the happy path alone.

Cross-backend sync `LINUX-BASELINE-2604-LLVM231-2026-08-29` — **not applicable to SM120; the CUDA host is already on 26.04.**
The Linux baseline moves to **Ubuntu 26.04 LTS** and the compiler-backbone pin
tightens from "LLVM/MLIR 23.x" to **23.1.x exactly**; `scripts/setup_ubuntu.sh`
now FAILS on any other Ubuntu release rather than warning. `CLAUDE.md`'s host
record moved in the same change, because leaving it at 24.04 pointed this
project's own instructions at a bootstrap command that exits immediately.

Measured on the migrated box (`Princess-Luna`): Ubuntu 26.04.1, LLVM/MLIR
23.1.0 (assertions OFF), ROCm 10 series (HIP 7.15), repo at
`~/programming/tessera`, ssh on the default port.
*NVIDIA outcome.* No NVIDIA impact. The CUDA host (The-Super-Bear) was
already Ubuntu 26.04 with LLVM/MLIR 23.1.0, so the tightened pin and the new
`setup_ubuntu.sh` gate match it as-is; the branch built and ran there with
`TESSERA_ENABLE_CUDA=ON` (unit failure set identical to main, `lit` 429/429).
Unrelated pre-existing snag worth carrying: `TESSERA_ENABLE_CUDA=ON` fails to
configure because `examples/advanced/power_retention/src/extension` has no
CMakeLists.txt — use `-DTESSERA_BUILD_EXAMPLES=OFF` until that is repaired.
Cross-backend sync `SM120-REGRESSION-VALIDATION-2026-08-29` — **branch
validated on the RTX 5070; no regressions, and no P0 was ever owed here.**

Built and run on The-Super-Bear (RTX 5070 cc 12.0 / sm_120, Ubuntu 26.04,
CUDA 13.3, LLVM/MLIR 23.1.0 assertions OFF) with `TESSERA_ENABLE_CUDA=ON`,
from clean worktrees at the branch and at `f65f9b3b`:

* unit suite **13503 passed / 53 failed / 5414 skipped**, and the failure set is
  **byte-identical to main's on the same box (53 = 53)** — no regressions, none
  fixed. `lit tests/tessera-ir/` **429/429**.
* Confirm before reading the numbers: NO P0 from the 2026-08-29 review touches
  NVIDIA code. The CUDA item is a P1 (`nvidia_cuda.py:1137` shared reduction
  scratch reused without a barrier).
* That P1 remains **only partially proven**. The fixed pattern is correct and
  deterministic over 300 runs on sm_120, but the race was NOT reproduced, and
  `compute-sanitizer --tool racecheck` **cannot initialize under WSL2**
  ("Failed to initialize WDDM debugger interface") — its summary reports its own
  init failure identically for both variants, so it is not evidence either way.
  Nsight Compute (`ncu`) and Nsight Systems (`nsys`) ARE installed here and are
  the obvious next instrument.
* Build note, pre-existing on main: `TESSERA_ENABLE_CUDA=ON` fails to configure
  because `examples/advanced/power_retention/src/extension` has no
  CMakeLists.txt. Work around with `-DTESSERA_BUILD_EXAMPLES=OFF`; the example
  tree needs repair independently.

Cross-backend sync `SHARED-CONTRACTS-P1-REVIEW-2026-08-29` — **assessed; SM120 execution unchanged, one device proof attempted.**
PR #638 changes four SHARED contracts, so each backend records its own
outcome rather than letting the queues drift:
1. **Float `ne` is now UNORDERED** (`TesseraToLinalgPass`). `arith.cmpf one`
   is false when either operand is NaN; IEEE-754 and numpy define `!=` as the
   negation of `==`, so `NaN != NaN` is true and `x != x` — the idiomatic NaN
   test — silently never fired. eq/lt/le/gt/ge stay ordered.
2. **Control-flow predicate forms** (`LowerControlFlowToSCFPass`): boolean and
   signless-integer conditions lower instead of crashing; explicitly
   signed/unsigned integer predicates are refused, because `arith.cmpi`
   requires signless operands and cannot express them at all.
3. **Symbolic-dim while results** (`SymbolicDimEqualityPass`) are seeded from
   the condition's forwarded values, not the init/yield position.
4. **Recompute purity is derived** (`InsertRecomputePass`): an op with no
   effect attribute must be provably memory-effect-free, so an RNG draw or an
   opaque call is no longer marked recomputable.
*NVIDIA outcome.* All four are host-free contract changes; no PTX, cubin,
SM120 selector, or numeric policy moves. The emitted-CUDA barrier fix that
rides in this PR (`nvidia_cuda.py`: `tsr_sum` returning `s[0]` with no barrier
before the next call's `s[t]=v`, and the softmax max broadcast) WAS exercised
on the RTX 5070 (cc 12.0) this cycle: the fixed pattern computes correctly and
deterministically over 300 runs. **The race itself was NOT reproduced** —
`compute-sanitizer --tool racecheck` cannot initialize under WSL2 ("Failed to
initialize WDDM debugger interface"), so its output is not evidence, and 300
direct runs of the pre-fix pattern produced identical results. The defect
stands as a CUDA memory-model argument (an unsynchronized read of `s[0]`
against a write of `s[t]`), not a measured failure. A racecheck-capable host
would settle it.

Cross-backend sync `FOUNDATION-LLVM231-REVIEW-P0-2026-08-29` — **no NVIDIA P0
in this batch; foundation actions apply, and four sm_120-gated lanes plus one
confirmed CUDA-emitter P1 are owed by this box.**

*Foundation (all backends).* The LLVM/MLIR major pin is unchanged at **23**;
the Mac moved from a manual pre-release `23.1.0git` prefix to Homebrew's
production `llvm` keg **23.1.0** (old prefix deleted). This box stays on
apt.llvm.org `/usr/lib/llvm-23`. **No fleet box has an assertions-enabled LLVM
any more**, so an MLIR promise/contract claim can currently be falsified
nowhere (Decision #19) — relevant before recording any "no longer reproduces"
here. Python/deps: the `numpy<2.2` cap **can be dropped** — `pyproject.toml`
now skips numpy/scipy stubs (`follow_imports = "skip"` **plus**
`follow_imports_for_stubs = true`), keeping `python_version = "3.10"` while
making the mypy ratchet independent of the installed numpy (the fleet spans
three versions). Also: `check-{clifford,ebm,spectral}` no longer hardcode
`llvm-lit`, and driver discovery in `tests/_support/compiler_tool.py`,
`compiler/driver.py` and `_jit_boundary.py` now requires a candidate binary to
**start**, not merely exist.

*Actions on this box.*
1. Sweep for build trees pinned to a removed toolchain:
   `grep -l llvm-23.1.0-rc1 build*/CMakeCache.txt`, then `ldd` the drivers.
   On the Mac, four stranded trees produced ~86 unit failures whose messages
   were all about a missing dylib rather than about any code under test.
2. This host is x86_64, so the new `CMAKE_SYSTEM_PROCESSOR` gate in
   `src/CMakeLists.txt` must still select the native x86 kernels — confirm
   `cmake` prints no "x86 native kernels skipped" line (see the x86 todo).
3. Run the four sm_120-gated lanes that cannot be evaluated on the Mac and
   currently fail there as hardware-absent:
   `tests/unit/test_scheduled_matmul_consumers.py` (3) and
   `tests/unit/test_nvidia_compiler_artifacts.py::test_sm120_tile_fragment_lowers_to_real_nvvm_mma`.

*The confirmed CUDA-emitter finding this box owns (P1, unfixed).*
`python/tessera/compiler/emit/nvidia_cuda.py:1137` — several emitted
block-reduction kernels read the reduction result from `scratch[0]` and then
rewrite the same shared buffer for the next phase **with no
`__syncthreads()` between**, a data race under the CUDA memory model. The
reported instance is `run_row_norm(x, 'layer_norm', eps)` at K=4096 with a
256-thread block (8 warps): after `tsr_sum` returns, warp 0 can re-enter the
scratch write while slower warps are still reading. This is the same defect
class fixed in the HIP paged-attention kernel this cycle (see the ROCm todo),
which is corroborating but **not** transferable evidence — it needs an sm_120
run. Two adjacent confirmed P2s in the same emitter: `:622` split-reduced
backward recomputes softmax statistics `Sk` times, and `:879` recomputes the
decay product `O(S)` per key for `O(S^3)` total.

Cross-backend sync `IKF-INTRA-KERNEL-CONTRACT-2026-08-27` — **follow-up
required at IKF-P6; SM120 execution unchanged.** The IKF-1 intra-kernel
measurement plan (`docs/audit/compiler/INTRA_KERNEL_FEEDBACK_PLAN.md`,
PR #634) defines a shared artifact schema, Tile IR trace ops, and a runtime
buffer contract, proven first on ROCm gfx1151. The NVIDIA lowering (planned
P6) maps the constant-rate clock rule to `%globaltimer`; the slot index's
wave-in-role coordinate must be re-derived from SM120's own role structure
(no wgmma/tcgen05 — consumer-Blackwell schedules differ from the sm90 plans
the key was designed against). No gfx1151 timing or schedule evidence
transfers; promotion requires an exact-SM120 clock-validation packet
mirroring IKF-P0 before any NVIDIA lowering lands.

Cross-backend sync `X86-AVX512-IMAGE-ADMISSION-2026-08-27` — **shared runtime
load safety repaired; CUDA execution unchanged.** The AVX2 Threadripper that
hosts the RTX 5070 Ti now rejects Tessera's monolithic AVX-512 CPU image through
the canonical complete-feature authority before `ctypes.CDLL`; the image itself
also performs no AVX-512 work in ELF initialization. This prevents unrelated
CUDA/solver collections from dying with SIGILL while preserving fail-closed
x86 execution. No PTX, cubin, SM120 selector, CUDA numeric policy, or RTX
certificate changes or inherits evidence from this host-side x86 repair.
The independently rebuilt Zen 5 image passes its complete 79-case FFT/solver
packet; that CPU evidence likewise transfers no SM120 execution claim.

Cross-backend sync `CUDA-SOLVER-KRYLOV-SCALE-2026-08-27` — **arbitrary dense
operator Arnoldi/GMRES, non-diagonal CG, explicit low-precision solver matmul,
and multi-CTA performance ratchets exact-SM120 closed.** A content-addressed
dense-operator v2 package represents any finite-dimensional linear map as a
row-major matrix and launches one cooperative CUDA grid for the complete
solve. Restarted GMRES retains its basis, Hessenberg matrix, Givens rotations,
work vectors, dot partials, and convergence state on device; twice-modified
Gram-Schmidt limits loss of orthogonality and an fp32 recomputed `b-Ax`, never
the Givens estimate alone, establishes convergence. The CG route admits an
authored SPD promise, checks positive curvature on device, periodically
replaces the recursive residual with the true residual, and rejects indefinite
operators. Dot and norm reductions use deterministic two-level CTA partials
with a cooperative grid barrier; the exact RTX 5070 packet exercises 2-3 CTAs
and f32/f16/bf16 storage. The live performance packet covers orders 513, 1025,
and 2049, scales reduction geometry 3->5->9 CTAs, and ratchets both complete
host calls and CUDA-event kernel time. Separately, solver residual matmul now
requires `{storage=f16|bf16, accum=fp32, math_mode=ieee}` before selecting the
native SM120 `mma.sync` route; missing or contradictory storage fails closed,
while f32 retains the scalar IEEE route and never substitutes TF32. Krylov
matrix-vector products intentionally convert their declared storage to fp32
FMA; the tensor-core claim belongs only to the registered rank-2 matmul child.
Open boundaries are compiler-fused matrix-free child callbacks, sparse/
structured operator encodings, preconditioners, and NCU-guided matvec tuning;
none is implied by the dense package.

Cross-backend sync `CUDA-SOLVER-FAMILY-2026-08-27` — **typed residual children,
broader storage policy, and dedicated device-resident CG exact-SM120 closed.**
Compiler-emitted CUDA children now execute sqrt/reciprocal/exp/log/tanh/
sigmoid/sin/cos, sum/mean/max/min, eq/ne/lt/le/gt/ge, predicate `where`, and
rank-2 matmul products. Matmul uses the resident scalar IEEE-f32 route only
when the residual product preserves an explicit `math_mode="ieee"`; missing
mode or non-fp32 accumulation fails closed and never selects TF32. Residual,
JVP, and VJP SSA replay consumes all of these target-owned children, including
pure predicate recomputation and exact reduction duals. f16/bf16 storage is
admitted with explicit package-boundary widening and fp32 Krylov arithmetic;
leaf unary/comparison/where/reduction and the dedicated CG input ABI also
execute true two-byte storage. A distinct content-addressed positive-diagonal
SPD CG package retains solution, residual, direction, matvec, dot reductions,
and convergence state in device memory for one complete CUDA launch. The
f32 packet converges in 17 iterations with max solution/equation error below
`3.6e-7`; f16/bf16 device oracles also pass. General residual GMRES remains
host-orchestrated, and device-resident arbitrary-operator Arnoldi/GMRES,
multi-CTA reductions, non-diagonal CG operators, and performance promotion
remain open.

Cross-backend sync `CUDA-SOLVER-IFT-PILOT-2026-08-27` — **diagonal-sqrt and
the first general matrix-free CUDA solver envelope exact-SM120 closed.**
The shared content-addressed IFT contract now admits `nvidia_sm120`/`sm120`
and preserves its residual, matrix-free solve, JVP/VJP mode, and parameter
product lineage into one Tile artifact. An NVIDIA-owned compiler-emitted CUDA
package executes all three phases on the RTX 5070 under CUDA 13.3. The
reproducible 30-sample packet records a maximum absolute error below `4.8e-7`,
complete-backward timing, stale-lineage rejection, and correctness-only
promotion. Compiler-generated affine residual/JVP/VJP SSA now replays through
the registered CUDA binary carrier, and a digest-bound parent executes
restarted GMRES with true-residual checks. Its separate 20-sample packet has
zero numerical error for both product directions and binds all five child
digests. This is a binary-f32 envelope: unary, reduction, comparison/where,
matmul, fully device-resident Krylov state, CG-specific execution, and broader
dtype policy remain open and fail closed.

Cross-backend sync `CUDA-BINARY-SPECTRAL-JVP-2026-08-27` — **compound spectral
JVP dispatch hole and the first general CUDA binary-math family exact-device
closed.** The public compound plugin previously declared NVIDIA ownership for
`spectral_filter` and `spectral_conv`, but two active tangents selected the
ROCm binary executor while carrying an NVIDIA artifact. `nvidia_binary_compiled`
now owns matching-shape add/sub/mul/div/pow/max/min/mod/floor-div with
f32/f16/bf16 storage and fp32 evaluation; NaN propagation, signed-zero min/max,
and floor-quotient semantics are explicit. Filter and convolution JVP tangent
terms consume that CUDA add route, while logical complex64 filter storage is
recorded as interleaved fp32. Exact RTX 5070 CC 12.0 / CUDA 13.3 execution
passes the full binary dtype matrix, public filter/convolution bilinear laws,
and public f16/bf16 STFT JVP oracles. Unsupported shape, dtype, operation, or
target still fails before launch. This is correctness closure, not a selector
or performance promotion.

Cross-backend sync `CUDA-SPECTRAL-JVP-NUMPOL-2026-08-26` — **remaining SM120
spectral admission and numeric-policy rows exact-device closed.** Public
`native_jvp` now constructs a content-addressed STFT/ISTFT child whose Graph,
Schedule, Tile, and CUDA identities are distinct and digest-bound. Production
Schedule→Tile admits the `nvidia_sm120`/`sm120` cuFFT policy rather than routing
through a sibling profile. f16/bf16 DCT, STFT, ISTFT, JVP, and VJP storage use
explicit two-byte ABIs with fp32 framing, cuFFT accumulation, overlap-add, and
window-gradient reduction. The RTX 5070 Ti packet passes 8/8 independent
forward, centered-difference, low-precision parity, and JVP/VJP adjoint checks;
unsupported or stale policy still fails closed. This closes the three
follow-ups in `TSOL-CUDA-POLICY-V1` below without transferring ROCm/x86/Metal
evidence.

Cross-backend sync `CUDA-OPTIMIZER-VJP-2026-08-26` — **SM120 optimizer reverse
execution exact-device closed for the ordered package.** The shared
non-reexecuting state-lineage carrier now admits NVIDIA as a physical owner and
stamps `nvidia_sm120`/`sm_120` through Schedule→Tile. CUDA-owned PTX packages
execute SGD, Momentum, Nesterov, Adam, AdamW, and full/factored Adafactor;
factored reductions have one deterministic owner and every output is a fresh
no-alias write. The RTX 5070 Ti packet passes all seven numerical variants and
requires runtime-origin `sm_120` attestations in content-addressed execution
certificates. This is correctness-first f32 execution, not a performance
promotion or a transfer of gfx1151/AVX-512 schedules.
The consolidated all-family packet now observes exact equality for all nine
registered NVIDIA VJP families, including `optimizer_vjp` and
`adafactor_vjp`; omitting either new family fails the packet.

Cross-backend sync `TSOL-CUDA-POLICY-V1-2026-08-26` — **SM120 f32 physical
forward, inverse, reverse, DCT, and streaming rows exact-device proven.** The
new `tessera.nvidia.spectral_policy.v1` ABI owns DCT-I/II/III/IV, STFT/ISTFT
device framing, cuFFT execution, normalization, deterministic overlap-add,
analytic signal/spectrum and broadcast-window adjoints, and causal streaming
state transitions. Exact RTX 5070/SM120 packets cover arbitrary axes, true
element strides, `n_fft >= window`, centered constant/reflect padding,
explicit inverse cropping, one-sided/full spectra, trailing batch-window
broadcast, all FFT normalization modes, DCT types, streaming artifact/parent
lineage, independent forward/VJP oracles, and the native forward/adjoint
inner-product law. NVIDIA transform capability now records complex64 as
logical interleaved-fp32 storage instead of rejecting the shipped ISTFT Graph
path. At this synchronization point the public content-addressed JVP child,
Schedule→Tile admission, and f16/bf16 storage were still open; the later
`CUDA-SPECTRAL-JVP-NUMPOL` row above closes them with exact-device evidence.
This synchronization supersedes the CUDA-physical-open clauses in
`TSOL-POLICY-PHYS-1-8C8G`, `TSOL-POLICY-PHYS-1-8B`,
`TSOL-POLICY-PHYS-1-8A`, and `AD-TSOL-STFT-GFX1151` below; those entries remain
as the history of the shared carrier landing; their subsequent closure is
recorded by the newer synchronization key.

Cross-backend sync `TSOL-POLICY-PHYS-1-8C8G-2026-08-26` — **shared spectral
carrier assessed; CUDA implementation remains open.** Schedule→Tile now binds
the runtime-stride ABI, independent transform/window lengths, and full versus
one-sided spectrum policy; streaming state has digest-chained lineage. The x86
numerical packet supplies no SM120 evidence. CUDA still needs its own
true-stride forward/adjoint package, broader/full transforms, broadcasting,
streaming package, and exact-device oracle packet. The independent AVX-512 and
gfx1151 forward/streaming packets and their artifact-bound state certificates
transfer no SM120 schedule or execution evidence. The later exact-gfx1151
expanded reverse/VJP packet likewise transfers no CUDA adjoint implementation;
SM120 retains every architecture-owned reverse row named above.

Cross-backend sync `TSOL-POLICY-PHYS-1-8B-2026-08-26` — **shared logical-axis
contract assessed; CUDA implementation remains open.** Centered STFT and
centered/cropped ISTFT now preserve arbitrary normalized logical axes and
`outer`/`inner` indexing through Schedule→Tile, while non-C-contiguous storage
still fails closed. The AVX-512 and exact-gfx1151 packets supply no SM120
execution evidence; CUDA needs its own architecture-owned forward/adjoint
package and oracle packet. Full spectrum, broader lengths, broadcasting,
streaming, and true stride support remain open.

Cross-backend sync `TSOL-POLICY-PHYS-1-8A-2026-08-26` — **shared policy
carrier assessed; CUDA implementation remains open.** Center, pad mode, crop,
and explicit ISTFT length are digest-bound through Schedule→Tile. The AVX-512
and exact-gfx1151 centered/cropped packets supply no SM120 execution evidence.
CUDA still needs an architecture-owned forward/adjoint package and exact-device
oracle packet before any policy row can be promoted.

Cross-backend sync `AD-TSOL-STFT-GFX1151-2026-08-26` — **shared carrier
assessed; CUDA implementation and evidence remain open.** gfx1151 n=16/n=18
forward/adjoint numerics and certificates do not transfer to SM120. CUDA still
needs an architecture-owned STFT/ISTFT package and exact-device oracle packet.
The structured spectral `numeric_policy` now survives Schedule→Tile, closing
the generalized carrier ceiling, but SM120 `math_mode` consumption and device
proof remain NVIDIA-owned Order 3b follow-ups.

Cross-backend sync `E2E-REAL-6F-EXACT-CERT-2026-08-26` — **initial seven-family
SM120 certificate packet exact-device closed, subsequently expanded to nine
under `CUDA-OPTIMIZER-VJP`.** The target-owned single-process
packet executes binary loss, class loss, Lion, normalization, regression loss,
sequence mixer, and spectral backward on the RTX 5070 and requires
runtime-origin `sm_120` attestations plus exact family-set equality. The
spectral row now also exercises STFT and ISTFT analytic adjoints. A CUDA-mode
string without that attestation remains `runtime_unattested` and cannot close
a row; no x86 or gfx1151 result was transferred.

Cross-backend sync `BOUNDED-GATE-RELAXATION-2026-08-26` — **shared
control-scan normalization assessed; CUDA physical outcome not applicable.**
The paired reverse pass now admits the statically bounded symbol-body scan and
keeps payload/dynamic/malformed forms closed. The f16-accumulate WMMA consumer
is exact-gfx1151 and transfers no Tensor Core claim. The AVX-512 packed-C2R and
gfx1151 direct-DFT STFT/ISTFT packages each transfer no CUDA spectral claim.
The shared
family-plugin certificate carrier is portable, but the new factored Adafactor
certificate is x86 evidence, not an SM120 optimizer execution. The bounded MPI slice now consumes all five
explicit Schedule→Tile collective SSA forms and has exact two-process x86 host
evidence, including artifact/communicator/subgroup binding. That evidence does
not transfer to NCCL: process-rank ownership and an sm_120 multi-rank packet
remain NVIDIA follow-ups; no MPI or mock result satisfies them.
NVIDIA's duplicated composed-layout materializer was also aligned with the
shared CuTe rule by retaining the slowest mixed-radix quotient. The lowering
fixture passes in an isolated hardware-free NVIDIA compiler build on this box,
but there is no RTX exact-device run, so CUDA numerical parity remains
follow-up required rather than promoted.

Cross-backend sync `W4-EFFECTS-1-E5-2026-08-25` — **one physical family carrying an admissible effect, end to end; NVIDIA outcome: not applicable — no row claimed.** E5's physical acceptance was scoped to x86 + gfx1151 and executed there. No sm_120 evidence exists or is implied; an NVIDIA row would need its own exact-device replay on NR2 Pro, since no result transfers between architectures.


Cross-backend sync `W4-EFFECTS-1-E4-2026-08-25` — **ordered-collective
recorded products (identity only); NVIDIA outcome: not applicable today, inherited on adoption.** The product
binds communicator, issue order, reduction algorithm and topology; the
verifier rejects a permuted order and a changed tree under an identical
order. Order evidence comes from the deterministic mock-mesh executor.
When an NVIDIA collective family adopts recorded products it inherits the same requirement, and NCCL deterministic-algorithm selection becomes the gating evidence for any result claim.


Cross-backend sync `W4-EFFECTS-1-E3-2026-08-25` — **shared state-lineage
identity change; NVIDIA outcome: not applicable today, inherited on
adoption.** The lineage is host-side package identity, not target codegen; no
sm_120 artifact changes. Worth knowing when NVIDIA stateful packages adopt
recorded products: the dtype field is now real, so a bf16 or fp8 optimizer
state gets its own identity rather than aliasing the f32 one.


Cross-backend sync `W4-EFFECTS-1-E2-2026-08-25` — **shared autodiff gate
change (AutodiffPairedPass); NVIDIA outcome: not applicable, no behaviour
change.** The pass is target-neutral and the change is a diagnostic split
over a fail-closed check, so no sm_120 artifact or numerical result moves and
no NVIDIA-owned surface needs revalidation. What NVIDIA inherits when a
stochastic family is admitted on its lane: the same product requirement, and
its own exact-device replay evidence.


Cross-backend sync `W4-EFFECTS-1-2026-08-25` — **UPDATED 2026-08-25 (slice E1
landed): shared recorded-product carrier + verifier implemented in Python;
NVIDIA outcome: not applicable at this slice, follow-up on adoption.** E1
introduces no target-owned surface and E5's physical acceptance stays scoped
to x86 + gfx1151, so no sm_120 row is claimed or implied. Obligations on
adoption are unchanged: exact-device replay evidence of its own, and NCCL
deterministic-algorithm selection, since the carrier requires the reduction
tree to be bound before bit-identity may be claimed.

Cross-backend sync `SPECTRAL-PAYLOAD-CHAIN-2026-08-25` — **shared
Schedule->Tile spectral identity contract + pipeline carrier ordering; NVIDIA
outcome: not applicable today, inherited on adoption.** No NVIDIA Target
consumer exists for the scheduled spectral program (its dialect is off by
default in this build and the spectral physical lanes are x86/gfx1151), so no
sm_120 artifact or numerical result changes. When an NVIDIA spectral consumer
lands it inherits the same required preimages; the pipeline-carrier ordering
fix is target-neutral and applies unchanged.


Cross-backend sync `SCHEDULE-AUTHORITY-RESHARD-2026-08-24` — **shared SO-3 and W5.4 parity validated; no CUDA physical change.** Pipeline and compound-spectral lowering now consume one digest-bound Schedule Object, inferred producer edges, roles, and resource evidence without scalar reconstruction. Placement emits exact mesh-sized local-shard/collective SSA and all movement forms execute on the deterministic mock mesh. The carrier and verifier changes are shared; Zen 5/gfx1151 spectral evidence and mock transport transfer no CUDA schedule, NCCL proof, or RTX claim. `NUMPOL-CARRIER-1` owns the generalized S5 carrier; SM120 consumption remains a later architecture-owned assessment.


Cross-backend sync `SO3-INFER-EDGES-2026-08-24` — **shared W2.1/W5.2e
dependence-inference semantics + MegaMoE R3 producer; NVIDIA outcome: not
applicable today, inherited when NVIDIA adopts MoE plans.** The change is
host-side schedule ANALYSIS (Python R3 composition), not target codegen: no
NVIDIA-owned surface compiles through it and no sm_120 artifact or numerical
result changes. When an NVIDIA MoE transport lane adopts the overlap plan it
inherits the corrected inference unchanged; the exact-device evidence rules
are unaffected (no ROCm/x86 analysis result transfers a NVIDIA claim).


Cross-backend sync `NUMPOL-CARRIER-1-2026-08-24` — **shared Schedule→Tile
`numeric_policy` carrier contract (integrated-plan queue row 3b); NVIDIA
outcome: follow-up required, sequenced behind W1.1.** Newly owned row,
nothing implemented yet. NVIDIA will consume the same carrier, but its typed
fragment producers are still open under W1.1 — the carrier work here should
follow that, not race it, or the two will collide at the same seam. CAKE
(#32's original derivation) is an NVIDIA-facing consumer, so the barrier and
TCGen05 fragment paths are the ones to check first. sm_120 exact-device
evidence required before any numerical claim; no ROCm/x86 result transfers.


Cross-backend sync `LAYOUT-ALG-APPLE-PHYSICAL-2026-08-24` — **shared ABI
assessed; no CUDA physical change.** Apple now exports the existing C++ rank-2
plan through the native layout ABI for its MSL emitters and owns fresh M1 Max
proof. NVIDIA continues to consume the same header authority in CUDA; Metal
source templates, simdgroup scheduling, and Apple evidence transfer no CUDA
schedule, PTX, or RTX claim.

Cross-backend sync `LAYOUT-ALG-L5-X86-2026-08-24` — **shared admissibility
parity validated; no CUDA physical change.** The x86 consumer follows the same
canonical dynamic-leaf order and mixed-radix mathematics already proven by the
SM120 consumer. CPU assertions and AVX2/AVX-512 evidence transfer no CUDA
schedule, PTX, or RTX claim.

Cross-backend sync `LAYOUT-ALG-L4-X86-2026-08-24` — **shared rank-2 authority
parity validated; no CUDA physical change.** NVIDIA already consumes the same
C++ coordinate mapping in its proven SM120 matmul paths. AVX-512 host admission,
loop structure, intrinsics, performance, and Ryzen numerical evidence transfer
no CUDA schedule or RTX claim.

Cross-backend sync `LAYOUT-ALG-L3-L5-DYNAMIC-2026-08-24` — **L3/SO-4,
mixed-radix/static tuple materialization, and dynamic macro-CTA execution are
closed for the stated NVIDIA subset.** The native factorization/residency ABI
and Schedule Object v2 proof are shared contracts. CUDA now consumes the shared
rank-2 index authority, mixed-radix basis maps, and static tuple codomain
products. Both the corrected narrow dynamic route (`17x19 @ 19x13`, padded
`29/31/23`) and the alignment-safe scalar-shared macro route
(`257x127 @ 127x259`, padded `139/137/269`) match FP32 oracles on RTX 5070. The
post-correctness macro NCU row is 17.536 us, 40 registers/thread, 82.85% L2
sector hit rate, and 13.89% active warps. Dynamic/non-separable tuple codomains,
explicit pointer-offset alignment metadata, and bare-metal selector authority
remain open. Apple/x86 rank-2 index-template migration is closed by
architecture-owned evidence.

Cross-backend sync `DYNAMIC-COMPOSED-SM120-2026-08-24` — **bounded dynamic
Graph matmul, arbitrary leading dimensions, and scalar-affine nested
materialization are exact-device proven.** A dynamic rank-2 NVIDIA matmul is
admitted only with static `shape_bounds=[M,N,K]`. Its canonical scheduled Tile
producer carries runtime `M/N/K/LDA/LDB/LDD`, bounded `tile.view` operands, and
dynamic outer shape/stride leaves through `tile.materialize_composed_layout`;
the CUDA target re-runs the shared C++ affine proof before emitting address
arithmetic. The versioned strided f16/bf16 descriptor validates A `(LDA,1)`, B
`(1,LDB)`, and D `(LDD,1)` element strides and the launch bridge allocates and
copies their physical spans with overflow checks. RTX 5070 execution for
`17x19 @ 19x13`, `LDA=29`, `LDB=31`, `LDD=23` matches the independent FP32
oracle, including M/N bounds and the final K tail. The proof found and fixed a
bounded B-fragment defect where the second packed lane advanced by `LDB`
instead of contiguous K. The subsequent synchronization record closes
mixed-radix/static tuple materialization and the alignment-safe dynamic macro
route. Dynamic/non-separable tuple codomains and the bare-metal selector gate
remain open.

Cross-backend sync `SCHEDULED-MATMUL-TAIL-EPILOGUE-LDS-2026-08-24` — **SM120
static K-tail and scheduled epilogue ABI closed; dynamic stride proof and
bare-metal policy remain open.** The canonical package now distinguishes
16-byte-row-aligned partial K panels from arbitrary-alignment K tails. The
former uses `cp.async` source-size zero fill; the latter uses masked scalar
shared staging. RTX 5070 cases `257x513 @ 513x257` and
`257x520 @ 520x257` match the FP32 oracle. Graph matmul now carries optional
fp32 bias and residual as ordered SSA operands, while Schedule/Tile preserve
ReLU/GELU/SiLU and f16 reduced-output policy. The widened CUDA descriptor and
runtime launch path execute bias→ReLU→residual with an f16 store on RTX 5070.
The current host is WSL, so neither
these correctness results nor the retained NCU observations can authorize a
global selector change; the bare-metal packet is still required.
After correctness, Nsight Compute on the aligned `257x520x257` tail reports
scheduled/direct duration about `15.3/23.2 us`, `56/35` registers, `4.10 KiB/0`
static shared memory, and `14.0%/23.3%` achieved occupancy. Logical input
redundancy is `10.09x/26.19x`. Reports:
`/tmp/tessera-sm120-k520-{scheduled,direct}.ncu-rep`. These are diagnostic
profiler observations, not selector authority.

Cross-backend sync `SM120-MACRO-CTA-2026-08-24` — **NVIDIA-owned async
shared-panel contract implemented and exact-device proven; global selector
unchanged.** Canonical f16/bf16-to-f32 scheduled matmul now emits the registered
`tessera_nvidia.macro_cta_matmul` Target IR boundary. Its exact contract is one
`32x32` CTA tile, four warps with `quadrant_2x2_two_n_tiles` ownership,
`m16n8k16` MMA, two 2 KiB shared A/B slots, and `cp.async` commit/wait plus CTA
barriers. The 128 threads cooperatively transfer one 16-byte vector each.
Out-of-range M/N vectors use `cp.async` source-size zero, which zero-fills the
shared panel without dereferencing an invalid address; K remains a positive
multiple of 16. Exact RTX 5070 proof includes aligned and ragged FP16 plus
ragged BF16 (`257x512 @ 512x257`).

The retained nine-sample, 1000-repetition WSL CUDA-event packet admits only
cases at or above 67,108,864 FLOPs. Its three eligible scheduled/direct ratios
are 0.721 (`256x512x256`), 0.453 (`512x256x512`), and 0.429 (`512^3`); every
eligible route has CoV below 3% and all numerical rows are green. Smaller cases
remain on the typed fallback because repeated packets exposed launch-scale
variance through 33.6M FLOPs. The route threshold is pruning-only WSL evidence,
not global `target_perf` authority; a bare-metal packet is still required for
global selector promotion.

Post-correctness Nsight Compute at `256^3` records 8.22 us scheduled versus
9.57 us direct, 48/35 registers per thread, 4 KiB/0 static shared memory, no
spills, and L1/TEX throughput pressure of about 29%/80%. Achieved occupancy is
lower (11.1%/21.6%), but panel reuse reduces load pressure enough to win. The
reports are `/tmp/tessera-sm120-async-{scheduled,direct}-256.ncu-rep`; the
durations are diagnostic profiler observations, not selector timings.

Still open for this family: dynamic/non-separable tuple-codomain
materialization, explicit pointer-offset alignment metadata, and a bare-metal
selector packet. Static separable tuple products and the alignment-safe dynamic
macro-CTA specialization are closed by the synchronization record at the top
of this plan. Bounded dynamic extents and arbitrary runtime leading dimensions are closed by
`DYNAMIC-COMPOSED-SM120-2026-08-24`; static K tails and the bounded scheduled
epilogue ABI are closed by the synchronization record above.

Cross-backend sync `SM120-SCHEDULED-LICM-2026-08-24` — **superseded for the
macro route by `SM120-MACRO-CTA-2026-08-24`; narrow typed fallback retained.** The
native pipeline now applies LICM before SCF destruction, hoisting invariant lane
and composed-layout address terms out of the typed K loop. Five RTX 5070
numerical cases pass. Final v2-packet scheduled/direct CUDA-event ratios were
1.002 (`128x128x128`), 0.929 (`128x256x64`), and 0.901 (`256x256x256`). The
first is tied and the rectangular scheduled sample contains an outlier, so one
session does not define a safe selector boundary. On the largest case, separate NCU
observations were 9.088/10.528 us, 303104/634368 DRAM bytes,
187392/510976 executed instructions, and 40/35 registers per thread. The
benchmark's logical traffic model exposes a 24x input-reuse gap at 256 cubed.
A repeated largest-case ratio was 0.914, but scheduled/direct sample CoV was
3.89%/1.76%, failing the low-variance promotion requirement.
The target-owned macro-CTA follow-up described by this earlier checkpoint is
now implemented above. This LICM record remains the evidence for the narrow
typed fallback and changes no sibling physical schedule.

Cross-backend sync `SM120-BLOCK-COORDINATE-2026-08-24` — **NVIDIA-owned
coordinate boundary and macro numerical closure landed; selector unchanged.**
The registered pure `tessera_nvidia.block_coordinate` op returns typed i64 row
and column bases and verifies the sole contract `sm_120/16x8/column_major_xy`.
Only NVIDIA lowering interprets it as `ctaid.y*16, ctaid.x*8`; the canonical
scheduled producer consumes those SSA values in its two proven composed-layout
materializations and typed K-loop MMA. RTX 5070 exact-device cases
`16x32@32x8`, `32x32@32x16`, and `48x64@64x24` pass with max absolute error
`<=5.97e-7`. Seven-sample CUDA-event scheduled/direct ratios were 1.003, 1.025,
and 1.005, so no macro selector promotion is justified. Separate Nsight
resource captures for the largest case recorded scheduled/direct 44/36
registers per thread, 1024/1024 shared-memory bytes, and equal 2.08% active-warp
occupancy. Reports are retained at `/tmp/tessera-sm120-{scheduled,direct}-macro.ncu-rep`.


Cross-backend sync `CUTE-LAYOUT-MATERIALIZE-1-2026-08-23` — **SM120 static
affine view-address bridge landed; physical layout selection remains open.**
`tile.materialize_composed_layout` now takes one i64 coordinate per outer mode,
rechecks the recursive carrier through `tessera_layout_coalesce_v1`, and admits
only the static scalar-basis subset. SM120 lowers its canonical offset into the
existing `tile.view{tile.linear_base}` consumer, preserving ordinary bounds and
the selected fragment contract. It does not select a new register/shared-memory
layout. Nested or dynamic-residue carriers fail closed. This is compiler
lowering evidence plus exact RTX 5070 numerical proof for the static f16
m16n8k16 row-major-A/column-major-B subset: nonzero A-row and B-column origins
reached `tile.view → fragment_pack → mma.sync` and matched NumPy. It remains no
layout-performance claim. Dynamic and non-affine carriers, other dtypes, and
ROCm physical consumption remain follow-up required.

Cross-backend sync `ROCM-CI-HSACO-SERIALIZE-2026-08-23` — **ROCm-owned host-free
CI serialization lane; NVIDIA outcome: follow-up available, not required.**
The transferable idea is the technique, not the artifact: device-code
*serialization* is host work, so a GPU-less runner can prove the lane still
emits an object. NVIDIA's analogue would be a cubin/fatbin emission proof, but
it is **not** a drop-in — `runtime.py`'s NVIDIA device code is NVRTC-compiled at
load with no cubin lane (Decision #26a), and CUDA codegen needs the CUDA
toolkit rather than a stock `lld`. No PTX, cubin, sm_120 schedule, or
exact-device evidence transfers from gfx1151; nothing in the NVIDIA queue
changes.

Cross-backend sync `CI-BACKEND-CAPABILITY-SKIP-2026-08-23` — **Apple-owned
pytest capability gate; NVIDIA outcome: not applicable / no exposure measured.**
Measured 2026-08-23 on a host with `TESSERA_BUILD_NVIDIA_BACKEND:BOOL=OFF`:
`pytest -k nvidia -m "not slow"` reports **496 passed, 14 skipped, 0 failed**, so
the NVIDIA fixtures do not fail-instead-of-skip when their backend is absent and
need no equivalent guard. No sm_120 artifact, cubin, or exact-device evidence is
involved.


Cross-backend sync `NVIDIA-AOT-PACKAGE-V1-HARDEN-2026-08-22` — **NVIDIA-owned
runtime package hardened and exact-device validated on SuperBear.** The f16
SM120 peer now ships both versioned fatbin and cubin images plus a generated
package manifest. Each image embeds artifact-version, physical-ABI-version,
and canonical-source SHA metadata; the loader verifies those device globals
before admitting the kernel, binds the selected format into its cache key, and
uses identical-source NVRTC for missing, corrupt, incompatible, or stale
images. Forced fatbin, forced cubin, stale-image fallback, missing-image
fallback, and cache-key separation pass on RTX 5070 CC 12.0. This adds only
NVIDIA C-ABI inspection symbols; no shared IR, operation, dtype, numerical
policy, or sibling physical package changes.

Cross-backend sync `NVIDIA-FFT-WORKSPACE-1-2026-08-22` — **canonical CUDA
FFT/workspace ABI and first C2C consumers exact-device validated on SuperBear.**
`libtessera_nvidia_fft.so` exports the versioned
`tessera.nvidia.cuda_fft_workspace.v2` contract (superseded and extended by the
sync record below): reusable opaque cuFFT plans,
automatic allocation disabled, exact workspace-byte reporting, explicit
caller-owned device workspace allocation/free, and normalized inverse
execution. `nvidia_fft_compiled` consumes the contract for complex64-logical /
interleaved-f32 physical `fft` and `ifft`, including arbitrary positive length,
nonleading axes, and explicit pad/truncate `n`. SuperBear RTX 5070 (CC 12.0,
CUDA 13.3) exact-device proof is 7/7 across radix lengths 4/16, mixed length
100, prime length 257, forward/inverse comparison with NumPy, undersized
workspace rejection, and plan/workspace reuse. The initially deferred real,
compound, and autodiff consumers are promoted in
`NVIDIA-SPECTRAL-PHILOX-JVP-2026-08-22` below.

Cross-backend sync `NVIDIA-RNG-PHILOX-CORE-2026-08-21` — **typed NVIDIA
stateless compiler/runtime package exact-device validated on SuperBear.**
`tessera_nvidia.philox` is a registered Target IR directive with
closed `uniform_core`, `uniform_range`, `normal`, and `dropout` modes.
`--generate-nvidia-philox-kernel` consumes
the explicit `(seed_lo, seed_hi, counter_lo, counter_hi)` ABI and emits a
Philox4x32-10 `gpu.func` whose threads own disjoint 128-bit counter blocks. The
NVIDIA lit suite proves directive typing, fail-closed mode rejection, constants,
launch-index construction, four bounds-checked core stores, range scaling,
Box–Muller normal transforms, and dropout replay (51/51 host-free tests). This
is paired with the shipped `libtessera_nvidia_rng.so` four-symbol ABI and the
registered `nvidia_rng_compiled` executor/manifest row. SuperBear RTX 5070
(CC 12.0, CUDA 13.3) proof is 10/10: uniform core and range are bit-exact for
zero/ragged/large counts and explicit keys/counters, normal is tolerance- and
statistics-bounded, dropout replays the exact mask, and determinism/counter
separation hold. Compiler-JVP integration remains required.

Cross-backend sync `IEEE-MINMAX-CONTRACT-2026-08-23` — **NVIDIA
outcome: assessed, exact-device tie probe required (NR2 Pro).** The
fleet-wide IEEE-754-2019 ±0-tie contract for tessera.maximum/minimum
(rocm plan owns the key). Survey of the NVIDIA emitters: reductions and
elementwise max/min emit NaN-propagating `arith.maximumf` (no
numpy-emulating tie wrapper exists here, unlike the old ROCm binary
kernel), and `maxnumf`/`fmaxf` appear only in the Philox uniform floor
(`GenerateNVIDIAPhiloxKernel.cpp`, `tessera_nvidia_rng.cu`), whose
input cannot be NaN — the accepted pattern. So NVIDIA is expected
IEEE-conformant by construction, but the ±0 tie behavior of the
maximumf lowering on sm_120 is a hardware claim: run signed-zero tie +
NaN probes on the RTX 5070 Ti (mirror
`test_binary_max_min_signed_zero_ties_are_ieee_ordered`) before
recording parity. No evidence transfers from gfx1151 or the AVX-512
host.


Cross-backend sync `JIT-MATH-AUDIT-FIXES-2026-08-23` — **NVIDIA
outcome: assessed, no defective pattern found; adafactor_vjp NaN
follow-up open (NR2 Pro).** The NaN-laundering eps-floor defect fixed
on ROCm/x86 (rocm plan owns the key) was searched for in the NVIDIA
backend: no `maxnumf(statistic, eps)` floors exist in the C++ emitters
(the only maxnum is the Philox floor, input cannot be NaN). The sm_120
`adafactor_vjp`/`optimizer_vjp` training families route through
`tessera_nvidia.training_kernel`; when their update bodies are next
touched, add the NaN-gradient reference test on the device (mirror
`test_adafactor_factored_nan_gradient_propagates_like_reference`). The
softmax running-max maxnumf optimization is ROCm-kernel-only; NVIDIA's
softmax/attention paths were not changed.


Cross-backend sync `JIT-ELEMENTWISE-LINALG-2026-08-21` — **shared
`tessera_jit` pipeline change; NVIDIA outcome: not applicable.**
`tessera_jit` is the host-CPU JIT lane; the NVIDIA backend's device
paths (NVRTC / tessera-opt pipelines) do not consume it, and NR2 Pro's
host-CPU lane is the x86 follow-up recorded in the x86 plan. No
NVIDIA-owned surface compiles through the changed pipeline; the module-scope
correction and residual legality gate therefore change no CUDA IR, kernel,
runtime ABI, or exact-device claim.


Cross-backend sync `JIT-VECTORIZE-UNGATED-2026-08-23` /
`JIT-CACHE-BLOCK-2026-08-23` / `JIT-MATH-AUDIT-2026-08-23` — **shared
`tessera_jit` boundary/pipeline changes; NVIDIA outcome: not
applicable.** The x86 plan owns these keys. `tessera_jit` is the
host-CPU JIT lane; the NVIDIA device paths (NVRTC / tessera-opt
pipelines) do not consume it, and NR2 Pro's host-CPU lane is the same
x86 lane those keys validate — though on a different x86
microarchitecture, so if NR2 Pro's host lane is ever cited as
evidence, rerun the signature-guard + totality + vectorize packet
there rather than transferring the Strix Halo result. No CUDA IR,
kernel, runtime ABI, or exact-device claim changes.

Cross-backend sync `AD-DATUM-POLYGAMMA-2026-08-21` — **autodiff reference
numerical policy, wave 3; NVIDIA outcome: follow-up required (sm_120).**
Same contract change as the rocm entry. Expected parity-neutral for the
same reason as the previous two keys (both sides of every parity
comparison read the same updated reference; dtype preserved; lgamma/
digamma primals mirror the canonical forwards bit-for-bit). Run the
CUDA-marked autodiff/loss parity tests on NR2 Pro and record here — one
NR2 Pro session can now close all three open keys (this one,
AD-RETIRE-1-POINTWISE-2026-08-20, AD-RETIRE-2-2026-08-20).
Supplemental SuperBear evidence (2026-08-22): the complete CUDA training /
loss-autodiff package lane passed 55/55 on RTX 5070 CC 12.0. This is useful
independent SM120 parity evidence but does not close the NR2 Pro-owned row.


Cross-backend sync `AD-RETIRE-2-2026-08-20` — **autodiff reference numerical
policy, wave 2; NVIDIA outcome: follow-up required (sm_120).** Same contract
change as the rocm entry. Expected parity-neutral for the same reason as
`AD-RETIRE-1-POINTWISE-2026-08-20` (both sides of every parity comparison
read the same updated reference; dtype preserved); run the CUDA-marked
autodiff/loss parity tests on NR2 Pro and record here. Note this key AND the
still-open AD-RETIRE-1 key can be closed by one NR2 Pro session.
The same 55/55 SuperBear supplemental run above is green; NR2 Pro confirmation
remains required by the owning-host declaration.


Cross-backend sync `AD-RETIRE-1-POINTWISE-2026-08-20` — **autodiff reference
numerical policy; NVIDIA outcome: follow-up required (sm_120).** PR #600
retires the ODE-family pointwise hand rules behind the `DerivativeContract`
datum (dtype-preserving reference rules; unified log/sqrt boundary guard —
see the rocm entry for the full contract statement). Expected impact on
sm_120: parity-neutral — the CUDA loss/optimizer backward lanes compare
against the same updated reference on both sides — but per the
no-evidence-transfer rule that expectation is not a claim: run the
CUDA-marked autodiff/loss parity tests on NR2 Pro and record the outcome
under this sync key. Boundary inputs below 1e-12 are outside every sampled
parity envelope; the dtype change moves the reference TOWARD the fp32 native
lanes, not away.
The same 55/55 SuperBear supplemental run above is green; NR2 Pro confirmation
remains required by the owning-host declaration.


Cross-backend sync `APPLE-RUNTIME-SINGLE-IMAGE-2026-08-19` — **Apple runtime
loading; NVIDIA outcome: not applicable.** The single-image slice fixes duplicate loading of the Apple GPU runtime.
`_apple_gpu_dispatch._prebuilt_candidate` decided whether a prebuilt dylib was
current by `ctypes.CDLL`-ing it and probing sentinel symbols. Loading is not a
read-only probe: it registers the runtime's Objective-C classes process-wide,
and skipping the candidate afterwards does not unregister them (the ObjC
runtime pins an image that has defined classes). A stale candidate therefore
stayed resident and the from-source dylib compiled next registered the same
classes again -- two images of one runtime, each with its own copy of every
file-static, including the thread_local last-error channel that
`_apple_gpu_run_checked` reads to decide whether a kernel failed. Staleness is
now read from the file's symbol table via `nm`, so a stale candidate is never
loaded; an undecidable probe (no `nm`) keeps the previous load-and-probe
behaviour rather than rejecting a library it cannot fault.
NVIDIA impact: none. The loader is Apple-specific (`_apple_gpu_dispatch.py`,
`libTesseraAppleRuntime` / `libtessera_apple_gpu_runtime`) and no NVIDIA path
reaches it. No sm_120 retest required and no device evidence is produced or
claimed.


Cross-backend sync `APPLE-STUB-BINARY-OPCODES-2026-08-19` — **shared runtime
contract; NVIDIA outcome: not applicable.** The portable-stub opcode slice fixes a silent wrong-answer class in the Apple
GPU elementwise-binary lane. `apple_gpu_runtime_stub.cpp` — compiled on every
NON-Darwin host so the C symbol exists — implemented opcodes 0-8 and its
`default:` arm assigned `out[i] = x`, so `mod`(9), `floor_div`(10), the six
comparisons(11-16), and the logical/bitwise ops(17-22) returned the LEFT operand
instead of computing. Because the symbol exists,
`_apple_gpu_dispatch_mpsgraph_binary` takes the kernel branch rather than its
numpy fallback, so those values came back as if computed, with no diagnostic
(Decision #21). Fixed by implementing opcodes 9-22 to match
`mpsg_binary_node` and the declared host reference
`runtime._apple_gpu_binary_numpy`, and by rejecting an unknown opcode through
the stub's last-error channel (new kind 3) so it routes to the host fallback
instead of returning a plausible buffer.
NVIDIA impact: none. `tessera_apple_gpu_mpsgraph_binary_f32` is referenced only
by Apple files (`runtime.py`, `_apple_gpu_backend.py`, `_apple_gpu_dispatch.py`,
`apple_exact_device_proofs.py`, the two Apple runtime TUs, and
`SiluMulToAppleGPU.cpp`); no NVIDIA lowering or runtime path reaches it. No
sm_120 retest required and no device evidence is produced or claimed.
Cross-backend sync `ZERO-FUNCTION-CANDIDATE-2026-08-19` — **shared frontend ABI
and diagnostics; NVIDIA outcome: not applicable today; parity by construction
for future migrations.** The zero-function-candidate slice (PR #590) changes `JitFn`'s call ABI recovery
and adds one diagnostic code. A `@jit` function whose AST lowering produced no
function raised a bare `IndexError` from `_establish_tracer_authority`, and the
same absence left `_call_arg_names`/`_constraint_ir_args` empty — silently
mis-binding keyword calls and skipping call-time constraint re-checking. The ABI
is now derived from the Python signature via the shared
`graph_ir.ir_args_from_signature` (Decisions #30/#31), and the apple_gpu tracer
lane lifts foreign interpreter exceptions into the new registered
`JIT_APPLE_GPU_TRACE_FAILED` code (Decision #21). `TesseraTraceError` passes
through unwrapped.
NVIDIA impact: none today. The diagnostic is apple_gpu-scoped, and the
trace-defer route that reaches it is gated on `target == "apple_gpu"`. The ABI
recovery is target-independent and strictly widens what was previously an empty
tuple, so no NVIDIA behaviour changes. No sm_120 retest required and no device
evidence is produced or claimed. Future sm_120 frontends inherit the corrected
keyword binding and constraint re-check.


Cross-backend sync `SCALAR-SIDE-ORDERING-2026-08-19` — **shared Graph IR
runtime contract; NVIDIA outcome: not applicable today.** The `scalar_side` slice (PR #589) makes the Graph IR lifted-scalar form carry
operand order. `graph_ir._OpExtractor._try_map_binop` lifts a literal out of
either side of a `BinOp` into the `scalar` attribute and records the side; until
now no code in `python/`, `src/`, or `tools/` read that record (Decision #29), so
`2.0 - x` and `x - 2.0` emitted indistinguishable IR and any consumer binding
`scalar` as the right operand computed `x - 2.0` for both — sign-flipped for
`sub`, reciprocal for `div`, with no diagnostic. Shared contract changed: a lone
`scalar` means the RIGHT operand, `scalar_side="left"` requests the mirrored
binding, and any other value is rejected rather than guessed (Decision #21).
NVIDIA impact: none. No NVIDIA lowering or runtime path consumes the `scalar`
kwarg — an exhaustive sweep for `get("scalar"`/`["scalar"]`/`get("other"` across
`python/tessera/` finds consumers only in `runtime._apple_gpu_dispatch_mpsgraph_binary`,
`runtime._execute_runtime_cpu_op`, and `matmul_pipeline._execute_op`, none of them
NVIDIA-specific. No sm_120 retest required and no device evidence is produced or
claimed. If an NVIDIA elementwise lane later accepts the lifted-scalar form, it
inherits this contract and must honor `scalar_side` for its non-commutative
opcodes.


Cross-backend sync `AD-LAW-SERIES-2026-08-19` — **shared reference rules and
test infrastructure; Nvidia outcome: not applicable today; parity by construction for future migrations.** The AD-LAW series (PR #588)
closes the swallowed-kwarg class registry-wide and adds the AD-WEIL-1 algebra
substrate plus all six executable laws. Reference-rule changes in this slice:
`stft`/`spectral_conv` JVPs **deleted** in favour of derivation from the
forward (both are bilinear, so every configuration key is honored by
construction); `jvp_istft` rewritten to honor axis/center/length/onesided/norm
while preserving its window quotient; `dequantize_nvfp4` fixed to accept the
per-block scale array its canonical forward takes (both modes previously
crashed); `jvp_lgamma`/`jvp_digamma` replaced (a dead zero-returning stub and
an identity placeholder); `jvp_cast` fixed for canonical dtype strings; the
shared polygamma helpers given reflection formulas (previously an O(n) loop
that hung on valid negative input — a live defect in the REVERSE path too).
NVIDIA impact: no native binding consumes these reference rules yet, so nothing retests. Future sm_120 backward family migrations inherit the law-checked oracle lane and the E2E-REAL-6 Law-3 gate. The previously recorded open spectral/quantize swallow findings are
therefore CLOSED; `_OPEN_FORWARD_KEY_SWALLOWS` (42 entries from the tape
positional-routing scan) remains the open set.


Cross-backend sync `AD-LAW-1-SHARED-ORACLE-2026-08-18` — **shared test
infrastructure; NVIDIA outcome: not applicable today, parity by
construction for future migrations; no sm_120 evidence changed.** AD-LAW-1
(PR #584) adds law oracles (adjoint + canonical-forward chain) over the
shared numpy reference JVP/VJP registries and the byte-gated
`autodiff_law_audit` dashboard. NVIDIA has no native binding to the two
fixed reference rules (`jvp_rmsnorm` eps default, `jvp_clamp` kwarg names),
so nothing retests; future sm_120 backward family migrations inherit the
law-checked oracle lane and the E2E-REAL-6 Law-3 gate (#583). Open shared
follow-up: the pinned swallowed-kwarg findings in `test_autodiff_laws.py`.
*Triage update, same key (AD-LAW-1b):* reference JVP fixes landed for
`clip` alias deafness, `add`/`mul` unary-`scalar`, and fft/ifft/rfft/irfft
`norm` handling; five entries benign-classified; the open set is now the
stft/istft/spectral_conv and quantize families only. *Spec-growth update, same key (AD-LAW-1c):* law coverage roughly doubled (109 adjoint / 87 chain rows green incl. attention, spectral-complex, structural, and loss families); two more silent reference JVP defects found and fixed — `lgamma` (derivative was a dead stub returning 0) and `digamma` (whole JVP was an identity placeholder) — plus `jvp_cast` crashing on canonical dtype strings. Forward-mode reference oracles for those three ops changed; reverse-mode VJPs are untouched, so no backend backward package is affected. Reflection formulas added to the shared polygamma helpers (`_digamma_positive`/`_trigamma_positive`): the upward recurrence advanced by 1 per step, so a valid input like -1e9+0.5 spun for ~10^9 iterations — a live defect in the **reverse** path too, since `vjp_lgamma`/`vjp_digamma` already call these. Now O(1) on the whole real line, exact against the canonical forward, poles -> nan.

Cross-backend sync `W4-DYNAMIC-EFFECT-NONLINEAR-CFG-2026-08-18` — **shared
contract parity; CUDA follow-up required.** Dynamic saved-slot data/shape
tapes, exact polynomial witness guards, and variadic branch CFG state are
target-neutral contracts. Effectful replay remains fail closed except for
compiler-owned extent assertions. NVIDIA has no native region-product consumer
or SM120 evidence and inherits no x86/gfx1151 proof.

Cross-backend sync `W2.4-E2E6-SYMBOLIC-2026-08-18` — **shared legality and
frontend authority landed; CUDA evidence unchanged.** Production pipeline,
WarpSpec, barrier-reuse, and derived Tile dataflow relations now run through
one staged `TileDataflowLegalityPass`; legacy CLI names are wrappers. Pure
static annotations abstract-trace before AST compatibility capture. The
existing SM120 Lion and causal DeltaNet launchers are now selected by explicit
family plugins instead of `JitFn` target dispatch. This is an ownership move,
not new device evidence; CUDA state-lineage package unification and an exact
SM120 packet remain open.

Cross-backend sync `W1-W3-AUTHORITY-CLOSEOUT-2026-08-18` — **shared contracts
landed; CUDA producer proof remains open.** CUDA-math kinds and TCGen05 operands
are ODS-typed, bare Tile fragments are rejected, and WarpSpec no longer trusts
legacy ancestor markers. Existing SM120 loss packages are reached through
explicit VJP plugins. The final tensor-valued MMA producers, barrier-at-birth
emission, SM120 Target proof, and Lion/DeltaNet shared-lineage proof remain
NVIDIA-owned; host-free verification transfers no device evidence.

Cross-backend sync `W4-PRODUCT-1-RESIDUAL-CONTRACT-2026-08-17` — **shared
contract parity validated; CUDA physical follow-up required.** SAVE/HYBRID
selection now has a digest-bound Graph→SCF carrier. Shared paired AD consumes
dynamic branch-local residual extents, bounded SAVE and sparse HYBRID `while`
state tapes, and bounded-dynamic counted-loop tapes. The target-neutral
compiler now classifies source-CFG SCCs and structurizes bounded pure reducible
or irreducible native CFGs as a typed program-counter state machine while
preserving CFG/Presburger identity. Nested canonical structured bodies and
mixed control/tensor state are admitted. Saved dynamic slots require total
data/shape-tape envelopes; unbounded, unsupported-region, and unrecorded
effectful forms remain fail closed. SM120
still needs its own region-product correctness/performance packet and inherits
no sibling proof.

Cross-backend sync `E2E-REAL-6F-OPTIMIZER-VJP-2026-08-17` — **shared
optimizer lineage validated; NVIDIA not applicable for the bounded physical
set.** The new `schedule.optimizer_vjp` → `tile.training_kernel` authority
does not declare CUDA consumers because no matching native reverse package
was proved in this slice. SM120 remains fail closed and inherits no AVX-512 or
gfx1151 evidence.

Cross-backend sync `E2E-REAL-6E-STATEFUL-VJP-2026-08-17` — **shared
Adafactor/sequence-mixer authority validated; NVIDIA target follow-up
required.** Factored/full Adafactor and causal DeltaNet backward now have a
non-reexecuting Graph→Schedule→Tile package for x86/gfx1151. The existing
SM120 DeltaNet path is now plugin-owned but remains a compatibility package;
NVIDIA has no
Adafactor Target owner. Neither inherits sibling numerical evidence; each must
adopt the shared lineage carrier and produce CUDA exact-device proof.

Cross-backend sync `E2E-REAL-6D-LION-VJP-2026-08-17` — **shared flat Lion
Graph and non-reexecuting proof parity validated; CUDA plugin follow-up
required.** The existing SM120 PTX Lion VJP remains numerically valid, but it
is now selected by the family plugin but does not yet consume the shared
`schedule.lion_vjp` state-lineage artifact. A CUDA follow-up must extend that
typed package to SM120 and rerun its owning-device packet. No x86/gfx1151 evidence
transfers.

Cross-backend sync `E2E-REAL-6C-ATTENTION-VJP-2026-08-17` — **shared
rank-4 authority and bottom-right ragged-causal semantics validated; CUDA
plugin follow-up required.** The shared family registry now owns
`flash_attn`/GQA/MQA reverse through tracer Graph →
`schedule.attention_backward` → `tile.attention_backward_kernel`. NVIDIA’s
existing SM120 package is unchanged and inherits no x86/gfx1151 evidence. A
CUDA plugin consumer and independent SM120 numerical packet are still required.
The rank-3 `multi_head_attention` wrapper remains outside this bounded rank-4
migration, and active dropout needs keyed, non-reexecuting replay proof.

Cross-backend sync `E2E-REAL-6B-SPECTRAL-VJP-2026-08-17` — **shared tracer
and package-contract parity validated; CUDA physical follow-up required.**
Concrete AST specialization now resolves the same shape-derived spectral
identity as tracing, and compound spectral reverse products have one declared
Graph/Schedule/Tile/Target family-plugin boundary. NVIDIA is deliberately not
a Target consumer in this slice: no PTX package, CUDA runtime path, or SM120
evidence was added. A future CUDA consumer must own its package and independent
device proof.

Cross-backend sync `GFX1151-CALIB-BAREMETAL-2026-08-16` — **shared calibration
authority parity validated; no CUDA evidence transfers.** `target_perf` now
rejects explicitly provisional and WSL-hosted corpora from its measured
selector registry while exposing a non-mutating pruning reader. NVIDIA code,
SM120 selectors, and CUDA packets are unchanged. A future SM120 corpus still
requires its own CUDA-event/CUPTI-correlated evidence.

Cross-backend sync `REF-TIER-PHYS-2026-08-16` — **shared Schedule contract
received; CUDA physical follow-up required.** Batched tridiagonal solve and
the four coalition zeta/Mobius transforms now have content-addressed
Schedule→Tile carriers. The coalition transforms share one parameterized
Yates butterfly rather than four emitters. NVIDIA gains no PTX consumer or
SM120 claim in this slice; a future lane must select an architecture-owned
parallel solver/butterfly schedule and provide independent device evidence.
The shared Schedule Object now snapshots nested resource metadata before
digesting, and dynamic rearrange/GQA-fold inference preserves ranked `?`
dimensions; parity is validated without transferring a physical schedule.

Cross-backend sync `LAYOUT-SCHEDULE-OBJECT-2026-08-16` — **shared carrier
parity validated; physical producer follow-up required.** The native layout ABI
and GQA fold transfer no CUDA layout or raster decision. SO-1 now owns the
content-addressed action/edge/role/residency value; SO-2 registers symbolic
producer/consumer roles on Tile mbarriers and proves the Hopper split with the
same rule as AMD ping-pong. Shared loop-carried role provenance and role-bearing
pipeline state are implemented. Barrier-at-birth now lands on the canonical
typed producer path: WarpSpecialization creates the role-bearing barrier beside
the roles, AsyncCopyLowering binds that exact SSA value to copies and waits,
and NVTMA assigns region-local slots without synthesizing a function-global
barrier. Nested streaming and flattened single-device paths are both covered;
multi-region isolation is regression-tested. Deletion of the remaining legacy
WarpSpec ancestry assumptions and exact-device SM120 proof remain open. No
raster output or selector evidence changed.

Cross-backend sync `ATTN-BWD-ARCH-2026-08-16` — **no NVIDIA result transfers.**
ROCm's canonical split backward program was re-audited and x86 gained a
deterministic parallel implementation. The SM120 architecture-owned backward
package and performance evidence remain independent.

Cross-backend sync `X86-PASS-DIALECT-DEPENDENCY-2026-08-16` — **parity
validated; no NVIDIA physical follow-up.** The shared pass library now models
the optional hardware-free x86 Target dialect as a declared MLIR pass
dependency and fails closed when it is absent. No NVIDIA dialect, pipeline,
CUDA ABI, schedule, selector, or SM evidence changed.
The same closeout removes a private permissive `schedule` dialect from the
shared transform library; all transforms now consume the canonical ODS
`TesseraScheduleIR` authority. NVIDIA receives the build-parity fix only.

Cross-backend sync `PDE-EXACT-CONTRACT-2026-08-14` — **shared semantic parity
validated; CUDA physical follow-up required.** Exact-rational PDE
classification and the first fail-closed diffusion stability certificate are
backend-neutral. No PTX stencil/boundary/halo package or evidence is claimed.

Cross-backend sync `DIST-SHARD-HVP-2026-08-14` — **shared compiler parity;
native NVIDIA follow-up required.** Reshard plans now materialize bulk
collectives as Graph→Schedule→Tile SSA with subgroup/region identity and
deterministic all-to-all matching rounds. Exact forward-over-reverse HVP now
exists as a compiler Graph product. NVIDIA receives those shared contracts but
adds no NCCL launch binding or SM120 HVP package in this slice; native
multi-rank and product evidence remain independent gates.

Cross-backend sync `E2E-REAL-6-NATIVE-VJP-2026-08-14` — **normalization parity
validated; no new SM120 evidence claimed.** The existing SM120 normalization
backward launch is now owned by the shared native-VJP family registry with
explicit Graph/Schedule/Tile/Target declarations instead of package
construction inside `JitFn`. Runtime semantics and the existing SM120 evidence
row are unchanged. Other NVIDIA backward families remain compatibility paths
until migrated independently.

Cross-backend sync `AMD-ISA-DTYPE-2026-08-14` — **parity assessed; no NVIDIA
physical change required.** The new selector is AMD-specific and changes no
Graph dtype spelling, public operation, Schedule contract, NVVM/PTX ABI, or
SM120 capability. AMD FP8/BF8/FP4/MX and sparse-WMMA evidence cannot be used as
CUDA evidence. The shared `OP-DTYPE-FLOW-1` generator now audits SM120 by
operator and storage dtype; target-wide derived legality remains `legal_only`
unless an NVIDIA manifest owns the physical kernel.

Cross-backend sync `CI-LIT-BACKEND-DIALECTS-2026-08-12` — **not applicable to
NVIDIA, by existing architecture.** The `Validate / lit` lane was dead from
2026-08-11 to 2026-08-12 (pytest collection aborted on a missing `ml_dtypes`,
fixed in #554); the first green-collection run failed 27 of 367 fixtures on
unregistered `tessera_x86` / `tessera_rocm` dialects. No NVIDIA fixture is in
the failure set and this PR changes nothing for NVIDIA: the NVIDIA dialect is
off by default and its fixtures run under the separate `tessera-nvidia-opt`
driver (`%tnv`), not the `tessera-opt` binary this lane builds — the same
separation `test_target_ir_contract.py` already records as NVIDIA's documented
skip under Decision #19.

Architecture-specific reason this stays not-applicable rather than
follow-up-required: adding `TESSERA_BUILD_NVIDIA_BACKEND=ON` to the shared lane
would be actively harmful. With `ENABLE_CUDA` off — the only option on a
CUDA-less runner — `tools/tessera-opt/CMakeLists.txt:95-96` forces
`TESSERA_OPT_LEAN_ARTIFACT_DRIVER`, dropping core `TesseraIR`/`TesseraPasses`
and the x86 Target IR. That is the same trap documented for ROCm under this
sync key; NVIDIA's separate driver is the existing, correct answer. Revisit
only if a CUDA-capable runner joins CI, which would also be the point at which
sm_120 exact-device evidence becomes schedulable.
Cross-backend sync `CI-LIT-DEPS-2026-08-12` — **parity validated; no physical
follow-up required.** PR 554 made the shared opt-in MLIR lit lane install the
workflow-owned Python dependency set before `lit`/FileCheck collection. This is
backend-neutral test infrastructure, changes no compiler/runtime contract, and
requires no CUDA package or exact-device evidence.

Cross-backend sync `PDE-STENCIL-FOUNDATION-1-2026-08-12` — **shared semantic
parity validated; CUDA physical follow-up required.** Explicit coefficients,
scheme/order, and per-axis spacing are compiler requirements. The absent
NVIDIA stencil/halo/boundary symbols are artifact-only rather than callable
Target records. No SM120 package or CUDA packet was added.

Cross-backend sync `BLOCK-ATTNRES-ROCM-2026-08-12` — **follow-up required.**
The shared Block AttnRes plan establishes portable balanced-partition,
epsilon-qualified numeric, VJP, and softmax-merge oracle contracts. This PR
adds Phase-1 stats/merge/finalize references, the Phase-2 stdlib recurrence,
Phase-3 typed Graph/VJP/JVP contracts, and the Phase-4 content-addressed
Schedule→Tile artifact, but no SM120 package or CUDA evidence. NVIDIA must
later provide its own
stats-attention/merge Target consumer and exact-device packet; the small depth
shapes do not by themselves justify tensor-core lowering, and ROCm proof does
not transfer.

Cross-backend sync `MODEL-FUSED-PHYS-1-2026-08-12` — **shared MiniMax MSA
package lineage landed; SM120 consumption remains follow-up required.** x86
and gfx1151 now consume exact digest-bound MSA artifacts without Graph
redispatch. No CUDA package or SM120 evidence was added, and those schedules do
not transfer. NVIDIA must bind the same parent-digest contract to its canonical
NVVM/PTX package; DeepSeek MLA/DSA remain independently open.

Cross-backend sync `MODEL-WEIGHT-PHYS-1-2026-08-12` — **shared physical-byte
weight ABI landed; SM120 FP8/INT4 consumption remains follow-up required.** The
carrier preserves genuine checkpoint bytes and separate fp32 scales under a
content digest and forbids full-weight materialization. No CUDA package or
SM120 packet was added, and gfx1151 INT4 evidence does not transfer. NVIDIA
must bind this ABI to its architecture-owned FP8/INT4 package and exact-device
performance proof.

Cross-backend sync `W4-PRESBURGER-SHARD-2026-08-12` — **shared analysis
contracts landed; SM120 consumption remains follow-up required.** Graph IR now
carries typed integer-affine plus exact modular/divisibility constraints into
the C++ Presburger consumer. The shared sharding layer has a fail-closed
replicated/tiled/partial-reduction fixed point and explicit reshard planner;
lowered `control_scan` owns shared recompute-all JVP/VJP products. This adds no
CUDA region product, reshard lowering, NCCL packet, or SM120 evidence; other
architecture results do not transfer.

Cross-backend sync `W4-CFG-RESIDUAL-W5.2G-2026-08-14` — **shared compiler
carriers and scalable scheduler landed; CUDA follow-up required.** The
tracer-owned structured CFG, block-wide Presburger identity, and executable
SAVE/HYBRID residual ABI change shared lineage only. The action-DAG model now
uses deterministic critical-path/list scheduling with safe lower-bound pruning
and a small-DAG exhaustive oracle. No SM120 region product, physical producer
wiring, calibrated packet, or selection claim was added; architecture evidence
does not transfer.

Cross-backend sync `E2E-AUTH-DAG-2026-08-12` — **shared native-product v3 and
automatic dependency consumption landed; SM120 remains follow-up required.**
Reduction and normalization now have truthful Schedule/Tile product carriers,
native JVP/VJP paths require cached tracer/AST differential proof for pure
programs, and Graph-derived R3 candidates consume compiler-generated edges.
No CUDA product child, physical dependency consumer, or SM120 packet was
added; x86/gfx1151 evidence does not transfer.

Cross-backend sync `E2E-AUTH-DAG-2026-08-11` — **shared frontend authority and
automatic dependence-edge contracts landed; SM120 remains follow-up
required.** Pure straight-line tensor signatures now cache tracer-owned Graph
IR and can be differentially certified against the retained AST candidate.
Native-JVP plugins declare their Graph/Schedule/Tile/Target disposition and own
package construction; compatibility gaps are explicit. W2.1 facts now generate conservative Tile action-DAG
edges with reason and analysis digests. This adds no CUDA family package,
edge-consuming physical pipeline, selector promotion, or SM120 evidence.

Cross-backend sync `AD-SOLVER-ISTFT-PHYSICAL-2026-08-11` — **shared product
contracts landed; SM120 consumption remains open.** Graph IR now represents
the exact ISTFT spectrum/window product, and the general solver parent binds
residual plus solution/parameter JVP/VJP children under restarted GMRES with a
true-residual gate. The shared compiler now derives those five children for
typed pointwise, sum/mean, rank-2 matmul, bounded-dynamic/mixed-storage,
distinct parameter-space, and statically counted-region residual Graphs.
Pure scalar `if`/bounded-`while` predicates now have explicit compare/select
replay in the shared child contract. The AVX-512 and gfx1151 packages and
packets do not transfer to CUDA. NVIDIA still needs PTX children and independent SM120 correctness and
performance evidence.

Cross-backend sync `E2E-REAL-6-JVP-SOLVER-2026-08-11` — **shared frontend and
family-plugin boundary landed; SM120 remains follow-up required.** Native
forward-product specialization now originates in tracer-produced canonical
Graph IR, and explicit family plugins own reduction, normalization, FFT, and
compound-spectral planning outside `JitFn`. General solver contracts bind exact
residual/JVP/VJP identities and can execute matrix-free reference products
without finite differences. This adds no CUDA family plugin, native package,
or SM120 evidence; the AST lane remains for unmigrated families.

Cross-backend sync `AD-FWD-DIST-3-2026-08-11` — **shared exact JVP,
structured-region products, and typed point-to-point transport landed; SM120
evidence remains open.** Public JVP/jacfwd no longer substitutes finite
differences. Compiler forward mode carries primal/tangent state through bounded
SCF. `collective_permute` reaches the existing one-process/multi-device NCCL
launcher as grouped send/receive with an explicit peer map. SM120 still needs
architecture-owned JVP packages, subgroup communicators, and exact multi-GPU
correctness/performance packets.

Cross-backend sync `W4-SOLVER-REGION-2026-08-11` — **shared bounded-region
adjoints and general matrix-free solver policy landed; CUDA consumption is
follow-up required.** Portable tracing now emits bounded SCF, and the paired
compiler differentiates effect-safe single-block `if`, counted `for`, and
canonical bounded `while` with implicit captures. General residual execution
uses restarted GMRES/CG policy in shared IR. This adds no SM120 region executor,
solver package, checkpoint packet, or performance claim.

Cross-backend sync `COMP-GRAPH-DATAFLOW-W2.1-2026-08-11` — **shared
analysis substrate landed; CUDA behavior and evidence are unchanged.** Graph
IR now has one fail-closed, invalidatable shape/alias/liveness/memory-
dependence/activity analysis with C++ and Python query surfaces. Reverse AD and
await sinking consume it. This is target-independent legality infrastructure;
it transfers no SM120 schedule or performance evidence. Region-aware clients
and native CUDA overlap proof remain separately owned.

Cross-backend sync `AD-FWD-FAMILY-2-2026-08-11` — **shared affine,
compound-spectral, solver-product, and native-collective contracts landed;
SM120 consumption remains open.** Compound spectral Graph operations now own
direct tangent interfaces, including an exact ISTFT window-product carrier, and solver
artifacts distinguish JVP/non-transposed from VJP/transposed solves. The
multi-rank product accepts only a live NCCL hardware adapter, but no SM120
correctness/performance packet is claimed by this shared-contract slice.

Cross-backend sync `AD-FWD-NATIVE-1-2026-08-11` — **shared native-product
lineage landed; SM120 consumption is follow-up required.** The parent artifact
schema binds paired-JVP IR to immutable ordered child packages and detects
child substitution. Only x86/AVX-512 and ROCm/gfx1151 are executable in this
slice; no schedule or evidence transfers to CUDA. SM120 needs family-owned
product packages and exact-device numerical/performance packets before a
native JVP matrix row may be added.

Cross-backend sync `COMP-EFFECTS-W2.2-2026-08-10` — **shared registered-effect
analysis closed; no CUDA evidence claim.** Canonical Graph records now carry
effect, alias, mutation, and stochastic identity; Python and C++ consume the
same fail-closed facts and internal calls reach a fixed point. Await sinking
uses that shared query. This changes scheduling legality only; SM120 still owns
native overlap execution and exact-device correctness/performance packets.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R4-2026-08-10` — **shared functional
MegaMoE plan consumption landed; NCCL/SM120 evidence remain open.** The
content-addressed plan binds chunk slices, per-expert capacity, two-live-frame
workspace limits, true-use dependencies, ordered collectives, and deterministic
combine order. R3 only prunes complete measured plan records; scalar CUDA-event
latency selects. Mock multi-rank execution transfers no performance claim.
SM120 still needs native NCCL stream/event binding and exact-device packets.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R3-2026-08-10` — **shared prune-only
Tile action-DAG model landed; no CUDA selection claim.** R3 validates explicit
dependencies and calibration identity, uses deterministic critical-path/list
scheduling, and composes compute/memory/communication lanes with queue
serialization. Exact small DAGs and proven lower-bound losers may be pruned;
every estimate is promotion-ineligible and scalar
measured latency remains authoritative. SM120 needs its own calibrated vectors
and exact-device packet before using this analysis; R4 production consumption
does not transfer from another backend.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R2-2026-08-10` — **shared measured
resource-vector schema landed; CUDA evidence remains architecture-owned.**
Successful measured autotune rows may record compute time, dtype-correct bytes
moved, communication bytes, queue/resource identity, timing provenance, and
the measured-candidate digest. Analytical rows cannot claim the vector, and
scalar measured latency remains selector authority. No gfx1151 or x86 timing,
queue, or resource identity transfers to SM120; CUDA event/activity providers
must populate their own provenance before R3 composition analysis can use it.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R1-2026-08-10` — **shared explicit
async lineage and fail-closed await sinking landed; CUDA consumption remains
architecture-owned.** Python Schedule→Tile no longer emits internal
`tessera.queue.*` compatibility markers: async copies produce named tokens and
waits consume them. Collective awaits move only across operations proven
memory-effect-free; mutation, RNG, aliases/casts, regions, and ordered
collectives are barriers. Existing typed CUDA token lowering is unchanged;
SM120 needs independent executable overlap and exact-device proof.

Cross-backend sync `AD-STOCHASTIC-RNG-1-2026-08-10` — **shared stochastic JVP
contract available; CUDA core consumption started.** Explicit key/counter Graph ops,
estimator provenance, dropout replay, fixed-key EGGROLL JVP, and derivative
proof obligations are target-independent. The typed NVIDIA generator and its
four native modes now consume the explicit key/counter ABI, but x86 and gfx1151
runtime evidence still does not transfer to CUDA; SM120 needs compiler-JVP
packaging, runtime launch, and exact-device proof.

Cross-backend sync `AD-FWD-PRODUCT-2-2026-08-10` — **public JVP ABI landed;
CUDA execution remains follow-up required.** Forward/JVP requests now carry
mode-neutral provenance and stable `wrt_indices`, and the compiler emits only
requested tangent terms. Tanh/sigmoid add direct CPU-oracle proof. No PTX
package, selector, or SM120 evidence transfers; native JVP remains fail-closed.

Cross-backend sync `AD-FWD-CORE-1-2026-08-09` — **shared compiler JVP
foundation landed; NVIDIA physical consumption remains architecture-owned.**
The Graph dialect now exposes compiler-owned tangent rules and a paired
`--tessera-autodiff-forward` function contract. Matmul/mul has independent CPU
IR numerical proof, while unsupported active operations and regions fail
closed. The generated ledger distinguishes compiler `ir_tangent` evidence from
Python JVP registration. This changes no PTX package or SM120 evidence; NVIDIA
must lower and prove any native JVP package independently.

Cross-backend sync `X86-TYPED-FAMILY-PLUGIN-2026-08-09` — **shared schema
parity assessed; no CUDA physical change.** x86 now validates a closed Tile
family and registered Target marker before selecting its prebuilt AVX-512
image. NVIDIA may reuse the schema and fail-closed family discipline, but no
x86 ABI call, schedule, package, or Zen 5 evidence transfers to SM90/SM120.
The x86 backward-family allowance is narrowly one explicit forward-recompute
companion, not a general multi-carrier escape hatch. The canonical NVVM/PTX
image boundary remains NVIDIA-owned.

Cross-backend sync `EGGROLL-ES-LOWRANK-2026-08-09` — **the shared Graph,
Schedule, Tile, lineage, member-RNG-v1, and fp32 numeric-policy contract has
landed; NVIDIA physical consumption is follow-up required.** gfx1151 owns the
first GPU exact-device rank-1 proof and Zen 5 now owns an independent AVX-512
fp32 package; neither is a portable SM schedule. NVIDIA remains a lead
performance target: an SM-owned SGMV / `mma.sync` implementation and numerical
packet are open. The `s32` lane maps to native int8→s32 tensor cores and remains
a separate EGG expansion. W4 scalar-gather/member reconstruction passes
mock-mesh proof; native NCCL multi-rank execution and a target packet remain
open. Contract:
`docs/audit/compiler/EGGROLL_SUPPORT_PLAN.md`.

Cross-backend sync `COLLECTIVE-RCCL-ADVANCED-LANES-2026-08-09` — **shared
artifact discrimination adopted; AMD transports not applicable.** Advanced
collective artifacts now distinguish Copy Engine, GIN/RMA, and gfx1250 DDA and
bind target architecture plus selector evidence where required. These RCCL
lanes do not transfer to SM120. NVIDIA device-initiated communication remains
an architecture-owned NCCL follow-up with its own public API, legality, and
exact-device packet; it must not reuse an AMD transport or selector claim.
The shared Target dialect now registers explicit window-lifecycle and
put/signal/wait operations, but those records are RCCL GIN-gated and do not
imply NCCL device-initiated support on SM120.
The native multi-process harness binds only the RCCL GIN ABI; its launcher
metadata and evidence schema may be reused, but no AMD operation or result
transfers to NVIDIA's separately gated NCCL device-initiated lane.

Cross-backend sync `COLLECTIVE-NATIVE-FOUNDATION-2026-08-09` — **host adapter
and artifact contract landing; SM120 evidence open.** The C++ NCCL adapter now
executes all-reduce, reduce-scatter, all-gather, and grouped send/receive
all-to-all from an explicit communicator and CUDA stream instead of compiling
to successful no-ops. The shared Target artifact binds initiation,
registration, ordering, capture compatibility, backend/source identity, and a
capability digest. Shared communicator-property discovery, move-only symmetric
window ownership, and runtime-digest rejection are available, but still need a
CUDA-enabled build and SM120 packet. No AMD LSA, Copy Engine, GIN, or DDA claim
transfers.

Cross-backend sync `COLLECTIVE-ASYNC-UNIFY-2026-08-09` — **shared software
contract closed; SM120 NCCL evidence open.** The legacy unregistered
`tessera.collective.*` markers are gone from active producers and fixtures.
Forward and adjoint passes emit registered futures, await their payloads, and
rewire SSA uses. Runtime topology validation now fails closed for unknown or
unsupported subgroup mesh axes, and native v1 forbids implicit non-fp32
conversion. Exact multi-GPU NCCL correctness/performance remains required; no
PTX selector or device claim changes.

Cross-backend sync `DIST-SHARD-ALIAS-1-2026-08-09` — **shared alias mapping
available; SM120 evidence open.** Five public reduction/broadcast aliases now
resolve to the registered all-reduce/all-gather transport; three sharding
entries remain compile-time placement/region contracts. `collective_permute`
correctly remains a distinct point-to-point gap rather than being mislabeled
as all-to-all. CUDA frontend capture and exact multi-GPU NCCL proof remain
architecture-owned; no SM120 claim transfers from portable execution.

Cross-backend sync `AD-SOLVER-RESIDUAL-EVAL-2026-08-08` — **bounded x86/ROCm
pilot landed; CUDA follow-up required.** The shared IFT chain now has a
content-addressed Schedule→Tile physical contract, and counted-region treeverse
can execute checkpoint replay before a row becomes eligible. SM120 has no
consumer for the diagonal-sqrt pilot, no general iterative solver, and no
complete-backward packet. This PR changes no PTX package, selector, policy, or
device claim.

Cross-backend sync `AD-CORE-EFFECT-CONTROL-COLLECTIVE-2026-08-08` — **shared
Graph/Tile/portable-Target contracts available; CUDA follow-up required.** Compiler activity,
effects, `stop_gradient`, stochastic rejection, and fail-closed region
adjoints are target-independent. The four collectives now lower into one
content-addressed asynchronous Target queue and execute through the shared
runtime-adapter ABI. SM120 still needs exact multi-GPU NCCL execution and a
device packet. No PTX selector, native performance, or device evidence is
claimed or transferred.

Cross-backend sync `GRAPH-VERIFY-SIGNED-1-2026-08-08` — **shared legality
parity validated; no CUDA physical claim.** Graph and canonical-attention
integer bounds now consume signed `IntegerAttr` values, preventing MLIR 23
unsigned accessors from accepting negative schedules, seeds, cache windows, or
control bounds. Direct negative IR cases cover both dialects. No PTX ABI,
SM120 schedule, selector, package, or exact-device evidence changes.

Cross-backend sync `AD-TSOL-SPECTRAL-1-2026-08-08` — **shared Graph contract
available; CUDA follow-up required.** Compiler spectral identity and the
FFT/RFFT/DCT transpose rules are target-independent and CPU-oracle proven.
SM120 still needs its own Schedule→Tile/native compound-backward package and
exact-device evidence; no CUDA support or performance claim follows from the
x86/gfx1151 carrier.

Cross-backend sync `AD-TSOL-SPECTRAL-NATIVE-2026-08-09` — **SM120 follow-up
still required.** The bounded spectral-filter/convolution consumers and proof
land only for AVX-512 and gfx1151. No PTX image, schedule, correctness, or
performance evidence transfers to NVIDIA; its compound-backward path remains
fail-closed until an architecture-owned package lands.

Cross-backend sync `AD-CORE-LINEAR-1-2026-08-08` — **shared Graph-IR follow-up
available; no CUDA physical claim.** Compiler-owned linear transposition now
covers structural views, broadcast, and operand-wise matmul in both autodiff
passes with CPU numerical proof. SM120 backward packaging remains
architecture-owned; no CUDA image, selector, schedule, or device evidence is
transferred by this shared interface.

Cross-backend sync `COMPILER-DASHBOARD-PROOF-TRUTH-2026-08-08` — **SM90 and
SM120 proof separated; no CUDA physical change.** SM90 compile/artifact rows
are no longer added to SM120 runtime counts, and only exact-device statuses
close a hardware op×target grain. No package, selector, or device evidence is
changed.

Cross-backend sync `X86-BUILD-ARTIFACT-DISCOVERY-2026-08-08` — **shared
fail-closed selection assessed; no CUDA package changes.** The x86 runtime and
native packager now honor `TESSERA_BUILD_DIR` and reject a missing selected
tree rather than loading a stale default image. NVIDIA keeps its own CUDA tool
and image discovery, and inherits no AVX-512 implementation or timing result.

Cross-backend sync `STANDALONE-COVERAGE-TRUTH-2026-08-08` — **registry truth
adopted; no CUDA execution claim changes.** The standalone dashboard now
generates its counts, compiler-layer rollup, exact-target manifest summary,
and open queues from the live registries. It explicitly separates aggregate
best-available evidence from per-target support. SM120 still owns every CUDA
physical and benchmark follow-up in its manifest; x86 and gfx1151 TSOL or
Adafactor evidence does not transfer.

Cross-backend sync `TSOL-NATIVE-REAL-FFT-2026-08-08` — **shared artifact
follow-up required; no CUDA schedule transfers.** The target-neutral FFT
contract now binds logical/physical length, Hermitian layout, and packed-real
versus full-complex policy. Only x86 and gfx1151 own physical N/2 consumers and
evidence. SM120 must implement and measure its own real-transform package
before selecting the packed policy; it inherits no AVX-512 or RDNA schedule.

Cross-backend sync `ROCM-BUILD-ARTIFACT-DISCOVERY-2026-08-07` — **parity
validated; no CUDA physical change.** Shared compiler-test discovery now
accepts fail-closed `TESSERA_BUILD_DIR` selection while retaining explicit
`TESSERA_OPT` precedence. The migrated runtime-library and backend-tool users
are ROCm-owned; SM120 packages, schedules, evidence, and selectors are
unchanged.

Cross-backend sync `AUTODIFF-RELAXATION-1-2026-08-07` — **shared
Python-reference contract; CUDA physical follow-up required.** `sparsemax`,
`entmax15`, `soft_top_k`, `gumbel_softmax`, and `perturbed_argmax` now have
storage-preserving reference semantics and autodiff rules, but no CUDA lowering
or SM120 evidence. They remain explicitly reference-only until a CUDA-owned
physical package is selected and proven.

Cross-backend sync `MATH-PHYSICAL-2-2026-08-06` — **shared dtype contract
assessed; CUDA follow-up required.** Physical binary math packages now require
matching input storage dtypes. The Zen 5 scan selector and gfx1151 HIP module
cache are architecture-owned and transfer no PTX schedule, dtype, or
performance claim. NVIDIA must run the reduced-storage and difficult-domain
corpus on its canonical CUDA math packages before claiming parity.

Cross-backend sync `TSOL-CONTRACT-GENERALIZE-2026-08-06` — **shared semantic
contract adopted; physical consumer remains follow-up.** Bounded dynamic
dimensions, arbitrary axes, storage policy, and normalization are now explicit
before an exact TSOL specialization is emitted. Zen 5 and gfx1151 now consume
that wider contract, but their ABI and evidence do not transfer. NVIDIA still lacks the
prerequisite promoted FFT package, so Schedule→Tile lowering rejects CUDA
physical consumption and records no dtype, numerical, or performance claim.
The architecture-owned sequence remains: close canonical CUDA FFT, define an
SM120 workspace/residency ABI, implement the compound package, then gather
exact-device evidence.
Cross-backend sync `TPROF-MULTICLOCK-2026-08-06` — **shared evidence schema is
executable on ROCm/x86; CUDA adoption remains follow-up.** The ROCm plan now records each
clock independently and forbids fallback relabeling. CUDA should adopt the same
provenance, validity, calibration, instrumentation, and verdict fields for host
wall, CUDA events, an architecture-qualified device clock, and CUPTI activity.
HIP `wall_clock64()`, `rtg_hsa_dispatch`, ROCprofiler, and gfx1151 evidence do
not transfer to SM120. CUDA clock choice and CUPTI/SM120 promotion remain
architecture-owned exact-device work.
The native-evidence extension adds content-digested provider captures,
clean-versus-instrumented image/resource comparisons, and exact-machine event
maps. These are shared evidence concepts only: ROCprofiler, RTG, Linux perf,
IBS, gfx1151, and Zen 5 records transfer no CUDA result. NVIDIA must populate
the corresponding fields from CUPTI activity/metrics/PC sampling on SM120.

Cross-backend sync `TSOL-ROCM-E2E-1-2026-08-06` — **shared ODS vocabulary
adopted; CUDA physical execution remains follow-up.** The target-neutral
`schedule.spectral_program` and `tile.spectral_program_kernel` contract is
registered in production. NVIDIA still lacks a promoted canonical FFT package,
so it cannot consume the compound artifact or inherit ROCm/x86 evidence. A
future CUDA implementation must first close its FFT package gap, then bind its
own workspace/residency policy and SM120 device evidence.

Cross-backend sync `TSOL-GFX1151-FUSED-BATCH-2026-08-08` — **not applicable to
CUDA execution.** The content-addressed FFT vocabulary now carries gfx1151's
batched fused-LDS residency explicitly, but the HIP image dependency, AMD LDS
kernel, and WSL timing evidence establish no CUDA package or SM120 selector.
NVIDIA's architecture-owned FFT/TSOL follow-up is unchanged.

Cross-backend sync `TSOL-SPECTRAL-POLICY-2026-08-08` — **shared DCT and
streaming policy adopted; CUDA physical follow-up unchanged.** DCT-I/II/III/IV
now carry distinct API, autodiff, Graph, Schedule, and Tile identities. The
target-neutral causal chunked-STFT state binds its policy digest and overlap
lineage, while centred streaming fails closed pending explicit lookahead.
NVIDIA still lacks the prerequisite CUDA FFT/compound package and inherits no
x86/gfx1151 physical or performance evidence. The length-one convolution and
one-sample STFT/ISTFT physical boundary repairs transfer no CUDA implementation
claim.

Cross-backend sync `ROCM-MATH-EVIDENCE-2026-08-06` — **not applicable to
NVIDIA codegen.** Centered Welford and the scalar boundary fixes alter ROCm C++
generators plus shared host-side atan2 quadrant semantics; no PTX, CUDA ABI,
math mode, or NVIDIA capability changed. NVIDIA must evaluate the same domains
on its own canonical math/reduction lanes.

Cross-backend sync `ROCM-FFT-PREBUILT-2026-08-05` — **not applicable; NVIDIA
still has no promoted canonical FFT package.** The ROCm opaque plan ABI and HIP
allocation policy are not transferred. A future CUDA package must define and
measure its own artifact-bound plan/workspace contract on NVIDIA hardware.

Cross-backend sync `FFT-PERF-2-2026-08-05` — **not applicable to the unproven
CUDA lane; follow-up remains required.** The new cached Bluestein, Rader,
mixed-radix AVX-512 codelets, and rejected Bailey candidate are x86-owned.
They do not establish SM120 code generation or evidence. NVIDIA's existing
mixed-radix/Bluestein and exact-CUDA gaps remain unchanged.

Cross-backend sync `FFT-PERF-FOUNDATION-2026-08-05` — **follow-up required;
radix-17 source changed without CUDA evidence.** The shared planner now admits
radix 17 and the CUDA generic-stage private array was widened accordingly, but
this Ubuntu/gfx1151 host cannot compile or execute the SM120 lane. NVIDIA must
compile and compare radix-17 direct execution against its prior rejection path
on CUDA before claiming parity. The expanded Schedule→Tile FFT identity remains
outside NVIDIA until its Bluestein gap closes.

Cross-backend sync `E2E-REAL-FFT-2026-08-05` — **follow-up required; no support claim.**
ROCm corrected its public FFT authority to the proven Stockham/Bluestein
package and identified `schedule.fft`→launch Tile as the remaining shared
boundary. That shared content-addressed contract is now implemented and
consumed by x86/gfx1151, while NVIDIA remains deliberately outside its target
set. NVIDIA must join it only after its existing mixed-radix
hook gains Bluestein and passes exact CUDA/SM120 evidence; ROCm evidence does
not promote this lane.

Cross-backend sync `FFT-MIXED-RADIX-BLUESTEIN-2026-08-03` — **follow-up required — mixed-radix only, no Bluestein, unverified.**
Tessera's own FFT (Stockham, `TargetHooks/`) extends from powers of two to
every length: a generic radix-r stage for the odd small primes and Bluestein
for the rest. Shared contracts changed, so all four backends are affected:

* **Planning is now one implementation** (`TargetHooks/Common/FFTPlan.h`).
  CPU, AMD and NVIDIA each carried their own `while (n%4) ... while (n%2)`
  driver loop, and all three silently returned a HALF-FINISHED transform for
  any other N while reporting success. `LegalizeSpectral::pickRadixSequence`
  was a fourth copy, factoring over radices 7/5/3/4/2 and pushing a residual
  prime as a "stage" of that radix -- a stage nothing could execute.
* **Compiler routing was wrong independently of the kernels.**
  `LowerToTargetIR::stageSymbolFor` mapped every radix other than 4 to
  `ts_stockham_r2_*`, so a static N = 12 = 4x3 emitted a radix-2 call for a
  radix-3 stage. The runtime driver was correct; the compiler path was not, and
  direct driver tests could not see the difference.
* **New C ABI surface:** `ts_stockham_rn_<backend>(in, out, N, L, r, sign)`
  (note the extra radix argument, which r4/r2 do not take), plus
  `tessera.target_ir.stage_radices` carrying it, and a
  `tessera.target_ir.bluestein` marker routing those lengths to the driver.

NVIDIA gets the shared plan and the generic radix-r stage, so every
mixed-radix length is routed and emitted correctly. It does NOT get Bluestein.

There is no CUDA toolchain on the development box, so ~60 lines of device code
written for it could not be compiled, let alone checked against a reference.
Shipping unverifiable device code is the same unproven-claim pattern the silent
truncation was an instance of, so the driver DECLINES instead:
`ts_fft_supported_nvidia(N)` answers the question and the driver returns
without writing `d_out` rather than truncating.

**Nothing in this change has been compiled for NVIDIA.** The generic radix
kernel is a mechanical mirror of the gfx1151-verified AMD one, which lowers but
does not remove the risk. First task on the CUDA box: compile the `.cu`, run
the mixed-radix sizes against numpy, then implement and verify Bluestein.
Registering the lane as a `spectral_fft` arbiter candidate is a separate,
still-open item.


Cross-backend sync `SHAPE-RULE-REGISTRY-2026-08-03` — **follow-up required - scale operands changed, and NVIDIA has no FFT lane.**
PR #493 closed the Graph IR shape-rule registry: **303 declared / 6 deliberately
undeclared / 0 unexamined**, with the `MAX_UNCLASSIFIED` ratchet dropped 106 -> 0.
Shared contracts changed; all four backends are affected equally at the
reference level:

* **Result contracts.** Multi-result ops now emit every SSA result
  (`kv_cache.read -> (K, V)`, `top_k`, `qr`/`svd`/`lu`/`nonzero`), and tuple
  destructuring (`v, i = ...`) lowers. The emitter previously called the
  single-result `_infer_result_type`, so a declared multi-result contract
  stopped at Graph IR.
* **Stateful handles.** `!tessera.kv_cache` is now reachable from Python; the
  emitter had been printing `tensor<*x?>` for a type the ODS has always
  declared.
* **dtype policy.** An integer input to a float-producing op promotes to the
  declared `COMPUTE_FLOAT_DTYPE` (fp32) instead of NumPy's width-derived float
  (`cos(int8) -> f16`, `cos(int32) -> f64`); index/count results use a declared
  `INDEX_DTYPE`; complex is a LOGICAL dtype carried in an interleaved real pair,
  not a storage format.
* **Diagnostics.** The whole `GRAPH_IR_*` family (17 codes) is registered - the
  drift gate's scanner did not know the prefix, so it reported green while the
  family accumulated unregistered.

**This is the Python reference lane, not generated device code.** The NVFP4/MX matmul lane carried `scale_a`/`scale_b` as ATTRIBUTES holding SSA
names; they are now real operands 2 and 3. This corrects Graph IR toward what
NVIDIA already declared everywhere else: the ABI is
`tessera.nvidia.nvfp4.a_b_scale_a_scale_b_d_m_n_k.v1` and the kernel is
`tile.matmul_kernel %a, %b, %scale_a, %scale_b, %d`, so Tile IR modelled the
scales as operands and only Graph IR demoted them. `nvidia_native.py`'s
packagers and its `requests_`/`supports_` predicates were updated; `bias` was
never affected (x86, ROCm and the unscaled NVIDIA lanes all append it as an
operand). Parity validated at the Python packaging level; **device evidence is
missing** - no exact-device run confirms the packaged buffer order end-to-end on
sm_120.

Recorded plainly: **complex FFT is REJECTED on nvidia_sm120**, because no NVIDIA
target declares an `fft` capability entry and zero NVIDIA source files mention
FFT at all. An earlier cut synthesised capability entries for absent ops and made
the sm_90 dashboard assert `fp8_e4m3` and `int8` FFT kernels - nine
`artifact_only` rows for a backend with no FFT. That was backed out. A target
with no `fft` entry is stating it has no `fft`.


Cross-backend sync `SUBBYTE-STORAGE-PATH-2026-08-03` — **follow-up required; NVIDIA is the target where this matters most.**
The quantize family is now correctly declared as MULTI-RESULT `(codes, scale)`,
and `quantize_nvfp4` has its own rule because its scale is per-BLOCK (one per 16
elements along the last axis) rather than per-tensor — the micro-scaled form
Blackwell implements. A shared per-tensor rule would have misstated the format
for exactly the architecture that motivates it.
**The open gap is a backend-path one, not a shape-rule one.** The reference
returns codes as **f32** — fake-quant. `fp8_e4m3`, `fp8_e5m2`, `fp4_e2m1` and
`nvfp4` are canonical dtypes and the Graph IR type system can carry them, so
nothing in the compiler prevents real sub-byte storage; no lowering produces it.
NVIDIA owns this first: consumer/datacenter Blackwell has native FP8 and NVFP4,
so it is the one target where "the backend upcasts anyway" is NOT the answer.
The Target IR must carry fp8/fp4 as real storage into the mma path.

Cross-backend sync `REDUCED-PRECISION-COMPUTE-2026-08-03` — **follow-up required, reference-level only.**
The shared reduced-precision policy changed: ops whose declared rule preserves
storage dtype now upcast reduced-precision inputs to f32, compute, and store
back. This repaired six ops whose INTERNAL arithmetic left fp16 range while
their answers fit easily — including `flash_attn` and `mla_decode`, both hot
SM120 paths, which previously returned float64 for f32 AND bf16 inputs.
**This is the Python reference lane, not generated CUDA.** The same hazard
class applies to NVIDIA kernels — a QK^T contraction overflowing fp16 before the
softmax rescales — and nothing here proves the generated kernels handle it.
NVIDIA owns verifying the accumulate-in-f32 contract on device; the reference
now states what the kernels must match.

Cross-backend sync `TILE-MMA-DATA-OPERANDS-2026-08-03` — **parity validated, and the prior NOT-VALIDATED status is now CLOSED.**
`MMAOp::verify()` now counts DATA operands, so the typed `tile.mma` fragment
form and the warp-spec `!tile.async_token` edge can coexist. NVIDIA needed the
same correction and had not received it: `NVIDIALowering.cpp` compared raw
`op->getNumOperands()` against 3/5 and then indexed raw operands, so the exact
typed-plus-token form this change unblocks would have hit `emitError` +
`signalPassFailure` during SM120 lowering — and operand 3 would have been the
token rather than an NVFP4 scale. Both the count and the indexing now use
`tessera::tile::dataOperands`.
**Verified by building it**, not by inference: ROCm and NVIDIA cannot both
register in one `tessera-opt`, so a second tree (`build-nvidia`,
`-DTESSERA_BUILD_NVIDIA_BACKEND=ON`) was configured and built. With
`tessera_nvidia` actually registered, the W0.9 parse/verify gate **passes for
sm90 / sm100 / sm120, plain and probe-annotated** — it had been SKIPPING every
run since PR #490. The contract test now discovers either build, so NVIDIA is
no longer silently unmeasured.

Cross-backend sync `TARGET-IR-CONFORMANCE-2026-08-02` — **NVIDIA host-free
conformance validated (2026-08-21); exact-device execution remains separate.**
W0.9 added a real parse + dialect-load + verifier gate over every Target-IR
emitter, and it found that no Python-emitted Target IR was valid MLIR
(undialect-prefixed module attributes, an invented `<dialect>.func` container,
ops emitted with signatures their ODS rejects, and several undeclared op names).
Those defects were fixed and verified for `cpu`, `x86`, `rocm`, and `apple`.
The test harness now discovers this workspace's NVIDIA-enabled
`build-nvidia-cuda/` compiler rather than silently skipping it. Its real MLIR
parse/load/ODS lane passes for sm90, sm100, and sm120, including probe-annotated
multi-op IR and committed NVIDIA goldens. `tessera_nvidia.profiler_probe` and
the Python wrapper/call surface are registered. The former unrestricted
`AnyType` envelope is now a Target-value union: tensor/memref data, scalar or
vector fragments, and LLVM pointer/aggregate ABI values. A negative NVIDIA lit
fixture proves `!tile.async_token` cannot enter an MMA Target IR op. Validation:
NVIDIA lit 48/48 and `test_target_ir_contract.py` 28 passed; the 10 skips are
only Apple/ROCm dialects absent from this build. This is compiler conformance,
not SM120 execution evidence.

Cross-backend sync `CORE-ATTENTION-TRAINING-X86-2026-07-30` — **follow-up
required, no NVIDIA contract change.** X86 adopted the shared rank-4 forward
and tensor backward loops and closed its optimizer adjoints. No Zen 5 ABI,
schedule, LSE policy, or timing transfers to SM120. NVIDIA retains direct
shared-loop forward/backward consumption, architecture-owned LSE selection,
and its remaining backward materializers.

## NVIDIA-SPINE-1: make the completed SM120 package the canonical default

Cross-backend sync `EXECUTION-SPINE-2026-07-29` — **landing.** The SM120
Graph/Schedule/Tile → NVVM/PTX image and launch-descriptor path was already
complete under NVIDIA-E2E-1/-2, but `canonical_compile()` did not select it by
default. NVIDIA now owns one `native_package_kind` / `package_native` entry
point; the shared driver no longer duplicates the vendor family-dispatch table,
and eligible static modules auto-promote when the complete SM120 toolchain is
available. Explicit `package_native = false` remains a stable opt-out.

Host-free focused validation covers canonical selection, typed package
production, runtime projection, and the native artifact contract. Existing
RTX 5070 Ti evidence remains the exact-device proof for the unchanged lowering,
PTX, ABI, and schedules; this slice changes selection authority, not emitted
code. The ROCm, Apple, and x86 selector reconciliations subsequently landed
under the same synchronization key. Apple keeps its explicit Value Target-IR
compatibility/probe route outside descriptor promotion. X86 has since separated canonical MLIR/native target
`x86` from its `x86_c` source candidate. NVIDIA already has that separation;
no PTX, ABI, schedule, or exact-device evidence changes.

APPLE-RASTER-1 subsequently consumed the shared map in emitted MSL and retained
row-major after mixed Apple7 timing. This is Apple-specific evidence, not an
NVIDIA selector or measurement result.

## NVIDIA-CALIB-1: supply the sm_120 corpus to the hardware-free score calibration

Cross-backend sync `COSTMODEL-CALIB-2026-07-29` — **superseded by the terminal
ROCm home-architecture rejection.** No NVIDIA arbiter-score adoption work remains.

The independent Zen 5 hierarchical T1 packet now also rejects T1 for latency
ranking (median rho -0.4062, 0/3 winner matches). Its x86 cache hierarchy,
bandwidths, candidates, and verdict do not transfer to SM120; NVIDIA's own
descriptor-complete correlation packet remains the only valid local decision.

**Correction that created this item.** `APPLE_AUDIT.md` originally scoped this
calibration to Apple alone, on the stated grounds that ROCm and NVIDIA kernels
"cannot be measured". That was false for NVIDIA: this backend already has a
committed, **consumed**, device-keyed `nvidia:sm_120` autotune corpus covering
64/256/512/1024/2048 square buckets plus fused GEMM and causal attention,
generated by `benchmarks/nvidia/record_autotune_corpus.py`. Excluding it would
have discarded the deepest per-shape latency evidence in the fleet.

**Historical calibration result and current subject.** The original two
static, device-free scores from
[`../../compiler/AMD_KERNEL_COMPILER_SURVEY.md`](../../compiler/AMD_KERNEL_COMPILER_SURVEY.md)
§3.7–3.8 — a step-distance locality histogram and an N-way bank-conflict
analyzer — were not retained after the step-distance line failed on its ROCm
home architecture. Do not tune or transplant that score. The current subject is
the shared T1 GEMM model: symbolic tile identities, capacity-bounded LRU reuse,
cache-derived DRAM traffic, and explicit target compute/bandwidth inputs
([`TILESIGHT_ASSESSMENT.md`](../../compiler/TILESIGHT_ASSESSMENT.md) §3).

**NVIDIA's role: shape depth.** The committed corpus already varies the shape
axis within GEMM and attention at fixed op kind, which is exactly the axis Apple
cannot supply and the one a cache/reuse score most needs to be tested against —
locality changes with shape at constant op. Apple supplies op breadth
(`APPLE-CALIB-1`); ROCm supplies a second, independent architecture
(`ROCM-CALIB-1`). Fitting on any one of the three reproduces the single-arch
overfit the assessment records for NeuSight.

**Note on translating the retired bank metric.** It was derived for AMD LDS with a known
bank count and a wave64 4-phase access pattern (survey §5.1). CUDA shared memory
is 32-bank and warp-synchronous; the *method* transfers but every constant must
be re-derived for sm_120 before a conflict number here means anything. Do not
port AMD constants.

**Fleet outcome (2026-07-29).** ROCM-CALIB-1 tested the metric where it
originated and reproduced 0/6 committed gfx1151 winners (median rho -0.1381, 0%
positive). The agreed home-architecture failure rule ends this latency-ranking
line without coefficient or target retuning. NVIDIA therefore owes no sm_120
promotion analysis for this score; CUDA-specific bank diagnostics remain a
separate future model and must not inherit AMD constants.

**Live-input update (2026-07-30).** The owning RTX 5070 Ti now reports through
CUDA 13.3 `cudaDeviceGetAttribute`: 70 SMs, 2.497 GHz core, 14.001 GHz memory,
a 256-bit bus, and `cudaDevAttrL2CacheSize = 50,331,648` bytes (48 MiB). The
registry consumes the measured L2 capacity; the observed memory clock and bus
corroborate the existing 896 GB/s peak-bandwidth derivation.

**Counter rerun (2026-07-30).** Nsight Compute 2026.2.1 is now installed on
the owning WSL host. A resource-only capture of the existing 512³ f16
production-route launcher retained `/tmp/nvidia-calib-sm120-2026-07-30.ncu-rep`
(`sha256:e0d0a1e2650dbfc1da921f72ebf0f36c27a1ec1165b278c51905336d02d6c79d`).
For the two Tile implementations, `tessera_tile_matmul_direct_f16` reported
97.65% L2-sector hit rate and 1,273,856 DRAM bytes, while
`tessera_tile_matmul_shared_f16` reported 91.24% and 1,101,312 bytes. This
confirms that CUDA counter collection works and that the two schedules have
distinct cache/traffic behavior. The profiler's replay duration is explicitly
not timing evidence, and these are different schedule shapes rather than a
controlled candidate-rank packet; neither value changes selection or supplies
a T1 correlation verdict.

**Correlation verdict: not identifiable from the committed corpus.** The
corpus retains latency keyed by named implementation but does not serialize the
candidate Tile M/N/K/raster descriptor needed to replay T1 for each competitor.
It therefore cannot supply an honest within-shape rank correlation, even with
the now-measured cache input. Retain T1 solely as a legal pruning estimator;
measured latency remains selection authority. A future CUDA corpus revision
must persist candidate schedule descriptors and profiler counter availability
before this item can produce a correlation verdict. Do not fabricate a rank or
revive the rejected AMD step-distance metric.

## NVIDIA-RASTER-1: consume the shared block-rasterization contract

Cross-backend sync `RASTER-CONTRACT-2026-07-28` — **follow-up required, owning
host NR2 Pro (RTX 5070 Ti, sm_120).**

**Shared contract changed.** Schedule IR gained two attrs and two knobs —
`raster_order` (`row_major` | `column_major` | `grouped_m` | `grouped_n`) and
`raster_group` — carried on `schedule.tile` and `schedule.knob`, mirrored by
`TuningConfig.raster_order`/`raster_group` and persisted in the SQLite tuning
cache. The order is a *permutation of block ids onto the tile grid*, defined in
the arch-neutral `compiler/tile_rasterization.py` with a `remap()` reference, an
`emit_c()` snippet valid identically under CUDA and HIP, and `is_bijection()` as
a total hardware-free oracle. Rationale and the 35%→72% L2 figure that motivated
it: [`compiler/TILESIGHT_ASSESSMENT.md`](../../compiler/TILESIGHT_ASSESSMENT.md)
§3.2.

**Implementation landed (2026-07-30); selection remains open.** The SM120
`mma.sync` fused-GEMM and gated-matmul emitters now consume `raster_order` and
`raster_group`. `row_major` retains their established 2-D launch and direct
`blockIdx.x` / `blockIdx.y` coordinate arithmetic. Non-default orders flatten
the same block count to one dimension and inject the shared `emit_c()` mapping,
including ragged final panels. The compiled-artifact cache key includes both
knobs, so a swizzled binary can never alias a row-major one. Focused host-free
tests cover source selection and the shared permutation oracle; exact RTX 5070
Ti execute-and-compare covered grouped-M fused GEMM and grouped-N gated matmul
on ragged dimensions. No selector change is implied.

**Why it did not land in the contract PR.** Changing a hardware-verified
`mma.sync` kernel without sm_120 silicon to measure the result would be an
unverified edit to a proven path for no demonstrable gain. `row_major` is the
default and reproduces the existing index arithmetic exactly, so today's emitted
code is byte-identical.

**Validation performed (host-free).** `tests/unit/test_tile_rasterization.py`
proves the permutation property over ragged grids and **compiles the emitted C
with host clang, running it against the Python reference for every block id**
under `-Wall -Wshadow -Werror`. That covers the arithmetic and the emission's
scoping, on any host.

**Remaining exact-device evidence.** Whether a swizzle moves sm_120 latency, and
at which `raster_group`, for the GEMM shape buckets in the perf ratchet. This
needs a committed repeated-median CUDA-event matrix plus `ncu` L2 hit-rate
deltas on the NR2 Pro. Until then the axis is **carried, not selected**. T1 can
score the order symbolically, but it has not earned an sm_120 raster retain
verdict and cannot promote a choice.

**SuperBear timing packet (2026-08-22): row-major retained.** Seven repeated
CUDA-event medians across square, rectangular, and ragged buckets swept
row-major, column-major, grouped-M, and grouped-N at groups 2/4/8. The three
per-shape winners disagreed (column-major, grouped-N/2, grouped-N/2) and
improved over row-major by 0.57%, 3.42%, and 2.54%; only one bucket crossed the
recorded 3% promotion floor. The committed packet is
[`nvidia_sm120_superbear_raster_2026_08_22.json`](../../../../benchmarks/baselines/nvidia_sm120_superbear_raster_2026_08_22.json).
Nsight Compute 2026.2.1 captured the exact 512x512x512 kernel: row-major was
96.65% L2 / 1,305,600 DRAM bytes, column-major 96.81% / 1,298,944,
grouped-M/8 94.08% / 1,054,720, and grouped-N/8 95.85% / 1,054,976. The lower
grouped traffic did not produce a stable all-shape timing winner, so the
selector remains row-major. This closes SuperBear's measured timing + counter
decision but leaves the NR2 Pro packet open under the owning-host rule.

## NVIDIA-AOT-1: decide whether NVRTC needs a precompiled peer — complete

Cross-backend sync `APPLE-AOT-METALLIB-2026-07-28` — **follow-up required**.
Apple added `apple_gpu_air`, a precompiled-artifact lane behind the shared
`register_compiler(target, compile_fn)` seam, measured against its compile-on-
launch lane (cold pipeline creation 29.7 ms -> 15.2 ms, ~1.95x; host-wall
timing on Apple M1 Max, not device-event evidence). NVIDIA is the backend
closest to Apple's position, not a distant one: its device code is NVRTC-
compiled at load (`nvrtc_jit.cpp`; runtime.py describes the mma.sync lane as
NVRTC-compiled for the device arch) and `runtime.py` has no cubin/fatbin
precompiled lane. So the AOT-vs-JIT question is genuinely open here. An earlier
version of this note said CUDA had 'nothing to catch up on' because
`emit/nvidia_cuda.py` contains no nvrtc reference — that was inferred from
absence of evidence in one file and is withdrawn. Follow-up: decide whether
SM120 wants a precompiled artifact lane, and if it is measured, reuse
benchmarks/apple_gpu/benchmark_aot_vs_jit.py *with its cache control* (a never-
before-compiled kernel per sample) — the driver's own cache is what made the
first Apple number 13x too good. No shared IR, ABI, dtype/op registration, or
numerical contract changed.

**Decision (2026-07-30): a precompiled peer is warranted.** The exact SM120
probe [`benchmark_aot_vs_jit.cu`](../../../../benchmarks/nvidia/benchmark_aot_vs_jit.cu)
uses a unique CUDA source and entry symbol for every sample, so neither NVRTC
nor the driver can serve a prior module. Both lanes load, launch, and verify the
same device result. On the RTX 5070 Ti (CUDA 13.3, driver 610.62), seven-sample
medians were **18.266 ms** for NVRTC compile + module load + launch and
**0.867 ms** for a precompiled cubin load + launch: **17.399 ms** saved per cold
request. Offline cubin construction was **173.004 ms**, amortizing after about
**10 cold launches**. The retained packet is
[`nvidia_sm120_aot_vs_jit_2026_07_30.json`](../../../../benchmarks/baselines/nvidia_sm120_aot_vs_jit_2026_07_30.json).

This closes the decision, not productization: a follow-on must add a versioned
SM120 cubin/fatbin artifact to the native package/runtime seam, preserve the
current NVRTC fallback for unsupported or stale artifacts, and execute-compare
the production matmul ABI before any selector promotion.

**Productization landed (2026-07-30).** `libtessera_nvidia_gemm.so` now ships
the first package-owned precompiled peer: the versioned
`tessera_nvidia_mma_f16_sm120_v1.cubin` beside the runtime image. Its canonical
`.cu` input is also generated into the library as the NVRTC fallback source, so
the AOT and JIT lanes have one kernel body, entry symbol, and physical
`A:u16[M,K], B:u16[K,N], D:f32[M,N], M,N,K` ABI. The loader admits the cubin
only on exact CC 12.0 and CUDA-driver >= 13000; absent, corrupt, incompatible,
or explicitly disabled artifacts use NVRTC. `TESSERA_NVIDIA_AOT_MODE=require`
is a strict deployment check and never falls back silently. Fresh-process RTX
5070 Ti execution proves forced-AOT, forced-NVRTC, and missing-artifact
fallback are numerically equivalent on ragged `17x31x9` f16 GEMM; the version
and canonical-source SHA are queried from the shipped C ABI. This is NVIDIA
runtime packaging only: no shared IR/ABI, selector, Apple, or ROCm contract
changed, so sibling backend plan changes are not applicable.

**Product hardening (2026-08-22).** The native package now carries both
`tessera_nvidia_mma_f16_sm120_v1.fatbin` and `.cubin` plus a generated manifest.
Both images embed artifact/ABI versions and the canonical-source SHA. Admission
reads and compares those globals before resolving `gemm`; a loadable but stale
image therefore reports `nvrtc_stale` and takes NVRTC in auto mode, while
`require` remains fail-closed. Format and source identity are part of the
runtime cache key. Exact SuperBear tests execute forced fatbin and cubin,
distinguish their keys, mutate an embedded SHA in an otherwise loadable image,
and prove the stale route numerically equals the fallback.

Cross-backend sync `TESSERA-OPT-CAPABILITY-SKIP-2026-07-27` moves the last 43
self-resolving test files onto the shared `tests/_support/compiler_tool.py`
driver contract, adds `--pass-pipeline=` inner-pass capability checking, and
folds `CompilerToolchain` onto one resolver and one capability check. NVIDIA is
**not applicable** for an architecture-specific reason: the NVIDIA lit and
compiler lanes drive the *separate* `tessera-nvidia-opt` binary through
`TESSERA_NVIDIA_OPT` / `CompilerToolchain.require_nvidia_opt` (the `%tnv`
substitution), which this resolver does not govern and which this change leaves
byte-for-byte untouched — `require_nvidia_opt` keeps its own `_tool_path`
lookup and its own skip. No CUDA registration, PTX or SM120 schedule, runtime
ABI, selector, or device evidence changed, and **no exact-device evidence is
claimed or required**. Should the NVIDIA lane later want the same
build-capability skip behaviour for `tessera-nvidia-opt`, that is a separate
follow-up owned by this plan, not a debt created here.

Cross-backend sync `ROCM-BF16-ATTENTION-2026-07-27` validates that the shared
BF16 attention carrier and canonical forward/backward loop contracts can be
consumed by a second physical backend. ROCm now has exact ragged-GQA
bias+softcap+causal-window+dropout forward proof and deterministic five-entry
backward proof on gfx1151, with dedicated resident BF16 timing ratchets. This
is parity validation at the shared semantic boundary only. AMD BF16 WMMA,
LDS scheduling, HSACO packaging, HIP workspace and launch ABI, numerical
evidence, and timing do not transfer to CUDA; NVIDIA retains its own SM120
BF16 package and exact-device evidence requirements.

Cross-backend sync `TESSERA-OPT-BUILD-CAPABILITY-2026-07-27` is **closed**.
The shared lit resolver now accepts `TESSERA_OPT_BIN`, `TESSERA_OPT_PATH`, and
`TESSERA_OPT_CPP` after the canonical `TESSERA_OPT` override, and the validation
script forwards its selected binary through that contract. Exact gfx1151
verification proves the full ROCm driver, legitimate lean ROCm artifact
driver, conflict rejection, both named streaming-attention fixtures, the
seven-fixture filter, and the complete 50-test ROCm backend lit suite. This is
shared test/build infrastructure only; no CUDA registration, PTX schedule,
runtime ABI, device evidence, or selector changes.

Cross-backend sync `LSE-CHECKPOINT-CONTRACT-2026-07-27` lands the real shared
checkpoint vocabulary: explicit memref source/destination, SSA row offset,
identity, memory space, lifetime scope, cache policy, and read/write effects.
Default forward lowering no longer emits a destination-less save. ROCm
validates saved versus recompute on gfx1151 and retains the provisional
128+ policy, but the newer dual-clock packet is explicitly fail-closed on WSL:
HIP events are positive yet non-transferable, and FP16 at 256 is not a stable
saved winner. Bare-metal gfx1151 confirmation remains required. NVIDIA is
**follow-up required**: consume the same shared contract, measure its own CUDA
forward-store/backward-load package, and retain or replace its zero-workspace
policy using exact SM120 evidence. AMD WMMA, HSACO size, threshold, and WSL
host-wall results do not transfer.

Cross-backend sync
`ROCM-ATTENTION-SHARED-BACKWARD-CONSUMER-2026-07-26` makes ROCm gfx1151 the
first direct physical consumer of the shared tensor-valued attention backward
phase loops. NVIDIA remains **follow-up required** to validate the same
dQ/split-dK/dV/fixed-reduction contract and map it to a CUDA-owned package.
The AMD WMMA schedule, five-entry HSACO, HIP launch workspace, gradient
evidence, and host-wall timing do not transfer. No shared IR or NVIDIA
capability state changed in this ROCm-owned closure.

Cross-backend sync `CORE-ATTENTION-TENSOR-LOOPS-MODIFIERS-2026-07-26`
materializes the deterministic split/reduced backward contract as tensor-valued
shared `scf.for` bodies with explicit dQ ownership, split dK/dV workspace
tensors, and ascending reduction. Registered shared score-bias and softcap
operations now preserve `softcap(scale*QK^T + bias)` inside the forward
KV-block recurrence, including rank-4 per-head bias. NVIDIA is **follow-up
required** to consume these phase operations through its SM120 package and
direct forward schedule. AMD HIP ABI code, HSACO, exact-device gradients, and
resident timing do not transfer.

Cross-backend sync `CORE-ATTENTION-BACKWARD-CONTRACT-2026-07-26` adds verified
split count, launch-owned workspace, block-loop metadata, ascending reduction
order, and canonical `softcap(scale*QK^T + bias)` semantics to the shared
carrier/oracle. NVIDIA is **follow-up required** to consume this form through
its SM120 schedule and validate dropout replay; AMD code and evidence do not
transfer.

## Cross-backend sync `E2E-REAL-5C-STATE-LINEAGE-2026-08-05`

The shared training spine now defines content-addressed logical-buffer lineage,
mutation identity, and typed Schedule→Tile contracts for Lion VJP,
factored/full Adafactor VJP, and sequence-mixer backward. **NVIDIA outcome:
follow-up required.** Existing SM120 packages remain NVIDIA-owned and do not
yet consume these exact artifacts. CUDA evidence and CUDA-owned buffer bindings
remain required.

Cross-backend sync `ROCM-E2E-ATTENTION-BACKWARD-2026-07-26` is not applicable
to NVIDIA physical execution. It adds a ROCm-owned five-entry HSACO and
gfx1151 split/reduced launch workspace without changing the shared launch
descriptor schema or canonical backward loop. AMD WMMA kernels, workspace
topology, exact-device gradients, timings, and selector state do not transfer.

The ROCm optimized-attention feature follow-up under
`ROCM-E2E-ATTENTION-CARRIERS-2026-07-26` adds AMD-only deterministic dropout
replay and combined bias+softcap consumption to the gfx1151 WMMA schedule,
plus a host-wall resident performance ratchet. It changes no shared carrier,
ABI, NVIDIA Target IR, CUDA schedule, capability, or selector. NVIDIA parity
at the semantic carrier remains validated independently; AMD counter code,
HSACO evidence, and WSL timing do not transfer.

Cross-backend sync `SSA-STATEFUL-TRANSPORT-2026-07-26` removes the last active
shared and ROCm `#tile.buffer_ref` compatibility readers after their fixtures
migrate to `!tile.buffer`; the deprecated attribute is parser-only. NVIDIA was
already SSA-only, so its SMEM/TMEM schedule and evidence are unchanged. The
shared ReplaySSM lifecycle schema now keys Apple and ROCm resident ABIs while
preserving session-private ring ownership, flush/rollback, ordered submission,
and drain-before-release. MoE metadata now owns launch-lifetime workspace and
can bind a canonical NCCL/RCCL rank/device fingerprint. NVIDIA consumes the
same local descriptor as before; no CUDA schedule, selector, or timing changes.

Cross-backend sync `ROCM-E2E-ATTENTION-CARRIERS-2026-07-26` lands an
AMD-owned consumer, native HSACO package, descriptor, and exact gfx1151 proof
for the already-shared `tile.attention_kernel` contract, plus a direct
correctness consumer for `tile.attention_backward_kernel`. NVIDIA parity at the
shared semantic carrier remains validated by its existing SM120 forward and
backward packages. ROCm's wave32 WMMA descriptor, LDS allocation, HIP ABI,
resource counts, timings, selector boundary, and direct scalar recurrence do
not transfer to CUDA. The ROCm v2 benchmark's operation-total and resident
synchronized HIP host-wall domains do not replace CUDA-event or CUDA
end-to-end evidence. No NVIDIA plan state or exact-device claim changes.

Cross-backend sync `ROCM-SSA-LDS-PIPELINE-2026-07-26` lands the AMD consumer of
the already-shared `!tile.buffer`, `!tile.async_token`, and
`!tile.pipeline_state` ownership vocabulary. It changes no shared operation,
type, verifier, ABI, or NVIDIA lowering. NVIDIA parity is therefore validated
at the existing SSA contract: WarpSpecialization continues to own SMEM/TMEM,
TMA/mbarrier, and architecture-specific pipeline mechanics. AMD LDS layouts,
waitcnt/s_barrier semantics, gfx1151 evidence, compiler timings, and selectors
do not transfer to CUDA or SM120; no NVIDIA follow-up is required.

Cross-backend sync `PACKED-LEGALIZE-CAPABILITY-2026-07-26` expands terminal
storage legalization without making sub-byte storage global. For `nvidia_sm120`,
the pass now proves the complete operation-specific consumer before stamping a
physical pack: packed load to ordinary store (explicit unpack/format
conversion), matching unscaled packed-load/store round trips, and packed
matmul whose A/B MMA descriptor agrees with the logical storage format.
Orphan or mixed-use packed loads, descriptor disagreement, arbitrary
operations, and the public shape-preserving Graph quantize/dequantize ABI stay
logical. The standalone empty-target transform remains available for explicit
IR inspection; named pipelines use the capability decision. Apple and ROCm
retain architecture-owned physical schedules; their evidence is not inferred
from SM120. The deprecated `#tile.buffer_ref` attribute is parser-only and no
active pass consumes it. No selector or timing disposition changes.

Cross-backend sync `CORE-STREAMING-ATTN-2026-07-26` replaces the shared
rank-2 FlashAttention whole-KV lowering with an explicit KV-block `scf.for`.
The loop carries the FP32 output accumulator, running maximum, normalization
sum, producer and consumer `!tile.pipeline_state` values, and an absolute
boundary offset. Each block consumes K and V through typed async tokens; the
online update now takes V explicitly, while causal/window/ragged masking and
counter-based dropout consume the loop offset rather than replaying block zero.
NVIDIA TMA descriptor hoisting traces each block slice to its kernel argument
and retains typed coordinates plus logical source extents, enabling
out-of-bounds zero fill for the ragged tail. WarpSpecialization no longer emits
name-based `#tile.buffer_ref` or annotation-only `#tile.pipeline_state`
metadata, and Schedule→Tile consumes structured per-operand `#tile.layout`
directly. The SM90 structural pipeline is lit-green. Follow-up sync
`CORE-STREAMING-ATTN-RANK4-ROCM-2026-07-26` adds shared rank-4 batch/head
distribution and proves a direct ROCm consumer. A direct NVIDIA
Target-IR/runtime consumer of the shared loop remains open; gfx1151 LDS/WMMA
schedules, resources, wall timing, and selector evidence do not transfer.
Existing launch-level attention images and selectors are unchanged.

Cross-backend sync `CORE-GEMM-KLOOP-2026-07-25` is **landing**, owned by
NVIDIA under the `NVIDIA-E2E-2` continuation. The shared compiler now forms a
target-neutral M/N/K `scf.for` nest with FP32/INT32 loop-carried accumulation,
zero-pad ragged guards, structured copy layouts, asynchronous SSA
dependencies, and threaded `!tile.pipeline_state`. The SM120 launch-level
FP16/BF16/TF32 images serialize the same `tile_m/tile_n/tile_k` contract and
reject descriptor disagreement before NVIDIA materialization. Two exact RTX
5070 Ti SM120 runs each pass all 12 FP16/BF16/TF32 square, rectangular,
ragged-K, and fully fragment-misaligned rows with FP32 accumulation,
matmul→bias→activation→residual ordering retained by the shared epilogue
contract, warm image/descriptor identity, and numerical comparison. The
checked-in 12-row repeated-median packet discards the first complete call,
amortizes 1,000 resident launches and 50 complete calls, and retains 31
interleaved observations per cohort in both timing domains. Every row meets the
4% two-run WSL gate (maximum device-event delta 1.39%, maximum end-to-end delta
3.47%); shared FP16/BF16 uses 42 registers and 10 active blocks/SM, direct TF32
uses 38 registers and 24 active blocks/SM, and all rows retain zero local
memory and zero spills. INT8 and packed formats remain follow-on after the
ordinary loop is stable. WSL timing is selector-ineligible and no selector
changes in this slice.

Cross-backend sync `ROCM-CORE-GEMM-KLOOP-2026-07-27` is **parity validated**
for NVIDIA. The only shared edit preserves the existing canonical
ragged-zero-fill guarantee across `tessera.matmul` → `tile.mma`; NVIDIA's
already-proven SM120 consumer and twelve-row packet are unchanged. AMD LDS,
wait/barrier, WMMA, HSACO resource, and gfx1151 wall-clock evidence do not
transfer. No NVIDIA route, capability, execution state, or selector changes.

Cross-backend sync `COMPILER-LIT-BACKEND-GATING-2026-07-24`: retired eleven
never-runnable CUDA13 pseudo-IR fixtures whose undefined `tessera_opt_built`
feature masked stale CLI options and unregistered operations. Core named
pipeline aliases now run in the ordinary LLVM lit lane; typed
Tile→NVIDIA→NVVM contracts remain owned by the NVIDIA backend lit suite.
`validate_nvcc_compile.py` now labels and runs its handwritten instruction
catalog strictly as a CUDA-toolchain probe, not evidence of Tessera emission.
The two integrated core+NVIDIA control-flow fixtures retain their precise
backend gate and still require the CUDA-enabled build on the NVIDIA host.

Cross-backend sync `COMPILER-PYTEST-PLATFORM-SKIPS-2026-07-24`: shared
compiler-owner markers now report foreign compiler proofs as skipped with the
required Apple, CUDA, ROCm, X86, or AVX512 system and a per-system count. This
is test-harness observability only; NVIDIA compiler ownership, CUDA evidence,
and selector state are unchanged.

This is the execution plan for evaluating, repairing, and then restructuring
the CUDA compiler tests on the NVIDIA box. It complements
[`NVIDIA_AUDIT.md`](NVIDIA_AUDIT.md); it does not reopen completed sm_120 feature
work unless a test exposes a real defect.

Baseline state on the NVIDIA box (2026-07-15, commit `ecf9483f`):

- The repository collects **264 exact-device CUDA tests** under
  `pytest -m hardware_nvidia`: **246 correctness** cases and **18 measured
  performance** cases. Eight of the correctness cases also require external
  compiler tools.
- The required CPU PR lane now excludes hardware and measured-performance
  states while retaining host-free CUDA emit, selector, validation, rejection,
  registry, and source-contract tests.
- Live CUDA tests carry `hardware_nvidia`; measured tests additionally carry
  `performance`; compiler/toolchain crossings carry `compiler_tool`.
- The WSL NVIDIA host is an RTX 5070 Ti (UUID
  `GPU-5072cda5-509a-008c-93c8-dc06e105f307`, CC 12.0), driver 610.62, CUDA
  13.3.73, LLVM/MLIR 23, and Python 3.14.4. At collection it was idle;
  observed graphics clock/power were 375 MHz / 23.18 W with a 300 W limit.
- The compiler-artifact lit suite passed **19/19**. The exact-device
  non-performance lane passed twice: **224 selected, 0 failed, 0 errored, 1
  Apple-only skip** (54.783 s and 53.658 s). This covers the required
  execute/compare layer, but it does not constitute performance evidence.
- The serial measured lane passed **18 CUDA tests** (plus the same unrelated
  Apple-only collection skip; JUnit: `/tmp/nvidia-performance-ecf9483f.xml`).
  The production hot-path ratchet, device-resident event timing, convolution
  routes, and Tile/autotune selection were included. Verbose `ptxas` and
  `cuobjdump --dump-resource-usage` for the compiler-produced Tile kernels
  report zero spills: 36 registers and no shared/local memory for direct,
  GELU, and SiLU; 42 registers and 2 KiB static shared memory from `ptxas`
  (3 KiB in the cubin resource table) for shared-staging f32/bf16/bias-ReLU.
  Nsight Compute 2026.2.1 profiled the f16 device-resident GEMM proof: the
  19x13x29 case launches four one-warp blocks (2x2 grid), uses 40
  registers/thread and no kernel shared memory, and reports 50% theoretical
  versus 2.08% achieved occupancy. The low achieved value is expected for this
  deliberately tiny ABI fixture: four blocks cannot occupy all 70 SMs. It is
  launch-shape evidence, not a production-throughput claim.
- Nsight captures for the production hot-path, convolution-route, and Tile
  schedule tests are retained under `/tmp/nvidia-*-ncu.ncu-rep`. Hot-path
  GEMM/fused/attention use 40 registers/thread and no shared memory (the
  attention kernel has 22.92% theoretical occupancy); fp quantization uses 18
  registers/thread, no shared memory, and 74.36% achieved occupancy. The
  direct/shared convolution routes use 40/30 registers and 0/32 bytes static
  shared memory, respectively. The selected Tile candidates report direct:
  36 registers, 0 shared, 50% theoretical occupancy; shared: 42 registers,
  2.05 KiB shared, 83.33% theoretical occupancy. The measured achieved
  occupancies are shape-dependent and low for the micro-grid fixtures, so they
  are resource evidence rather than selector-retuning evidence.
- The hot-path ratchet intentionally fails when run under Nsight Compute:
  profiler replay raises the wall-clock samples above its uninstrumented
  repeated-median caps. Its ordinary serial run remains the timing proof; the
  Nsight run is resource-only and must never update or relax the ratchet.
- The serving benchmark completed 20-repetition device-event and end-to-end
  rows for ReplaySSM `1x128x64`/`1x256x128` and fused/staged paged-KV decode
  at 128/512/2048 tokens (`/tmp/nvidia-sm120-serving-test5.json`). A temporary
  candidate corpus records both timing domains for rectangular `128x256x64`
  and ragged `127x259x63` f16 GEMM: end-to-end selects shared Tile (0.506 and
  0.490 ms), while device timing selects direct Tile (0.00657 and 0.00703 ms).
  No committed selector or timing cap changed.
- Nsight reduction coverage passed 49 native cases: f32/f16 kernels use 28
  registers and 1.02 KiB static shared memory with 100% theoretical occupancy.
  MoE transport passed three native cases: gather/combine use 20 registers and
  no shared memory; grouped GEMM uses 40 registers and no shared memory. The
  two live MoE transport tests were missing `hardware_nvidia` and are now in
  the canonical exact-device collection; its host-only rejection test remains
  unmarked. Their repeated-median rows and resource evidence are now committed
  under the TEST-5 baselines described below.
- NVIDIA-TEST-4 now has a shared storage/accumulation tolerance contract for
  f32, f16, bf16, TF32, FP8, int8, and NVFP4 semantics. The shipped MMA and
  compiler-produced Tile proofs consume that contract; exact integer/NVFP4
  contracts remain bit-exact. The reduction matrix now also proves f16/f32
  non-finite propagation and rejects empty, rank-invalid, unsupported-storage,
  and unknown-operation contracts before launch. Two WSL exact-device runs
  recorded 243 tests with zero failures/errors (one expected Apple-only skip).
- NVIDIA-TEST-5 now productizes repeated-median reduction and MoE transport
  rows in `record_reduction_transport_baseline.py`: every route records both
  end-to-end and CUDA-event timing through the production generated kernel.
  Two 20-sample sm_120 runs were recorded, the committed ratchet baseline
  covers reduction sum/mean/max plus MoE dispatch/combine/grouped-GEMM, and
  the first expanded serial performance lane passed 19 tests (one expected
  skip). The wider corpus and parsed resource evidence have since landed.
- The TEST-5 D2 corpus now includes measured square `512x512x512`, rectangular
  `128x256x64`, and ragged `127x259x63` f16/bf16 GEMM rows in both timing
  domains, alongside fused GELU, forward attention, gated MLP, and convolution
  routes. Two 20-sample WSL runs were taken before retaining the second corpus;
  the initial end-to-end winners varied between runs, so no selector was
  promoted from that evidence.
  Serving was likewise refreshed from two 20-sample runs for ReplaySSM and
  fused/staged paged-KV at 128/512/2048 tokens, retaining device-event and
  end-to-end medians separately.
- **NVIDIA-TEST-5 is closed (2026-07-16).** Two fresh high-sample sweeps
  (50 end-to-end repetitions after 10 warmups; 200 device-event repetitions
  after 20 warmups) converge for all 20 retained D2 rows under the declared 3%
  noise policy. Every row is selector-eligible only because both runs share a
  near-winner consensus and the selected route has a committed resource
  fingerprint. Backward attention adds regular `1x8x128x64` and ragged
  `1x8x257x64` dual-domain ratchets. Parsed Nsight evidence records registers,
  static/dynamic shared memory, theoretical/achieved occupancy, and explicit
  local-load/store spill counters for GEMM/Tile, fused and forward/backward
  attention, convolution, reductions, MoE transport, paged-KV, and ReplaySSM.
  The backward VJP uses 48 registers and measurable local-memory traffic; this
  is retained evidence, not hidden by a zero-spill claim. All other selected
  rows in the resource manifest recorded zero local spill traffic. The final
  serial performance lane passed 20 tests with one expected Apple-only skip.
- NVIDIA-TEST-6 has begun with `tests/_support/nvidia.py` (with a retained
  `tests/unit/_nvidia_testutil.py` compatibility import): it centralizes
  CUDA-toolchain, MMA-runtime, and bare CUDA-host probes without conflating
  their skip semantics, and supplies a common native-provenance assertion.
  The MoE transport, reductions, paged-KV, and ReplaySSM families migrated in
  the first batch; 70 focused tests passed and the canonical device collection
  remains 243 nodes. This is NVIDIA-only test infrastructure; Apple and ROCm
  plan states are unaffected.
- Cross-backend sync `LLVM23-NVIDIA-2026-07-16`: NVIDIA exact-device parity is
  now validated on the RTX 5070 Ti after the shared LLVM/MLIR 23 migration.
  A clean `sm_120a` build required the MLIR bytecode interface include and the
  `NVVM::Barrier0Op` to `NVVM::BarrierOp` API migration. NVIDIA lit passes
  19/19, two stable collections contain the same 268 nodes, the host-free
  compiler-artifact proof passes, exact-device correctness passes 248/248
  twice, TEST-4/TEST-6 focused gates pass 190/190, and the isolated TEST-5
  lane passes 20/20. Explicit Tile tool paths now take precedence over stale
  build-tree binaries. ROCm receives only the LLVM 23 lit-shell compatibility
  update; Apple has no affected physical schedule or runtime contract.
- **NVIDIA-TEST-7 is closed as local WSL release ownership; GitHub runners are
  intentionally not used.** The release command exposes independent `cpu`,
  `compiler`, `device`, and `performance` layers, rejects overlapping runs with
  a host lock, writes a fail-closed status record, retains timestamped machine,
  JUnit, and baseline bundles, and keeps performance serial. The finalized
  all-layer invocation passed 410 host-free/shared-registry tests (one explicit
  skip), 20/20 lit, 1/1 compiler artifact, 268/268 correctness twice, and 20/20
  performance. Its retained bundle is
  `artifacts/nvidia-release/20260717T003224Z-18866bbb/all/`.
- The second batch removed the same local MMA-runtime probe from norm, softmax,
  matmul-ReLU, matmul-softmax, compiled KV-cache, forward/backward Flash
  Attention, and convolution tests. Their 89 focused exact-device tests passed
  on the RTX 5070 Ti. Specialized compiler and Tile availability probes remain
  local until their stronger capability contracts can be preserved explicitly.
- The third batch migrated control flow, DeltaNet, dequant GEMM, FP quant,
  local collectives, optimizers, positional encoding, and SSM to the shared
  MMA-runtime probe. It also classified their live CUDA tests with
  `hardware_nvidia` while leaving host-only negative tests unmarked. The 32
  focused tests passed; collection increased from 243 to 264 exact-device
  nodes (246 correctness, 18 performance).
- The next helper-deduplication batch replaced private ordinary MMA-runtime
  probes in linear attention, MLA decode, and sparse attention with the shared
  capability-specific helper. The E3 hand-tuned GEMM proof now uses the shared
  MMA-plus-PTX-launch predicate; Tile tool/runtime checks remain local because
  they prove a stronger compiler-path capability.
- The second physical relocation split the mixed NVIDIA MMA launch file into
  two host-free execution-matrix contracts and five exact-device launch/JIT
  proofs under `tests/device/nvidia/`. The mapped cohort passed 21 focused
  tests, 19/19 compiler lit plus its compiler pytest contract, exact-device
  correctness twice (246 passed, one Apple-only skip each), and serial
  performance (18 passed, one Apple-only skip).
- The third physical relocation moved the two device-only DSA sparse-attention
  proofs to `tests/device/nvidia/test_sparse_attention.py`. Its node map,
  focused execute/compare run, compiler artifact lane, two exact-device runs
  (246 passed, one Apple-only skip each), and serial performance lane (18
  passed, one Apple-only skip) all passed without changing the 264-node
  NVIDIA marker topology.
- NVIDIA compiler-artifact selection no longer relies on the
  `test_nvidia_*.py` filename pattern: the `compiler_nvidia` marker owns the
  CUDA artifact lane and its release-gate selection. `NvidiaDeviceSession`
  now frees all tracked buffers and destroys its stream even after a
  synchronization failure, and destroys a successfully-created timing event
  if its partner event cannot be created. Host-free fault-injection tests pass
  (2/2); the marker artifact lane passed and the real stream/event ABI fixture
  passed 15/15 on the RTX 5070 Ti.
- **NVIDIA-TEST-6 is complete (2026-07-16).** The closure audit found no
  remaining ordinary private MMA/PTX probe
  implementation: plugin and hot-path-ratchet compatibility names now delegate
  to shared predicates, while the Tile probe remains intentionally specialized.
  Running the hot-path ratchet immediately after the broad plugin matrix
  exceeded two f16 caps (512³ and 1024³); two isolated serial reruns both
  passed. The disposition is test-state contamination outside the canonical
  isolated performance lane, not a tolerance change or performance regression.
  An AST ratchet now rejects any future exact-device test under `tests/unit`.
  The final topology collects 333 NVIDIA nodes; compiler artifacts pass 20/20,
  exact-device correctness passes 313/313 twice, and the serial measured lane
  passes 20/20. New backward-attention and epilogue families landed directly
  in `tests/device/nvidia`, and the backward nodes are recorded in the
  executable post-migration map.
- The control-flow cohort is now accepted: source/rejection contracts remain
  host-free under `tests/unit`, while the bounded-control and runtime-binding
  execute/compare proofs moved to `tests/device/nvidia/test_control_flow.py`.
  Its mapped nodes passed focused validation, 19/19 compiler lit plus the
  NVIDIA compiler marker lane, exact-device correctness twice (246 passed,
  one Apple-only skip each), and serial performance (18 passed, one skip).
- The first device run exposed a product defect in Tile GELU: NVPTX could not
  select LLVM's `ftanh`; after its arithmetic lowering, SiLU exposed the same
  issue for `fexp`. Both now lower through a bounded Pade tanh expression, so
  no unsupported transcendental libcall is emitted. Focused Tile and related
  CUDA correctness tests passed **38/38** after the fix.

## Completion definition

This plan reaches `closed` only when all of the following are true on the
NVIDIA box:

1. Host-free CPU PR tests, NVIDIA compiler-artifact tests, CUDA device
   correctness tests, and CUDA performance tests run as separate commands with
   separate reports.
2. Every exact-device test proves `native_gpu` provenance and compares against
   the same numerical oracle used by the CPU/ROCm paths; no fallback earns a
   pass.
3. The full non-performance CUDA device matrix passes twice from a clean build.
4. Performance tests run serially after warmup and commit repeated-median
   kernel-only and end-to-end evidence. Timing under xdist is forbidden.
5. Tool, dtype, diagnostic, op, target, execution-state, and generated-doc
   registries remain green.
6. Duplicate/source-scan tests are removed only after an equal or stronger
   semantic, FileCheck, object/SASS, or execute/compare proof replaces them.
7. The NVIDIA release gate owns this lane and preserves logs plus machine
   identity for each proof run.

## NVIDIA-box preflight

Record this before interpreting any failure:

```bash
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total \
  --format=csv,noheader
nvcc --version
ptxas --version
python3 --version
git rev-parse HEAD
```

Required target is RTX 5070 Ti / compute capability 12.0. NVFP4/block-scale
tests compile the architecture-specific `sm_120a` target. Record driver, CUDA
toolkit, LLVM/MLIR, Python, GPU UUID, clocks/power mode, and whether another
process is using the device.

### Install LLVM/MLIR 23 on Ubuntu

Use the repository bootstrap on Ubuntu 24.04; it installs one matched LLVM,
Clang, LLD, MLIR, and Polly 23 toolchain from apt.llvm.org:

```bash
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

For a toolchain-only manual installation, use a dedicated versioned source
file rather than replacing the distribution LLVM packages:

```bash
sudo install -d -m 0755 /etc/apt/keyrings
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
  | sudo gpg --dearmor --yes -o /etc/apt/keyrings/apt.llvm.org.gpg

. /etc/os-release
LLVM_SUITE="llvm-toolchain-${VERSION_CODENAME}-23"
if ! wget -q --spider \
  "https://apt.llvm.org/${VERSION_CODENAME}/dists/${LLVM_SUITE}/Release"; then
  LLVM_SUITE="llvm-toolchain-${VERSION_CODENAME}"
fi
echo "deb [signed-by=/etc/apt/keyrings/apt.llvm.org.gpg] https://apt.llvm.org/${VERSION_CODENAME}/ ${LLVM_SUITE} main" \
  | sudo tee /etc/apt/sources.list.d/llvm-23.list >/dev/null
sudo apt-get update
sudo apt-get install -y \
  clang-23 lld-23 llvm-23 llvm-23-dev llvm-23-tools \
  mlir-23-tools libmlir-23-dev libpolly-23-dev

export LLVM_ROOT=/usr/lib/llvm-23
export PATH="$LLVM_ROOT/bin:$PATH"
export CMAKE_PREFIX_PATH="$LLVM_ROOT${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

llvm-config --version
mlir-opt --version
mlir-tblgen --version
FileCheck --version
```

All four commands must report major version 23. Remove or disable any stale
pre-23 toolchain source selection from the build environment; keeping
multiple apt repositories installed is acceptable, but Tessera's CMake cache,
compiler executables, MLIR tools, and CMake package directories must all resolve
to `/usr/lib/llvm-23`.

Build the compiler and CUDA runtime from a clean NVIDIA build directory:

```bash
cmake -S . -B build-nvidia-cuda -G Ninja \
  -DCMAKE_C_COMPILER=/usr/lib/llvm-23/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/lib/llvm-23/bin/clang++ \
  -DLLVM_DIR=/usr/lib/llvm-23/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-23/lib/cmake/mlir \
  -DTESSERA_BUILD_NVIDIA_BACKEND=ON \
  -DTESSERA_ENABLE_CUDA=ON \
  -DTESSERA_CUDA_ARCH=sm_120a \
  -DTESSERA_BUILD_EXAMPLES=OFF
ninja -C build-nvidia-cuda tessera-opt tessera-nvidia-opt \
  tessera_nvidia_gemm tessera_runtime
```

Export explicit tool paths rather than relying on a previous build:

```bash
export TESSERA_OPT="$PWD/build-nvidia-cuda/tools/tessera-opt/tessera-opt"
export MLIR_OPT=/usr/lib/llvm-23/bin/mlir-opt
export PYTHONPATH="$PWD/python:$PWD"
```

Adjust `TESSERA_OPT` to the actual Ninja output reported by the build if the
generator places it under `build-nvidia-cuda/tools/tessera-opt/` differently.

The 2026-07-16 shared compiler migration raises the project floor to matched
LLVM/MLIR 23 and updates portable Tile/NVIDIA TableGen plus greedy-rewrite
compatibility. The shared sources compile in the LLVM/MLIR 23 ROCm build, and
NVIDIA exact-device parity is now validated independently on the `sm_120`
host. No CUDA execution status was inferred or promoted from the ROCm run.

## Ordered work

| Order | ID | Work | Engineering action | Completion gate |
|---:|---|---|---|---|
| 1 | NVIDIA-CALIB-1 | Validate T1 reuse/cache pruning against the committed sm_120 corpus | Supply evidence-backed sm_120 bandwidth/cache inputs, then compute per-family and per-shape rank correlations without reviving the rejected step-distance score. | The analysis records model version, corpus identity, rank correlations, and a retain/reject verdict; no new device run is required. |
| 2 | NVIDIA-RASTER-1 | Consume and measure the shared raster contract | Wire the emitter with row-major identity preserved, then sweep only on sm_120 with device timing and `ncu` L2 evidence. | Exact-device correctness remains unchanged and a measured raster-order/group decision is recorded without hardware-free arbitrary selection. |
| 3 | NVIDIA-AOT-1 | Decide whether the NVRTC lane needs a precompiled peer | Reuse the Apple AOT/JIT harness and its never-before-compiled-kernel cache control; first decide whether the expected cold-start use case justifies implementation. | A documented not-applicable/retain decision or a measured precompiled candidate with equivalent numerics and explicit offline-build amortization. |
| 4 | NVIDIA-TEST-1 | Establish a reproducible baseline | Run collection and each proof layer separately; save JUnit, skip reasons, duration report, machine identity, and the exact commit. Classify every failure as product defect, test defect, environment defect, or stale claim. | Two collections return the same node set; no unknown markers; every skip has an explicit unavailable capability. |
| 5 | NVIDIA-TEST-2 | Compiler-artifact layer | Run `check-tessera-nvidia` plus CUDA pytest files carrying `compiler_tool`; migrate private tool probes to `compiler_toolchain`; split artifact assertions from the eight tests that currently continue into device execution; replace large textual snapshots with named diagnostics, FileCheck, or focused IR/object invariants. | Clean build passes without a GPU; missing-tool simulation skip-cleans; no compiler test invokes a nonexistent path. |
| 6 | NVIDIA-TEST-3 | Exact-device correctness | Run `hardware_nvidia and not performance`; group failures by GEMM/Tile, attention, reductions/norms, control flow, KV/ReplaySSM, collectives, and ABI/conformance. Require native provenance and execute/compare. | Entire correctness matrix passes twice; fallback-injection negatives fail to earn native proof. |
| 7 | NVIDIA-TEST-4 | Numerical policy | Centralize dtype/op tolerances from accumulation/storage behavior. Add ragged, rectangular, boundary, non-finite, misalignment, and invalid-contract cases where absent. | f16/bf16/tf32/FP8/int8/NVFP4 cases use documented tolerances; no default zero-`atol` checks near zero. |
| 8 | NVIDIA-TEST-5 | Measured performance | Run `hardware_nvidia and performance` serially. Warm up compilation and caches; use repeated medians; measure kernel-only and end-to-end separately; record registers, shared memory, occupancy, spills, and selected route. | Stable baselines cover square/rectangular/ragged GEMM, fused epilogues, attention, paged KV, ReplaySSM, reductions, and transport. Each ratchet identifies the selected implementation. |
| 9 | NVIDIA-TEST-6 | Refactor and deduplicate | Move mature families toward `tests/compiler/`, `tests/device/nvidia/`, `tests/integration/`, and `tests/performance/nvidia/`. Consolidate repeated CUDA availability, compilation, launch, oracle, and cleanup code. | No central filename allowlist; no duplicated private CUDA probe/loader; process trees and device allocations clean up on failure. |
| 10 | NVIDIA-TEST-7 | Local release ownership | Own the NVIDIA-box release gate locally in WSL with a host concurrency lock and retained artifacts; GitHub runners are intentionally not used. Keep two-run device correctness required for NVIDIA promotion and performance serial. | A clean branch run reports NVIDIA host-free/shared registries, compiler artifact, device correctness, and performance independently and retains the fail-closed evidence bundle. |
| 11 | NVIDIA-LSE-1 | Consume and measure the real shared LSE checkpoint on CUDA | **Landing, SM120 P0 complete.** Compiler-owned f32 paired physical ABIs now carry `Q/K/V -> O,row_lse` and `dO/Q/K/V,row_lse -> dQ/dK/dV` through Tile operands, descriptors, bridge allocation/copies/argument order, runtime validation, and the native-package route. Exact RTX 5070 Ti proof covers oracle equality, saved-vs-recompute forward/backward equality, and malformed rank rejection. | At `[1,2,1,3,4,4,3]`, saved lowers backward event time (0.01716 vs 0.02777 ms) but the paired save/load e2e median loses (1.56149 vs 1.26335 ms); NCU/resource packet is retained in `benchmarks/baselines/nvidia_sm120_lse_checkpoint_2026_07_30.json`. Retain recompute default; repeat representative sequence/shape sweeps before promotion. |
| 12 | NVIDIA-E2E-1 | Canonical SM120 compiler spine | Under sync `E2E-SPINE-2026-07-18`, compose Graph/Schedule/Tile lowering with `LowerTileToNVIDIA(sm=120)`, NVVM/PTX/native-image packaging, and the existing register/invoke launch bridge. Prove f16 and NVFP4 first, including non-origin scale tiles and general-shape dispatch. | One canonical driver request returns a typed image artifact plus launch descriptor, registers and launches on `sm_120`, compares numerically, and retains compiler/ABI/device/resource evidence without a selector change. |
| 13 | NVIDIA-E2E-2 | Per-SM and operation breadth | Replace shared-alias/hardcoded target behavior with architecture-specific pipelines, then move supported CUDA families through the same typed image/launch seam. | Every enabled SM/family has the four-layer proof on its exact device or an explicit unsupported/planned terminal state; `sm_90` and `sm_100` are never inferred from `sm_120`. |

### High-risk NVIDIA-TEST-6 migration

**NVIDIA-TEST-6-HIGH — Relocate mature CUDA families without breaking the
proof contract.** Move mature compiler, device, integration, and performance
families toward `tests/compiler/`, `tests/device/nvidia/`,
`tests/integration/`, and `tests/performance/nvidia/`. This is high risk because
pytest node IDs, import roots, marker collection, CI selection, and retained
JUnit history can all change even if individual assertions still pass.

Before accepting the migration, record an old-to-new node map, preserve every
`hardware_nvidia`/`performance`/`compiler_tool` classification, prove that the
old paths have no duplicate collection, and run the host-free, artifact,
exact-device, and serial-performance layers. Do not combine this migration with
backend behavior, tolerance, or selector changes.

**Pilot evidence (2026-07-15, `landing`).** MoE transport is the first
relocated family. Its two native CUDA execute/compare nodes now live in
`tests/device/nvidia/test_moe_transport.py`; its host-free invalid-partition
contract remains in `tests/unit/test_nvidia_moe_transport_contract.py`. The
checked-in old-to-new map is `tests/device/nvidia/node_migrations.json`, and
`tests/unit/test_nvidia_test_location_migration.py` prevents restoration of
the old file or duplicate destinations. The second cohort applies the same
contract to the former mixed `test_nvidia_launch_execute.py`: two host-free
execution-matrix nodes remain under `tests/unit/`, and five native launch/JIT
nodes move to `tests/device/nvidia/test_launch_execute.py`. The combined roots
collect exactly **264** `hardware_nvidia` nodes (246 correctness, 18
performance). The two device-only DSA sparse-attention nodes are also mapped
to `tests/device/nvidia/test_sparse_attention.py`. Every relocated node
preserves its `hardware_nvidia` classification; none gained `performance` or
`compiler_tool` classification.

The compiler-artifact proof passed (19/19 lit and 1 compiler-tool pytest
contract), exact-device correctness passed twice (246 passed, 1 unrelated
Apple-only skip, zero failures/errors on each run), and the serial performance
lane passed (18 passed, 1 unrelated Apple-only skip). The second cohort
repeated those artifact, two-run correctness, and serial-performance proofs;
its executable node-map and retained host-free contracts passed (4/4). The
complete host-free PR command is **not an
NVIDIA-host acceptance gate** when it exercises Apple/ROCm compiler passes:
this WSL checkout's generic `build/` is intentionally NVIDIA-only, so 274
foreign-backend compiler tests cannot run here. This is not a relocation
failure or an NVIDIA-TEST-6-HIGH blocker. `APPLE-CI-2` and `ROCM-TEST-1` own
validation of their respective host-free compiler configurations on the correct
backend hosts; the NVIDIA host retains the focused host-free migration guard
plus its artifact and exact-device proof layers.

**Completion evidence (2026-07-16, `complete`).** The executable map now covers
**286** relocated node IDs. Mature execute/compare families are collected from
`tests/device/nvidia/`; paged-KV, ReplaySSM, and the MMA bridge are in
`tests/integration/`; and hot-path, Conv2D, MMA-symbol, and plugin timing
proofs are in `tests/performance/nvidia/`. The mixed plugin implementation is
shared through a non-discovered support module, while its 20 host-free
contracts, 53 native nodes, and 8 measured nodes are collected only from their
respective unit/device/performance entry points. The only remaining
`hardware_nvidia` references under `tests/unit/` are release-gate and
marker-policy structural assertions.

The final migrated plugin cohort passed its focused mapping/architecture guard
(93 tests), compiler artifacts (19/19 lit; one compiler pytest pass and one
hardware-excluded skip), exact-device correctness twice (246 passed and one
Apple-only skip each), and serial performance (18 passed and one skip).
Relocating Conv2D exposed an order-dependent product defect: automatic f32
dispatch admitted the explicit `im2col_tf32` candidate under a looser internal
tolerance. Automatic dispatch now selects only f32-accurate direct/shared
routes; explicitly requested TF32 performance coverage remains intact. The
post-fix exact-device matrix passed twice (246 passed and one skip each), and
the serial performance lane passed (18 passed and one skip). This is a product
correctness fix with retained before/after numerical evidence, not a tolerance
relaxation.

The final static audit finds no `hardware_nvidia` test function under
`tests/unit`; structural marker/release assertions remain host-free. The
expanded map contains 292 relocations plus 23 post-migration nodes. The final
four-layer proof is 20/20 compiler lit, 313/313 exact-device correctness twice,
and 20/20 serial performance on the RTX 5070 Ti. This closes the migration;
future native tests must land directly in device, integration, or performance
roots and satisfy the same AST/node-map ratchets.

## Canonical commands on the NVIDIA box

```bash
# 0. State/collection contract (currently 334 nodes)
python3 -m pytest tests/unit tests/device/nvidia tests/performance/nvidia tests/integration \
  -m hardware_nvidia --collect-only -q --no-header

# 1. Host-free PR contract, including CUDA emit/validation/rejection tests
python3 scripts/run_unit_tests.py --timeout=180 -q

# 2. Compiler artifacts without claiming device execution
ninja -C build-nvidia-cuda check-tessera-nvidia
python3 -m pytest tests/unit tests/device/nvidia tests/integration \
  -m "compiler_nvidia and not hardware_nvidia" -q --durations=50 \
  --junitxml=/tmp/nvidia-compiler-tool.xml

# 3. Exact-device correctness; run twice from the same clean build
python3 -m pytest tests/unit tests/device/nvidia tests/integration \
  -m "hardware_nvidia and not performance" -q --durations=100 \
  --junitxml=/tmp/nvidia-device-correctness.xml

# 4. Measured lane: serial only
python3 -m pytest tests/unit tests/device/nvidia tests/performance/nvidia tests/integration \
  -m "hardware_nvidia and performance" -q -n 0 --durations=0 \
  --junitxml=/tmp/nvidia-performance.xml
```

Do not use `-x` for the first baseline: the complete failure topology is needed
to design the migration. After triage, use focused files for the edit loop and
rerun the complete layer before marking an item complete.

## Failure triage contract

For every failure, record:

- node id, proof layer, target/dtype/shape, seed, selected route, and native
  provenance;
- whether it reproduces alone, serially, and on the second clean run;
- compiler stdout/stderr and named diagnostic code;
- numerical maximum absolute/relative error and first failing index;
- kernel-only versus end-to-end latency for performance cases;
- register/shared-memory/occupancy/spill evidence when a kernel changes;
- disposition: fix product, fix test state, replace weak test, merge duplicate,
  or document an exact environment blocker.

Never relax a tolerance or timing cap solely to make the lane green. Recompute
it from dtype semantics or a stable repeated-median baseline, and retain the
before/after evidence.

## Initial family matrix

| Family | Representative coverage | Required follow-up on NVIDIA box |
|---|---|---|
| Tile/GEMM | compiler-generated SM120 fragments, shipped MMA symbols, ragged/grid GEMM, f16/bf16/tf32/FP8/int8, NVFP4 OMMA | Verify exact SASS/instruction family, lane maps, ragged stores, allocation cleanup, and kernel/device timing separation. |
| Fusion | bias, ReLU, GELU, SiLU, gated SwiGLU, matmul-softmax | Cross-check epilogue order and dtype accumulation against CUDA and ROCm shared oracles. |
| Attention | MHA/GQA/MQA, backward, sparse/DSA, window/bias/softcap | Separate compiler artifact from live execution; cover global decode positions and non-finite policy. |
| Reductions/norms | sum/mean/min/max, softmax, RMSNorm/LayerNorm | Validate non-power-of-two/ragged widths, NaN policy, large-offset variance, and dtype tolerances. |
| Stateful serving | paged KV and ReplaySSM async ring | Long decode, flush, rollback, rejection/backpressure, remapped pages, native provenance, and leak-free teardown. |
| Control/collectives | bounded for/if/while/scan and single-device collectives | Validate one-launch ABI, bad-shape rejection before launch, and explicit multi-rank deferral. |
| Performance | GEMM routes, convolution routes, device timing, hot-path ratchet | Run isolated and serial; record winner, resource evidence, kernel-only, and end-to-end rows. |

## ROCm-derived CUDA parity work

The completed ROCm work raised the proof standard for several features that
already exist on CUDA. These are CUDA audits and measured retunes, not literal
ports of AMD schedules. Share logical fixtures, ABI contracts, numerical
oracles, benchmark schemas, and decision rules across backends; keep physical
fragments and schedules architecture-owned.

In particular, an RDNA wave is not a CUDA warp, LDS is not evidence about
shared-memory behavior, VGPR pressure does not predict the CUDA register file,
and WMMA/MFMA winners do not select `mma.sync` or OMMA winners. Every production
selector change below requires fresh `sm_120a` measurements on the NVIDIA box.

| Order | ID | ROCm lesson and CUDA work | Current CUDA state | Completion gate |
|---:|---|---|---|---|
| 1 | NVIDIA-PARITY-TILE | Re-run the same logical portable-Tile fixture through the NVIDIA architecture-owned fragment selector. Cover direct/shared schedules, grid and ragged edges, supported f16/bf16/tf32/FP8/int8/NVFP4 forms, and bias/ReLU/GELU/SiLU epilogues. Add a CUDA fragment resource record containing registers, shared memory, occupancy, spills, and the selected SASS instruction family. | Compiler-generated SM120 fragments, layout oracles, and direct/shared execution tests exist. | Fixtures never author physical fragments; pack/execute/unpack/store matches the shared oracle; emitted instructions and resource rows match the selected `sm_120a` contract. |
| 2 | NVIDIA-PARITY-GEMM-RATCHET | Extend the hot-path recorder into a repeated-median schedule matrix covering square, rectangular, ragged, dtype, and fused-epilogue cases. Record kernel/device-event and end-to-end time separately, then capture registers, shared memory, occupancy, and spills before changing the production tile selector. | `record_hot_path_baseline.py` provides a useful but narrow latency ratchet. | A committed device-keyed baseline identifies every candidate and winner; two stable runs agree within the declared noise policy; no selector change lands without before/after resource evidence. |
| 3 | NVIDIA-PARITY-LEGACY-RETUNE | Re-evaluate older f32/tf32 GEMM, grouped GEMM, grouped SwiGLU, KV movement, and MoE transport now that the compiler and fragment selection are stronger. Compare compiled, shipped, and staged/direct candidates without conflating launch/transfer cost with kernel time. | Individual CUDA paths and hot-path rows exist, but there is no ROCm-equivalent wide retune corpus. | All candidates match one oracle; kernel-only and end-to-end winners are recorded independently; grouped and transport rows include launch collapse and achieved-bandwidth evidence. |
| 4 | NVIDIA-PARITY-ATTN-FWD | Apply the G6-B methodology to CUDA forward attention: evaluate occupancy-aware multi-warp CTA schedules with online softmax at D=128, plus ragged, causal/window, bias, softcap, and MHA/GQA/MQA cases. Do not assume ROCm's two-wave shape is the CUDA winner. | Compiled CUDA forward-attention paths and exact-device tests exist. | Candidate schedules match the shared oracle; traffic and resource evidence explain the winner; the selected route wins repeated-median kernel timing without regressing end-to-end timing. |
| 5 | NVIDIA-PARITY-ATTN-BWD | Apply the G6-C methodology to dK/dV backward. Measure the existing path against atomic and split-workspace/reduction candidates, including deterministic behavior and workspace limits. | Compiled CUDA backward attention is covered, but has not been re-ratcheted against the split/reduced design space. | Forward-derived gradients pass the shared tolerance matrix; determinism and workspace caps are explicit; resource, kernel-only, and end-to-end rows select the production route. |
| 6 | NVIDIA-PARITY-PAGED-KV | Re-prove the stable paged-KV ABI with non-identity/permuted pages, remaps, causal offsets, and boundary lengths. Compare direct resident page-table attention with staged/gather-to-FA using the same oracle and retain both timing domains. | Direct fused and staged paged-attention candidates plus an SM120 serving baseline already exist. | Every candidate consumes the same ABI and matches the same permuted-page oracle; device-event and end-to-end rows may choose different winners and the cache keys preserve that distinction. |
| 7 | NVIDIA-PARITY-REPLAY | Re-run CUDA ReplaySSM against the closure matrix exposed by ROCm: long decode, flush, rollback, speculative rejection, block submit, ordered async ring, backpressure, and teardown. Expand B/D/N/M shapes and record state traffic as well as latency. | CUDA is the reference persistent ReplaySSM implementation and has serving rows, but needs the wider proof and benchmark matrix. | All transitions match `SSMStateHandle`; rejected work cannot mutate committed state; ring ordering and cleanup survive stress; traffic plus kernel/end-to-end latency are committed. |
| 8 | NVIDIA-PARITY-EPILOGUE | Make the common Tile epilogue contract explicit for bias, ReLU, GELU, and SiLU. Check accumulator precision, operation order, optional bias/residual guards, ragged stores, and all supported storage dtypes against shared CUDA/ROCm fixtures. | CUDA emits fused epilogues and plugin tests cover representative forms. | One backend-neutral oracle drives both backends; every supported fusion executes natively; unsupported dtype/op pairs reject with registered diagnostic codes rather than silently de-fusing. |
| 9 | NVIDIA-PARITY-AUTOTUNE | Align CUDA and ROCm corpus schemas around device-keyed candidates, timing domain, compiler/resource fingerprint, cold/warm compile state, and cache behavior. Promote a winner only after the relevant correctness and schedule ratchets pass. | CUDA has autotune and serving corpus writers, but their evidence must be reconciled with the newer ROCm records. | Corpus validation rejects stale devices, compilers, resources, and timing domains; cold/warm behavior is reproducible; selector decisions cite a retained measurement row. |
| 10 | NVIDIA-PARITY-TRANSPORT | Close KV-movement and MoE-transport parity with direct/staged routes, ragged/grouped loads, bandwidth attainment, and launch-amortization measurements. Feed any winner into the legacy retune only after ABI and correctness closure. | CUDA transport operations exist but lack one consolidated exact-device performance proof. | Byte counts and achieved bandwidth are auditable; kernel-only and end-to-end winners are separate; awkward sizes and grouped routes match their reference without leaks or hidden host staging. |

### CUDA parity execution record

The parity queue uses ROCm's logical coverage and proof methodology, not its
physical schedules. CUDA owns warp/register packing, `HMMA`/`QMMA`/`IMMA`/OMMA
selection, shared-memory staging, barriers, occupancy limits, and every selector
winner. An AMD wave shape, LDS strategy, or VGPR result is never a CUDA default.

- **NVIDIA-PARITY-TILE — complete on sm_120a.** The architecture-owned SM120 fragment
  selector now describes f16 (f16/f32 accumulation), bf16, TF32, FP8 E4M3/E5M2,
  int8, and block-scaled NVFP4 separately. C++ lowering consumes the descriptor
  for physical input packing and per-lane register-count validation. The exact
  compiler path passes the shared numerical oracle for f16/bf16/TF32/FP8/int8,
  direct/shared grids, ragged edges, and bias/ReLU/GELU/SiLU. The reproducible
  13-row `nvidia_sm120_tile_fragment_resources.json` record retains cubin hashes,
  registers, shared memory, theoretical occupancy, spills, and observed SASS:
  `HMMA` for f16/bf16/TF32, `QMMA` for FP8, `IMMA` for int8, and block-scaled
  `OMMA` for NVFP4. Portable typed Tile now carries NVFP4's two logical UE4M3
  scale tiles. C++ consumes nibble-packed logical A/B storage, materializes the
  backend-owned scale selectors, emits real block-scaled inline PTX, assembles
  for `sm_120a`, and passes the non-uniform-scale numerical oracle without
  fixture-authored physical fragments. The resource row now comes from that
  typed compiler artifact rather than the original CUDA spike.
- **NVIDIA-PARITY-GEMM-RATCHET — measured complete; no promotion.** The
  device-keyed `nvidia_sm120_gemm_schedule_matrix.json` contains 34 exact-case
  rows spanning square, rectangular, ragged, f16/bf16, and bias plus
  none/ReLU/GELU/SiLU epilogues. Every row has two stable repeated-median runs,
  separate CUDA-event and rotated-interleaved end-to-end timing, complete
  per-candidate resource fingerprints, and an explicit 3% noise policy. The
  smallest ragged rows require 50 untimed device warmups to remove clock-ramp
  drift. The record intentionally leaves the production selector unchanged.
  CUDA 13.3's renamed local/shared spill-request metrics are normalized alongside
  the legacy Nsight metrics, and the synthesized fused fallback now has retained
  production-sized Nsight evidence.
- **NVIDIA-PARITY-LEGACY-RETUNE — stable complete; no selector promotion.** The
  device-keyed `nvidia_sm120_legacy_retune.json` now compares compiled exact-f32
  and shipped TF32 GEMM on square/ragged rows, one grouped GEMM launch against
  the retained per-expert decomposition, and a new grouped SwiGLU route whose
  four launches are independent of expert count against the legacy `4E` route.
  All candidates use one f32 oracle and retain separate event/end-to-end rows,
  byte and achieved-bandwidth accounting, launch counts, and linked resources.
  SwiGLU rows retain both the grouped-GEMM cubin fingerprint and the exact
  generated SiLU-gate registers, occupancy, and spill record.
  The final corpus uses production-scale 512-square and 509x773x257 ragged
  GEMM, 1024x384x256x5 grouped GEMM, and 512x256x384x8 grouped SwiGLU rows.
  Exact-route resident warmup occurs after allocation in the timed session,
  and two disjoint interleaved cohorts retain every device and end-to-end batch.
  All eight rows pass 3% (maximum device/end-to-end deltas 2.47%/0.77%). TF32
  and the launch-collapsed grouped routes win both retained runs, but this
  evidence intentionally leaves selectors unchanged.
- **NVIDIA-PARITY-ATTN-FWD — stable complete; CUDA 4-warp candidate leads kernel time.**
  CUDA-owned 4- and 8-warp CTA candidates now cover D=128 MHA, causal sequence
  1009, ragged GQA windowing, and MQA bias+softcap. Each warp owns one query,
  uses warp shuffles for QK, and keeps distributed online-softmax/PV state;
  this is not ROCm's two-wave LDS schedule. All rows match the shared oracle
  within `7e-8`. Both candidates use 56 registers with zero spills; modeled
  occupancy is 75% for four warps and 66.67% for eight. Four warps win CUDA-event
  timing on both retained runs for every case. Ten disjoint, sample-interleaved
  end-to-end batches remove run-order aliasing without sharing observations;
  all eight rows now pass 3% with maximum device/end-to-end deltas of
  0.22%/1.84%. Small end-to-end rows do not have unanimous winner consensus,
  so production selection remains unchanged.
- **NVIDIA-PARITY-ATTN-BWD — measured complete; atomic retained.** The atomic
  incumbent and deterministic two-part split/workspace/fixed-order-reduction
  candidate share one forward-derived oracle across D64 MHA, causal D128 MHA,
  and ragged windowed GQA. The split route is bitwise repeatable, rejects
  unsupported f16 storage, and enforces an exact one-extra-dK+dV f32 workspace
  cap (524,288 bytes on MHA rows; 134,144 on ragged GQA). All six candidate
  rows pass 3% with maximum device/end-to-end deltas of 0.36%/1.67%. Resources
  retain atomic 48-register/83.33%-occupancy and split dQ 48-register,
  dK/dV 56-register/75%-occupancy, and reduction 12-register/100%-occupancy
  fingerprints plus spill evidence. Atomic wins both timing domains in every
  case by a large margin, so `selector_changed` remains false.
- **NVIDIA-PARITY-PAGED-KV — correctness and timing complete; no promotion.** Both fused
  and staged routes now pass the same permuted-page oracle at lengths 1, 3, 4,
  5, 7, 8, 9, and 13, including non-monotonic logical indices and global causal
  offsets. The 13-row transport corpus covers 127/128/129/511-token boundaries
  with separate device/end-to-end keys, byte formulas, resources, and no
  selector change. Repeated event batches now remain inside one warmed resident
  session; all eight fused/staged rows pass the 3% two-run policy, with maximum
  device and end-to-end deltas of 1.89% and 1.85% respectively.
- **NVIDIA-PARITY-REPLAY — canonical state contract and correctness complete;
  timing characterization retained.** Exact-device
  tests cover long decode across flushes, rollback, speculative rejection,
  block submit, reset, ordered ring backpressure, rejected-submit immutability,
  and teardown over wider B/D/N shapes. The 10-row replay corpus spans five
  geometries and 16/64 tokens with traffic, resources, and both timing domains.
  Each runtime handle now carries the shared `tessera.replayssm.state.v1`
  descriptor: exact persistent device and pinned-host byte formulas, session
  lifetime with preserved initialization, ordered stream/event slot ownership,
  consumer-wait-before-release, and teardown draining. Span checks reject before
  CUDA submission. The CPU oracle is outside the end-to-end interval; each
  retained run has disjoint four-route batch medians with recorded out-of-band
  clock conditioning. All errors remain below `1.5e-8`. Under the WSL 4%
  foundation policy 5/10 refreshed rows satisfy both domains; the remaining
  small/multi-batch rows range from 4.05% to 8.75%. No selector decision consumes
  these unstable rows.
- **NVIDIA-PARITY-EPILOGUE — execution matrix complete.** `FusedRegion` is the
  backend-neutral bias/activation/residual/order oracle and now emits registered
  `E_FUSED_EPILOGUE_*` diagnostics for unsupported dtype/op/order and missing
  operands. The exact-device matrix now executes all 43 supported combinations
  over f32/f16/bf16/FP8 E4M3/FP8 E5M2, optional bias, no activation or
  ReLU/GELU/SiLU, and f32 residual-after-activation ordering. Accumulation is
  f32; low-precision residual, activation-before-bias, repeated activation,
  and unsupported dtype/op pairs reject with the registered diagnostics.
- **NVIDIA-PARITY-AUTOTUNE — strict admission complete; no promotion.** Corpus
  admission can require exact device, timing domain, compiler fingerprint,
  resource fingerprints, compile state, and cache state. The committed
  reproducibility record admits all 20 selector-eligible NVIDIA rows, rejects
  stale device/timing/compiler/resource mutations, and reproduces one kernel
  cache key across two cold builds and warm hits (about 0.05 ms warm lookup).
- **NVIDIA-PARITY-TRANSPORT — correctness, evidence, and timing complete.**
  The consolidated 13-row paged-KV/MoE/grouped corpus retains auditable traffic
  formulas, achieved bandwidth, launch-amortization keys, exact resources, and
  independent timing domains. MoE dispatch/combine now consume one canonical
  `tessera.moe_transport.v1` int32/fp32 descriptor with stable expert grouping,
  capacity/drop semantics, and dispatch-before-compute-before-combine ordering;
  grouped GEMM consumes canonical ragged sizes/offsets and retains empty experts.
  Local-device scope is explicit; multi-rank collective execution remains a
  separate backend/runtime item. Maximum oracle error is below `3e-7`; all 13
  rows pass the WSL 4% foundation policy. MoE CUDA-event samples retain one
  native allocation set across repeated batches, and the tiny routes use 101
  medians per run. No selector or legacy-retune winner is promoted.

- **NVIDIA-SM120-LOWP-PRODUCTIZATION — complete (2026-07-18).** The shipped
  CUDA ABI adds general-shape block-scaled NVFP4: packed E2M1 A/B, raw UE4M3
  scale views, M16/N8 grid dispatch, K64 accumulation, ragged zero fill, and
  pre-launch shape/view rejection. Fixed 16x8x64, multi-tile 33x19x129, and
  sub-tile 7x5x31 non-uniform-scale cases match the exact NVFP4 oracle on the
  RTX 5070 Ti. Native one-kernel TF32 and FP8 E4M3/E5M2 fused-epilogue,
  QK-softmax-PV attention, and gated routes now coexist with the composed
  candidates. Two fresh runs use 20 end-to-end medians and 100 CUDA-event
  repetitions per route. The cross-domain 3% gate promotes 11 of 18 retained
  shape/dtype rows; long attention and disagreement rows remain unpromoted.
  The linked 12-row cubin record reports 40-register fused/attention kernels,
  47–48-register gated kernels, 8/32 KiB attention dynamic shared memory,
  shape-dependent 22.92%/6.25% modeled attention occupancy, zero compiler spill
  storage, and the expected TF32 HMMA / FP8 QMMA SASS. Evidence:
  `nvidia_sm120_low_precision_native_{routes,resources}.json`.
- **Audit-document reconciliation — complete (2026-07-18).** This plan,
  `NVIDIA_AUDIT.md`, and `sm120-kernel-guide.md` now agree that mature SM120
  fragments lower for real, general NVFP4 dispatch is executable, and native
  TF32/FP8 transformer candidates exist. The plan remains `landing` only for
  unrelated architecture-specific follow-ons such as sm_90 WGMMA and sm_100
  tcgen05 exact-device proof; landed SM120 work is no longer described as open.

Cross-backend sync `NVFP4-TILE-SCALES-2026-07-16` changes the shared typed Tile
operand contract only. NVIDIA supplies exact-device materialization evidence;
Apple and ROCm do not inherit its physical schedule and record their outcomes
in their own plans.

Cross-backend sync `PR420-REVIEW-2026-07-17` corrects the NVIDIA-owned NVFP4
scale materializer to apply both `tile.view` origins using the declared
row-major A-scale and column-major B-scale layouts. A live `sm_120a` fixture
selects nonzero A-row and B-column scale tiles and matches the NumPy oracle;
the NVIDIA compiler lit suite passes 21/21. The SM120 Target IR selector also
accepts canonical `fp16` as the existing f16 fragment contract. This is a
correctness/dispatch repair only: no physical fragment, resource record,
timing row, or production selector changes. The same sync makes Ubuntu LLVM
repository setup install its probe prerequisites before first use; sibling
backend outcomes are recorded in their plans.

Cross-backend sync `NVIDIA-SM120-LOWP-2026-07-18` is NVIDIA-owned. It changes no
shared dtype spelling, portable Tile scale layout, backend-neutral epilogue
order, or generic autotune schema. Apple has no enabled NVFP4 cooperative-matrix
route and ROCm gfx1151 has no FP8/FP4 WMMA instruction; neither inherits CUDA
packing, HMMA/QMMA/OMMA schedules, resource values, timings, or selector rows.

Cross-backend sync `E2E-SPINE-2026-07-18`: NVIDIA owns **NVIDIA-E2E-1** and
**NVIDIA-E2E-2**. Shared code owns only the image/launch schemas and canonical
orchestration; NVIDIA retains PTX/SASS generation, physical fragments, launch
geometry, resources, and route selection. Existing NVRTC, shipped-library, and
PTX-register/invoke paths remain valid candidates while the typed spine lands.
Host-free IR/object evidence cannot promote an SM or selector, and exact-device
proof for `sm_90`, `sm_100`, and `sm_120` remains architecture-specific. The
completed E2E-SPINE-0 foundation records SM80 as lacking an exact registered
pipeline and SM100/SM120 as shared-builder aliases; it also corrects the Python
pass inventory to match that builder without changing CUDA runtime selection.
E2E-SPINE-1 adds the portable image/descriptor and rejection contract only;
PTX/cubin contents, warp schedules, launch geometry policy, resources, and CUDA
selectors remain NVIDIA-owned and unchanged until NVIDIA-E2E-1.
E2E-SPINE-2 completes the shared typed carriers, stage ledger, cache join, and
descriptor-first exact-target launcher registry. It registers no CUDA hook and
does not reinterpret `nvidia_mma` or any shipped/NVRTC candidate; NVIDIA-E2E-1
still owns PTX packaging, `sm_120` registration/submission, numerical proof,
resources, cleanup, and the first Level-C row.

NVIDIA-E2E-1 is **complete**. The f16 slice makes an explicit canonical
driver request own the typed `tile.matmul_kernel`, runs the production
`LowerTileToNVIDIA(sm=120)` and NVVM/LLVM/PTX pipeline, validates the image with
`ptxas`, and returns the shared native-image plus exact A/B/D/M/N/K descriptor.
The descriptor registers and launches through the shipped PTX bridge on the RTX
5070 Ti; aligned `16x8x16` and ragged `37x29x23` rows match the f32 NumPy oracle.
The image retains compiler/toolchain fingerprints, cold/warm state, and ptxas
register/shared-memory/spill fields. This slice changes no production selector.
The same driver now selects a CUDA-owned general-shape NVFP4 descriptor with
packed E2M1 A/B, logical UE4M3 `scale_a`/`scale_b`, f32 output, and M/N/K. The
typed lowering owns M16/N8 origins, K64 accumulation, ragged zero fill, scale
word materialization, and guarded stores before LLVM 23 emits `sm_120a` PTX.
Exact RTX 5070 Ti rows `16x8x64`, `33x19x129`, and `7x5x31` match the block-scale
oracle; the multi-tile row uses nonuniform row/column scales to prove non-origin
scale views. Missing/malformed scales, wrong scale storage, bias, and malformed
launch shapes reject before CUDA submission. Both f16 and NVFP4 retain stable
cold/warm image identity and ptxas register/shared-memory/spill evidence. The
shared Tile verifier change is limited to the explicit eight-operand NVFP4
launch ABI; it transfers no CUDA schedule, layout, resources, or selector.

NVIDIA-E2E-2 is **closed for the available SM120 host**, with the unavailable
multi-GPU, SM90, and SM100 boundaries assigned the deferred terminal states
below. Its first dependency slice replaces the former
shared SM90 alias with exact SM90/SM100/SM120 Graph→Tile builders and registered
Tile→`tessera_nvidia`→NVVM producers. The exact target now reaches Tile IR,
the control-flow guard, async-copy lowering, and the target producer without
being rewritten to SM90. Hopper alone consumes the proven WGMMA and Hopper
FlashAttention markers; SM100 and SM120 retain target-tagged typed carriers for
architecture-owned lowering. Straight-line async copies mint typed completion
tokens, the matching wait retires them, and matrix consumers preserve those
edges through TMA lowering. Host WSL FileCheck proves the three distinct IR
routes; native SM90 and SM100 remain unsupported-by-evidence until exact-device
runs exist. No selector changes. That breadth statement described the first
landing slice and is now superseded by the implementation record below: SM120
canonical execution covers the complete matmul dtype matrix plus softmax,
reductions, fused epilogues, attention, paged-KV, ReplaySSM, and local MoE.

The next NVIDIA-E2E-2 family slice now gives static f16/f32 last-axis softmax a
canonical Level-C path. `tile.softmax_kernel` carries source/destination,
flattened Rows/K, `storage="f16"|"f32"`, `accum="f32"`, and `axis=-1`; the SM120
materializer emits a stable max-shifted row loop and target-native `nvvm.ex2`
instead of an unavailable NVPTX `fexp` libcall. LLVM 23 emits and ptxas
validates `sm_120a` PTX, while the typed descriptor registers and launches it
through the shipped CUDA-driver bridge. Exact RTX 5070 Ti proof covers shapes
`1x16`, `8x64`, `4x300`, and `2x3x48`, extreme logits, malformed output shape
rejection, stable cold/warm image identity, and resource/spill fields for both
storage types. f16 loads extend before the max/sum/normalization loops and
truncate only at output storage. This is a correctness-first 128-thread,
one-thread-per-row candidate. The existing cooperative CUDA-C route remains
selected. All four final canonical/production rows are stable in both timing
domains, and production wins both domains for the production-sized and ragged
cases.

The following NVIDIA-E2E-2 dtype-totality slice centralizes consumer-Blackwell
storage, math-mode, scalar/vector, Tensor Core, compiler, and runtime states in
`nvidia_dtype_contract.py`. Every canonical float storage type now has an
explicit row. CUDA 13.3 compile proof covers scalar/vector forms for fp64,
fp32, fp16, bf16, FP8 E4M3/E5M2, FP6 E2M3/E3M2, and packed FP4; TF32 remains
strictly an fp32 `math_mode`, never storage. Tensor Core Target IR/PTX rows now
cover the required TF32, bf16, fp16, FP8, FP6, FP4, and int8 families. The
canonical descriptor lane now executes BF16, explicit fp32-storage TF32 math,
FP8 E4M3/E5M2, and INT8 with int32 accumulation. FP64 m8n8k4 DMMA now owns a
distinct Tile lane map and f64 descriptor/bridge ABI; aligned and ragged RTX
5070 Ti rows match the f64 oracle with masked tails.
FP6 E2M3/E3M2 now assemble as `kind::mxf8f6f4`, m16n8k32,
UE8M0/`scale_vec::1X`; OCP/MXFP4 assembles as `kind::mxf4`, m16n8k64,
UE8M0/`scale_vec::2X`. Compiler-owned packed-memory Tile materializers,
five-buffer descriptors, CUDA-driver launch ABIs, and aligned/ragged numerical
proof now cover both FP6 encodings and MXFP4. In particular,
`fp4_e2m1` does not alias NVFP4: MXFP4's UE8M0 scale contract cannot reuse
NVFP4's UE4M3/`scale_vec::4X` scale words.
The shared MMA selector now requires explicit `math_mode="tf32"` for fp32 and
retains distinct `nvfp4` and `fp4_e2m1` K64 identities. No selector promotion
or production route changes.

The canonical dtype execution matrix records two disjoint, sample-interleaved
runs for square and ragged fp64/fp16/bf16/TF32/FP8/FP6/MXFP4/INT8 routes, with separate
CUDA-event and allocation/copy-inclusive timing, cold/warm image identity, and
ptxas register/shared-memory/spill fields. The retained 20-row collection
changes no selector. The final 31-sample, 10,000-device-launch and
50-end-to-end-launch run has 19/20 rows stable in both timing domains. The only
terminal miss is TF32 `256x256x256`: its device cohorts remain bimodal at 7.02%
while end-to-end is stable at 0.59%. That row is explicitly non-promoting; the
existing selector is retained rather than hiding the exact-device result.

The broader-family NVIDIA-E2E-2 reduction slice now carries
`tile.reduce_kernel(X,O,Outer,AxisExtent,Inner)` and an SM120-owned v2
materializer/descriptor ABI for f16/f32 sum, mean, and NaN-propagating max.
Normalized arbitrary axes and keepdims shape contracts execute through both a
single-owner serial schedule and a 128-thread cooperative shared-memory
candidate. Exact RTX 5070 Ti proof covers axes 0/1/2, keepdims on/off,
rectangular/ragged rank-3 inputs, f32 accumulation, non-finite values,
image/resource retention, and 42 numerical rows. The earlier last-axis record
remains historical evidence; the new comparative record applies the WSL 4%
foundation policy in both timing domains and changes no selector.

The canonical epilogue slice carries f16/bf16/TF32/FP8 E4M3/E5M2 bias,
ReLU/GELU/SiLU, optional f32
residual, and the explicit `matmul -> bias -> activation -> residual` order in
the Tile kernel plus launch descriptor. The CUDA materializer consumes distinct
bias/residual buffers and rejects unsupported dtype/order/shape contracts
instead of silently dropping epilogue semantics. The original 32 f16/bf16 rows
and the 48-case TF32/FP8 matrix pass exact-device execution. The comparative
record measures canonical single-kernel images against the existing production
composed routes, retaining both timing domains, cold/warm state,
image/resource fingerprints, spills, and raw disjoint cohorts. Production
selectors remain unchanged unless both domains select the same stable winner.

The first canonical attention slice adds a shared typed
`tile.attention_kernel(Q,K,V,O,B,Hq,Hkv,Sq,Sk,D,Dv)` carrier with explicit
f16/f32 storage, f32 accumulation/output, positive scale, and causal semantics.
The SM120 correctness-first materializer and four-buffer descriptor launch
through the shipped PTX bridge; exact RTX 5070 Ti proof passes 8/8
MHA/MQA, rectangular/ragged, causal/non-causal cases with zero spills. The
entry symbol includes the scale/causal semantic digest so incompatible images
cannot alias in the driver cache. Bias, window, softcap, dropout, and backward
are completed below. The retained eight-row
two-cohort baseline records CUDA-event and allocation/copy-inclusive timings,
cold/warm image identity, resources, and raw samples. A higher-amortization
rerun now has 8/8 rows within 3% in both domains. It remains historical
evidence; the final comparison below owns the production disposition.

The forward carrier now also owns optional dense f32 bias, signed left/right
window bounds, arithmetic softcap, and deterministic `lcg32_counter_v1`
dropout. These semantics participate in the image digest and descriptor
provenance. An exact-device advanced row proves causal+window+bias+softcap,
bitwise dropout replay, and malformed-bias rejection; the earlier 8-row
MHA/MQA matrix remains green. The f32 backward reference now also crosses the
compiler-owned seam through `tile.attention_backward_kernel` and a seven/eight-
buffer native descriptor. It assigns one dQ/dK/dV element to one thread,
performs fixed-order single-owner dK/dV reduction, requires
`deterministic=true`, and declares zero workspace. The exact-device GQA row
proves causal+window+bias+softcap derivatives, bitwise replay, descriptor-shape
rejection, and agreement with the shared Pade-softcap oracle. The final semantic
slice below adds matching f16 storage and dropout-mask replay.

The refreshed backward candidate matrix passes 6/6 exact-device oracle,
determinism, and workspace cases. All six atomic/split rows are stable in both
timing domains. Atomic wins both domains for MHA D64, causal MHA D128, and
ragged GQA; split/reduced remains the bitwise-repeatable option with one extra
dK+dV f32 workspace (134,144--524,288 bytes in the retained shapes). Production
already selects atomic, so the evidence retains that selector. The canonical
deterministic reference carrier is now landed; production selection continues
to be governed by the stable atomic/split corpus rather than the intentionally
serial reference materializer.

The paged-KV landing slice adds `tile.paged_kv_read_kernel` and a compiler-owned
f32-pages/i32-table direct descriptor ABI. Four exact-device boundary ranges,
two non-identity physical-page permutations, remap/reuse, and invalid-table
rejection pass. The existing 12-case fused/staged suite also remains green,
including causal offsets and page boundaries. The committed
`nvidia_sm120_e2e_spine_paged_kv.json` corpus compares canonical Tile-direct
against legacy CUDA staged gather at 128, 512-ragged, and 2048-ragged tokens.
It retains two repeated medians in both timing domains, cold/warm image and
cache state, registers, shared memory, occupancy, spills, and resource
fingerprints. This WSL foundation lane uses a 4% repeatability policy because
its graphics clocks are host-managed. All six candidate rows are accepted; the
legacy 2048 device-event row uses an explicit five-basis-point WSL margin at
4.02%, and margin-accepted rows are selector-ineligible. Timing-domain winners
also disagree at 512/2048, so the selector
remains unchanged. The SM120 foundation disposition is closed as retain-existing;
a future native-Linux controlled-host promotion attempt is a separate
hardware-environment follow-up, not an open migration dependency.

The stateful/MoE image slice adds compiler-owned Tile→NVIDIA→PTX packages for
ReplaySSM decode/flush and local f16/bf16/f32 MoE dispatch/combine/ragged
grouped GEMM.
The resident Replay handle no longer embeds those device kernels in its CUDA
host bridge: it loads the compiler-produced PTX functions while retaining the
session-persistent allocations, asynchronous ring, events, and ordering
contract. Compiler-owned MoE candidates launch through the generic descriptor
submission path. Exact RTX 5070 Ti tests cover
dispatch/combine numerical order,
zero-sized expert groups, ragged grouped GEMM, Replay transitions, persistent
workspace metadata, image identity, and resource retention.

The final comparative record contains 14 strict-stable rows for cooperative
softmax, four-warp forward attention, and local MoE dispatch/combine/grouped
GEMM. Every row retains two CUDA-event and allocation/copy-inclusive cohorts,
the discarded first lifecycle launch, 100-launch end-to-end amortization,
per-candidate clock conditioning, cold/warm image state, and exact resource
fingerprints. Production softmax and attention win both domains. MoE does not
produce cross-domain consensus across all three routes, so the existing MoE
selector is retained. ReplaySSM's higher-amortization 10-row matrix is now
10/10 stable in both timing domains. No selector changes.

The collective follow-on adds an explicit content-addressed rank/device
topology and a one-process/multiple-device NCCL executor for all-reduce,
all-gather, reduce-scatter, and grouped send/receive all-to-all. This host
exposes one CUDA device, so it proves deterministic topology/rejection and
records the two-device request as unavailable; it cannot supply the required
two-or-more-GPU numerical, topology, resource, or timing evidence. RCCL and
Apple mappings remain architecture-owned follow-ups. No collective or MoE
selector changes.

The remaining-dtype/reduction performance corpus uses production-sized square
and ragged TF32/FP8 fused epilogues plus f16/f32 arbitrary-axis reductions.
For every candidate it records first-use compilation/cache fill separately,
discards the first launch, and amortizes each device-event and end-to-end sample
over the next ten launches. Two disjoint time-interleaved 100-sample cohorts
retain raw samples, cold/warm or first/second-use state, image/resource
fingerprints, registers, shared memory, and spill fields. All 30 rows are
accepted under the WSL foundation rule: 29 pass the strict 4% gate and the
production fp16-mean reduction end-to-end row is explicitly margin-accepted at
4.099% under the user-approved 4.15% rounding bound. That row is
selector-ineligible. Seven strict rows have cross-domain winner consensus, but
the record changes no selector because stable consensus alone does not establish
a promotion policy or required material benefit.

The final SM120 semantic slice removes the remaining execution limitations.
The deterministic attention VJP now accepts matching f16 or f32 dO/Q/K/V and
gradient storage, accumulates in f32, and replays the forward
`lcg32_counter_v1` dropout mask from the semantic seed without a saved-mask
workspace. A compiler-owned `tile.paged_attention_kernel` consumes Q, K/V
pages, the i32 remap table, i64 logical token indices, and an explicit causal
offset in one fused descriptor; the offset is never inferred from allocation
capacity. MoE dispatch, deterministic combine, and ragged grouped GEMM now
accept f16, bf16, or f32 storage with int32 metadata, f32 combine weights, and
f32 grouped accumulation. Exact RTX 5070 Ti tests prove numerical agreement,
dropout bitwise replay, page remapping/causal boundaries, malformed metadata
rejection, and low-precision MoE execution. This shared Tile-carrier extension
transfers no CUDA schedule: Apple is not applicable because it owns a separate
resident paged-attention ABI and mature low-precision dispatch paths; ROCm
requires architecture-owned lowering before claiming these carrier variants.

Two hardware boundaries now have formal **deferred terminal** states for this
work item:

- exact two-or-more-GPU NCCL topology, numerical, resource, and timing proof is
  deferred because the available SM120 WSL host exposes one GPU;
- exact SM90 Hopper and SM100 datacenter-Blackwell Level-C evidence is deferred
  because neither exact target is available. Their compile-only Level-B
  artifacts do not inherit SM120 execution evidence.

Deferred hardware terminals do not authorize selector changes and do not hide
missing evidence. A future hardware follow-up must reopen its own exact-device
item under synchronization key `E2E-SPINE-2026-07-18`.

E2E-SPINE-3 is a shared-contract follow-up under the same synchronization key.
It is applicable to NVIDIA only as a family-granular evidence envelope around
the existing SM120 results: fixture identity, Level-C provenance, cold/warm
cache identity, benchmark metadata, and hash-sealed release-packet validation.
It changes no CUDA schedule, ABI, dtype capability, or selector. SM90, SM100,
and exact multi-GPU rows remain explicit hardware-deferred terminals and may
not inherit the SM120 packet.

The E2E-SPINE-3 exact-host recorder now packages all eight bounded SM120
families through compiler-owned image/descriptor seams: matmul, softmax,
reduction, fused epilogue, attention, paged-KV, ReplaySSM, and MoE. The six
formerly pending family rows add shared differential fixtures where needed,
prove cold/warm image and descriptor identity, retain selected route plus ptxas
resource fingerprints, and record independently conditioned repeated-median
device-event and allocation/copy-inclusive end-to-end rows. The hash-sealed
WSL RTX 5070 Ti packet is checked in against landed source commit
`9da32b78c37fc3bebf3f69d575e7b1eb4013a399`; all 16 timing rows pass the
unchanged 4% stability gate. Family-granular recording prevents one noisy
family from invalidating already-stable evidence while the final manifest
still seals one `(nvidia_sm120, sm_120a)` packet. No CUDA selector changed.

The LLVM-stage device-library follow-on makes CUDA `libdevice` an explicit
compiler dependency rather than accidental driver behavior. Native-image
identity now retains logical device-library name, content digest, and link mode
without serializing host paths. The SM120 packager fingerprints
`nvvm/libdevice/libdevice.10.bc` and uses `llvm-link --only-needed` whenever
translated LLVM IR retains an unresolved `__nv_*` call. A real `__nv_sinf`
fixture links through CUDA 13.3 libdevice, lowers with LLVM 23 `llc`, and
assembles with `ptxas -arch=sm_120a`. Intrinsic-only kernels retain an empty
linked-library set, while the available libdevice digest still participates in
the toolchain/cache fingerprint. This changes no runtime selector.

The CUDA floating-point follow-on separates three semantic routes: IEEE
arithmetic operators, function-specific CUDA libdevice calls, and explicit PTX
approximations. The shared softmax envelope now carries
`exp_mode="approx_exp2"` and `ftz=false`; SM120 accepts only that proven mode
and lowers it to `ex2.approx.f32`. The contract records PTX's full-range 2-ULP
bound, requires a nonzero near-zero comparison budget, and versions native
cache identity independently of `-O3`. It does not reuse the `__expf` accuracy
table for a different instruction and does not enable global fast math.
The semantic authority is NVIDIA's
[floating-point computation appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html);
instruction-specific accuracy comes from the
[PTX `ex2` specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2).

The CUDA Math API scalar/integer follow-on records representative integer math,
bit, packed-dot, numeric/bit-cast, and 2x16/4x8 packed-SIMD families. A CUDA
13.3 `nvcc -arch=sm_120a` fixture proves the documented symbols compile, while
this original synchronization point kept every Tessera Target-IR/runtime state
`planned`. `NVIDIA-PACKED-MATH-2026-07-25` first promoted a bounded subset; the
structured continuation recorded below now closes the listed bit, bit-cast,
and packed-SIMD families through typed Target IR and 27 exact-device cases.
The shared rounding vocabulary now represents CUDA's four conversion suffixes
RN/RD/RU/RZ exactly; nearest-away and stochastic modes cannot silently map to a
CUDA cast. Undefined signed-min absolute value, out-of-range float-to-integer
conversion, funnel-shift wrap/clamp, signedness, lane width, and saturation are
retained as contract boundaries. The later internal Tile route adds no public
Graph op or selector. Sources: [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html),
[integer intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html),
[integer math](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INT.html),
[casts](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html),
and [packed SIMD](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html).

The PTX 9.3 truth audit now separates CUDA C++ storage spelling from physical
PTX typing. Fundamental storage rows are fp64 `.f64`, fp32 `.f32`, fp16 `.f16`,
and int8 `.s8`; BF16, TF32, FP8, FP6, FP4, and NVFP4 are alternate instruction
formats carried in same-width bit registers. Tensor fragments explicitly name
`.f64` or packed `.b32` operands. This corrects BF16 scalar/vector status to
`conversion_only` and prevents CUDA header types from implying fundamental PTX
types. PTX operand compatibility never performs automatic numeric conversion.
Direct `ptxas -arch=sm_120a` proof assembles the fundamental register surface
and rejects `bf16`, `tf32`, `e4m3`, and `u8x4` as register declarations.

Cross-backend sync `ROCM-E2E1-SOFTMAX-2026-07-19` is ROCm-owned. It adapts the
shared `tile.softmax_kernel` envelope to `tessera_rocm.softmax`, packages
HSACO, and submits through an exact gfx1151 HIP descriptor hook. NVIDIA's
`tessera_nvidia` lowering, `ex2.approx.f32` math contract, PTX ABI, SM120
schedule, resource/timing evidence, and selectors are unchanged. No AMD
wave/LDS or OCML behavior transfers to CUDA. ROCm's subsequent use of the
shared device-library record for its driver-selected OCML/OCKL/OCLC set is
parity validated at the schema boundary and requires no CUDA record or cache
change.

Cross-backend sync `ROCM-DTYPE-TOTALITY-2026-07-19` is ROCm-owned and not
applicable to NVIDIA target state. It adds no canonical dtype or alias and does
not change the SM120 PTX storage/Tensor Core contract, fragment ABI, runtime
readiness, or selector; it only prevents RDNA3.5 ISA formats from being
conflated with Tessera gfx1151 execution support.

Cross-backend sync `ROCM-DTYPE1-CLOSE-2026-07-21` promotes signed `int4` and
alias `i4` into the shared canonical/Graph-IR vocabulary and adds signedness to
the shared packed-storage descriptor. NVIDIA parity is validated at that
logical contract; NVFP4 and NVIDIA packed-weight/Tensor Core ABIs remain
distinct and backend-owned. No PTX capability, fragment ABI, runtime route, or
selector is promoted by the gfx1151 proof, and unsigned packed-4 remains
unregistered.

Cross-backend sync `E2E-FROZEN-IDENTITY-CACHE-2026-07-19`: ROCM-E2E-1 memoizes
deterministic hashes for frozen runtime artifacts, native images, and launch
descriptors. Serialized identity values and required launch validation are
unchanged, so CUDA schema parity is validated; no NVIDIA ABI, schedule,
runtime route, performance claim, or selector changes.

Cross-backend sync `ROCM-E2E2-REDUCE-2026-07-19` is ROCm-owned. It consumes the
already-shared `tile.reduce_kernel` carrier and widens its portable verifier to
admit bf16. At that synchronization point NVIDIA's backend-specific
materializer still accepted only f16/f32, so the ROCm five-argument HSACO ABI
transferred no CUDA claim. That historical NVIDIA boundary is superseded by
`NVIDIA-BF16-CANONICAL-BREADTH-2026-07-25`, which owns its independent
`Outer/AxisExtent/Inner` PTX ABI, serial/cooperative-128 lowering, resources,
and exact-SM120 evidence without inheriting the ROCm schedule or selector.

Cross-backend sync `ROCM-E2E2-PAGED-KV-2026-07-19` is ROCm-owned. It consumes
the existing shared paged-KV carrier without changing its verifier or public op
schema. NVIDIA's existing direct PTX mapping remains parity validated; no ROCm
gather schedule, HSACO ABI, page-table validation evidence, timing, readiness,
or selector state transfers to CUDA.

Cross-backend sync `ROCM-E2E2-MOE-DISPATCH-2026-07-19` is ROCm-owned. It
consumes the existing shared MoE dispatch carrier and public operation without
changing their verifier or dtype registry. NVIDIA's typed PTX mapping remains
parity validated at the carrier boundary; no AMD gather schedule, HSACO ABI,
gfx1151 evidence, timing, readiness, or selector state transfers to CUDA.

The accompanying PTX memory contract records CTA/cluster/GPU/system scopes and
relaxed/acquire/release/acq_rel atomic semantics. Vector and packed memory
accesses are sets of scalar accesses in unspecified element order, not one
atomic unit; mixed-size races fall outside the model; `red` does not form an
acquire pattern; texture/`ld.global.nc` accesses are excluded; ordered CUDA
submission does not establish intra-kernel memory order. Sources:
[types and state spaces](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#state-spaces-types-and-variables),
[instruction operands](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-operands),
and [memory consistency](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model).

The first focused CUDA parity proof on the NVIDIA box is:

```bash
python3 -m pytest -q \
  tests/device/nvidia/test_tile_fragment_compiler_path.py \
  tests/unit/test_nvidia_fragment_layout.py \
  tests/integration/test_nvidia_paged_kv_native.py \
  tests/integration/test_nvidia_replay_ssm.py \
  tests/device/nvidia/test_flash_attention.py \
  tests/device/nvidia/test_flash_attention_backward.py

python3 benchmarks/nvidia/benchmark_serving.py \
  --shapes 1x128x64 1x256x128 \
  --tokens 64 --chunk 4 --slots 4 \
  --kv-tokens 128 512 2048 --heads 8 --dim 64 --page-size 16 \
  --reps 20 --output /tmp/nvidia-sm120-serving.json

python3 benchmarks/nvidia/record_hot_path_baseline.py --reps 20 --margin 2.0
```

The focused pytest command is a correctness loop, not a substitute for the
marker-separated full CUDA lanes above. Benchmark outputs under `/tmp` are
review artifacts only; update committed baselines or autotune corpora only
after two stable runs and an explicit before/after review.

## Next update

Cross-backend sync `X86-E2E1-NATIVE-CPU-2026-07-19` classifies shared native
descriptor results for host x86 targets as `native_cpu` with CPU-wall timing.
CUDA remains `native_gpu` with its existing event and end-to-end timing domains;
no PTX ABI, SM schedule, device evidence, readiness, or selector state transfers.
The x86 pilot consumes existing Tile softmax/reduction carriers without changing
their shared dtype or operation registration.

Cross-backend sync `X86-E2E1-BREADTH-2026-07-19` consumes the existing shared
matmul and attention carriers for f32 AVX-512 descriptors. NVIDIA inherits no
x86 ABI, vector schedule, host timing, readiness, or selector state. SM120
GQA/dropout and dtype breadth remain governed by NVIDIA-owned Target IR and
exact-device evidence; x86's narrower descriptor contract changes no CUDA row.

Cross-backend sync `E2E-SPINE-2026-07-18` records the 2026-07-20 scoped x86
selector retirement: eligible static X86-E2E-1 modules now use their canonical
descriptor by default. NVIDIA parity is not applicable; no NVIDIA pipeline,
PTX ABI, schedule, capability, or selector changes. X86-E2E-2 subsequently
closed the remaining inventory and reassessed NVIDIA at each shared-contract
boundary.

Cross-backend sync `X86-E2E2-ELEMENTWISE-2026-07-20` adds the internal shared
`tile.elementwise_kernel` semantic carrier for f32 unary/binary and f32-to-bool
predicate requests. NVIDIA parity is assessed at the carrier boundary only;
the AVX-512 ABI, CPU schedule/timing, 16K binary selector threshold, and exact
x86 evidence transfer no PTX implementation or CUDA selector claim. Existing
NVIDIA elementwise target and execution rows are unchanged.

Cross-backend sync `X86-E2E2-TYPED-LOGIC-2026-07-20` widens that internal
carrier with compare, logical, and bitwise semantics plus explicit f32/i8/i32
physical storage. The capability repair is x86-owned bool/int32 truth for
already-shipped AVX-512 ABIs. NVIDIA inherits no C ABI, null-operand convention,
32K selector threshold, CPU timing, PTX implementation, or CUDA selector
claim; NVIDIA target and execution rows remain unchanged.

Cross-backend sync `X86-E2E2-FLAT-FOLLOWON-2026-07-20` extends the shared
elementwise carrier with where, transcendental, and binary-math semantics.
NVIDIA parity is assessed at the carrier boundary only; AVX-512 approximations,
C ABIs, CPU-wall thresholds, exact-host evidence, PTX routes, and CUDA selectors
do not transfer. Existing NVIDIA rows remain unchanged.

Cross-backend sync `X86-E2E2-DTYPE-2026-07-20` adds an x86-only datatype/CPUID
contract and BF16, VNNI U8/S8, and FP64 descriptor ABIs. NVIDIA already owns
independent dtype, MMA, accumulator, PTX, and runtime contracts; no CUDA target,
execution, or selector row changes.

Cross-backend sync `ATTN-DIALECT-MLIR23-2026-07-20` corrects the internal MLIR
attention dialect namespace from the nested `tessera.attn` spelling to the
MLIR-23-compatible `tessera_attn` spelling. Public Graph IR operation names,
attention semantics, NVIDIA target capabilities, PTX ABIs, schedules, and
selector state are unchanged; NVIDIA parity is validated by the shared
attention lit coverage.

Cross-backend sync `LLVM23-BACKBONE-2026-07-20` makes LLVM/MLIR 23.x the sole
accepted compiler build environment. Top-level and standalone CMake entry
points reject every other major and mixed installations; NVIDIA uses the
versioned apt LLVM 23 packages alongside CUDA 13. NVIDIA target semantics, PTX
ABIs, and selectors are unchanged, and the LLVM 23 compiler/lit build validates
host-free parity; exact-device claims remain NVIDIA-owned.

The collection contract, compiler-artifact, exact-device correctness, and
serial measured lanes now have recorded baselines. NVIDIA-TEST-5 and
NVIDIA-TEST-6 are closed; the requested attention, epilogue, and legacy-retune
parity records are stable without a selector promotion. Keep `plan_state:
landing` while unrelated implementation or exact-device follow-ons remain.
Move this plan to the NVIDIA archive only after every completion gate is met.

Consumer plan `SEQUENCE-MIXER-2026-07-17`: the compiler-direction Sequence Mixer
track ([`../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md`](../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md))
consumes the NVIDIA families as a **lead performance target** (Decision #28 — its
`wgmma`/`mma.sync` candidates set the ceiling and are never capped by the shared
mixer framework). It adds candidates under existing families, opening no new
NVIDIA-TEST item: channel-wise KDA/GDN decode → **NVIDIA-TEST-3/-5 KV/ReplaySSM**;
`sliding_window`/full mixer fwd + backward → **attention** (split/reduced dK/dV,
G6-C-style); chunkwise-scan inner GEMMs → **GEMM/Tile** (`wgmma` sm_90 / `mma.sync`
sm_120, preferably via the NVIDIA Tile IR lowering target); NVFP4/MXFP8 mixer GEMMs
→ **NVIDIA-TEST-4** numerical policy (this is the executing FP4 lane — sm_120
`mma.sync`, not `tcgen05`). Inherits the TEST-3 native-provenance / TEST-5
kernel-vs-E2E evidence contract unchanged. Direction pointer only; no NVIDIA gate,
route, or exact-device claim changes here.

Cross-backend sync `X86-E2E2-COHORT2-2026-07-20` adds shared typed Tile
carriers for argreduce, inclusive scan, unweighted row normalization,
interleaved-pair RoPE, and ALiBi. NVIDIA parity is assessed at the semantic
carrier boundary only. AVX-512 ABIs, CPU schedules, Ryzen timing, and route
disposition transfer no PTX/CUDA implementation, device evidence, or selector.

Cross-backend sync `X86-E2E2-BREADTH-2026-07-20` adds an explicitly x86-owned
`tile.x86_abi_kernel` and cohort-3/4 C-ABI registry. It changes no portable
semantic Tile carrier, PTX/CUDA ABI, NVIDIA schedule, dtype capability,
execution row, or selector. NVIDIA parity is therefore not applicable.
X86-E2E-2 is now closed with measured x86-only selector thresholds; this does
not change the NVIDIA not-applicable disposition or transfer device proof.

Cross-backend sync `LLVM23-LOCAL-CLEANUP-2026-07-20` hardens the LLVM 23 and
Linux TSAN host environment and repairs an Apple-only capability row. NVIDIA
parity is not applicable: no CUDA dtype, PTX ABI, schedule, execution row,
selector, or exact-device evidence changes.

Cross-backend sync `ROCM-E2E-SPINE3-TEST1-2026-07-21` adds shared paged-KV and
MoE fixture identities to the E2E-SPINE-3 corpus. NVIDIA fixture-schema parity
is validated, but the gfx1151 HSACO, HIP ABI, resources, timing, and
exact-device evidence do not transfer to CUDA. The ROCm-owned compiler lane
explicitly excludes `compiler_nvidia`; no NVIDIA capability, test ownership,
schedule, execution row, or selector changes.

Cross-backend sync `CORE-COMPILER-1-2026-07-22` closes shared Graph/Neighbors
verifier gaps and records the shared `sm_120` MMA selection in NVIDIA manifest
rows. Equal-tier candidates may use its analytical accumulator footprint only
after route-tier precedence. This is parity validated at the host-free
compiler/manifest boundary; it changes no PTX instruction schedule, automatic
selector promotion, CUDA ABI, or exact-device evidence.

Cross-backend sync `CORE-COMPILER-2-2026-07-22` makes compute dtype
legalization the default in NVIDIA named pipelines. Terminal storage
legalization remains intentionally opt-in for the generic value-level CUDA
route because it has no block-scale operand ABI. The later
`NVIDIA-BF16-CANONICAL-BREADTH-2026-07-25` continuation closes the physical
consumer gap for the scale-bearing NVFP4/MXFP4/FP6 launch envelopes and wires
StoragePackConsume after opt-in terminal legalization; at that point generic
INT4 and a sub-byte default remained unsupported. The executable row-major layout
materializer and guarded
dynamic launch are x86-only and transfer no PTX schedule, CUDA ABI, bucket
policy, selector, or exact-device evidence.

Cross-backend sync `CORE-COMPILER-NEXT-2026-07-22` tightens shared Graph layout
propagation through agreed-layout pointwise chains and last-axis reductions,
preserves packed-storage attributes, and records source-layout provenance on
inserted casts. At that synchronization point the architecture-owned
Graph-cast materializer was open; it landed in the later NVIDIA continuation
recorded below. The pass stays opt-in and transfers no PTX layout, schedule,
selector, or device proof. The x86 dynamic last-axis reduction guard
is not applicable to bucketed tensor-core routes. Shared add/multiply/static-
broadcast adjoints change Graph IR only; no CUDA backward runtime or exact-
device promotion is claimed.

Cross-backend sync `CORE-COMPILER-FOLLOWON-2026-07-22` adds shared kind-aware
sum/mean, GELU/SiLU, and softmax Graph adjoints with host CPU oracle proof.
Dynamic mean, max/min, ReLU, and normalization remain explicit fallbacks for
the documented Graph-contract reasons. Guarded dynamic softmax, attention, and
growing KV-cache execution are x86-only and are not applicable to bucketed
tensor-core routes; no CUDA ABI, schedule, selector, backward runtime, or
exact-device claim transfers. NVIDIA's architecture-owned Graph-cast consumer
is host-validated: after shared legality it accepts row/column-major/BHSD/NHWC,
removes the Graph marker, and carries the binding into `tile.async_copy`.
This changes staging metadata only and claims no PTX schedule or device proof.

Cross-backend sync `CORE-COMPILER-ADJOINTS-2026-07-22` registers shared
tensor-to-i1 comparison contracts plus internal scalar-threshold,
rank-reduced normalization-statistics, and explicit broadcast-in-dimension
Graph carriers. ReLU and unweighted RMSNorm/LayerNorm paired adjoints are
static/dynamic Graph-native and CPU-IR oracle-proven; the static shared path
lowers through linalg. This shared sync added no PTX/CUDA execution; the
architecture-owned affine backward ABI, runtime binding, and exact-SM120 proof
land in `CUDA-TRAINING-MEMORY-FOUNDATION-2026-07-24` below. It does not imply a
tensor-core schedule or selector promotion.

Cross-backend sync `CORE-COMPILER-NORM-AFFINE-2026-07-22` makes integer
comparison signedness explicit in shared Graph IR and adds dynamic-dimension
carriers plus channel-affine RMSNorm/LayerNorm adjoints. The NVIDIA dynamic
affine materializer and backward runtime were still open at this sync and are
closed by the architecture-owned continuation below. The gfx1151 HSACO and
AVX-512 ABIs, schedules, and timings did not transfer to CUDA/PTX; no selector
promotion was inferred from sibling evidence.

Cross-backend sync `CORE-COMPILER-NORM-BWD-DETERMINISM-2026-07-22` changes only
the ROCm architecture-owned backward schedule and temporary-buffer ABI. The
shared affine adjoint and f32 accumulation contract are unchanged. The
CUDA/PTX backward materializer and exact-device proof were supplied later by
the NVIDIA-owned continuation below; the gfx1151 two-kernel schedule, bitwise
evidence, and timing did not transfer.

Cross-backend sync `CORE-COMPILER-NORM-BWD-2026-07-22` adds family-specific
RMSNorm/LayerNorm backward execution rows and public JIT binding for ROCm and
x86. The then-open NVIDIA execution row is now closed by the CUDA/PTX
continuation below. Neither the gfx1151 HSACO ABI nor the AVX-512 ABI,
schedule, timing, or device evidence transferred; the NVIDIA implementation
retains its own descriptor and exact-device proof.

Cross-backend sync `CORE-COMPILER-LAYOUT-AUTODIFF-MEMORY-2026-07-23` completes
the shared transpose/packed epilogue/reduction layout envelope and adds native
guarded-dynamic broadcast, runtime-extent mean, and equal-share max/min Graph
adjoints. NVIDIA parity is host-validated through the shared linalg contract.
All NVIDIA backend variants now execute Tile buffer reuse and materialize one
address-space-3 shared-memory arena with typed planned-offset views before
Tile-to-NVIDIA/NVVM lowering. Function-budgeted liveness-aware
rematerialization also runs in the shared production post-autodiff pipeline.
Exact CUDA/PTX shared-allocation assembly and occupancy were open at this
shared sync; the static and dynamic CUDA evidence is recorded in the NVIDIA
closeouts below. No selector promotion is implied.

Cross-backend sync `CORE-COMPILER-TRAINING-SPINE-2026-07-23` registers
`tessera.loss.mse` and its paired backward carrier as verifier-checked shared
Graph IR, with dynamic none/sum/mean Linalg lowering and FP32 compute for
FP16/BF16 storage. Shape-preserving MSE participates in shared layout
propagation, and post-autodiff rematerialization now distinguishes saved
forward activations from backward temporaries. NVIDIA parity is host-validated
at the shared IR boundary. The gfx1151 HIP composition/module cache and
AVX-512 execution do not transfer to CUDA/PTX. The NVIDIA-owned compiled MSE
backward launch and exact-device evidence land in the continuation below; no
tensor-core training schedule or selector promotion is claimed.

Cross-backend sync `CORE-COMPILER-DEEPENING-2026-07-23` adds shared
runtime-sized address-space-3 arena planning and a benchmark-fed
rematerialization cost contract. NVIDIA retains opt-in Graph layout assignment
through its existing materializer. The new MSE backward launch and numerical
proof are ROCm gfx1151-only. The architecture-owned CUDA VJP and dynamic
shared-allocation/occupancy proof land in the later NVIDIA closeouts; selector
evidence remains explicitly non-promoting on WSL.

Cross-backend sync `CORE-COMPILER-TRAINING-BREADTH-2026-07-23` adds shared
Graph-native MAE, Huber, SmoothL1, and SGD adjoints with dynamic Linalg and CPU
oracle proof. The architecture-owned CUDA/PTX backward materialization and
exact-device evidence land in the continuation below. The gfx1151 generated
HIP kernel, AVX-512 C ABI, boundary timing, caches, and selector state did not
transfer.

Cross-backend sync `CORE-COMPILER-TRAINING-SERIES-2026-07-23` adds shared
Graph-native stable BCE-with-logits, class-index/label-smoothed cross entropy,
KL/JS, explicit Momentum/Nesterov state, and explicit Adam/AdamW moment-state
adjoints. Dynamic shared Linalg contracts are live for BCE, Momentum/Nesterov,
and Adam/AdamW. The NVIDIA CUDA/PTX materializers and exact-device evidence
land in the continuation below, including KL/JS and FP16/BF16 storage. The
gfx1151 and AVX-512 loss and optimizer ABIs did not transfer.

Cross-backend sync `CORE-COMPILER-TRAINING-FUSION-2026-07-23` adds shared
single-use loss-backward to SGD/AdamW fusion carriers and one-loop dynamic
Linalg lowering for MSE, MAE, Huber, SmoothL1, and BCE-with-logits. This sync
validated only the shared Graph/Linalg contract; the architecture-owned
CUDA/PTX fused materializer and exact-device evidence land below. gfx1151 HIP
and AVX-512 ABIs, cache identities, timings, and selector decisions did not
transfer.

Cross-backend sync `CORE-COMPILER-MEMORY-LAYOUT-CLOSEOUT-2026-07-23` replaces
the shared static address-space-3 alloca with a workgroup global and supports
dominance-scoped dynamic arena cohorts. At that point exact CUDA assembly,
resource, occupancy, and performance evidence were open and not inferred from
gfx1151; the NVIDIA-owned static/dynamic closeouts below now provide them.
No NVIDIA selector or default policy changes.

Cross-backend sync `CORE-COMPILER-HONEST-BOUNDARIES-2026-07-23` broadens the
shared measured-rematerialization schema to exact consumer chains and
64/128/192 matmul shapes with ReLU/GELU/SiLU. The later NVIDIA packets provide
CUDA measurements; native-Linux policy selection remains deferred. ROCm dynamic
normalization epilogues, HIP launch-sized LDS materialization, and packed IU4
WMMA are architecture-owned and transfer no PTX ABI, shared-memory allocation,
packed consumer, performance, or selector claim. The existing NVIDIA
architecture-owned layout consumer remains unchanged.

Cross-backend sync `CORE-COMPILER-HONEST-BOUNDARIES-2-2026-07-24` extends the
shared rematerialization corpus schema with softmax, RMSNorm, and MSE producer
families plus measured workload-budget decisions. The later NVIDIA packets
provide CUDA measurements; native-Linux policy selection remains deferred.
ROCm's packed
multi-arena LDS ABI, GELU normalization epilogue, and terminal-pack
dequant-GEMM consumer are architecture-owned; no PTX shared-memory ABI, packed
consumer, timing, selector, or support claim transfers. CUDA path-max and
general serialized launch expressions are closed below; physical packed
consumers remain governed by their own dtype rows.

Cross-backend sync `CORE-COMPILER-HONEST-BOUNDARIES-3-2026-07-24` extends the
shared rematerialization evidence schema to a measured four-layer workload with
softmax, RMSNorm, MSE, Huber, SmoothL1, and BCE instances. CUDA measurements
now land in the NVIDIA continuation below; controlled native-Linux policy
selection remains deferred. ROCm's
branch-path dynamic-LDS expression, binary normalization epilogues, and packed
elementwise/sparse/cache ABIs are architecture-owned; no PTX shared-memory
expression, packed ABI, timing, selector, or support claim transfers.

Cross-backend sync `CORE-COMPILER-CFG-MEMORY-BUDGETS-2026-07-24` adds a shared
model/device-derived rematerialization budget contract with explicit override
precedence and bounded dynamic parameters. The exact CUDA-context capacity,
free-memory cap, reserve policy, parameter bounds, and measured packet now land
in the NVIDIA continuation below. ROCm's alias-aware nested/loop LDS slots and 40,208-byte
gfx1151 packet are architecture-owned; no PTX shared-memory expression,
occupancy, execution, or selector claim transfers.

NVIDIA-owned closeout `E2E-SPINE3-SM120-MEMORY-2026-07-24` completes the
static NVPTX boundary named by `CORE-COMPILER-MEMORY-LAYOUT-CLOSEOUT`.
Tile lowering turns three logical 512-byte allocations with disjoint lifetimes
into one 1,024-byte address-space-3 arena at offsets `[0, 512, 0]`, reducing
the unreused 1,536-byte plan. LLVM 23 emits an exact 1,024-byte NVPTX shared
declaration; `ptxas` reports the declaration and the retained executable route
has matching tool-reported shared-resource accounting. Exact SM120 execution
compares that retained route with a register-rematerialized expression:
both return 42, have complete no-spill resource records and 100% theoretical
occupancy, and pass the 4% two-run gate in device-event and end-to-end domains.
The evidence is intentionally selector-ineligible. Dynamic/path-expression
arenas and model-level CUDA rematerialization policy remain separate follow-ups.

NVIDIA-owned continuation `CUDA-TRAINING-MEMORY-FOUNDATION-2026-07-24`
closes the named SM120 execution gaps for dynamic-affine RMSNorm/LayerNorm
backward; MSE, MAE, Huber, SmoothL1, stable BCE-with-logits, KL/JS, and
class-index/label-smoothed cross-entropy backward; deterministic general
broadcast-gradient reduction; SGD, Momentum/Nesterov, Adam, and AdamW updates;
and fused loss-backward plus SGD/AdamW. FP32, FP16, and BF16 storage preserve
their physical dtype while every arithmetic/reduction path accumulates in FP32. The
architecture-owned path generates CUDA, compiles an immutable PTX image,
binds a launch descriptor, and executes through the shipped CUDA-driver
bridge. Public runtime artifacts and `@jit(...).native_backward()` use the
same descriptor seam. Exact SM120 tests cover dynamic/ragged extents,
transition boundaries, extreme logits, none/sum/mean cotangents, optimizer
state, fused-versus-composed numerics, invalid labels, cache identity, and
live resources.

The same continuation completes CFG-forwarded and locally computed dynamic
shared-memory sizing. A post-LLVM NVIDIA pass replaces runtime-sized
address-space-3 byte arenas with slices of an external NVPTX shared symbol,
colors mutually exclusive lifetimes into one slot, keeps simultaneously live
arenas in distinct aligned slots, and serializes checked constant, argument,
add, multiply, cast, select, and path-max launch expressions. The v2 CUDA
launch ABI evaluates the descriptor expression and passes the resolved byte
count to `cuLaunchKernel`. Exact SM120 execution proves the original
12,289/32,001-byte branch paths and a CFG-forwarded
`max(4096*4+17, 12289)` expression with a 16,416-byte allocation, rejects
undersized descriptors, and retains driver-JIT register/static/local-memory/
occupancy evidence.

The native bridge also reports total/free bytes from the same retained CUDA
context. The compiler policy caps usable capacity by current free memory,
marks static or explicitly bounded dynamic Graph IR model parameters, and
stamps reserve, gradient-copy, optimizer-state, and persistent-state inputs
consumed by the shared activation-rematerialization pass.

The checked-in 26-row repeated-median packet discards the first/JIT launch,
uses 1,000 device-event repetitions and 20 end-to-end iterations, and retains
cold/warm image identity plus artifact and resource fingerprints. Fourteen
rows meet the 4% two-run gate in both timing domains on this WSL2 host; twelve
retain explicit unstable dispositions, with the two tiny dynamic-shared probes
remaining especially host-noisy. Fused MSE+SGD and MSE+AdamW are stable and
beat their composed references in both timing domains, but their references
are not stable and WSL is not the controlled native-Linux promotion host.
Therefore no production selector changes. Rerunning this exact packet on
controlled native Linux is the remaining performance/selector boundary.

NVIDIA-owned closeout `NVIDIA-BF16-CANONICAL-BREADTH-2026-07-25` extends the
compiler-owned SM120 Tile-image seam from FP16/FP32 to BF16 input storage with
FP32 accumulation and output for reduction, stable row softmax, and attention
forward/backward. The reduction contract covers sum, mean, max/min and
amax/amin aliases, arbitrary static axes, keepdims true/false, ragged extents,
NaN propagation, and both serial and cooperative-128 physical schedules.
Every BF16 descriptor carries a distinct typed ABI and two-byte input binding;
the CUDA-driver bridge retains four-byte FP32 result bindings. The canonical
Graph verifier, target capability, backend manifest, execution matrix, Tile
verifier, NVIDIA lowering, runtime registration, and launch bridge agree on
that boundary.

Exact RTX 5070 Ti SM120 execution proves 44 focused BF16 softmax, reduction,
and attention cases through MLIR to NVVM, immutable PTX image, descriptor, and
CUDA-driver launch. Compiler lit proves BF16 extension into FP32 arithmetic,
serial/cooperative reduction assembly, min lowering, and attention
forward/backward materialization. The checked-in 12-row reduction packet
compares both schedules across all six public kind spellings, discards the
first launch, amortizes 500 resident launches and 10 complete calls, and
retains two disjoint repeated-median cohorts, cold/warm image and entry-symbol
identity, numerical error, ptxas registers/shared-memory/spills, and live
driver local-memory/occupancy. WSL timing remains an explicit non-promotion
result: only 3/12 candidates meet the 4% gate in both timing domains and no
candidate has stable cross-domain winner consensus. Production selectors are
therefore unchanged; controlled native-Linux comparison remains required
before any schedule promotion.

The continuation of `NVIDIA-BF16-CANONICAL-BREADTH-2026-07-25` closes the
remaining normalization envelope. Compiler-owned immutable PTX images now
execute unweighted RMSNorm/RMSNorm-safe and LayerNorm with f32, f16, or bf16
storage, f32 row statistics/accumulation, immutable nonnegative epsilon, and
same-storage output. Eighteen exact SM120 cases cover ragged/non-power-of-two
rows, multiple ranks, both normalization kinds, all three storage types,
resource/no-spill inspection, warm image/cache/descriptor identity, and FP32
oracles. Dynamic affine BF16 backward remains on the previously proven
training descriptor, so forward and backward now share the BF16 storage
boundary without claiming an affine forward fusion in this unweighted kernel.

The same synchronization point lands NVIDIA's first physical consumer of the
shared terminal packing descriptor. Canonical NVFP4, MXFP4, FP6-E2M3, and
FP6-E3M2 launch images carry `tessera.storage_pack`; NVIDIA lowering requires
the logical format, int8 container, packing factor (2 for four-bit, 1 for
six-bit), and format-defined signedness to agree with the selected
scale-bearing fragment ABI before generating packed byte/nibble loads.
Descriptor drift rejects in compiler lit, while exact general-shape/ragged
SM120 tests traverse the descriptor-driven loaders. Opt-in terminal storage
legalization now runs `StoragePackConsume` in the NVIDIA named pipeline.
The later capability expansion admits only the proven value-level decode,
unscaled round-trip, and matching packed-matmul def-use paths. It does not make
terminal packing the default for arbitrary FP4/FP6 values or for Graph-level
quantize/dequantize operations.

NVIDIA-owned continuation `NVIDIA-PACKED-MATH-2026-07-25` adds the missing
canonical signed-INT4 consumer. Its compiler-owned Tile image and typed CUDA
launch ABI accept only an int8 container, factor two, signed two's-complement
nibbles, low logical index in the low nibble, i32 accumulation/output, and no
scale or fused-epilogue operands. The correctness-first general-shape schedule
decodes packed A rows and packed B columns, guards ragged M/N/K edges, and
rejects container, factor, signedness, scale/operand, and epilogue disagreement
before PTX materialization. Exact aligned and ragged RTX 5070 Ti execution
matches an integer oracle, retains zero ptxas spills, and proves cold/warm
image and descriptor identity. NVFP4/MXFP4/FP6 keep their distinct
scale-bearing fragment semantics; no physical schedule is transferred between
those formats or from ROCm.

The 2026-07-25 continuation replaces the dictionary-only pack record with
portable `#tile.packed_format`, `#tile.scale_layout`, and
`#tile.packed_view` attributes plus generic `tile.packed_load`/
`tile.packed_store`. The contract records logical bit width independently of
the int8 container (notably FP6 factor one), signedness/encoding/lane order,
packing axis/strides, alignment/offset, and an explicit scale operand/layout.
SM120 lowering decodes signed INT4, FP4/NVFP4, and both FP6 encodings, applies
origin-aware scale indexing for non-origin views, guards ragged bounds, and
supports unscaled packed round trips. Descriptor, axis, scale, alignment, and
store disagreement fail closed. Terminal marking is capability-filtered by
target + operation + format + available consumer; unsupported operations stay
logical rather than inheriting launch-envelope support.

The generic value-level consumer now also owns a CUDA runtime ABI rather than
ending at compiler lit. Exact RTX 5070 Ti execution covers non-origin/ragged
NVFP4 with explicit UE4M3 scale binding, signed INT4, and FP6-E2M3/FP6-E3M2
with explicit UE8M0 scale binding. All four cases match their format oracles,
retain zero ptxas spill bytes, and reproduce cold/warm image plus launch
descriptor identity. Source/scale byte extents are launch scalars checked
against the serialized physical view; the bridge rejects negative origins,
nonpositive extents, overflow, or buffer/descriptor disagreement. Generic
packed stores retain host-free compiler round-trip proof; no unsupported
format is default-enabled by this runtime addition.

The checked-in SM120 packed-storage packet measures the production-sized
`4097x4099` ragged view for signed INT4, NVFP4, FP6-E2M3, and FP6-E3M2. Each
device observation amortizes ten resident launches, each cohort retains seven
device-event observations and ten end-to-end submissions, and the first
allocation/copy/JIT-inclusive submission is discarded. Every row reproduces
its compiler image, launch descriptor, cache fingerprint, and 18-register,
zero-local-memory resource record. On this WSL host all four rows meet the 4%
two-run gate in both timing domains: the device-event deltas are 0.62%, 0.57%,
0.32%, and 0.08%, while the end-to-end deltas are 0.27%, 0.09%, 3.46%, and
0.09%. WSL evidence remains selector-ineligible, so every row records an
explicit retain disposition. No selector or terminal-legalization default
changes from these timings.

CUDA Math now crosses a registered `tessera_nvidia.cuda_math_kernel` seam
instead of extending an open five-pointer string dispatcher. Its closed
verifier covers scalar `brev`, `prmt`, `clz`, `ffs`, `popc`, numeric and
bit-preserving f32/i32 casts, packed 2x16 and 4x8 signed/unsigned wrapping or
saturating arithmetic, byte absolute difference, and both lane-mask and
predicate-bit comparisons. All 27 exact RTX 5070 Ti cases launch, match
bit-exact/numerical oracles, retain zero spills, and reproduce warm image and
descriptor identity. No production selector changes.

The structural continuation registers `!tile.buffer`, `tile.alloc`,
`tile.dealloc`, `!tile.pipeline_state`, `tile.pipeline_init`, and
`tile.pipeline_advance`. It now also registers SSA TMA descriptor, mbarrier,
mbarrier-token, TMEM, and TCGen05 types/operations. WarpSpecialization allocates
SMEM/TMEM handles in the parent region, threads them into staged copies and
consumers, threads producer/consumer pipeline states, and deallocates only
after CTA synchronization. AsyncCopy lowering emits registered TMA descriptor
and copy operations; descriptor deduplication assigns slots, creates one SSA
mbarrier, binds it to every copy, and threads copy completion tokens into the
typed wait. FlashAttention emits typed arrive/try-wait token chains.
Barrier-reuse legality passes on this real WarpSpec output. NVIDIA no longer
emits or consumes name-based `#tile.buffer_ref`; the shared reader remains only
for Apple/ROCm migration fixtures. Annotation-only `#tile.pipeline_state`
compatibility metadata is rejected, and the structured Schedule→Tile layout is
consumed directly. TCGen05/TMEM has host-free structural and SM120 fail-closed
proof only; exact execution remains SM100-owned and cannot be inferred from
consumer Blackwell.

Cross-backend sync `ROCM-TRAINING-MEMORY-FUSION-2026-07-27` adds ROCm-owned
Adam/AdamW and KL/JS physical backward execution plus a ROCm normalization
softcap epilogue; none of those HIP kernels, gfx1151 timings, or selector
evidence transfers to CUDA. The shared change is the target-neutral,
serializable dynamic-local-memory expression field on `LaunchDescriptor`.
NVIDIA's existing SM120 add/multiply/path-max/alignment probe now consumes
that field and retains its CUDA-owned PTX, launch-v2, resource, and exact-device
evidence. No NVIDIA execution row or selector changes.

Cross-backend sync `ROCM-LION-BACKWARD-2026-07-27` adds only the ROCm-owned
physical consumer of the already-shared Lion stop-sign VJP policy and extends
the gfx1151 operation-total benchmark packet. HIP code objects, AMD launch ABI,
and WSL timings do not transfer to CUDA. NVIDIA remains follow-up required for
an architecture-owned compiled Lion backward materializer; no SM120
capability, execution row, PTX schedule, or selector changes.

**NVIDIA follow-on update (2026-07-30):** the CUDA-owned Lion materializer now
has landed as `nvidia_lion_bwd_compiled`: an SM120 PTX package with an explicit
eight-buffer ABI (`p/g/m`, two output cotangents, and `dp/dg/dm`), CUDA bridge
layout, runtime executor, execution-matrix row, f32 oracle, and RTX 5070 Ti
device validation. It implements the shared stop-sign VJP without importing an
AMD code object or schedule. This is a correctness-first 128-thread package;
no optimizer selector changed and a timing packet remains a separate SM120
measurement task.

Cross-backend sync `CORE-SCHEDULE-1F1B-MATERIALIZE-2026-07-27` emits a shared
unique-clock warmup/steady/cooldown dependency order after pipeline legality.
At this synchronization point CUDA runtime consumption and collective overlap
remained NVIDIA-owned follow-up; the immediately following
`CORE-COMPILER-RUNTIME-CLOSEOUT-2026-07-27` record supersedes that structural
gap with the shared runtime consumer. The carrier itself changes no SM120
capability, PTX schedule, selector, or exact-device claim.

Cross-backend sync `CORE-COMPILER-RUNTIME-CLOSEOUT-2026-07-27` supplies a shared
runtime consumer for emitted 1F1B steps with independent collective transport,
and makes measured schedule records alter the actual Schedule/Tile M/N/K,
warp-count, and stage attributes after target/evidence validation. NVIDIA's
named pipelines now default layout assignment on because the architecture-owned
Graph-cast materializer immediately consumes the markers; focused structural
proof covers ordering. A real CUDA multi-rank transport packet and measured
SM120 selector application remain NVIDIA-owned exact-device follow-ups.

The same sync replaces DeltaNet-family finite-difference reverse mode with an
analytic carried-state recurrence and explicit reverse-token scheduling.
Directional-derivative fixtures validate shared semantics only; CUDA backward
packaging and device scheduling remain follow-up required. ROCm factored
Adafactor HSACO, HIP capacity query, gfx1151 numerics, and WSL timing do not
transfer to CUDA.

Cross-backend sync `CORE-PRODUCTION-EVIDENCE-2026-07-27` serializes collective
descriptors on emitted pipeline steps and binds OptimizerShard ownership
transitions to the shared runtime. NCCL remains the CUDA-native executor, but
this ROCm-host continuation contains no real multi-rank CUDA packet and changes
no SM120 selector. The gfx1151 Adafactor adjoint and two-entry DeltaNet
reverse-chunk HSACO (later superseded by the five-entry AMD package) are AMD
schedules and do not transfer. CUDA sequence-mixer
backward packaging and a refreshed measured selector packet remain
NVIDIA-owned exact-device follow-ups.

Cross-backend sync `CORE-SEQUENCE-MIXER-PHYSICAL-BACKWARD-2026-07-28` adds the
exact modified-Delta normalization VJP to physical ROCm and AVX-512 backward
paths and proves an affine parallel chunk-composition algorithm for
`erase=false`. This changes shared algorithm evidence, not CUDA execution. The
five-entry gfx1151 HSACO, AMD workgroup schedule, and WSL resident timings do
not transfer to SM120. CUDA sequence-mixer backward packaging, nonlinear/erase
chunk scheduling, and a refreshed exact-CUDA-host selector packet remain open.

**CUDA DeltaNet reverse architecture (2026-07-30):** NVIDIA owns a separate
four-stage package: (1) fp32 state checkpoints per `(batch, head)` trajectory;
(2) an erase-free affine chunk-summary/prefix path; (3) a serial nonlinear or
`erase=true` checkpoint-fill path; and (4) reverse-token gradients with unique
`(batch, head)` ownership for Q/K/V/gate/beta/decay. The modified-normalization
derivative is part of stage 4, not a reuse of an AMD or AVX implementation.
The current CUDA forward recurrence is intentionally unchanged. Do not add a
`nvidia_deltanet_bwd_compiled` execution row or selector candidate until this
CUDA package has exact f32 numerical tests plus SM120 device-event and
end-to-end timing cohorts; ROCm/x86 packets are non-evidence for that decision.

**Implementation update (2026-07-31):** the versioned v2 CUDA package now
keeps its fixed 13-buffer/10-scalar ABI while executing the validated bounded f32
`Dqk,Dv <= 8` reverse recurrence. It has CUDA-owned analytic gate, beta, and
decay derivatives, followed by serial replay for `erase` and modified-update
normalization; direct-package and public JIT exact-device tests compare all six
gradients against the shared oracle. This is correctness evidence only: no
execution-matrix promotion or selector choice has been made until the required
SM120 timing and counter packet exists. Apple, ROCm, and x86 are **not
applicable to this physical package**; their differently scheduled packages
and evidence remain sibling follow-ups rather than CUDA proof.

**Timing/NCU packet (2026-07-31):**
[`nvidia_sm120_deltanet_backward_2026_07_31.json`](../../../../benchmarks/baselines/nvidia_sm120_deltanet_backward_2026_07_31.json)
records two repeated CUDA-event and end-to-end cohorts at `[B,H,S,Dqk,Dv] =
[1,2,16,8,8]`: plain `0.64318 / 1.53015 ms`, affine `0.76717 / 1.54674 ms`,
and erase+modified serial-fill `0.92778 / 1.69752 ms`. All variants match the
oracle (maximum errors `7.45e-09`, `1.86e-09`, and `1.35e-04`). The packet
also records live resources (255 registers/thread, 1328 B local/thread, two
active blocks/SM) and two NCU L2/DRAM cohorts. Only serial-fill counters were
stable (94.77/94.70% L2 and 779,776/715,008 DRAM B); plain and affine counter
cohorts diverged materially, so they are evidence of collection instability,
not a schedule result. The packet changes no selector: there is one
correctness-first implementation and no controlled alternative to promote.

## Cross-backend sync `TILE-FRAGMENT-TYPE-PARAM-2026-08-03` — `!tile.fragment` parameterized (W1.1 step 1)

Shared Tile IR type changed: `!tile.fragment` gained `(m, n, k, elem, acc, role, layout, family)` and a domain verifier. **No behaviour changes in this PR** — the bare `!tile.fragment` still parses AND still prints bare, so every existing producer and fixture is unaffected. All 7 C++ `FragmentType` uses are `isa<>` checks, so there were no construction sites to migrate.

**Outcome: follow-up required.** 8 files under this backend reference `FragmentType` / `!tile.fragment`. Two obligations, both already scoped in [`W1_1_TYPING_DESIGN.md`](../../compiler/W1_1_TYPING_DESIGN.md):

* **Step 2b (blocking for the motivating GEMM).** `NVIDIALowering.cpp` requires the accumulator's direct defining op to be `FragmentZeroOp` (3 sites). A K-loop accumulator is an `scf.for` iter-arg with no defining op, so a typed K-loop will verify and still fail to lower until this is block-argument-aware. Needs a per-backend *lowering* fixture, not just a verifier one.
* **Step 3.** `GenerateWMMA*`-equivalent producers migrate to the typed form one PR at a time.

`family` is a type parameter partly because `NVIDIALowering.cpp` gates on it before matching (m, n, k, dtype) to an `mma.sync` variant — leaving it on the attribute would have let a fragment packed for one family feed an op selecting another.

No exact-device evidence in this PR; none required, since no generated code changed.

## Cross-backend sync `TILE-FRAGMENT-KLOOP-ACCUM-2026-08-03` — typed `tile.mma` K-loop (W1.1 step 2)

Shared Tile IR contract changed: `MMAOp::verify()` (and the `fragment_pack` / `fragment_zero` producers) now read the operand contract from the fragment TYPE when it is parameterized, falling back to producer-chasing for the bare form. `#tile.mma_desc` is optional on the typed path and cross-checked when present. **The canonical K-loop now verifies.** No lowering changed in this PR, and no existing IR is affected — the bare form keeps its old path.

**Outcome: follow-up required — and larger than previously recorded.**

Same finding as ROCm under this key. `NVIDIALowering.cpp` synthesizes zero constants for the accumulator (`operands.append(4, zero)` for f32, and the f16/s32 equivalents) and never reads `mmaData[2]` as a value. So step 2b is accumulator threading plus an `scf.for` region-signature conversion, not a relaxed check — and relaxing it alone would emit a silently wrong GEMM.

No sm_120 device evidence in this PR; no generated code changed. When 2b lands, its gate needs numerics on real hardware for the same reason ROCm's does.

## Cross-backend sync `NVWGMMA-ACCUMULATOR-GUARD-2026-08-03` — WGMMA accumulator drop (W1.1 step 2b guard)

A `tile.mma` carrying an accumulator was lowered by `NVWGMMALoweringPass` to a **two-operand** WGMMA call: the accumulator was discarded, the shape hardcoded `m64n64k16`, and the dtype inferred through `dyn_cast<ShapedType>` (which a `!tile.fragment` is not, so it defaulted to bf16) — with **rc=0 and no diagnostic**. A K-loop recomputed A×B from nothing each step and returned the last partial product.

Measured on merged main, this was **not** specific to the typed fragment form: a legacy bare `tile.mma(A, B, C)` — what `LowerKReductionAddToTileMMA` emits for the canonical K-step — was dropped identically. **No fixture in the tree covered either case**, which is how it survived. The guard therefore keys on *has an accumulator*, not *is typed*.

**Outcome: follow-up required — this backend owned the defect.**

`NVWGMMALoweringPass` now refuses such an mma with `NVWGMMA_ACCUMULATOR_DROPPED` and calls `signalPassFailure()`. The check lives in the PASS BODY, not the pattern: a pattern that emits an error and returns `failure()` only declines to match, so the diagnostic printed while the tool still exited 0 and the pipeline continued — an error that does not fail compilation is a warning in disguise, the same fail-open shape as the bug.

The two-operand lane is untouched: `nvwgmma_lowering.mlir`'s emitted call is byte-identical before and after (diffed, not assumed).

**Follow-up:** W1.1 step 2b threads the accumulator for real, which needs an `scf.for` region-signature conversion. This guard is replaced by lowering then, not relaxed. No sm_120 device evidence here; no working codegen changed, only a refusal added.

## Cross-backend sync `ROCM-COMPILED-STRICT-DISPATCH-2026-08-04` — compiled-lane failures stop masquerading

Runtime dispatch contract changed. A compiled-ROCm **failure** (tessera-opt ran and serialized no kernel, or emitted a non-ELF blob) now routes through the existing `_note_dispatch_fallback` funnel, so `TESSERA_STRICT_DISPATCH=1` raises instead of degrading. **Envelope limits** (no libamdhip64, hipInit failed, tessera-opt not built, dtype/rank/arch out of range) are unchanged and still degrade silently — making those raise would break strict runs on every CPU-only host.

Measured before the fix: a deliberately broken pass pipeline returned `ok=True, compiler_path="rocm_compiled", execution_kind="native_gpu"` with correct numbers. Strict-mode suite results are identical before and after (18 fail both ways, all pre-existing), so this adds no new failures.

**Outcome: not applicable — architecture-specific reason.** The changed sites are all `_RocmCompiledUnavailable` raise points inside ROCm compiled-lane hsaco builders. NVIDIA's compiled lanes do not raise that exception and are untouched.

**Follow-up worth recording, not created here:** NVIDIA has no equivalent failure/envelope split on its own compiled paths, so the same masking may exist there. Establishing that needs an sm_120 host, which this box is not — asserting it either way from here would be the guesswork this thread has been eliminating.

## Cross-backend sync `ROCM-PIPELINE-TILE-LOWERING-2026-08-04` — the compiled pipeline can lower `tile.mma`

Both ROCm compiled pipelines (plain and canonical) now run `lower-tile-to-rocm{arch=<chip>}` after `generate-wmma-gemm-kernel`. Verified byte-identical hsaco with and without the pass on the default path, so the production lane is unchanged.

**Outcome: follow-up required — recorded, not fixed here.** The equivalent NVIDIA seam is worse, not merely missing: `NVWGMMALoweringPass` lowered a `tile.mma` carrying an accumulator to a two-operand call and dropped it (`NVWGMMA-ACCUMULATOR-GUARD-2026-08-03`). That is guarded to fail closed; threading it for real is W1.1 step 2b on this backend and needs an sm_120 host for the numeric gate that ROCm just got.

## Cross-backend sync `TILE-VIEW-BOUNDED-CONTRACT-2026-08-04` — bounded `tile.view` is a shared contract

`ViewOp::verify` now defines the pointer-backed operand contract: exactly 3 `(base, rowOrigin, colOrigin)` or 5 with `(rowBound, colBound)`. It previously accepted any count >= 3, so a 4-operand view was legal and meaningless and the bounded form's validity was decided by whichever backend looked.

**Outcome: follow-up required — refuses the bounded form, and the refusal is UNVERIFIED on this box.**

`NVIDIALowering`'s fragment materializer emits an unguarded load, so ignoring `(rowBound, colBound)` would read past the edge of a ragged matrix. It now emits `NVFRAGMENT_BOUNDED_VIEW_UNSUPPORTED` naming op and target (Decision #21) instead of folding the case into the generic arity message, which would have read as "malformed IR" for IR that is well-formed and merely unsupported here.

**Explicitly not verified:** the NVIDIA dialect is off by default in this build, and neither `--tessera-lower-to-gpu` nor `--tessera-nvidia-pipeline-sm120` reached the materializer with a bounded view on this host — no diagnostic, no error. The code path is written and compiles; it has not been executed. Verifying it needs a build with `-DTESSERA_ENABLE_CUDA=ON` and, for the numeric half, an sm_120 host.

Until this materializer grows masking, a portable producer must emit the 3-operand form; only ROCm can consume the bounded one.

## Cross-backend sync `TILE-VIEW-LINEAR-BASE-2026-08-05` — should `tile.view` carry a precomputed linear base?

ROCm W1.1 step 3 (`W1_1_TYPING_DESIGN.md` §4.7) established that isolated
fragment address derivation could not express the direct lane's shared row
offset. Measurement selected an optional precomputed `linear_base` operand on
`tile.view`; logical row/column origins remain present for bounds.

**ROCm has implemented and measured the shared answer.** `tile.view` now accepts
an optional precomputed `linear_base`, and the ROCm producer uses it for A-base
hoisting and sibling-B address sharing. At 2048^3 the final rebuilt ratio is
0.711x (15.45/21.74 TFLOP/s), still insufficient for promotion. The
remaining ROCm evidence points to fragmented load scheduling and excess waits.

**Outcome for NVIDIA: FOLLOW-UP REQUIRED.** This backend consumes `tile.view`
and `tile.fragment_pack` (8 files each), so it must consume the optional base
form correctly or fail closed when a producer selects it. No exact-device
evidence from an NVIDIA host is claimed. This is also the owner of W1.1's final two untyped C++ `tile.mma`
construction sites: both are tensor-valued producers in
`TileIRLoweringPass`. They must migrate together with NVIDIA's typed
tensor-to-fragment materializer and accumulator-threading gate; ROCm/x86
hardware cannot validate that physical decision.

## Cross-backend sync `TILE-DYNAMIC-LEADING-DIM-2026-08-04` — generic typed fragment addresses

Shared `tile.view` / `tile.store` can now carry an SSA leading dimension when
`#tile.memory_layout` states zero. **Outcome for NVIDIA: FOLLOW-UP REQUIRED.**
The sm_120 fragment materializer fails closed on this valid shared form and
still requires a static leading dimension. No CUDA-enabled build or sm_120
device evidence is claimed from this ROCm/x86 host.

## Cross-backend sync `E2E-REAL-LINEAGE-SCHEDULE-2026-08-05`

Shared compiler orchestration now records explicit artifact ancestry and
production `tessera-opt` registers the generated Schedule dialect. **NVIDIA
outcome: follow-up required.** SM120 packaging still consumes `GraphIRModule`
and synthesizes NVIDIA-owned Tile, so its lineage truthfully records the Graph
fork and remains incomplete. This change does not migrate the two untyped
`tile.mma` producers, accumulator threading, bounded/dynamic views, PTX, or a
physical schedule; no SM120 evidence is claimed. Those remain NVIDIA-owned
gates after the shared x86/ROCm vertical slice establishes the consumer API.

## Cross-backend sync `E2E-REAL-SCHEDULED-MATMUL-2026-08-05`

Shared Graph→Schedule→launch-Tile lowering is now real for the initial x86-f32
and ROCm-f16/f32 matmul instances. **NVIDIA outcome: follow-up required, not
validated by this slice.** SM120 is deliberately outside the first bounded
dtype/descriptor selector and therefore fails closed rather than inheriting
another architecture's schedule. NVIDIA packaging still synthesizes its Tile
program from Graph IR. Its later consumer must accept the canonical scheduled
artifact only after the final two untyped `tile.mma` producers, accumulator
threading, and dynamic/bounded view gates pass in a CUDA-enabled SM120 lane.

## Cross-backend sync `E2E-REAL-PHYSICAL-CONSUMERS-2026-08-05`

The shared package boundary is now concrete: a validated
`ScheduledMatmulArtifact` carries exact Graph, Schedule, and launch-Tile text
plus content identities into x86 and ROCm physical consumers. **NVIDIA outcome:
follow-up required.** No CUDA code changed and no SM120 evidence is claimed.
The NVIDIA consumer remains blocked on its two untyped `tile.mma` producers,
accumulator threading, dynamic/bounded view support, and CUDA-enabled SM120
validation before it can adopt this boundary without recreating Graph intent.

## Cross-backend sync `E2E-REAL-PERFORMANCE-2026-08-05`

Schedule/Tile matmul now distinguishes instruction-tile shape from an
architecture-owned macro tile and carries that value through artifact identity
and launch provenance. **NVIDIA outcome: follow-up required.** This generic
contract is applicable, but no CUDA lowering, SM120 schedule, PTX, selector, or
device evidence changed. NVIDIA must choose its own macro tile after its two
untyped producers, accumulator threading, and bounded/dynamic view gates land;
the gfx1151 32x64 decision does not transfer.

## Cross-backend sync `E2E-REAL-SEMANTIC-KERNELS-2026-08-05`

The shared spine now has content-addressed `schedule.softmax` and
`schedule.reduce` SSA edges and atomically lowers the bounded canonical f32
contracts to launch-level Tile artifacts. **NVIDIA outcome: follow-up required
on a CUDA-enabled SM120 host.** Existing NVIDIA physical Tile softmax/reduction
lowering is unchanged, but SM120 packaging still synthesizes its Tile program
from `GraphIRModule`; no PTX, cubin, descriptor, schedule, selector, or device
claim changed. Its consumer must accept the exact scheduled artifact and run
the established SM120 numerical/performance gates. x86/gfx1151 schedules and
evidence do not transfer. Canonical Graph reduction currently excludes
mixed-output and keepdims forms; widening that shared contract is separate
from adopting this first f32 boundary.

## Cross-backend sync `E2E-REAL-ATTENTION-2026-08-05`

The shared spine now defines a content-addressed `schedule.attention` edge and
one launch-level Tile artifact for the bounded x86/gfx1151 instances.
**NVIDIA outcome: follow-up required on a CUDA-enabled SM120 host.** Existing
SM120 forward packaging still synthesizes an NVIDIA-owned Tile program from
Graph IR; no PTX, cubin, LSE policy, schedule, selector, or device evidence
changes here. NVIDIA must define its own schedule/LSE instance, consume the
exact artifact, and run its forward numerical/performance gates. x86 and
gfx1151 policy or evidence does not transfer.

## Cross-backend sync `E2E-REAL-ATTENTION-BACKWARD-2026-08-05`

The shared spine now defines a content-addressed three-result
`schedule.attention_backward` program carrying dQ, split-dK/dV, fixed reduction,
workspace, and LSE checkpoint identity. **NVIDIA outcome: follow-up required on
a CUDA-enabled SM120 host.** Existing SM120 backward packaging remains
NVIDIA-owned Graph-to-Tile synthesis. NVIDIA must define its own LSE identity,
consume this exact artifact, and validate MHA/GQA/MQA plus modifier/ragged
coverage before claiming parity; x86/gfx1151 schedules and evidence do not
transfer.

## Cross-backend sync `ROCM-TYPED-EXECUTABLE-PIPELINE-2026-08-07`

The shared orchestration direction now has a concrete typed configuration:
family, input artifact level, output artifact level, architecture, Tile
producer, Target-IR consumer, and backend code generator. **NVIDIA outcome:
follow-up required.** CUDA must define its own SM-specific family plugins around
the canonical Schedule/Tile artifact and NVVM/PTX/cubin code generator; this
change does not transfer gfx1151 scheduling or AMD wait semantics and supplies
no CUDA-enabled or SM120 evidence. The existing Graph-owned synthesis lane is
unchanged. NVIDIA accepts the shared strict-boundary policy (no surviving Tile
or Target IR, undefined result, or contract-marker symbol), but enforcement in
the NVVM/PTX/cubin pipeline remains CUDA-owned follow-up.
ROCm has now retired its final generic runtime pass-name helper. **NVIDIA
outcome: follow-up required:** CUDA family plugins must likewise expose a
closed semantic registry rather than an arbitrary pass option. No PTX/cubin,
SM schedule, selector, or device evidence changes here.

## Cross-backend sync `TSOL-PACKED-FUSION-2026-08-08`

The shared `schedule.spectral_program` contract now hashes packed-real fusion
topology and N/2 child identity. **NVIDIA outcome: follow-up required on a
CUDA-enabled host.** No NVVM/PTX/cubin consumer changed. NVIDIA must select its
own real-transform plan and carry the exact v5 artifact through its physical
package; Zen 5 and gfx1151 schedules, workspaces, and evidence do not transfer.

## Cross-backend sync `TILE-SYNC-RECONCILE-2026-08-10`

`tile.async_copy`/`tile.wait_async` now have one declared contract (ODS dual
form, `TileOps.td`): typed `!tile.async_token` SSA is production; legacy
grouping keys are the declared envelope, optional and conservative on absence.
New shared diagnostic `TILE_ASYNC_STAGE_NEGATIVE`. **NVIDIA outcome: parity
validated at the core-IR level.** The typed-token SSA edge is the NV warp-spec
production model (`TileIRLoweringPass::emitAsyncCopy`,
`tessera-warpspec-legality`); the previously contradictory required-stage
verifier was the one rejecting production NV Tile IR, and
`phase2/pm_verify_async_token.mlir` (red at baseline) now passes. No
NV-device-lane (`tessera-nvidia-opt`) fixtures changed.

## TILE-SYNC-TYPED-2026-08-15 — shared Tile sync ABI assessment (PR #566)

**Follow-up required.** The retyped family is NVIDIA's vocabulary:
`tile.mbarrier.wait` gains an optional `!tile.mbarrier_token` segment (the
operand-segment ABI is now barrier/token/dependencies), `tile.tma.copy_async`
and the wait grow fail-closed gates-on-nothing verifiers, the keyless legacy
wait is an explicit `tile.retire_all` marker (stamped by AsyncCopyLowering,
resolved — or preserved when no completion tokens exist — by
NVTMADescriptorPass), and `--tessera-tile-dataflow-legality` runs in every
NVIDIA pipeline after the post-NVTMA legality blocks. Host-free evidence:
full lit 324/0 including the pipeline-alias and NVTMA fixtures. **Open on
this queue:** sm_120-host revalidation of the NVTMA pipelines and any
SM90/Hopper device proof (Phase G/H). The barrier-at-birth compiler restructure
is host-free parity validated by the complete Phase 3 IR lane (24 supported
tests passed, 7 unsupported), including the named NVIDIA pipeline, streaming
FlashAttention, tokenless retire-all compatibility, and distinct barriers with
local slot zero across two schedule regions. Exact-device evidence is still
required before closing the hardware rows.

## REF-TIER-OPS-2026-08-15 — reference-tier op registration assessment (PR #568)

PR #568 registered ten new public operations through the canonical op catalog
and the primitive coverage registry — `tridiagonal_solve` (Thomas recurrence,
PDE plan §III.1 / TSOL-A1) and the nine-op coalition-lattice family
(`game_subset_zeta`, `game_subset_mobius`, `game_superset_zeta`,
`game_superset_mobius`, `game_coalition_marginal`, `game_semivalue`,
`game_boltzmann_value`, `game_coalition_excess`, `game_mex`). Op registration
is a shared contract, so this queue records the outcome per AGENTS.md
"Cross-backend work coordination"; PR #568 itself landed without these records.

**Follow-up required — no NVIDIA lane exists for any of the ten.** No
`tessera-nvidia-pipeline-{sm90,sm100,sm120}` stage, PTX emitter, or backend
manifest row consumes either family; the declared tier is the Python
reference. GAME_THEORY_PLAN.md G5 names the per-target arbiter registration
(SubsetZetaRegion beside SpectralFFTRegion, `boltzmann_value` on the
online-softmax emitter) as the NVIDIA-side entry point for the lattice family;
the solver's entry point is the PDE op-set admission, not G5. No sm_120 host
revalidation was run for this registration and no device evidence is claimed —
nothing here changes generated code.

## APPLE-SCHEDULED-REDUCE-NAN-2026-08-16 — shared reduce NaN semantics (PR #571)

**Shared contract changed; assess before relying on extrema reductions.** The
synthesizer's reduce vocabulary (`compiler/fusion_core.py::_PW_REDUCE_KINDS`)
emitted `max(acc, v)` / `min(acc, v)` for `amax`/`amin`. Metal's `max`/`min` are
IEEE maxNum/minNum-style and **suppress** a NaN operand, so the emitted kernel
disagreed with the table's own numpy reference (`a.max(-1)`, which propagates)
and with the `nan_mode = "propagate"` the reduce Schedule artifact declares.
With the `-INFINITY` seed an all-NaN row reduced to **`-inf`** — missing data
silently becoming a finite extreme. The accumulators now propagate explicitly.

**NVIDIA outcome: not applicable — no consumer.** `_PW_REDUCE_KINDS` supplies MSL
accumulate expressions consumed only by `compiler/emit/apple_msl.py`; SM120
reduction lowering is untouched. No CUDA schedule, PTX path, ABI, or exact-device
claim changes. The NaN-propagation contract is worth noting for any future
`emit/nvidia_cuda.py` reduce emitter: CUDA's `fmaxf` suppresses NaN the same way,
so a naive port would reintroduce the same divergence from the declared
`nan_mode = "propagate"`.

Also recorded for coordination: PR #571 admits Apple GPU into the shared
scheduled reduce contract (`scheduled_kernel.py`, last axis only) and closes
APPLE-DEVICE-EVENT-1 by giving the Apple MPSGraph BMM route an owned command
buffer. Both are Apple-guarded — the shared `scheduled_kernel` gate adds an
`apple_gpu` branch beside the existing x86/ROCm ones and changes neither — and
the runtime edit is in `apple_gpu_runtime.mm`, which no sibling links.

## APPLE-ATTN-BWD-PERF-1-2026-08-16 — backward row-prepass assessment (PR pending)

**Outcome: not applicable to this architecture — no shared contract changed.**
Apple's attention-backward dK/dV split kernel became key-parallel by finally
implementing the `row_prepass` stage that
`attention_contract.plan_attention_backward_workspace` **already declares**
(`row_lse`, `row_delta`, consumed by `dkdv_split` and `dq`). The change is
confined to Apple-private MSL in `apple_gpu_runtime.mm` and its dispatch; no
shared IR, Schedule contract, workspace plan, dtype, ABI, or sibling schedule
was modified. The declared schedule is unchanged — still `split_count = 2` with
ascending `reduction_order = (0, 1)`; only Apple's thread mapping changed, which
is architecture-owned.

**Worth reading anyway, because the shape of the bug is portable.** The declared
workspace stage existed and had no consumer, so every kernel recomputed the row
statistics inline. That is not merely duplicated work: because the statistics
reduce over the whole key axis, an inline-recompute kernel must own an entire
query stream, which capped the dK/dV split at one thread per (partial, KV batch)
— 4 threads at `B1 Hq4/Hkv2 S64`. Implementing the declared prepass raised it to
`2 * kv_outer * Sk` and moved backward from 195 ms to 6.0 ms (~32x), while
keeping single-owner determinism. Any backend whose backward recomputes row
statistics inside its dK/dV kernel should check whether it has the same
structural cap before attributing slowness to memory or instruction mix.

## APPLE-NORM-VJP-1-2026-08-16 — Apple native-VJP registration assessment

**Outcome: not applicable — no shared contract changed.** Apple registered as a
Target consumer in the existing native-VJP normalization plugin and added its own
Metal MSL VJP kernels. The shared registry schema, the `_parse_compiled_norm_backward`
contract, the Graph/Schedule contracts, and every sibling executor are unchanged;
the new code is Apple-private (`apple_gpu_runtime.mm`, one runtime executor, two
`execution_matrix` rows behind one new Apple executor id). This target's own
normalization VJP rows, evidence and selectors are untouched.

Recorded because the *scoping* lesson is portable: the item was scoped as
"registry wiring", but the target had no normalization-backward ABI at all, so
the entry would have been an unconsumed declaration (Decision #29). Any backend
asked to join this boundary should confirm it owns an executable VJP before
declaring a consumer.

## APPLE-NORM-VJP-2-2026-08-16 — reduced-precision normalization VJP assessment

**Outcome: not applicable — no shared contract changed.** Apple's normalization
VJP gained f16/bf16 storage: four new Apple-private C ABI exports
(`tessera_apple_gpu_{rmsnorm,layer_norm}_bwd_{f16,bf16}`), their MSL, the
non-Darwin stub returning 0, and Apple's own ABI registry rows. The shared
native-VJP registry schema, `_parse_compiled_norm_backward`, the Graph/Schedule
contracts, the execution-matrix executor id, and every sibling executor are
unchanged. This architecture's normalization VJP dtype support, evidence and
selectors are untouched.

Two portable notes, recorded because both are cheap to get wrong:

1. **Store rounding is part of the numeric contract.** The kernel accumulates in
   f32 and stores back at the operand's own dtype, and bf16 stores use
   round-to-nearest-even. Truncating instead would bias every gradient in one
   direction — a systematic error, not noise. Any sibling adding reduced-precision
   gradients should match its own runtime's established rounding convention
   rather than defaulting to a shift.
2. **A tolerance derived from storage epsilon is what proves the accumulator.**
   Measured error is ~0.5-0.65 ulp of each format's epsilon, i.e. one rounding on
   store; storage-format accumulation would land near `sqrt(cols)` ulps. A test
   asserting against that second level catches a silently dropped f32
   accumulator, where a hand-picked loose tolerance would absorb it.

A third note is Apple-specific but the shape recurs: adding an export obliges
updating the dylib **freshness gates**, or a runtime built between two landings
exports the older names, passes the staleness check, and then fails at the call
site — which reads as a broken consumer rather than the stale build it is.

---

## Cross-backend record — MC1 matrix-function family (PR #596, 2026-08-20)

**Owning item:** MATRIX-CALCULUS-MC1 · **synchronization key:** `MC1-LINALG-FAMILY`

**Shared contracts changed.** Ten public op registrations (`det`, `logdet`,
`inv`, `solve`, `trace`, `eigh`, `kron`, `vec`, `matrix_power`, `norm`); two
new Graph IR lowering kinds (`linalg_function`, `linalg_multilinear`) with four
new shape rules (`matrix_scalar`, `vec`, `kron`, `eigh`); numeric-policy
entries for both kinds; two diagnostic codes (`E_LINALG_CONTRACT`,
`E_METRIC_CONTRACT`); the `degeneracy_policy` semantic key extended to
eigenvalue gaps and to the nuclear norm's rank condition.

**Outcome for this backend: `not applicable — no lane attempted or implied`.**

The family landed as a *derivative contract* — closed-form VJP+JVP pairs under
the AD law sweep — because that was the gap in the AD stack. No Tile IR
lowering, Target IR op, or kernel was added for any backend, and none is
implied: every one of the ten is registered `backend_kernel: partial` /
reference tier, and each is listed in the Apple no-lane golden and the
single-GPU closeout classifier as an *intentional* reference-only decision with
a stated rationale, so none of them appears on this backend's promote queue as
a phantom blocker.

**What a sibling picking this up would need to decide, per op.** The reference
implementations delegate to LAPACK through numpy, so a native lane is a
vendor-library question rather than a codegen one for `eigh`/`inv`/`solve`
(rocSOLVER / cuSOLVER / Accelerate / MKL), and a "probably never worth it"
question for `det`/`logdet`/`trace`/`matrix_power`, which reduce a small matrix
to one number. `kron` and `vec` are layout/contraction shaped and would ride
existing lanes if anything.

**Numeric policy this backend must match if it does build a lane.**
`linalg_function` and `linalg_solver` declare **f64 accumulate with f32 storage
admitted** — the same conditioning-sensitive policy `linalg_decomposition`
already carries, because `det`/`logdet`/`inv`/`matrix_power` carry a factor of
the condition number. `linalg_multilinear` (`trace`, `kron`) declares the
ordinary f32 accumulator, since neither carries one. A lane that accumulates in
storage precision would silently lose the digits these rules exist to keep.

**Degeneracy contract that travels with the rules.** `eigh`'s eigenvector
coupling `1/(w_j - w_i)` and `svd`'s `1/(s_j^2 - s_i^2)` have no limit at a
crossing; both fail closed under the declared `degeneracy_policy` rather than
emitting `inf`/`NaN`. Any native lane must reproduce that refusal, not paper
over it with an epsilon — a damped coupling returns a finite, plausible, wrong
gradient, which is the failure mode the key exists to prevent.

**Exact-device evidence: none, and none claimed.** Everything in PR #596 was
validated on the host-independent Python reference lane on an M1 Max, where no
AMD GPU, no CUDA device and no AVX-512 exist. No parity claim is made for any
backend.

**NVIDIA-specific note.** cuSOLVER covers `eigh`/`inv`/`solve` directly; the
open design question is whether they enter through Target IR as `abi_call`-style
delegated work (Decision #19's x86 precedent) so the arbiter can still tell
compiler-generated work from delegated work.


## Cross-backend sync `NVIDIA-SPECTRAL-PHILOX-JVP-2026-08-22`

**Owning work:** native spectral sequencing and Philox compiler-JVP integration.
**Outcome: landed and exact-device validated on SuperBear RTX 5070 (sm120).**

The canonical CUDA FFT/workspace ABI is now v2 and owns reusable typed C2C,
R2C, and C2R cuFFT plans with automatic allocation disabled, exact caller-owned
workspace bounds, and on-device inverse normalization. Native `rfft`/`irfft`
cover odd, even, mixed, and prime lengths. DCT-II, STFT, ISTFT, spectral
convolution, and spectral filtering consume this package; framing, windowing,
pointwise work, and overlap-add remain explicit host orchestration and are not
reported as fused CUDA kernels.

The compound spectral reverse package now has an SM120 consumer. Real spectral
convolution VJPs use CUDA R2C/C2R and match direct correlation; the complex
spectral-filter adjoint is exact-package validated. Public NVIDIA Graph IR
`complex64` storage remains planned-gated, so filter VJP is artifact-level
physical evidence rather than a public Graph execution claim.

Compiler JVP packaging now admits exact `nvidia_sm120`. Seeded dropout binds
both primal and tangent children to the identical Philox key/counter attributes,
proving mask replay rather than resampling; unseeded training JVP fails closed.
Exact-device evidence: 22 tests in the three NVIDIA fixtures.


Cross-backend sync `NUMPOL-CARRIER-1-SCHEMA-AND-REDUCTION-2026-08-25` — **the
policy gets a schema and the reduction family carries its accumulator;
NVIDIA outcome: shared contract only; no CUDA change.** Two measured defects closed, both shared,
neither architecture-specific.

*Schema.* `numeric_policy` was a bare `DictionaryAttrBase` whose ODS predicate
checked only "is a dictionary". Measured before the change: five malformed
policies were all ACCEPTED while the documented TF32-as-storage violation
correctly failed, so the pass was running and simply had nothing to say. The
sharpest was a typo — `getAs<StringAttr>("accum")` returns null for a
misspelled key exactly as for an absent one, so an op carried a policy that
looked like it stated an accumulator contract and stated none. Seven new
diagnostics now refuse unknown keys, non-string values, unknown dtypes/modes, a
math_mode that does not reduce its storage, and an accumulator NARROWER in
significand bits than its storage.

*Reduction carrier.* `{storage="bf16", accum="fp32"}` on rmsnorm / softmax /
layer_norm lowered to `arith.addf … : bf16` — the emitted code contradicted the
declared contract on the very op that performs the accumulation. Executed on
Zen 5 through `--tessera-to-linalg` → LLVM → native object: a 4096-wide softmax
row summed to **1.466** (a 47% violation of the function's defining property)
versus **1.000169** once the declared accumulator is honoured. The whole
derived chain now runs in the accumulator with a single truncation at the
result — chosen by measurement over truncating the reduced value, which is 326x
worse. With no policy the emitted IR is byte-identical, so nothing widens
without being asked. The Graph→Linalg boundary is now bracketed by the
Decision #32 record/verify pair and declares `represented_in_type` /
`re_expressed`.

The schema and the reduction carrier are target-independent Graph-level contracts. Zen 5 execution transfers no sm_120 claim.

**`math_mode` now has a consumer on the runtime dispatch, and closing it exposed an INTERNAL CONTRADICTION on this backend — NR2 Pro proof owed.** This queue already records (above) that "the shared MMA selector now requires explicit `math_mode=\"tf32\"` for fp32". `runtime.py` did not: `_NVIDIA_GEMM_SYMBOLS` mapped `"float32"` to `tessera_nvidia_mma_gemm_tf32` and the dispatch took it **unconditionally**, with a comment citing Decision #15a as the reason. So two components held opposite policies for the same fact, and the permissive one is the one that executes. #15a says "TF32 is not a storage dtype. Model as `math_mode='tf32'` on fp32 via numeric_policy" — precisely so the reduced arithmetic is a choice the program makes; the storage dtype was making it instead, and the comment made the violation read as compliance.

Measured against an fp64 reference on 64xKx64 GEMMs (median relative error): fp32 1.64e-07 → tf32 2.93e-04 at K=128 (**1783x**); 3.02e-07 → 3.01e-04 at K=1024 (998x); 3.63e-07 → 2.91e-04 at K=4096 (800x). TF32 keeps 11 significand bits against fp32's 24 and rounds the OPERANDS, so no accumulator width recovers it. A program that asked for fp32 got tf32 numbers and no diagnostic.

Selection is now the pure function `runtime._nvidia_gemm_selection`, so its contract is host-testable on a box with no CUDA: explicit `math_mode="tf32"` selects the tf32 kernel, `"ieee"` (or any mode the lane lacks) is refused with `NVIDIA_MATH_MODE_UNAVAILABLE` rather than handed tf32 numbers under an fp32 label, and narrow-storage paths are untouched. This makes the runtime agree with the selector contract this backend already adopted — it is not a new policy.

**Absent `math_mode` deliberately keeps today's TF32 behaviour**, and that is the owed decision rather than an oversight. By #21a a semantic key should fail closed on absence; changing the default would alter every existing fp32 NVIDIA program from a host that cannot execute one, which is exactly the claim the fleet rule forbids. A test pins the current behaviour so the follow-up has a fixed baseline. **Owed on NR2 Pro (RTX 5070 Ti):** execute the tf32 and f16 lanes, confirm the selection reaches the intended kernel, and decide whether absent-`math_mode` should fail closed.

Also on this backend, not device-proven here: the sm_120 typed route's `!tile.fragment` accumulator now follows `selected->accum` instead of a hardcoded `"f32"` written three times beside that unused field, and a declared accum the schedule cannot provide fails closed with `MATMUL_SCHEDULE_ACCUM_UNSUPPORTED`. Byte-identical across all 169 Graph→Schedule→Tile fixtures under a before/after control.

---

## Cross-backend sync `DELTANET-BOUNDED-VJP-2026-08-31`

**Owning item:** the bounded ("modified"/Kimi) DeltaNet reverse rule ·
**PR:** #660 · **synchronization key:** `DELTANET-BOUNDED-VJP-2026-08-31`

**Shared contract changed — the bounded VJP's correction term.** The bounded
variant scales the rank-1 update by `f = 1/(1 + n)` with `A_de = k_d·target_e`
and `n = ‖A‖_F`, so the reverse rule for `U = b·A·f` is

```
∂L/∂A_ij = b·dU_ij·f  −  b·(Σ_de dU_de·A_de)·f²·A_ij / n
```

This is a *shared numerical contract*, not a per-backend schedule: three
backends implement the same closed form independently against one reference
(`get_vjp("modified_delta_attention")`). Two of them divided that correction by
**`max(n, 1)`** instead of `n`, understating it by a factor of `n` whenever
`n < 1` — which, with L2-normalised keys, is the ordinary case rather than an
edge case. No clamp is needed or wanted: the numerator is `O(n²)` (both
`update` and `projection` scale with `A`), so the quotient vanishes as `n → 0`,
and the existing `norm > 0` select already covers `n == 0` exactly.

**Why the failure was silent in four of six gradients.** At `erase=False` the
bounded update reaches only `dk` and `dv`; `dq`, `dgate`, `dbeta`, `ddecay` do
not consume `du` and stayed exact to ~1e-9. A wrong *bound* derivative
therefore presents as two gradients off by 19–33% while the other four are
perfect — which is what ruled out precision loss and pointed at a missing term.

**Outcome for this backend: `parity validated` — the formula was already
correct, but the exact-device evidence for it was not, and that was fixed
here.** `_deltanet_backward_source` in `python/tessera/compiler/
nvidia_training.py:1115` already divided by `(norm * denom * denom)`, so CUDA
never carried the defect. Confirming that by inspection is not evidence, so it
was measured on **Super-Bear / RTX 5070 (sm_120)**, and the measurement found a
second, separate problem.

**The NVIDIA device test could not have failed.**
`test_erase_modified_deltanet_serial_fill_backward_executes_on_sm120` asserted
at `rtol=5e-3, atol=5e-3` over gradients whose full scale is 1e-3..3e-2. Two of
the six — `dgate` (max 1.07e-03) and `ddecay` (max 2.20e-03) — are *entirely
below* that atol, so those assertions would pass for any kernel output
including all zeros.

Proven by mutation rather than argued: injecting this PR's exact defect
(`fmaxf(norm, 1.0f)`) into the CUDA source left the suite **green**. A control
mutation (a gross 2x on `dupdate`) failed immediately, which rules out a stale
cubin and locates the fault in the assertion rather than the toolchain. Both
tolerances are now `rtol=1e-5, atol=1e-8`, chosen from the measured
device-vs-reference deviation of **4.7e-10 abs / 2.0e-07 rel** (f32 round-off,
~50x margin) rather than picked by feel. Re-verified on sm_120: correct kernel
**2 passed**; same injected defect now **fails on `dk` at 13.6% relative**, the
same signature ROCm and x86 showed at 19–33%.

**Standing lesson.** A sibling backend that is green on a shared numerical
contract has not been assessed until its test is shown to be capable of going
red. This is another instance of the hollow-green pattern the fleet ledger
tracks, and the first found by mutating a device kernel. The cheapest general
form of the check needs no mutation at all: compare a test's tolerance against
the *magnitude of the quantity it asserts on*: an atol above full scale is a
vacuous assertion on its face.

---

## Cross-backend sync `NVIDIA-TIMER-DRAIN-2026-08-31`

**Owning item:** the NVIDIA half of the three-clock timing discipline
(`AUTOTUNE-RACED-FIELD-SYNC-2026-08-30`'s recorded follow-up) ·
**synchronization key:** `NVIDIA-TIMER-DRAIN-2026-08-31`

**Shared contract changed — what makes a device latency believable.** Every
backend's autotune verdict rests on one, and the corpus compares them across
architectures, so the rule for accepting a device clock is shared even though
each host's clocks are not.

**The finding: the drain and the cross-check are *ordered*, and adding the
second without the first is worse than adding neither.** Measured on sm_120
(RTX 5070) with 2500 2048³ GEMMs resident on a blocking stream, timing 40
launches of a 1024³ GEMM:

| start event recorded | wall ms/rep | event ms/rep | event/wall |
|---|---|---|---|
| without a preceding drain | 63.3338 | 0.3227 | **0.005** |
| after a drain | 0.3263 | 0.3255 | **0.998** |

The event is correct in both rows. Undrained, the start event is queued behind
the contending work, so the *wall* spans that drain and the event does not —
the two clocks bracket different regions and comparing them is meaningless.
Apply the two-sided band to the undrained row and it rejects the correct
0.3227 ms event (its lower bound is 31.7 ms) and falls back to the 63.33 ms
wall: a **196× overstatement**. In review, "adds a wall cross-check" reads as
strictly safer; here it would have been strictly worse.

**Outcome for this backend: `parity validated` — defect owned and fixed here,
on device.** `_nvidia_mma_gemm_device_latency` recorded two events around its
launches and returned the elapsed value with **no validation of any kind**: no
wall clock to compare against, and no drain that would have made a comparison
mean anything. It now goes through `_nvidia_timed_launch_ms`, which drains,
takes a wall witness, and band-checks. Every sm_120 shape reports
`device_event`, as expected.

**Two assumptions this plan carried by analogy did not survive measurement.**

* *"Whether sm_120's event clock is trustworthy is an open question."* It is
  trustworthy: `event/wall` measured **0.996–0.998** idle and contended. HIP's
  lying-clock rationale does not apply, so `_accept_nvidia_event_ms` is a
  separate function from ROCm's `_accept_device_event_ms` rather than a shared
  one. The band is identical; merging them would force one host's rationale
  onto the other, and the next person to widen one would silently widen both.
* *"Never time on the default stream"* does not transfer as stated. Legacy
  stream 0 **does** serialise against a blocking stream here — the wall
  inflated **161×** and the contending stream had drained by the end of the
  region, both signatures of it — but the CUDA **event was unmoved (1.01×)**,
  because the event pair brackets only its own stream's span. Moving the timed
  launches to a dedicated `CU_STREAM_NON_BLOCKING` stream measured **8.65×
  slower** (2.7667 vs 0.3199 ms): a truthful measurement of a kernel sharing
  the GPU, and the wrong number for an autotune verdict. For isolated latency
  the serialisation is a *feature*. A dedicated stream is required for
  concurrency benchmarks — which these timers are not.

**Correction to `NVIDIA-TIER-PRIORITY-IS-WRONG-AT-SCALE-2026-08-30`: "the
compiled kernel wins at every shape" is wrong at the smallest shape.** The
delegate half of that comparison came from the undrained timer, which
over-reads most where the kernel is shortest — measured **−25.4%** at 256³
against **+0.5…+2.6%** at 512³/1024³/2048³. Re-raced with the corrected timer,
12 interleaved samples per lane at 200 reps:

| shape | delegate | emitted PTX | verdict |
|---|---|---|---|
| 256³ | 0.01300 median (sd 14.5%) | 0.01057 median (sd 39.1%) | **not separated** — 18.7% apart against a 39.1% spread |
| 384³ | 0.02078 (sd 7.8%) | 0.01365 (sd 5.5%) | emitted, 1.52× |
| 512³ | 0.04402 (sd 2.2%) | 0.02695 (sd 0.6%) | emitted, 1.63× |
| 1024³ | 0.32102 | 0.19366 | emitted, 1.66× |
| 2048³ | 2.45369 | 1.47545 | emitted, 1.66× |

So the standing claim holds from 384³ up and **256³ is a tie**: that shape sits
at the launch-overhead floor, where a lane's own run-to-run spread exceeds the
gap between lanes. The earlier 1.63× there was an undrained delegate number
plus a three-sample read. The arbiter should not record a matmul winner at 256³
without a separation check — a median difference smaller than the spread is not
a verdict, and this is the second time a matmul ranking has been wrong at a
shape nobody re-measured.

---

## Cross-backend sync `AUTOTUNE-SEPARATION-2026-08-31`

**Owning item:** the arbiter's measured-verdict contract ·
**synchronization key:** `AUTOTUNE-SEPARATION-2026-08-31`

**Shared contract changed — a `MeasureRecord` must now say whether its ranking
survives its own noise.** All four backends write into and read from this
corpus, so the rule for what counts as a verdict is shared even though the
hardware producing the numbers is not.

**The defect.** `measured_arbitrate` took one median per candidate and picked
`min`. A margin was a margin; nothing recorded how noisy the lanes were, so a
19% gap between two lanes whose own spreads were 14.5% and 39.1% was stored
exactly like a 40% gap between lanes spread 2.2% and 0.6%. Measured on sm_120
at 256³:

| lane | median | spread |
|---|---|---|
| delegate (`nvidia_mma_gemm_shipped`) | 0.01300 ms | 14.5% |
| emitted PTX | 0.01057 ms | 39.1% |

That was recorded as a clean **1.63× win**. The same *ratio* at 2048³ is real.
The ratio is not what distinguishes them — the spread is, and the record did
not keep it.

**Measured over the committed corpus, so this is not a hypothetical:** 87 rows,
of which **75 assert a ranking** (two or more candidates timed), **none**
declares a separation, and **11 of the 75 — 15% — picked a winner that beat the
runner-up by under 2%**, the tightest at **0.07%**. That is inside ordinary
end-to-end wall jitter; those eleven verdicts record which lane was luckier.

**The rule.** The margin must exceed `SEPARATION_FACTOR` (2×) the *noisier* of
the two fastest lanes. `None` — not `True` — when fewer than two candidates were
timed: a sole candidate is chosen by applicability, not by a race, so there is
no margin to defend. As with `unmeasured`, absence is not the favourable
answer, and a publisher must treat `None` and `False` alike.

**A tie never blocks dispatch** — something has to run. It blocks two things:
*claiming* one candidate is faster, and re-picking by noise on every run. An
unseparated re-race now keeps the incumbent; a separated one still displaces it.

**Outcome for this backend: `parity validated` — the defect was found here and
the sm_120 evidence for it is this backend's.** All eleven under-2% rows in the
committed corpus are NVIDIA rows, which is an artifact of NVIDIA being the only
backend with four matmul candidates racing at every bucket, not of anything
CUDA-specific.

**Consequence for the corpus: it is now under-declared, not wrong.** Every
committed row reads `separation: None` — never asked — so no verdict has to be
retracted, but the eleven tight rows should not be cited as rankings until
re-raced. Re-generating them needs sm_120, and is the follow-up this key owns.

**Two NVIDIA-specific notes for whoever re-races.**

* The 256³ matmul bucket is at the launch-overhead floor and may simply never
  separate. That is a legitimate outcome to record, not a measurement to keep
  retrying at higher `reps` until one lane wins.
* `device_repeats` (default 3) triples device-timing cost, which matters most
  here because NVIDIA races the widest field. Three samples give a usable noise
  floor but a poor spread *estimate*; where a bucket lands near the bar, raise
  it for that run rather than trusting the boundary.

---

## Cross-backend sync `APPLE-TIMER-WITNESS-2026-08-31`

**Owning item:** the Apple half of `NVIDIA-TIMER-DRAIN-2026-08-31`, recorded
there as follow-up required · **synchronization key:**
`APPLE-TIMER-WITNESS-2026-08-31`

**Shared contract changed — a device latency must be checked against a clock
that did not produce it.** ROCm and NVIDIA already did; Apple did not, and had
no host clock at all to check against.

**Two findings from the measurement, both of which changed the design.**

*The witness must bracket the same region the device clock does.* The first
version instrumented `commit_and_wait_with_timeout` only — and the lane this
workload actually takes (`metal4_mpsgraph_envelope`) does not go through it.
It reported a null wall, which the acceptance rule reads as "no witness
available" and passes the device number through unchecked. The failure was
silent: telemetry looked healthy, the band existed, and it was checking nothing.

*A witness scoped to the wrong region produces a wrong rule, not an obvious
failure.* Measured device/wall against a **Python-level** wall — which carries
numpy marshalling and array conversion no GPU interval could contain — was
**0.35–0.60**, and that argued for a one-sided band, since 0.35 fails a 0.5×
floor. Against the **runtime** witness the same dispatches run **0.568–0.937**
over 100 samples from 8² to 2048². The symmetric band was fine all along; the
one-sided version was defending against an artifact of its own denominator.

**Corrected after review (2026-08-31): the band is ONE-SIDED, and the two-sided
version was wrong twice for the same reason — generalising from one route.** A
second route family, resident batched sessions on `metal_kernel_interval`,
measures **0.037–0.101** once warm: a 25 µs kernel inside a 265 µs
submit-to-signal window. Nothing is wrong there — `kernelStartTime`/
`kernelEndTime` is kernel execution only and legitimately excludes queueing.
Across routes the honest range is **0.037–0.937**, so no wall-derived floor can
separate a small kernel from an under-reading clock. That is exactly what ROCm
already states in `_select_rocm_latency_ms`.

What survives is containment, which is exact: GPU execution is a strict subset
of commit-and-wait, so **`device <= 1.25 × wall`** (1.25 rather than 1.0
because the two are independent clocks over nested regions and the measured max
of 0.937 leaves a strict bound only 6.3%). **The under-reading direction — the
dangerous one, since an under-estimate inflates throughput and gets published —
is now explicitly unguarded**, and asserted as such in the tests so the gap is
visible rather than assumed covered. It closes against Apple's *second* device
clock, not the wall; see the follow-up below.

**Outcome for this backend: `not applicable` — NVIDIA closed its half under
`NVIDIA-TIMER-DRAIN-2026-08-31` and nothing here changes it.** CUDA already
takes a wall witness across a drained region and band-checks the event against
it; the Apple work is the sibling item that key recorded as owed.

**The one transferable lesson, and it is a scoping one.** NVIDIA's finding was
that a band without a *drain* is worse than no band. Apple's is that a band
without a **correctly scoped witness** is worse than no band — the first
version measured against a Python-level wall carrying numpy marshalling, and
concluded a one-sided band was needed when it was not. Same failure at
different ends of the same sentence: the band is only as good as the region its
witness covers, and both ways of getting that wrong produce a plausible rule
rather than a visible error. Worth checking against the remaining twelve
duplicated timing loops in `tessera_nvidia_ptx_launch.cpp` when those are
de-duplicated — each one defines its own region.

**No sm_120 evidence is claimed** — no CUDA code changed.

---

## Cross-backend sync `APPLE-DEVICE-CLOCK-2026-08-31`

**Owning item:** Apple's device clock · **synchronization key:**
`APPLE-DEVICE-CLOCK-2026-08-31`

**Shared contract changed — what makes a device clock a *measurement*.**
`APPLE-TIMER-WITNESS` added a host witness and a containment bound. This closes
the direction that bound provably cannot reach: a clock that under-reads looks
exactly like a small kernel, since both sit far below the host wall.

**The defect.** `ts_record_dispatch_gpu_elapsed` preferred
`cb.kernelStartTime`/`kernelEndTime` and treated `GPUStartTime`/`GPUEndTime` as
a fallback, on a comment asserting the first pair was "the completed
compute-kernel interval". **The SDK says the opposite by omission**:
`GPUStartTime` carries an `@abstract` — *"the host time in seconds that GPU
starts executing this command buffer"* — and `kernelStartTime` is a bare,
undocumented declaration. Measured on an M1 Max with only a kernel's loop count
varying:

| iters | `kernelS/E` | `GPUS/E` | encoder stage | host wall | kern/wall |
|---|---|---|---|---|---|
| 5,000 | 54,583 | 498,375 | 498,375 | 764,417 | 0.071 |
| 320,000 | 65,833 | 9,390,833 | 9,390,792 | 9,833,750 | **0.007** |

`kernelStartTime` is **flat across a 64× workload**. `GPUStartTime` tracks the
wall *and* agrees with an independent stage-boundary counter-sample clock **to
the nanosecond** — two mechanisms agreeing that closely is what distinguishes a
measurement from a plausible number.

**A second bug hid behind the first.** `GPUStartTime` is documented to read zero
until the GPU starts and to be readable "in command buffer completion handler".
Every dispatch path here waits on a *shared event*, which proves the GPU
finished but does not publish those properties. Simply preferring the documented
pair therefore changed nothing — it read zero and fell straight back. The
recorder now forces publication itself (`ts_gpu_interval`), so no caller can
forget.

**The generalisable finding is the check, not the property.** No bound catches
an under-reading clock. What caught this is **metamorphic**: vary the workload
and require the device clock and the host wall to move *together*. They may
diverge in magnitude — the wall carries submission overhead — but not in
direction. Under the defect that ratio was 0.32–0.40; healthy it is 0.86–1.14.

**Outcome for this backend: `parity validated` on existing sm_120 evidence — no
CUDA change, and the check this key generalises was already satisfied here.**
`NVIDIA-TIMER-DRAIN-2026-08-31` measured `event/wall` at **0.996–0.998 across
shapes**, which is a tracking result and a stronger one than Apple had: two
clocks that agree to 0.2–0.4% at every size cannot be flat while the other
moves. The re-raced matmul corpus is the same story at the workload level —
0.01076 / 0.04434 / 0.32102 / 2.45369 ms across 256³–2048³, monotonic over a
512× work range.

**Why this is worth a row anyway.** NVIDIA has exactly one device clock and no
second one to cross-check, so its whole defence is the wall witness plus that
agreement. Apple's defect is the case where a device clock is *internally*
plausible and simply not measuring — and no bound detects that. If a CUDA lane
ever gains an in-kernel `%globaltimer` stamp (the `wall_clock64` analogue ROCm
uses), the metamorphic tracking check is what should gate it, not agreement at
a single shape.

**No new sm_120 evidence is claimed** — nothing CUDA-side changed under this key.

---

## Cross-backend sync `PACKET-PROVENANCE-2026-08-31`

**Owning item:** exact-device evidence provenance ·
**synchronization key:** `PACKET-PROVENANCE-2026-08-31`

**Shared contract: a packet may not claim a commit it was not generated from.**
Every lane's recorder stamps `tested_commit` from `git rev-parse HEAD` and
**none of the four checked that HEAD is what was actually measured.** Recording
from a modified working tree therefore produces a packet whose measurements
came from edited sources while its `tested_commit` names the parent — false
provenance that then propagates into `docs/audit/generated/e2e_fleet.*` as
though it were a device result for that commit (AGENTS.md:87-90).

**Found by doing it.** The Apple packet on PR #665 was sealed from a dirty tree:
its `source_fingerprint` hashed the *edited* `apple_gpu_runtime.mm` while
`tested_commit` named the parent, whose runtime hashes to something else. It was
review that caught it, not any gate.

**Apple is the only lane where the contradiction is visible at all**, because
only its packet carries a `source_fingerprint` of a runtime source file. The
other three fingerprint measured *resources*, not sources — so a packet built
from modified kernels is internally consistent and silently wrong. **The lane
with the strongest self-check is the one that got caught; the weaker three
would not have surfaced it.**

**Outcome for this backend: `follow-up required` — same defect, unfixed.**
`record_sm120_packet.py:426` stamps `tested_commit` from `git rev-parse HEAD` with no dirtiness
check, exactly as Apple's did.

**Why it is not fixed in this PR.** The Apple guard works because that packet
declares which file it fingerprints, so the set to check is unambiguous. This
lane fingerprints measured resources rather than sources, so choosing the right
file set — plausibly the CUDA runtime sources and `ptx_emit` (which generates the kernel text at record time) — is a judgement about what this backend's
measurement actually depends on, and getting it wrong fails in the worse
direction: a too-narrow set is a guard that passes while the provenance is
false, which reads as protection and is not. That call belongs with someone
looking at this backend's build, on **Super-Bear**.

**The cheap interim** is the whole-tree form: refuse when `git status
--porcelain` is non-empty for this backend's source directory. Cruder than
Apple's and more likely to be bypassed, but it cannot be wrong in the
dangerous direction.

---

## Cross-backend sync `SPECTRAL-CONV-RANK-2026-08-31`

**Owning item:** `tessera.spectral_conv` operand contract ·
**synchronization key:** `SPECTRAL-CONV-RANK-2026-08-31`

**Shared contract enforced (not changed): `spectral_conv` takes equal ranks.**
It was already stated in two places — the host reference in `__init__.py`, which
raises on a mismatch, and the `conv_full` shape rule, which returns *unknown*
rather than deriving `n + m - 1`. **No dispatch lane stated it.** Each computes
`rfft(x) * rfft(w)` and so inherited numpy broadcasting for free, silently
admitting a rank-1 kernel against a rank-2 signal.

**The divergence was in what the lanes ADMIT, not in what they compute.** The
broadcast and rank-matched forms are **bit-identical** (max |Δ| exactly 0.0),
and both match `np.convolve(..., mode="full")` to 2.4e-07. That is precisely why
it survived: nothing was ever numerically wrong, so no accuracy check could see
it. What it produced instead was a GPU lane that accepted input the CPU
reference rejects.

**And it had already caused a hollow test.**
`test_apple_gpu_spectral::test_composites_match_host_reference` was written
against the permissive lane, passing a rank-1 kernel — so its
`assert_allclose` **never executed**: the reference raised while building the
expected value. The test had been red long enough to be repeatedly triaged as
"pre-existing".

**Fixed by stating the rule once** — `_check_spectral_conv_ranks` in
`runtime.py`, called by every dispatch lane — rather than adding a check per
lane. There are **three** dispatch implementations of this op
(`_spectral_composite`, shared by NVIDIA and ROCm; `_apple_gpu_dispatch_spectral`;
and the host reference), and a rule copied three times is a rule that will hold
in two of them.

**Outcome for this backend: `follow-up required` — the fix reaches NVIDIA's
lane but no sm_120 evidence exists for it.** `_spectral_composite` is shared by
NVIDIA and ROCm, so `_nvidia_fftexec` now refuses a rank mismatch too. That is
a **behaviour change on this backend made from a Mac**, and it is host-free
only in the sense that the check itself is pure Python — whether NVIDIA's
spectral lane has any caller passing a rank-1 kernel is not something this host
can answer.

**What is owed on Super-Bear:** run the spectral suite against the CUDA lane and
confirm nothing was relying on the broadcast form. The risk is low (every
in-repo call site outside the corrected test uses equal ranks) but it is not
zero, and a refusal that fires in production is a hard error rather than a
degraded number.

---
## Cross-backend sync `HSACO-NEGATIVE-CACHE-2026-08-31`

**Owning item:** compiled-family build caching ·
**synchronization key:** `HSACO-NEGATIVE-CACHE-2026-08-31`

**`_build_rocm_family_hsaco` cached successes and not failures.** It invokes
`tessera-opt` through `subprocess.run`, so a host that HAS the binary but
cannot serialize ROCm — any dev box without the toolkit, and this Mac — forked
a process, waited for it to fail, and discarded the answer **on every single
launch**. **74 call sites** funnel through it.

**Measured on an M1 Max**, `rt.launch` of a draft-block workload:

| | direct | launch | ratio |
|---|---|---|---|
| before | 0.1042 ms | **70.5963 ms** | **677×** |
| after | 0.1141 ms | 0.1481 ms | 1.3× |

**477× faster**, and 0.347 s of a 0.354 s profile was the subprocess.

**It was hidden by a test that ratified it.** Five perf baselines assert
`launch_ms < max(75.0, direct_ms * 4.0)`. The oracle arm is ~0.4 ms, so `max()`
always selected **75.0** and the self-calibrating comparison was dead code —
while launch sat at **94% of that limit on an idle machine**. The constant had
been sized to accept exactly the overhead it claimed to bound, and tipped over
under any load, which is how it read as five flaky tests rather than one defect.

**The floor is now 2.0 ms, chosen by the only criterion that makes a constant
worth having: it FAILS on the old code.** Verified by mutation — with the
negative cache removed, all five report `71.3 < 2.0`. The old 75.0 accepted
70.6 ms in silence.

Caching the failure is sound because the causes are host properties, fixed for
the process. A genuinely transient failure would be remembered until exit — the
right trade against re-forking per call, and why the stored value keeps the
original message. Every attempt still records a dispatch fallback:
`_rocm_compiled_failed` is re-raised through its own funnel on a cache hit, so
`TESSERA_STRICT_DISPATCH=1` still raises. Only the subprocess is skipped.

**Outcome for this backend: `follow-up required` — the same asymmetry is likely
present and unmeasured.** Nothing NVIDIA-side changed. The question this key
raises for CUDA is whether its own compile paths cache failure: NVRTC compiles
at load (`nvidia_training.py`, `ptx_emit`), and a host without a usable CUDA
toolchain would retry per call in the same shape if the negative result is
discarded.

**Worth checking on Super-Bear**, and cheap: profile one `rt.launch` of a CUDA
family on a host where the build fails, and look for `subprocess`/NVRTC in the
cumulative time. The tell is the one this key was found by — a launch whose cost
is orders of magnitude above the work it performs.

---

## Cross-backend sync `APPLE-COMPLETION-HANDLER-2026-09-01`

**Owning item:** how a device timestamp is *obtained* ·
**synchronization key:** `APPLE-COMPLETION-HANDLER-2026-09-01`

**Shared lesson: when a value is only valid after an event, take it from the
event — do not block waiting for the value.** `APPLE-DEVICE-CLOCK` established
that `GPUStartTime`/`GPUEndTime` is the clock that measures work; it published
those values by forcing them with `[cb waitUntilCompleted]`, which has **no
timeout**. Both resident-session paths invoke the recorder *after* their 30 s
shared-event wait expires, so on a hung GPU that turned a bounded failure into
a permanent hang — in code whose only job is telemetry.

The first repair gated the wait on a `completed` flag threaded through six call
sites. That contained the hang but surrendered the clock on exactly the paths
that had timed out, and put a correctness-critical boolean into six places
where it could be passed wrongly (it was: one site passed a literal `true`).

**`addCompletedHandler` removes the dilemma instead of containing it**, and is
what the SDK prescribes — `GPUStartTime` is documented as readable *"in command
buffer completion handler"*. Metal fills a slot on completion; the reader waits
on a semaphore with a **bounded** timeout. A hung buffer never fires its
handler, the wait expires, telemetry reports no device number. No hang, and the
flag is gone.

**Two traps worth carrying to any backend doing the same.**

*The callback runs on another thread.* The slot is heap-shared and deliberately
**not** `thread_local` like the other 15 telemetry globals — a `thread_local`
write from Metal's callback thread lands in storage the reader never sees.

*Removing a wait can silently re-route the fallback.* Three tile recorders have
no slot of their own (they run after `commit_and_wait_with_timeout` already
waited) and would have fallen straight through to `cb.kernelStartTime` — the
clock measured **flat across a 64× workload**. The change would have undone
`APPLE-DEVICE-CLOCK` on three paths while every test stayed green. A direct,
non-blocking property read now precedes that fallback.

**Outcome for this backend: `not applicable` — nothing changed here, and the
specific API is Apple's.** Recorded because the *reasoning* transfers and the
question is cheap to ask of CUDA: `cuLaunchHostFunc` / a stream callback, against the event-elapsed pair it uses today.

**The test worth applying, whatever the API.** Does the timestamp read path
contain a wait that has no timeout? On Apple the answer was yes, twice — once
outright, once behind a flag that six call sites had to pass correctly. Neither
was visible as a failure until a reviewer traced what happens when the shared
event times out, because the hang only occurs on a GPU that is already broken.

**If this is ever checked on Super-Bear**, the control that made the Apple answer
trustworthy is the one to reuse: measure dispatch variance with and without the
change on the same box, before attributing instability to either.

---

## Cross-backend sync `CAPABILITY-GUARDS-2026-09-01`

**Owning item:** host-capability gating in the unit suite ·
**synchronization key:** `CAPABILITY-GUARDS-2026-09-01`

**Three tests failed on a host that cannot evaluate them, instead of skipping.**
CLAUDE.md's claim-integrity rule is explicit that a host lacking a device or
toolchain must say so; these did the inverse, and one of them (`gfx1151`) had
been carried as "pre-existing" for the whole session.

| test | what it needed | what happened |
|---|---|---|
| `..._sm120_tile_fragment_lowers_to_real_nvvm_mma` | a runnable `tessera-nvidia-opt` | `dyld: Symbol not found` from a binary linked against a prior LLVM keg |
| `..._attention_package_rejects_stale_parent_and_tile_lineage` | the AVX-512 shared image | `RuntimeError: X86 native packaging requires ...` |
| `..._gfx1151_scheduled_attention_backward_packages_exact_tile_program` | AMD clang | `RuntimeError: ROCm native packaging requires AMD clang ...` |

**One family: a guard that checks PRESENCE rather than USABILITY.** And in
every case **the helper that prevents it already existed, and the caller did
not use it** — which is the part worth carrying, because writing the helper
felt like fixing the problem:

* `compiler_tool.is_runnable`, whose docstring describes this exact dyld
  failure, gated only `tessera_opt` discovery. `_tool_path` — which resolves
  `mlir-opt` *and* `tessera-nvidia-opt` — checked `is_file()` alone.
* `rocm_native.native_packaging_available`, whose docstring says checking tools
  without AMD clang "reads as a broken test on any host without ROCm rather
  than an absent toolchain", was not called by the gfx1151 test.
* `rt._x86_elementwise_available` is used by a sibling test one function away.

**The skip has to be proven narrow, not just present.** A guard that skips too
much is the hollow-green pattern wearing a fix's clothes, so each was verified
on a host that HAS the capability:

| host | result |
|---|---|
| Princess-Luna (AVX-512 + ROCm) | both tests **PASSED**, not skipped |
| Super-Bear (CUDA) | nvidia-opt test **PASSED**, not skipped |
| Princess-Luna, resolver control | **771 passed / 85 skipped before and after** — the `_tool_path` change adds zero skips where tools work |

**Outcome for this backend: `parity validated` — fixed and verified on
Super-Bear.** `_tool_path` now requires `tessera-nvidia-opt` to start, not
merely to exist. An exported selector stays final: an unrunnable
`$TESSERA_NVIDIA_OPT` returns `None` (a clean skip) rather than silently
falling back to a different binary, because running one the developer did not
ask for is how a passing run stops meaning anything.

**The failure mode is specific to a machine that has built more than one
toolchain**, which is every developer box and no CI runner — so CI would never
have caught it. That is the same asymmetry recorded under
`fleet_llvm_assertion_asymmetry`: the host that can falsify a claim is often
the one nobody runs the check on.

---
## Cross-backend sync `AUTOTUNE-SEPARATION-NVIDIA-2026-09-01`

**Owning item:** `AUTOTUNE-SEPARATION`, NVIDIA half ·
**synchronization key:** `AUTOTUNE-SEPARATION-NVIDIA-2026-09-01`

**The corpus was re-raced on sm_120 with #663's separation verdicts recorded,
and 42 of 51 freshly-raced rankings (82%) turn out to be unsupported.** The
earlier estimate — "11 rows with margins under 2%" — understated it badly,
because a margin cannot be judged without the noise beside it. That is the
whole content of #663, now measured rather than argued.

**Two recorded verdicts are retired by evidence, not by opinion.** At 512³ and
1024³ device-timed matmul, the compiler-**emitted** PTX lane wins by ~38%
against **0.15–1.86%** noise, racing the full four-candidate field. The prior
rows named a *tile* lane and pinned the 1024³ field to exactly two candidates —
which encoded the biased race #655/#662 removed: the GEMM lanes had no device
timer, `_measure` scored them `inf`, and they lost silently. "The tile lane
wins" meant "the tile lanes were the only ones that could be timed".

**`device_repeats=3` overstates the noise floor it reports.** Measured at
128×512×64 bf16: sd **48.31% / 30.74% / 19.34%** over 3 / 10 / 30 whole
measurements, with a 2.3× min–max range even at 30. The lane genuinely is ~19%
noisy, so the *unseparated* verdicts hold either way — but a recorded floor
2.5× the truth is a number someone will act on. The corpus recorder now uses
10; `measured_arbitrate` keeps 3, which is the right cost trade for runtime
selection rather than published evidence.

**An independent mechanism agrees, which is what makes this trustworthy.**
`finalize_test5_corpus` replaces a row only when **two** runs pick the same
winner. The one row it refuses — `bfloat16 [128, 256, 64]` device — is exactly
the row separation flags at margin 9.92% against 102.96% noise. Two checks
built years apart, from different premises, rejecting the same ranking.

**Outcome for this backend: `parity validated` — measured on Super-Bear
(RTX 5070 / sm_120), and the corpus is committed.**

**Two near-misses on the way, both of which would have destroyed evidence
silently.**

*Without `--warm-start`, regenerating deletes other boxes' rows.* The recorder
writes the whole cache, so a bare run dropped **all 12 `rocm:gfx1151` rows** and
13 NVIDIA rows at shapes the default flags do not cover — 25 rows of
exact-device evidence, with no error. Re-run with `--warm-start`: 0 lost.

*Recording is two runs plus a finalizer, not one run.* `stable_runs == 2` is
literally two independent measurements agreeing;
`record_autotune_corpus.py` alone produces rows with no `evidence` block at
all. A single-run corpus therefore carries **less** evidence than the one it
replaces (41 rows → 0) while looking like an update. Done properly the count
goes 41 → **84** and nothing is lost.

**Both traps share a shape worth naming: a regeneration that succeeds while
producing weaker evidence than it replaced.** Nothing failed, nothing warned;
only a before/after row-and-evidence count catches it. Any future corpus
regeneration should print that diff before committing.

**Three consumer tests were rewritten to the measured reality**, and each now
pins a property the old assertion could not express:

* the 1024³/512³ rows assert the **full four-candidate field** and that the
  winner's margin is **separated** — not merely that a particular name won;
* the stability matrix asserts **eligible ⟹ stable** rather than "every row is
  stable", because reproducibility at the launch-overhead floor is not a
  property this hardware has;
* the fused/attention rows assert a **subset** of the known candidate pair,
  because `applies_to` makes them mutually exclusive by contract (below).

All three are mutation-verified: shrinking the field, un-separating the verdict,
and offering an unstable row to the selector each fail.

**Found on the way, and it is the `applies_to` item:**
`NvidiaGenericCudaCandidate.applies_to` returns True only when the epilogue
contract *selects* it, so `nvidia_generic_cuda` and `nvidia_mma_fused` are
never competitors in one race. Such a row records `unmeasured: {}` — "nothing
was skipped" — which is true and misleading: the other candidate was not
skipped, it was **contract-excluded**, and no field distinguishes "one
candidate exists" from "a second was excluded before the race". `unmeasured`
(#655) closed the *timing* half of this; the *applicability* half is still open.

---

## Cross-backend sync `AUTOTUNE-SEPARATION-ROCM-2026-09-01`

**Owning item:** `AUTOTUNE-SEPARATION`, ROCm half, and the dispatch tightening
it unblocks · **synchronization key:** `AUTOTUNE-SEPARATION-ROCM-2026-09-01`

**All 16 gfx1151 rows now carry a verdict — zero never-asked**, where all 12
previously had `separation: None`. 15 separate cleanly (margins 34–99.9%
against 0.10–14.24% noise); the single refusal is `paged_kv_decode 8192
end_to_end` at 6.52% margin vs 5.59% noise, which is genuinely marginal.

**That is the opposite of sm_120's 82%-unsupported result, and the reason is
structural rather than a hardware difference.** ROCm races two candidates that
are far apart (generic HIP vs WMMA); NVIDIA races four that often sit within a
few percent. A backend with a *narrow* field gets clean verdicts almost for
free — which is worth knowing before reading either number as a quality signal
about the backend.

Verified the numbers come from `device_event`, not the wall-clock fallback this
backend is prone to, so the noise floor is genuine rather than inflated.

**The dispatch rule is now tightened, and it is deliberately not "reject
`None`".** `corpus_winner` refuses a row that ranks **two or more** candidates
and has no verdict. `separation_verdict` returns `None` when fewer than two
were timed — a sole candidate is chosen by *applicability*, not by a race, and
has no margin to defend. Refusing those would be a category error: 12 of the 23
remaining `None` rows are exactly that shape. `inf` is likewise not a
competitor (it marks "could not be timed"), so a row with one latency and one
`inf` is a sole-candidate row wearing a pair's clothes.

Committed corpus: **113 rows — 67 refused as dispatch hints, 34 with a
supported verdict, 12 sole-candidate.**

**Outcome for this backend: `follow-up required` — 11 sm_120 rows are now
inert.** The tightening refuses them: they rank two or more candidates and
carry no verdict, at shapes the recorder's default flags do not cover
(`attention`, `conv2d`, `ssm_replay_decode`, and a handful of matmul buckets).

**They are recoverable by widening the recorder's shape flags and re-running**,
which is a mechanical job on Super-Bear rather than a design question. Until
then those buckets fall back to lead-safe tier priority, which is the correct
degraded behaviour and not a regression — before this change they served a
ranking nothing had checked.

**Read the ROCm/NVIDIA contrast carefully.** 15/16 separated here against 19/74
there is *not* evidence that gfx1151 measurements are better. It reflects field
width: two far-apart candidates separate almost automatically, four close ones
rarely do. A backend that adds candidates should expect its separated fraction
to fall, and that is the field getting more honest, not the hardware getting
worse.

## Cross-backend sync `APPLIES-TO-SHAPE-BLIND-2026-09-01`

**Owning item:** `applies_to(region)` shape-blind (NVIDIA plan, "Follow-up
owned here", now closed) · **synchronization key:**
`APPLIES-TO-SHAPE-BLIND-2026-09-01`

**Shared contracts changed** — arbiter selection and autotune measurement, so
all four backends are assessed here per AGENTS.md:

* `Candidate.applies_to_inputs(region, *inputs)` — new, additive, defaults
  `True`. Every existing `applies_to` implementation is untouched.
* `candidate.live_candidates(region, op, target, inputs)` — one statement of
  "who is racing", replacing the copy that `arbitrate`, `measured_arbitrate`
  and `corpus_winner` each kept.
* `arbitrate(..., inputs=())` — additive keyword; omitted, selection is
  byte-for-byte what it was.
* `autotune._measure` now reads the execution tag it was already producing.

**The defect: a region carries structure, not dimensions.** `MatmulRegion` has
a dtype and transpose flags; `FusedRegion` an epilogue chain. M/N/K arrive with
the operands and are inferred separately. So `applies_to(region)` could not
express "aligned shapes only", and the F4 oracle could not cover for it either
— its probe shape is fixed (32×16×32 for matmul) and its verdict is cached
under a key with **no shape in it**. An aligned-only lane therefore declined
inside `run`, by returning the numpy reference, *after* it had already won.

**Two harms, both reproduced against the real
`NvidiaMmaGemmEmittedCandidate` before the fix:**

| | before | after |
|---|---|---|
| ragged-shape winner | `nvidia_mma_gemm_emitted`, tag `reference` | the lane that can run the shape, real tag |
| its recorded latency | `0.00525 ms` (numpy) vs a real `0.00196 ms` rival | absent from the field |

The second is the one that propagates. A fabricated latency does not sit
inert — it **ranks**. With the backstop disabled the same record comes back
`separated: True, margin 0.59, runner_up: nvidia_mma_gemm_emitted`: the
separation machinery from #663/#671 confidently certifies a 2.4× loss for a
kernel that never ran. Separation judges the numbers it is given, and this is
a second, independent mechanism for the corpus bias recorded in
`NVIDIA-TIER-PRIORITY-IS-WRONG-AT-SCALE-2026-08-30` — where a biased race hid
the fastest kernel in the registry.

**The tag was already there and nothing read it.** The D3 arbiter log has
described this since it was written: *"the arbiter selects a candidate, but
that candidate's `run` may still decline to the numpy reference at execution
time (a device error, **an unsupported shape**) — a silent degrade the tag
reveals."* Observability without a consumer, which is Decision #29 wearing a
diagnostic's clothes.

**Fix, layered deliberately.** `applies_to_inputs` fails **open** on absent or
malformed operands — the question cannot be answered there, and refusing would
disable every shape-anonymous caller, while a malformed pair is an operand
error that must still raise through `run` rather than be silently excluded
(Decision #21). The fail-**closed** backstop sits one level down: `_measure`
refuses to record a latency for a run that came back with a reference tag,
whether or not the candidate declared itself. A lane that never adopts the
hook is therefore still safe from the fabricated-measurement half.

**Host-free evidence:** `tests/unit/test_arbiter_workload_applicability.py`
(10 tests). Mutation-verified — four independent mutations, each killing only
its own tests: disabling the selection filter (3 fail), disabling the tag
backstop (1), removing the NVIDIA producer (1), removing the ROCm producer (1).

**Outcome for this backend: `follow-up required` — the producer landed, the
device proof has not.** `NvidiaMmaGemmEmittedCandidate.applies_to_inputs` now
states the aligned-only contract its own class docstring has always carried and
its `measure_device_latency` already enforced; the declaration was the only
place the three disagreed. Its 20 sibling lanes were audited and need nothing:
their run-time guards test `ndim`/contraction agreement, which is operand
*validity*, not shape *support*, and those correctly stay in `run`.

Owed from Super-Bear (RTX 5070 / sm_120): re-race a ragged bucket and confirm
(a) the emitted lane is absent from the field rather than present with a numpy
latency, and (b) the ragged winner now carries a real execution tag. Neither is
claimed here — this Mac has no CUDA, so the aligned F4 probe declines for the
wrong reason and cannot falsify the device-present case. Harm 2 above was
therefore reproduced under a simulated device (aligned path stubbed to succeed,
ragged left to decline), and is labelled as such.

**Blast radius, measured: no committed corpus row is invalidated.** Every
persisted matmul bucket is aligned (0 of 28 rows ragged) and every attention
bucket uses head_dim 64 (0 of 18 ragged), so neither producer's decline was
ever exercised during a recorded race. The defect was live in the *dispatch*
path and had not yet reached the evidence.

That is luck, and it points at the real gap: the recorders only ever race
power-of-two shapes, which is why nobody hit this and also why **the ragged
path has no measured coverage at all**. The device follow-ups above should
add a ragged bucket rather than only re-checking an aligned one — a re-race
of the existing buckets would pass identically before and after this change.

**Two follow-on findings from the same code path, both now covered.**

*The two consumers had to move together, and it is not obvious why.* Buckets
are coarse — `bucket_key` maps both `(24,12,20)` and `(32,16,32)` to
`(32,16,32)` — so a ragged workload genuinely reads the aligned workload's
corpus row, and `run_arbitrated` passes a corpus hint to `arbitrate` as
`force`, which restricts to that one name. Making `arbitrate` shape-aware
*without* `corpus_winner` therefore converts the silent degrade into an
`ArbiterError` (verified, not inferred). `corpus_winner` withholds the hint
because its own `live` set excludes the lane — that coupling is now pinned by
a regression test rather than left to be rediscovered.

*The `force` diagnostic named the wrong gate.* One message — "not available" —
covered not-registered, wrong-region and unavailable-here alike, and once a
shape axis existed it was actively wrong for the commonest case: the lane IS
available, on a host that has it, for a shape it cannot serve. It now names
which of the four gates rejected the candidate (Decision #21).

## Cross-backend sync `ROUTE-LEDGER-RULES-UNCONSUMED-2026-09-01`

**Owning item:** Apple strict route ledger re-seal · **synchronization key:**
`ROUTE-LEDGER-RULES-UNCONSUMED-2026-09-01`

Apple's `promotion_rules` block turned out to be a declaration no code read:
sealed into twelve ledgers for audit, and `status: "promote_candidate"` was
self-certifying at load. Fixed there (see the Apple plan). All four backends
are assessed because the *pattern* is what travels, not the Apple code.

**Outcome for this backend: `follow-up required` — the same gap exists here,
in a milder form.** `nvidia_sm120_legacy_retune.json` and
`nvidia_sm120_low_precision_native_routes.json` declare `noise_policy`
(0.03 / 0.04) and `selector_promotions`, and the ratchets assert those values
**equal a constant** without ever comparing a promoted row's margin against the
policy the file declares. So a promotion inside the noise band would pass every
existing check.

Credit where due: `test_nvidia_low_precision_native_routes.py` is otherwise
well built for this class — it cross-checks the summary count against the rows,
explicitly guards the vacuous-pass case (`len(promoted) == selector_promotions
> 0`), and requires timing-domain consensus plus resource fingerprints on every
promoted row. The missing piece is exactly one: **the margin is never held to
`noise_policy`.** That is a smaller job than Apple's was, and it needs a
decision about what "margin" means for these rows before it is written.
## Cross-backend sync `PROMOTION-EVIDENCE-REDERIVED-2026-09-01`

**Owning item:** the NVIDIA/ROCm half of
`ROUTE-LEDGER-RULES-UNCONSUMED-2026-09-01` ·
**synchronization key:** `PROMOTION-EVIDENCE-REDERIVED-2026-09-01`

Apple's `promotion_rules` was a declaration nothing read; its loader now
re-derives each promotion from the evidence the ledger retained. The same
question was then put to the other backends' evidence artifacts, and the
answers differ enough to be worth stating one by one.

**A near-miss that shaped the whole exercise, recorded because it would have
been convincing.** The first NVIDIA checker modelled promotion as *"the winner
beats the runner-up by more than `noise_fraction`"* — the obvious reading — and
it flagged **7 of 11 committed promotions as violations**, complete with a
tidy table. Reading the producer settled it: `finalize_low_precision_native_routes._near`
promotes a candidate that is **within** `noise_fraction` of the fastest in
every run of every domain, tie-broken by total time. Every one of those 7 is
correct under the rule that was actually applied. A plausible model of someone
else's gate, checked against committed evidence, produces confident and wrong
findings — so both checkers here mirror their producer's own predicate rather
than a reasonable-looking substitute.

**Outcome for this backend: `parity validated` — both evidence files now
re-derive, 0 mismatches.**

* `nvidia_sm120_low_precision_native_routes.json` — every recorded conclusion
  (`near_winner_consensus`, `run_winners`, `timing_domain_consensus`, `winner`,
  `selector_promoted`) is recomputed from `timings`, the only field that is
  measurement rather than verdict. **18 rows, 11 promotions, 0 mismatches.**
* `nvidia_sm120_legacy_retune.json` — `noise_policy` was asserted to *equal
  0.03* and never compared to a measurement, so a regressed recording whose two
  runs disagreed by 40% would keep `stable: true` and pass. Stability is now
  `|run0 - run1| / max(run0, run1) <= noise_policy` per domain, re-derived, plus
  cross-candidate winner consensus per case. **8 rows, 4 cases, 0 mismatches.**

Both confirm their recorders rather than accusing them. Checkers live in
`tests/_support/nvidia.py`; each is mutation-verified against forged rows
(invented winner, stripped resources, widened consensus, lying `run_winners`,
drifted runs, deleted timings).

## Cross-backend sync `MATRIX-LANE-RAGGED-SHAPES-2026-09-01`

**Owning item:** the matrix-core lanes decline ragged shapes ·
**synchronization key:** `MATRIX-LANE-RAGGED-SHAPES-2026-09-01`

**One gap, found three times while fixing other things.** Every backend's
matrix-core lane — the *fast* one — declined ragged shapes and fell back to a
much slower path:

| backend | lane | gate | fallback |
|---|---|---|---|
| NVIDIA sm_120 | emitted `mma.sync` GEMM | `M%16, N%8, K%16` | numpy |
| ROCm gfx1151 | WMMA flash-attention | `head_dim % 16` | numpy |
| Apple M1 Max | coopmat `simdgroup_matrix` reduce | `N % 8` | scalar path |

What made it worth doing now is the measurement from the corpus work: the
compiler-**emitted** PTX GEMM beats the hand-tuned delegate **1.5–1.7x at every
shape** on sm_120. The alignment gate was not costing a little — it was
excluding the fastest kernel in the registry from every ragged workload. The
`applies_to_inputs` declines added in #672 made that *honest*; they did not make
it *fast*.

Both fixes share one idea and split on which axis a dimension plays:

* **A contraction dimension is zero-padded** — exact, because a zero operand
  contributes nothing to the dot product.
* **An output dimension is store-suppressed** — zero-padding is wrong there; a
  lane past the edge would write a correct-but-zero value into someone else's
  slot.

Getting that backwards is silent corruption rather than a fault, which is why
it is stated as the rule rather than left implicit in each kernel.

**Outcome for this backend: `parity validated` — measured on Super-Bear
(RTX 5070 / sm_120).** `emit_mma_sync_gemm_ptx` now clamps its M/N load index
and suppresses the out-of-range store, so the hot K loop is byte-for-byte the
proven aligned kernel: no per-iteration predicate, no divergence, cost confined
to a few instructions outside the loop. The K remainder is a genuinely
predicated extra slab.

**25 ragged shapes x 2 dtypes, 0 failures, <= 1.3e-7 relative error**; both
dtypes assemble under `ptxas --gpu-name=sm_120a`.

**K must stay EVEN, and that is hardware, not laziness.** The first run faulted
and `compute-sanitizer` named it: `CUDA_ERROR_MISALIGNED_ADDRESS`, not an
out-of-bounds. `ld.global.b32` needs a 4-byte-aligned address while the
fragments address 2-byte elements, so `row*K + k` must be even — with K odd,
every odd row starts misaligned, in the **main loop** and not only the tail.
Every data point fits once that is known: 24x12x20 passed (K even), 16x8x17
faulted (K odd), 1x1x7 passed (row 0 only). Lifting it needs a padded row
stride (the strided ABI could already carry it) or a paired `ld.global.u16`
slow path; both are real designs and neither is a one-line change, so K%2 is
recorded as the boundary rather than claimed as working.

**A dead seam turned out to be exactly this work.** `invokeMmaGemm16` has had a
`bool ragged = false` parameter since it was written that **no caller ever
passed true**, so the guard behind it rejected every unaligned shape with rc=5.
Decision #29's unconsumed declaration, in C++.

**The benchmark path needed the same guard, separately** — it does not route
through `invokeMmaGemm16`, so a first fix placed only there let an odd-K
*measurement* fault the device, which then poisons the CUDA context for every
later launch in the process. The autotuner calls that path. The capability now
lives in `tileLaunchConfig`, the per-kernel geometry table, so both paths get
one answer; the Tile-direct and scheduled kernels stay aligned-only, which is
why it is per-entry rather than a global relaxation.

**Aligned-path regression, measured rather than asserted** (A/B against the
unmodified emitter and launcher, same box, same session):

| shape | baseline ms | ragged ms | spread |
|---|---|---|---|
| 512³ | 0.02678 | 0.02762 | 1.1–1.5% |
| 1024³ | 0.19358 | 0.19764 | 0.3–0.5% |
| 2048³ | 1.48253 | 1.49670 | 0.3–0.5% |

**1–3% at large shapes** — real, outside noise, and the price of ragged shapes
going from numpy to full speed. At 64³/256³ the first sample looked like a 1.4x
regression, but against **52–58% spread**, so not a supported comparison; a
15x3000-rep re-measure pushed the spread *worse* (85%, 73%) and the medians
crossed over. Those shapes are unresolvable on this box and the apparent 1.4x
was noise, exactly as its own spread warned.

## NVIDIA-RAGGED-TIMING-FIELD-2026-09-01: the ragged flag was doing two jobs *(fixed)*

**Review finding on #675, verified rather than accepted — and it is a
regression that PR introduced.** `tileLaunchConfig` defaulted `ragged` to false
and set it true only for the two PTX GEMM entries. But **every** entry the
invoke dispatch routes through `invokeMmaGemm16` passes `ragged=true`: the
Tile-direct, Tile-shared and scheduled-SM120 kernels have masked their
out-of-bounds loads and stores in `NVIDIALowering.cpp` since they were written.
The comment #675 added — *"the Tile-direct and scheduled kernels are still
aligned-only"* — was simply false.

**Measured A/B on Super-Bear**, same PTX, same shapes, two libraries:

| kernel | shape | merged main | fixed |
|---|---|---|---|
| `tile_matmul_direct_f16` | 256×256×256 | accepted | accepted |
| `tile_matmul_direct_f16` | 255×129×258 | **rc=5 refused** | accepted + timed |
| `tile_matmul_shared_f16` | 255×129×258 | **rc=5 refused** | accepted + timed |

A refused measurement is a candidate that returns `None` and leaves the race —
the exact corpus bias `_record_raced_the_live_field` exists to refuse, and the
one this session spent PRs #655/#662/#670 removing. #675 put a small version of
it back on the ragged path.

**Two capabilities, not one.** Even-K is a property of *this emitter's fragment
layout*, not of ragged shapes: `ld.global.b32` needs a 4-byte-aligned address
while the fragments address 2-byte elements. The Tile kernels declare
**element alignment (2)** on their masked loads
(`unsigned alignment = (f16Storage || bf16Storage) ? 2 : 4;`), so an odd
element offset is legal for them. Folding the two into one flag would have
imposed an odd-K refusal on kernels that have no such limit — trading one
regression for another. `tileLaunchConfig` now reports `ragged` (default
**true**) and `requiresEvenK` (true only for `kGemmF16`/`kGemmBf16`).

**One honest limit on the evidence.** The A/B above proves the *gate* changed,
using real PTX that JITs — a first probe used a stub that failed to compile,
and rc=3 (JIT failure) happens *before* the shape guard, which made that run
silently inconclusive rather than informative. What is **not** separately
device-proven here is that the real Tile kernels compute correctly at odd K;
that rests on the alignment-2 masked loads above **and** on the fact that
merged main's own invoke path already dispatches them at odd K. This change
makes the benchmark path agree with the invoke path rather than making a new
claim about the kernels.

Follow-up: a ragged-bucket device re-race is still owed here (recorded under
`MATRIX-LANE-RAGGED-SHAPES-2026-09-01`), and it should now include the Tile
lanes, which this fix returns to the field.

## MATRIX-LANE-RAGGED-SHAPES device evidence *(sm_120, 2026-09-01)*

The re-race owed by `MATRIX-LANE-RAGGED-SHAPES-2026-09-01`, run on Super-Bear
against `origin/main` at #676's merge. Two claims were owed and they are
different: that the emitted lane now *measures* at ragged shapes (it declined
before #675), and that the Tile lanes are still *in the field* (#676 — the
benchmark path refused them as #675 landed it).

**The field is complete at every shape: 4 timed / 0 absent.** At odd K the
emitted lane is not *absent* from the race but not *live* — `applies_to_inputs`
excludes it, which is the honest form: 3 live, 3 timed, 0 absent.

**And three of the four shapes have no supportable winner.** Nine whole-device
repeats per candidate, medians with spread beside them:

| shape | fastest | margin | noise | separated |
|---|---|---:|---:|---|
| 256³ aligned | `mma_gemm_emitted` | 14.9% | 14.6% | **no** |
| 255×129×258 ragged | `mma_gemm_emitted` | 7.6% | 17.9% | **no** |
| 100×50×70 ragged | `tile_matmul_direct` | 3.6% | 13.2% | **no** |
| 1000×999×1002 ragged | `mma_gemm_emitted` | **19.9%** | **0.1%** | **yes** |

So the only ranking this run supports is the large ragged one: at
1000×999×1002 the compiler-**emitted** PTX GEMM beats `tile_matmul_direct` by
19.9% (0.18343 vs 0.22910 ms) and the shipped delegate by 34%, at 0.1% spread.
The small shapes are launch-overhead dominated and unresolved — the same
conclusion the 64³/256³ A/B reached under `MATRIX-LANE-RAGGED-SHAPES`, reached
again by a different route.

**Why the field composition mattered, concretely.** At 100×50×70 the ordering
puts `tile_matmul_direct` first. Under #675-as-merged both Tile lanes were
refused by the benchmark path, so that shape would have raced two candidates
instead of four and recorded the emitted lane as winner with no visible
competitor. The verdict would not have been provably wrong — it is unseparated
either way — but it would have been drawn from a field missing the candidate
that happens to lead it.

**Method notes, because two of them nearly produced a false result.** The first
run reported `0 timed / 0 absent of 0 live` — an empty registry, not a device
answer: the scratch worktree lacked `libtessera_nvidia_gemm.so`, so every
candidate probed unavailable. The second run had 2 of 4 live because the Tile
lanes additionally require `tessera-nvidia-opt`, `mlir-opt`, `mlir-translate`
and `llc`; pointing `TESSERA_NVIDIA_OPT` at the box's existing build (my changes
touch no MLIR pass) completed the field. **An empty or partial field reports as
a clean-looking table**, which is exactly why the harness prints
`N timed / M absent of L live` rather than just the winner.

Run from a detached `git worktree` at `origin/main` so the box's own checkout,
branch and 16 untracked study files were never touched; the worktree was
removed afterwards and the box verified back to `verify/sep` with the same 16.

## Cross-backend sync `PRIMITIVE-ROUTE-MAP-2026-09-01`

**Owning item:** the coverage ↔ MLIR/LLVM route join
(`generated/primitive_route_map.md`) · **synchronization key:**
`PRIMITIVE-ROUTE-MAP-2026-09-01`

A shared registry and generated dashboard reporting, per primitive and per
target, whether the mainline compiler or the Python bootstrap packager serves
it. All four backends are assessed here per AGENTS.md — the first landing
(#677) updated only the NVIDIA plan, which is the omission this entry closes.

**It shipped with six false rows, and how they got there is the useful part.**
`depth_attn` was published `compiled` on NVIDIA and x86 although `driver.py`
dispatches it only under `target_kind == "rocm_gfx1151"`; `min`/`amin` were
published as ROCm and x86 reduction routes although both contracts accept only
`sum/mean/max/amax`. Both came from the same mistake: a membership that was
described as "grounded in the `tessera.*` literals each backend names" but was
actually one global tuple applied to every target, plus a compiled-route
fan-out whose comment claimed it "keeps the claim no stronger than the source"
when the source is target-aware and the fan-out made it strictly stronger.

Three guards now make that class of error mechanical rather than editorial:

* **membership is per target**, and every declared member is cross-checked
  against the `tessera.*` literals of that target's own native module (either
  the canonical Graph IR name or the public alias — `sum` is `tessera.reduce`
  in coverage and `tessera.sum` in `x86_native`, and both are real);
* **a compiled route is claimed only where `driver.py` dispatches it**
  (`COMPILED_ROUTE_TARGETS`), never fanned across targets;
* **a target that cannot be classified says so** rather than vanishing.

**Outcome for this backend: `parity validated`.** `nvidia_sm120` is fully
classified by the audit. Its rows were corrected: `depth_attn` is no longer
claimed here (rocm-only dispatch), while `min`/`amin` legitimately remain —
`nvidia_native` is the one reduction contract that names them.

## MSW-4A-CODIFF-SIGN-1 — cross-backend assessment (recorded 2026-09-02)

`ga.calculus.codiff` changed from the unsigned `⋆d⋆` composition to the true
codifferential `δ = (-1)^(n(k+1)+1) ⋆d⋆` (PR #688), and its `clifford_codiff`
VJP changed with it. That is a shared numerical contract, so each backend gets
an explicit verdict.

**NVIDIA — not applicable; there is no NVIDIA codiff to correct.** No
`clifford_codiff` entry exists for this backend in `backend_manifest.py` and no
CUDA kernel implements the operator; a JIT'd Clifford program on NVIDIA
executes `tessera.ga.*`, which is the signed Python path. If a native NVIDIA
codiff is ever added it must apply the sign at its ABI boundary the way the
Apple symbol now does — the exported name promises δ, and the mistake this
entry records is precisely a symbol named `codiff` returning `⋆d⋆`.
