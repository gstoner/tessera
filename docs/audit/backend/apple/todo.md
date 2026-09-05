---
audit_role: plan
plan_state: landing
owner: Apple backend
target: apple_gpu
last_updated: 2026-09-04
---

# Apple compiler, exact-device, and performance plan

## Cross-backend sync `FRONTEND-DTYPE-BOUNDARY-2026-09-03`

A **shared Graph IR diagnostic boundary and dtype annotation contract** landed
in [PR #706](https://github.com/gstoner/tessera/pull/706), so all four backends
are assessed here per the integrated plan's PR rule 4.

Three shared changes, none backend-specific:

1. **`GRAPH_IR_UNRESOLVED_ELEMENT_TYPE`** (`graph_ir.unresolved_element_type_diagnostics`)
   — a tensor with no element type renders `tensor<...x?>`, which MLIR rejects.
   The Apple value lane now consults this preflight *before* rendering, so the
   recorded reason names the argument and the missing semantic key (Decision
   #21a) instead of the parser's symptom. Renders are byte-unchanged.
2. **Tracer `loc`** — every traced op carries the user's call site, emitted as
   repo-relative `loc("file":line:col)` in the canonical (parser-bound) render
   only. Decision #13; the paren/golden render is byte-identical.
3. **Dtype annotations** — `Tensor["M","K","bf16"]` binds a trailing dtype
   instead of reading it as a third dim name; `tessera.bf16["M","K"]` keeps its
   `dim_names` and renders symbolic dims as `?`. `tf32` and the planned/gated
   set (`uint*`, `complex*`, `mxfp*`) are refused **by name** rather than
   demoted to a dimension (#15a/#21a).

Verification for all three was on the **Mac**, host-independent lanes only. No
device claim is made or transferred by this entry; the outcomes below are
contract assessments, not exact-device results (Decision #26).

**Apple outcome: parity validated (host-free), one behavior change to know.**
Apple is the only backend whose lowering the preflight actually gates today:
`compile_graph_module`'s value lane (`apple_target_ir_mode="value"`) refuses an
unresolved element type before invoking `tessera-opt`, and records
`apple_value_target_ir_error` with the named diagnostic. The failure was
already observable (S4); what changed is the *quality* of the reason. The
artifact route is untouched, and `tests/unit/test_apple_value_target_ir.py`
passes 123/123 on the Mac. The dead reason-discarding wrapper
`driver.lower_apple_value_target_ir` (zero callers) was deleted — Decision #29.
`loc` now rides in the Graph IR the Apple pipelines parse; `-mlir-print-debuginfo`
confirms it survives. **No follow-up owed.**

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

**Apple outcome: not applicable today — zero arbiter candidates — but this is
the backend where the contract would bind hardest.**

Apple registers no candidates in the Decision #28 arbiter, so there is nothing
to time and no evidence owed.

Recorded because the Apple analogue is unusually sharp. Metal's device timing
is command-buffer `GPUStartTime`/`GPUEndTime` plus counter sampling, and Apple
has no in-kernel constant-rate clock equivalent to `wall_clock64` — so the
independent third witness that validated the HIP event clock here would not be
available. An Apple lane entering the arbiter would be back to "the event
clock agrees with the host clock", which is the weaker check this contract was
written to move past.

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

**Apple outcome: not applicable today, same structural reason as
`DELEGATE-CONTRACT-SYNC-2026-08-30`.**

Apple registers zero arbiter candidates, so it has no rows in this corpus and
nothing to refuse. No Apple device evidence is owed.

It is worth recording what this backend would inherit if it entered. Apple's
~123 hand-written MSL kernels are the Tier-3 population Decision #28 exists to
score, and Metal command-buffer timestamps are the device-timing primitive they
would need. The NVIDIA result is the cautionary one: the *emitted* lane beat
the hand-tuned lane by 1.5–1.7× and had been invisible because it was the one
without a timer. An Apple lane brought into the arbiter without a device timer
would be excluded from its own races in exactly the same way — and on Apple the
untimed candidates would be the hand-written kernels, so the bias would run the
other direction.

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

**Apple outcome: parity validated, on device (M1 Max).** Apple owned the
reported defect: `_apple_gpu_dispatch_delta_attn` read the optionals from
kwargs only, so a `@jit` call arrived with them as trailing operands, both
resolved to `None`, and the **unweighted** rule was computed and returned as
the requested one — silently, with the extra operands dropped. It now decodes
from the flags and fails closed on trailing operands it cannot bind.

JIT vs the `erase=True` reference: max|d| **1.955e+00 -> 2.571e-07**.
`test_apple_gpu_delta_erase_routing.py` 16 passed on the M1 Max, including the
`@hardware_apple_gpu` end-to-end cases with all three optionals and with a
single optional bound to the correct slot.

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

**Apple outcome: not applicable today, for a structural reason worth stating.**

**Apple registers zero arbiter candidates.** Both changes are to the Decision
#28 candidate/arbiter layer, and no Apple lane enters it: the ~123 hand-written
MSL kernels in `runtime/apple_gpu_runtime.mm` are reached through
`runtime.launch()` and the dispatcher tables, not through `arbitrate()`. So
there is nothing on this backend for either change to alter, and no Apple
device evidence is owed for #652.

That is a gap rather than a property. Those MSL kernels are exactly the Tier-3
population Decision #28 exists to score — hand-tuned, fast, and currently
selected by dispatcher routing rather than by measurement. Until an Apple lane
registers as a candidate, the arbiter cannot compare a synthesized or emitted
Apple kernel against a hand-written one, and `emit/apple_msl.py`'s synthesized
kernels have no path to displace them.

Sequencing note: Apple is the backend where a delegate contract would bite
hardest on `determinism`, because the MSL reduction kernels' accumulation order
is threadgroup-dependent. Re-derive that claim from the MSL rather than
assuming the CUDA answer; the sm_120 delegate's `deterministic` claim rests on
one warp owning each output tile, which is not the Apple kernels' structure.

Cross-backend sync `AVX512-MARKER-AND-AMX-CONSUMER-2026-08-30` — **shared
marker vocabulary and conftest boundary changed; per-backend outcome below.**
`hardware_avx512` joins `policy.MARKERS`, the PR marker expression and its
four verbatim copies, `pyproject.toml`, and the device-accounting families.
`conftest` now consumes `hardware_avx512` and `hardware_amx` centrally,
matching the existing `hardware_nvidia` / `hardware_apple_gpu` boundaries, so
a marker that states a hardware requirement produces an honest skip rather
than a failure on a host that cannot meet it (Decision #29).

*Apple outcome: not applicable — no Apple lane carries `hardware_avx512` or
`hardware_amx`, and the `hardware_apple_gpu` / `metal4` boundaries in
`conftest` are unchanged by this. Re-verified on the M1 Max: the new
conftest arms do not intercept Apple markers, and `tests/device/x86/`
skips all three lanes there rather than failing one.*

Cross-backend sync `HOLLOW-GREEN-GATES-2026-08-30` — **shared test infra
changed; per-backend outcome below.**
A pytest session ledger (`tests/_support/device_accounting.py`) now tallies
executed-vs-skipped per hardware family and **fails the session** when a
family skipped everything on a host that plausibly has the device.

*Apple outcome: parity validated 2026-08-30, and this backend is where the
gate was proven.* `hardware_apple_gpu` probes for Darwin-on-arm, which always
has Metal, so it cannot false-positive on a Linux box.

**`metal4` is a separate family with its own capability-aware probe**
(review finding, PR #645). Folding it into `apple_gpu` was wrong in both
directions: merged, one generic Metal test that ran would set `executed > 0`
and mask a Metal 4 lane that skipped entirely; and on an Apple-silicon host
whose runtime does not report Metal 4, `require_apple_metal4()` skips
correctly and the generic Darwin-on-arm probe would have converted that
honest capability skip into a session failure. The `metal4` probe mirrors
that gate exactly — `runtime.apple_gpu_metal4_caps()["available"]` — so the
presence check and the skip decision cannot drift apart. This matters here
specifically because parts of the Metal 4 surface are macOS 27.0-gated, so
capability skips on a Metal-capable Mac are normal, not a defect.

Two pieces of evidence, in the order they matter:

* **The gate fires.** A test marked `hardware_apple_gpu` that skips on this
  Mac drives the session to **exit 1** with a named lane and remedy, while a
  normal run exits 0. This used the real probe with no mocking, because the
  Mac genuinely has the device — the one host in the fleet where that
  end-to-end proof is available without simulating anything.
* **It does not fire spuriously.** `pytest tests/unit -m "not slow"` on the
  M1 Max is **27 failed / 15475 passed / 3656 skipped** with the gate
  silent, which is correct: the Apple lanes executed. Those 27 are
  pre-existing — the subset I could isolate is identical on `main`, and
  neither conftest hook can reach the rest (no applied `hardware_nvidia`
  marker in `tests/unit`).

The Apple-specific failure mode this guards is the stale dylib: it makes
Apple GPU lanes skip *or hang* rather than fail, so `ninja -C build
TesseraAppleRuntimeShared` is named directly in the remedy text the gate
prints.

*Adafactor (`ADAFACTOR-BIAS-CORRECTION-2026-08-30`) remains not applicable
here* — unchanged, and stated with its reason below.

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

*Apple outcome: not applicable.* The Apple backend exposes no Adafactor
kernel — neither the Accelerate CPU lane nor the MSL/MPS GPU lane registers
one, so there is no Apple code path this policy reaches. The reference
`optim.adafactor` runs as host numpy here like any other pure-Python lane and
is covered by `tests/unit/test_s10_optim.py`; nothing Apple-specific to
validate.

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

*What this queue must run.* Nothing Apple-specific changed, but the Apple lane
lowers through every pass listed above, and the Mac is where the 437-test
fixture evidence came from. The one Apple-adjacent item is that this batch's
`lit` run is the only fixture evidence for all four backends' shared passes —
treat it as necessary, not sufficient, for the other three.

Cross-backend sync `LINUX-BASELINE-2604-LLVM231-2026-08-29` — **not applicable to Apple execution, but the macOS install line changed.**
The Linux baseline moves to **Ubuntu 26.04 LTS** and the compiler-backbone pin
tightens from "LLVM/MLIR 23.x" to **23.1.x exactly**; `scripts/setup_ubuntu.sh`
now FAILS on any other Ubuntu release rather than warning. `CLAUDE.md`'s host
record moved in the same change, because leaving it at 24.04 pointed this
project's own instructions at a bootstrap command that exits immediately.

Measured on the migrated box (`Princess-Luna`): Ubuntu 26.04.1, LLVM/MLIR
23.1.0 (assertions OFF), ROCm 10 series (HIP 7.15), repo at
`~/programming/tessera`, ssh on the default port.
*Apple outcome.* The Ubuntu baseline does not touch this lane, but the same
change corrects the macOS instructions, which matters here: `GETTING_STARTED.md`
told users to `brew install llvm --HEAD`, and HEAD tracks LLVM's development
branch — under the new 23.1.x pin that aborts configure. Homebrew's **stable**
`llvm` formula is 23.1.0 (verified: `brew info` reports stable 23.1.0, and it is
what this Mac runs), so the stable formula is now what the doc installs.
Cross-backend sync `SHARED-CONTRACTS-P1-REVIEW-2026-08-29` — **assessed; Apple execution unchanged, two Apple-specific fixes verified on device.**
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
*Apple outcome.* The four shared contracts are host-free here. Two
Apple-specific P1s in the same PR were verified on this Mac's Metal runtime:
the tiled MSL kernel no longer raises an uncaught KeyError for a valid
`layer_norm` region (it takes the documented reference fallback, exact match,
while `softmax` still runs on the tiled GPU lane), and mixed-dtype attention no
longer bit-reinterprets an f32 Q against f16 K/V — measured error fell from
3.42 to 4e-4 against the f32 reference, where the wrong result had been
returned tagged `metal_runtime` as if correct.

Cross-backend sync `FOUNDATION-LLVM231-REVIEW-P0-2026-08-29` — **this box is
the one the foundation moved on; done here, and it is where the review's
verification was performed.**

*What changed on this host.* LLVM/MLIR is now Homebrew's **production `llvm`
keg 23.1.0** at `/opt/homebrew/opt/llvm`; the manual pre-release
`/opt/homebrew/llvm-23.1.0-rc1` prefix (`23.1.0git`, assertions **ON**) was
deleted by owner decision. The keg is **NDEBUG**, so this box is no longer the
fleet's MLIR-assertion falsifier — see the Decision #19 update in `CLAUDE.md`
and the `APPLE-VECTORIZE-1` note below; the from-source assertions recipe is
preserved in this file. The Metal 4 evidence lane's pinned prefix and
`TESSERA_LLVM23_PREFIX` default now point at the keg;
`validate_apple_metal4_evidence.py` accepts **either** pin so packets sealed
before 2026-08-29 stay valid as recorded.

*Fallout that had to be repaired, and the general lesson.* Removing the rc1
prefix stranded four build trees — `build-apple`, `build-asan`, `build-tsan`,
`build-ubsan` — whose `CMakeCache.txt` still pin it and whose binaries link
`@rpath/libLLVM.23.1git.dylib`. That alone caused **~86 unit failures**, 69 in
`test_apple_value_target_ir.py`, every message about a missing dylib rather
than about anything under test. `build_artifacts.py` had been hardened for
exactly this in Aug 2026, but three *other* discovery paths still checked only
existence — and two are **product** code that actively *preferred*
`build-apple` on Darwin: `compiler/driver.py::_resolve_tessera_opt` and
`_jit_boundary.py::_find_tessera_opt` (plus `tests/_support/compiler_tool.py`).
All now require the binary to start. Lesson worth carrying: fixing one
discovery path does not fix the class — after removing any toolchain prefix,
sweep `grep -l <prefix> build*/CMakeCache.txt` and `otool -L` the drivers.

*Also repaired here:* `ninja check-{clifford,ebm,spectral}` were dead because
the brew keg ships no `llvm-lit` (they now fall back to a standalone `lit` and
pass `BUILD_DIR` so the suite locates its own driver); and
`TESSERA_BUILD_X86_BACKEND=ON` is now safe on this Mac — `src/CMakeLists.txt`
skips only the native AVX-512/AMX kernel subdirectory on a non-x86 host, so
`ninja -C build` completes and `lit tests/tessera-ir/` goes **414/425 →
425/425**. Enable it here: this is the only fleet host *without* AVX-512, hence
the only one whose green result on the x86 Target IR fixtures is evidence of
host portability (Decision #19).

*Verification performed on this box for the whole review batch:* unit suite
**114 → 27 failures** (15263 passed), the 27 all citing `sm_120`, `gfx1151` or
`x86_64_avx512`; a stash-and-rerun over exactly the failing files gave
**identical failure sets (22 = 22), i.e. zero regressions**; `lit
tests/tessera-ir/` 425/425 and Clifford 16/16; `mypy` restored from a hard
abort to **0 errors over 480 files**; `ruff` clean on `python/tessera/` and
every touched file. A pre-existing hard **bus error** in
`test_numeric_policy_carrier_execution.py` was also fixed: it hand-rolled the
x86-64 large-struct-return convention (result pointer as first argument), which
is wrong on AArch64 where the indirect return slot is `x8`. It now calls MLIR's
`_mlir_ciface_` wrapper, whose ABI is explicit and target-independent — the
oracle still reproduces the bf16-accumulator defect it exists to catch.

Cross-backend sync `IKF-INTRA-KERNEL-CONTRACT-2026-08-27` — **follow-up
required at IKF-P6; Apple execution unchanged, and the clock primitive is
unverified.** The IKF-1 intra-kernel measurement plan
(`docs/audit/compiler/INTRA_KERNEL_FEEDBACK_PLAN.md`, PR #634) is proven
first on ROCm gfx1151. Two Apple-specific gates before any P6 scoping:
(1) an in-kernel constant-rate timestamp primitive readable from MSL is
**not established** — ground it in the on-machine SDK headers per Decision
#27 before writing a lowering or a "blocked" verdict (encoder-boundary
counter sample buffers are not a per-instance in-kernel clock); (2) the
lowering must cross the Python-synthesizer/MLIR seam, since the executing
Apple GPU lane is runtime-delegated — IKF instrumentation of synthesized MSL
lands in `emit/apple_msl.py` territory, not in the MLIR passes alone. No
gfx1151 evidence transfers.

Cross-backend sync `X86-AVX512-IMAGE-ADMISSION-2026-08-27` — **x86-only image
repair; Metal outcome not applicable.** Canonical CPU-feature admission now
guards the legacy AVX-512 shared-library loader and the x86 FFT translation
unit performs no vector work during ELF initialization. This changes no MSL
producer, Metal runtime ABI, Apple selector, or Apple-device evidence. Apple
must continue to own its architecture admission independently; neither the
AVX2 safe-decline result nor the closed 79-case Zen 5 FFT/solver packet
transfers to Metal.

Cross-backend sync `CUDA-SOLVER-KRYLOV-SCALE-2026-08-27` — **not applicable to
Metal physical execution.** The additive dense-Krylov contract is explicitly
owned by `nvidia_sm120`. CUDA cooperative-grid synchronization, deterministic
CTA partials, SM120 `mma.sync` low-precision matmul, and RTX device/performance
evidence create no MSL producer, Metal ABI, or Apple-device claim. A Metal
solver must choose an Apple-owned synchronization/reduction architecture and
provide independent correctness and performance packets.

Cross-backend sync `CUDA-SOLVER-FAMILY-2026-08-27` — **not applicable to Metal
physical execution.** Shared residual-product packaging now retains authored
matmul numeric policy and refuses to substitute GMRES for a declared CG
contract. NVIDIA added CUDA-owned unary/reduction/predicate/where/IEEE-matmul
children and a single-launch diagonal-SPD CG package. No MSL producer, Metal
runtime ABI, Apple selector, or exact-device claim follows; Apple solver/CG
adoption requires an architecture-owned package and Apple device packet.

Cross-backend sync `CUDA-SOLVER-IFT-PILOT-2026-08-27` — **not applicable to
Metal physical execution.** The shared diagonal-sqrt and general solver
contracts gained an explicit SM120 consumer, CUDA binary residual replay, and
exact CUDA evidence. Apple receives no Metal
solver implementation, selector, or device claim from that architecture-owned
package; a future Metal solver must provide its own admission and exact-device
packet.

Cross-backend sync `CUDA-BINARY-SPECTRAL-JVP-2026-08-27` — **NVIDIA-owned
physical closure assessed; Metal outcome not applicable.** The compiler-emitted
CUDA binary family and its spectral filter/convolution tangent accumulation
change no MSL producer or Apple runtime contract. SM120 f16/bf16 storage and
RTX numerical evidence transfer no Apple7 claim; Metal spectral adoption still
requires an architecture-owned package and exact-device proof.

Cross-backend sync `CUDA-SPECTRAL-JVP-NUMPOL-2026-08-26` — **NVIDIA-owned
closure assessed; Metal outcome not applicable.** The new public JVP child,
cuFFT Schedule→Tile profile, and f16/bf16 CUDA storage ABIs change no Apple
Target consumer. Exact SM120 numerics and certificates transfer no Apple7
claim; a future Metal spectral package still needs architecture-owned
implementation and exact-device evidence.

Cross-backend sync `CUDA-OPTIMIZER-VJP-2026-08-26` — **shared optimizer plugin
ownership assessed; Metal follow-up required.** NVIDIA was added as an exact
owner for the existing state-lineage carrier, but CUDA PTX schedules and
`sm_120` attestations do not transfer to Metal. Apple still has no registered
optimizer-VJP or Adafactor-VJP Target consumer; adoption requires an Apple-owned
package and Apple7 numerical certificates.

Cross-backend sync `TSOL-CUDA-POLICY-V1-2026-08-26` — **NVIDIA-owned physical
package assessed; Metal outcome not applicable.** The SM120 CUDA ABI and
logical interleaved-complex transform capability introduce no shared Metal
schedule or Apple runtime contract. Apple still has no native spectral Target
consumer, so CUDA device numerics, adjoints, streaming certificates, and
performance evidence transfer no Apple row.
The later `CUDA-SPECTRAL-JVP-NUMPOL` assessment above closes the NVIDIA-only
follow-ups without changing this Metal disposition.

Cross-backend sync `TSOL-POLICY-PHYS-1-8C8G-2026-08-26` — **shared spectral
carrier assessed; Metal outcome remains not applicable.** The shared artifact
now binds runtime-stride ABI identity, independent transform/window lengths,
full versus one-sided spectrum policy, and digest-chained streaming state.
Apple still has no native spectral Target consumer, so the independent x86 and
gfx1151 broadcast/streaming execution packets transfer neither a Metal schedule
nor correctness claim. The exact-gfx1151 expanded reverse/VJP packet also
transfers no Metal adjoint evidence. Broadcasting and all Metal physical rows
remain open.

Cross-backend sync `TSOL-POLICY-PHYS-1-8B-2026-08-26` — **shared logical-axis
contract assessed; Metal outcome not applicable.** Centered STFT and
centered/cropped ISTFT now carry arbitrary normalized logical axes plus
`outer`/`inner` physical indexing through Schedule→Tile, while admission still
requires C-contiguous storage. Apple has no native spectral Target consumer,
so the independent AVX-512 and gfx1151 packets transfer neither a Metal
schedule nor device evidence. True non-contiguous strides, full spectrum,
broader lengths, broadcasting, and streaming remain open shared policy rows.

Cross-backend sync `TSOL-POLICY-PHYS-1-8A-2026-08-26` — **shared policy
carrier assessed; Metal outcome not applicable.** Center, pad mode, crop, and
explicit ISTFT length now participate in the compound Schedule→Tile identity,
but Apple still has no native spectral Target consumer. The independent x86
and gfx1151 numerical packets transfer no Metal schedule or device evidence;
an Apple package requires its own implementation and exact-device oracle run.

Cross-backend sync `AD-TSOL-STFT-GFX1151-2026-08-26` — **shared carrier
assessed; Metal outcome not applicable.** The independent gfx1151 n=16/n=18
STFT/ISTFT implementation and exact-device certificates transfer no Apple
schedule or evidence. The compound-spectral Schedule→Tile artifact now binds a
structured storage/fp32-accumulator policy, but Apple has no Target consumer;
any future Metal row needs its own package and exact-device proof.

Cross-backend sync `E2E-REAL-6F-EXACT-CERT-2026-08-26` — **shared
family/target gate assessed; Apple normalization remains one blocking row.**
The x86 and gfx1151 packets provide no Metal evidence. Apple must add and run
an Apple7 normalization packet whose certificate carries a runtime-origin
Metal physical attestation; an execution-mode string or mocked launch is
explicitly insufficient.

Cross-backend sync `BOUNDED-GATE-RELAXATION-2026-08-26` — **shared
control-scan normalization assessed; Apple physical outcome not applicable.**
The paired reverse pass now admits only the statically bounded symbol-body
`control_scan` form by first consuming the canonical SCF lowering; payload,
dynamic, and malformed forms keep the named refusal. Reduced-precision WMMA is
gfx1151-only; the AVX-512 packed-C2R and gfx1151 direct-DFT STFT/ISTFT packages
transfer no Metal claim. The shared execution-certificate carrier changes host
evidence structure, but Apple has no Adafactor Target consumer and inherits no
x86 certificate. The
rank-local MPI transport now consumes all five explicit Schedule→Tile
collective SSA forms and has two-process x86 host evidence, but no Metal
launcher consumes it and no Apple transport claim transfers. The
composed-layout quotient correction is confined to
duplicated ROCm/CUDA materializers; Apple has no such physical consumer, so it
is not applicable and no M1 Max kernel or device evidence is implied.

Cross-backend sync `W4-EFFECTS-1-E5-2026-08-25` — **one physical family carrying an admissible effect, end to end; Apple outcome: not applicable — no row claimed.** As with NVIDIA: no M1 Max row was run and none is implied. The keyed-RNG class rests on counter-based purity, which holds on any target, so an Apple row is a matter of running it rather than of new design.


Cross-backend sync `W4-EFFECTS-1-E4-2026-08-25` — **ordered-collective
recorded products (identity only); Apple outcome: not applicable today.** The product
binds communicator, issue order, reduction algorithm and topology; the
verifier rejects a permuted order and a changed tree under an identical
order. Order evidence comes from the deterministic mock-mesh executor.
No Metal collective surface consumes this; no M1 Max evidence is implied.


Cross-backend sync `W4-EFFECTS-1-E3-2026-08-25` — **shared state-lineage
identity change; Apple outcome: not applicable today, inherited on
adoption.** Same reason as the nvidia entry — host-side package identity, no
Metal surface consumes it, no M1 Max evidence implied.


Cross-backend sync `W4-EFFECTS-1-E2-2026-08-25` — **shared autodiff gate
change (AutodiffPairedPass); Apple outcome: not applicable, no behaviour
change.** Same reason as the nvidia entry — a target-neutral diagnostic split
over a fail-closed gate. No Metal surface changes and no M1 Max evidence is
implied.


Cross-backend sync `W4-EFFECTS-1-2026-08-25` — **UPDATED 2026-08-25 (slice E1
landed): shared recorded-product carrier + verifier implemented in Python;
Apple outcome: not applicable at this slice, follow-up on adoption.** E1 is
host-neutral Python; no Metal surface consumes it and no M1 Max row is
claimed. Noted for adoption: the keyed-RNG class rests on counter-based
purity, which holds for the S4 generator on any target, and the carrier is a
single serialized payload so Apple's Python-synthesizer / C++-pipeline seam
does not need a third representation.

Cross-backend sync `SPECTRAL-PAYLOAD-CHAIN-2026-08-25` — **shared
Schedule->Tile spectral identity contract + pipeline carrier ordering; Apple
outcome: not applicable today, inherited on adoption.** Apple has no Metal
consumer for the scheduled spectral program, so nothing in the Apple lane
compiles through the changed verification and no M1 Max evidence is implied.
Noted for whoever adds one: the contract now requires BOTH preimages on the
module, so an Apple producer must carry them rather than only the digest.


Cross-backend sync `SCHEDULE-AUTHORITY-RESHARD-2026-08-24` — **shared SO-3 and W5.4 contracts assessed; Apple physical follow-up not applicable yet.** Pipeline and compound-spectral lowering now consume one digest-bound Schedule Object, inferred producer edges, roles, and resource evidence without scalar reconstruction. Placement now emits exact mesh-sized local-shard/collective SSA and executes all movement forms on the deterministic mock mesh. These Graph/Schedule/Tile contracts are portable, but x86/gfx1151 spectral numerics and mock transport transfer no Metal schedule or Apple-device claim. A future Metal distributed consumer must bind its own mesh/runtime proof. `NUMPOL-CARRIER-1` (queue row 3b) owns the generalized S5 carrier; no Apple Target change lands in this synchronization.


Cross-backend sync `SO3-INFER-EDGES-2026-08-24` — **shared W2.1/W5.2e
dependence-inference semantics + MegaMoE R3 producer; Apple outcome: not
applicable today, inherited on adoption.** Same reason as the nvidia entry:
host-side schedule analysis only, no Metal surface consumes it, no Apple
artifact or numerical result changes. Apple's MoE/transport families would
inherit the corrected inference if and when they adopt the overlap plan.


Cross-backend sync `NUMPOL-CARRIER-1-2026-08-24` — **shared Schedule→Tile
`numeric_policy` carrier contract (integrated-plan queue row 3b); Apple
outcome: follow-up required.** Newly owned row, nothing implemented yet.
Apple consumes the contract through the MSL emitters and the value-preserving
Target-IR lane; its `simdgroup_matrix` coopmat path is the natural first
carrier site (the accumulator choice there is exactly what the policy must
survive to reach). Note the standing seam: the Python MSL synthesizer and the
C++ MLIR pipeline are two disconnected compilers, so the carrier must be
designed not to require a third policy representation. M1 Max evidence
required before any numerical claim.


Cross-backend sync `LAYOUT-ALG-APPLE-PHYSICAL-2026-08-24` — **reachable Metal
physical-consumer tail closed.** The versioned native layout ABI now exports
the existing `Rank2Index.h` coordinate plan to source emitters without a Python
fallback. Steel Tile GEMM plus the scalar, tiled, cooperative-matrix,
reduction, normalization, attention, and gated-matmul MSL families request
their A/B/C rank-2 order from that authority before emitting address text.
All 20 reachable `(raster_order, raster_group)` combinations retain the
established row-major expressions. A fresh Apple7/M1 Max process using the
rebuilt runtime passed the compiler-selected canonical GEMM cohort 3/3,
including `13x16 @ 16x11` ragged zero-pad/store, and the fused cooperative
matrix pointwise/reduction cohort 38/38 against independent NumPy oracles.
Dynamic non-separable tuple codomains remain fail closed; no raster choice or
production-route promotion is implied.

Cross-backend sync `LAYOUT-ALG-L5-X86-2026-08-24` — **shared contract assessed;
reachable Metal rank-2 consumer now closed independently.** The x86 target pass now consumes static,
bounded-dynamic, nested, and tuple-product layouts through the shared proof,
but its scalar CPU `div/rem`, runtime assertions, and AVX2/AVX-512 evidence do
not define MSL address arithmetic or prove an Apple device. Non-affine and
non-separable tuple codomains remain fail closed fleet-wide.

Cross-backend sync `LAYOUT-ALG-L4-X86-2026-08-24` — **shared authority parity
and Metal migration closed.** AVX-512 evidence did not transfer: Apple now
consumes the same C++ rank-2 plan through the no-fallback native ABI, retains
the established emitted arithmetic, and owns fresh M1 Max numerical proof.

Cross-backend sync `LAYOUT-ALG-L3-L5-DYNAMIC-2026-08-24` — **shared proof and
carrier parity assessed; reachable Metal rank-2 adoption closed.**
Factorization/residency and Schedule Object v2 proof fields are portable. The
new tuple-product Tile carrier is registered and static components retain the
same native algebra proof; Apple has no reachable tuple-product schedule and
keeps that carrier fail closed. CUDA/ROCm
device results, shared-panel/LDS schedules, and index-template changes do not
transfer. Apple's reachable MSL indices now use shared authority with independent
device proof; this does not admit a dynamic or non-separable tuple layout.

Cross-backend sync `DYNAMIC-COMPOSED-SM120-2026-08-24` — **shared carrier
parity assessed; unsupported Metal dynamic layouts remain fail closed.** Nested
outer shape/stride trees with dynamic scalar-affine leaves now verify through
the common Tile contract, but Apple has no reachable schedule that requests
that dynamic materialization. CUDA's six-scalar strided ABI, CTA mapping, and RTX
5070 numerical proof transfer no MSL address rule or Apple-device evidence.
Tuple-valued/non-affine carriers continue to fail closed.

Cross-backend sync `SCHEDULED-MATMUL-TAIL-EPILOGUE-LDS-2026-08-24` — **shared
Graph lineage assessed; CUDA/ROCm physical work not applicable.** Optional
matmul bias and residual are now real ordered Graph SSA operands rather than
attribute-only names. Apple's scheduled consumer rejects that fused form until
a Metal-owned descriptor and epilogue materializer exist. SM120 K-tail
`cp.async`, the CUDA reduced-output ABI, gfx1151 multi-wave LDS staging, and
both devices' evidence transfer no MSL schedule or Apple proof. Dynamic/nested
composed-layout materialization remains an Apple follow-up if a Metal consumer
is introduced.

Cross-backend sync `SM120-MACRO-CTA-2026-08-24` — **not applicable to Apple
physical lowering.** The macro operation is registered in NVIDIA Target IR and
now owns f16/bf16 storage, CUDA warp assignment, two-stage `cp.async` shared
panels, M/N zero-fill tails, barriers, launch geometry, and RTX 5070 evidence.
Apple retains its target-owned SIMD/threadgroup mapping; no MSL ABI, Metal
schedule, dtype promotion, or device result transfers. The shared composed-
layout carrier itself is unchanged.

Cross-backend sync `SM120-SCHEDULED-LICM-2026-08-24` — **not applicable to
Apple codegen.** The pass ordering and all numerical/performance evidence are
owned by NVIDIA's private Tile-to-PTX package. Apple retains its target-owned
SIMD/threadgroup layout and Metal pipeline; neither the CUDA `16x8` coordinate
ABI nor a future CUDA macro-CTA staging schedule transfers.

Cross-backend sync `SM120-BLOCK-COORDINATE-2026-08-24` — **not applicable.**
The new operation belongs to NVIDIA Target IR and encodes CUDA CTA axes; it
changes no Metal threadgroup mapping, MSL ABI, or Apple exact-device evidence.


Cross-backend sync `CUTE-LAYOUT-MATERIALIZE-1-2026-08-23` — **shared static
affine coordinate ABI; reachable Apple rank-2 subset closed independently.**
Metal source emitters now obtain their coordinate order from the same native
C++ plan and have M1 Max proof. The general dynamic/tuple Tile producer remains
fail closed because no Apple schedule requests it. NVIDIA's SM120 view mapping
and RTX 5070 proof transfer neither an MSL address mapping nor Mac evidence.

Cross-backend sync `ROCM-CI-HSACO-SERIALIZE-2026-08-23` — **ROCm-owned host-free
CI serialization lane; Apple outcome: not applicable.**
Apple's device-code path cannot be exercised this way on a Linux runner: MSL →
AIR → `.metallib` packaging shells out to `xcrun metal` / `xcrun metallib`
(Decision #26a), which requires macOS and the Metal toolchain, so there is no
GPU-less hosted-runner equivalent of the `ld.lld` serialization proof. Apple
lit fixtures already gate on the `tessera-apple-backend` feature. No `.metallib`,
MSL, or M-series evidence transfers.

`CI-BACKEND-CAPABILITY-SKIP-2026-08-23` — **Apple-owned; pytest fixtures now
skip when `tessera-opt` lacks the Apple backend. Landed.**
`tessera-opt` registers the Apple pipelines only under
`-DTESSERA_BUILD_APPLE_BACKEND=ON`. The lit suite already derived a
`tessera-apple-backend` feature by probing `tessera-opt --help`
(`tests/tessera-ir/lit.cfg.py:129`), but the **pytest** fixtures had no
equivalent: their only guard asked whether the binary exists, not whether it
carries the backend. On the ROCm/x86 boxes that produced **57 failures** in a
full sweep — `Unknown command line argument '-tessera-lower-to-apple_cpu-full'`
and the `--pass-pipeline` parser's equivalent — noise that says nothing about
Apple and hides real signal (this is how the ROCm serializer outage in #619 went
unnoticed). `tests/_support/apple.py::skip_if_apple_pipeline_unregistered()`
brings pytest in line with what lit already did.

Narrow by construction, so it cannot mask a defect: it fires only on the two
"pipeline not registered" signatures, only when the probed binary genuinely
lacks the backend (if it HAS it, an unregistered pipeline stays a loud
registration regression), and never on a nonzero exit for any other reason. The
probe takes the **invoked** tool path — the fixtures disagree on how they find
`tessera-opt` (`TESSERA_OPT` vs `TESSERA_OPT_PATH`, different default build
dirs), so resolving independently could inspect a different build than the one
that failed (#620 review).

Evidence (Strix Halo, Apple backend absent): probe returns False for the Apple
pipeline and True for `tessera-lower-to-x86` on the same binary, and gives
different answers for two different binaries, so a False is real signal;
CMakeCache agrees independently of `--help`; **with the probe forced True the
fixtures FAIL rather than skip**, which is the property that keeps this from
silently disabling the Apple suite on the Mac. Sweep 59 → 17 failures.

**Open / not closed by this item:** the remaining 17 (all
`tests/unit/test_apple_value_target_ir.py`) fail through the Python front door
(`KeyError: 'compiler_path'`, `target_ir_artifact` vs `value_target_ir`) and a
`--help` introspection assertion, not through a pipeline invocation. Guarding
those needs an Apple build to confirm they still RUN; the ROCm box cannot build
the Apple backend, so a guard written there could mask a real regression.
**Requires a Mac run** — exact-device Apple evidence is unchanged by this item
and none is claimed.


Cross-backend sync `NVIDIA-AOT-PACKAGE-V1-HARDEN-2026-08-22` — **NVIDIA-owned
fatbin/cubin runtime admission; Apple outcome: not applicable.** Embedded CUDA
image metadata, CUDA-driver compatibility checks, NVRTC fallback, and NVIDIA
C-ABI cache-key inspection transfer no metallib, Metal ABI, selector, or Mac
evidence. Apple's architecture-owned AOT package remains unchanged.

Cross-backend sync `NVIDIA-FFT-WORKSPACE-1-2026-08-22` — **NVIDIA-owned cuFFT
ABI; Apple outcome: not applicable.** The reusable cuFFT plan and explicit CUDA
device-workspace lifecycle transfer no Metal/MPSGraph code, plan, workspace, or
evidence. Apple's spectral package retains its architecture-owned lifecycle and
Mac proof requirements.

Cross-backend sync `JIT-MATH-AUDIT-2026-08-23` — **Apple outcome: CLOSED
on the M1 Max 2026-08-23; both follow-ups audited exact-device, with three
contract violations found and fixed and one recorded open.** The original
statement of the follow-ups is kept below; the measured outcomes are in
`APPLE-MINMAX-1` and `APPLE-VECTORIZE-1`. Summary: the optimizer eps-floor
pattern does **not** exist on Apple (no `max(stat, eps)` anywhere in the
runtime — every eps is additive), but the same NaN-suppressing-`max` family
was found in three other places; the ±0 tie contract was violated on device
and is now fixed and pinned; and the vectorize lane aborted on Darwin on the
first compile until a missing dialect-extension registration was fixed.

Original follow-up statement (2026-08-23, pre-audit):
The x86/ROCm math-correctness loops (see those plans under this key)
produced changes whose Apple siblings are unaudited: **(1) NaN
laundering in optimizer eps floors** — ROCm's Adafactor kernels and the
x86 AVX-512 shim both floored second-moment statistics with
NaN-suppressing max (`maxnumf` / `std::fmax`), silently converting a
NaN statistic into eps while the optim.py reference (`np.maximum`)
propagates it; both are fixed with exact-device NaN-propagation tests.
The hand-written MSL optimizer kernels in `apple_gpu_runtime.mm` (and
any `fmax`-style eps floors in the softmax/norm MSL) need the same
audit on the Mac — MSL `fmax` is also NaN-suppressing. **(2) The
`TESSERA_JIT_VECTORIZE` lane was un-gated on LLVM 23** and its
runner-utils dlopen removed (x86 plan, `JIT-VECTORIZE-UNGATED` key);
the Mac is also LLVM 23, so the Darwin vectorize compat test in
`test_native_cpu_jit.py` will now exercise the actually-vectorized
path for the first time there. AVX-512-host proof does not transfer:
run the native CPU JIT packet + that test on the M1 Max before
claiming the vectorized lane on Apple. Also note the softmax
`maximumf → maxnumf` running-max switch is ROCm-kernel-only and does
not touch Apple's MSL softmax. **(3)** The ±0 tie contract for
`tessera.maximum`/`minimum` is now IEEE-754-2019 fleet-wide
(`IEEE-MINMAX-CONTRACT-2026-08-23` in the rocm plan): gfx1151 and the
x86 shim are fixed and pinned; the Apple MSL binary max/min kernels
(MSL `fmax`/`fmin` are NaN-suppressing AND tie-order unspecified) need
the same audit + explicit-expectation tie tests on the Mac.

Cross-backend sync `NVIDIA-RNG-PHILOX-CORE-2026-08-21` — **NVIDIA-owned
Target/runtime addition exact-device validated; Apple outcome: not applicable.** The typed four-mode
`tessera_nvidia.philox` directive and GPU kernel generator are confined to the
CUDA backend. They reuse the shared explicit key/counter semantics but transfer
no Metal code, schedule, ABI, or evidence. Apple's existing Philox alignment
and exact-Mac proof obligations are unchanged.

Cross-backend sync `JIT-VECTORIZE-UNGATED-2026-08-23` /
`JIT-CACHE-BLOCK-2026-08-23` / `JIT-MATH-AUDIT-2026-08-23` — **shared
`tessera_jit` boundary/pipeline changes; Apple outcome: CLOSED (M1 Max,
2026-08-23) — see `APPLE-VECTORIZE-1`. The lane aborted the process on the
first vectorized compile here; fixed in the shared `tools/tessera-jit`, then
measured 1.8 → 53.7 GFLOP/s at n=256.** Original statement: The x86 plan owns these
keys. Four changes reach the Mac's `@jit(target="cpu")` lane directly:
(1) the invoke signature guard (new `tessera_jit_signature` ABI +
fail-closed shape/dtype/arity validation); (2) the fastmath narrowing
to reassoc|contract (the old `fast` stamp was NEON-relevant too — the
original stamp rationale was measured on M1); (3) the vectorize lane
un-gating — the Mac is also LLVM 23, so the previously-Darwin-gated
compat test in `test_native_cpu_jit.py` (now host-agnostic) will
exercise the actually-vectorized path there for the first time; and
(4) the removal of the `libmlir_c_runner_utils` dlopen (which used a
hardcoded Homebrew path on the Mac) in favor of the in-process
`memrefCopy`. Rebuild `tessera_jit` on the M1 Max and run the
signature-guard, totality, native-CPU-JIT, and vectorize tests there —
AVX-512-host evidence transfers for none of this; NEON codegen through
the same MLIR pipeline is expected-portable but unproven.

Cross-backend sync `JIT-ELEMENTWISE-LINALG-2026-08-21` — **shared
`tessera_jit` pipeline change; Apple outcome: parity validated (M1 Max).**
The fresh `build-apple` JIT now runs MLIR's
`convert-elementwise-to-linalg` at module scope, so nested control flow and
generated AD tensor arithmetic share the same pointwise index-space lowering;
an explicit postcondition rejects any residual tensor-valued elementwise op
before bufferization.  The native CPU packet (native CPU JIT, production phase
1/3, paired state machine, and the new arithmetic/math totality matrix) passed
**74/74**.  Coverage includes add/sub/mul/div, min/max and NaN-number variants
with bit-level NaN/signed-zero checks, cmp/neg/select, scalar-condition
whole-tensor select, six tensor math ops, dynamic extents, and an executed
compiler-generated forward JVP.  The separate
exact-Metal cross-target packet passed **12/12** using the fresh CPU JIT as its
oracle.  This is Apple CPU-JIT closure plus Metal parity evidence; it does not
transfer an x86 or GPU-device implementation through `tessera_jit`.


Cross-backend sync `AD-DATUM-POLYGAMMA-2026-08-21` — **autodiff reference
numerical policy, wave 3; Apple outcome: parity validated (M1 Max).** The
lgamma/digamma datum-derived pairs and the RMSNorm gamma envelope are covered
by the switched-registry retirement/datum tests (**51 passed**).  On the exact
Apple GPU route, a fresh owned `libTesseraAppleRuntime.dylib` passed the
GA/EBM native-lane and value-lane backward follow-up: `test_apple_gpu_ebm_lane`,
`test_apple_gpu_clifford_lane`, `test_scheduled_attention_backward_consumers`,
and `test_apple_flash_attn_backward` reported **37 passed, 23 deselected**
under `hardware_apple_gpu`.  The latter is execution evidence for the native
consumers, not a claim that an Apple kernel evaluates the scalar polygamma
reference directly; as in AD-RETIRE-1, those lanes use their own analytic
references and fp32 MSL routes.


Cross-backend sync `AD-RETIRE-2-2026-08-20` — **autodiff reference numerical
policy, wave 2; Apple outcome: parity validated (M1 Max).** The structured
softmax/logsumexp/RMSNorm-core jet-derived pairs and the six datum-grown scalar
rules pass the switched-registry retirement tests (**51 passed**, shared with
the wave-3 closure).  The fresh-runtime GA/EBM and value-lane backward packet
recorded above also passed **37/37** selected exact-Metal tests.  This closes
the Apple follow-up without transferring ROCm or x86 evidence; the device lanes
remain analytically referenced and execute their architecture-owned fp32 MSL
implementations.


Cross-backend sync `AD-RETIRE-1-POINTWISE-2026-08-20` — **autodiff reference
numerical policy; Apple outcome: parity validated (M1 Max), and neutral by
construction — stronger than the expected reasoning.** PR #600 retires the 13
ODE-family pointwise hand rules behind the `DerivativeContract` datum
(dtype-preserving reference rules; unified log/sqrt boundary guard — see the
rocm entry for the full contract statement). Verified on the M1 Max
(2026-08-20), with a method note that corrects the sync note's premise:

* **The retirement is not on the Apple execution path at all.** A per-file
  probe that wraps the 13 ODE `_JVPS`/`_VJPS` registry entries and counts
  invocations recorded **0 hits across all 165 Apple-marked files** — Apple's
  native VJP lanes (normalization / flash_attn backward / EBM losses) compute
  an analytic reference inline and run fp32 MSL kernels; they never dispatch
  through the scalar-pointwise numpy registry the retirement touched. So the
  note's "both sides read the same updated reference" is not what happens —
  the Apple lane reads *neither* rule for these ops, and is invariant to the
  swap by construction. The probe is not blind to a real call: the same probe
  counts 4 VJP hits on an end-to-end `grad(tanh∘log∘sqrt∘exp)` and 62 on
  `test_retired_pointwise.py` (falsifiability control).
* **A/B over the whole Apple-marked suite is byte-identical.** All 165 files
  were run per-file (one process each — the full-process marker run trips a
  pre-existing segfault in the Apple `kv_cache_read` native path, unrelated to
  this change and passing in isolation) under BOTH the derived production
  registry and the retired #31 oracle re-registered. **1175 passed under each,
  0 differences** in pass/fail counts across all 165 files.
* **The contract changes were corroborated directly on-host anyway**, even
  though the Apple lanes bypass them: in-domain **bit-parity, 0/13** value
  divergence derived-vs-retired; the dtype fix reproduced — **10/13** retired
  rules promote fp32→fp64 in forward mode while all 13 derived rules stay fp32
  (the 3 that already matched are the tanh/sin/sigmoid hand rules the commit
  named as dtype-preserving); the log/sqrt boundary adjoint identity
  `⟨Jv,u⟩=⟨v,Jᵀu⟩` holds for the derived pair at `x<1e-12` and was violated by
  the retired pair. This is the sense in which the change "moves the reference
  TOWARD the fp32 native lanes", shown on the Mac rather than assumed.


Cross-backend sync `APPLE-RUNTIME-SINGLE-IMAGE-2026-08-19` — **Apple runtime
loading; Apple outcome: parity validated, no Metal evidence changed.**
The single-image slice fixes duplicate loading of the Apple GPU runtime.
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
Apple impact: this is the Apple runtime loader. Measured on the M1 Max with a
stale `build/libTesseraAppleRuntime.dylib` (2026-08-16, missing the four
low-precision `*_bwd_{f16,bf16}` sentinels): two resident images and a
duplicate-class warning on every run before, one image and no warning after,
with a current prebuilt still accepted without a from-source rebuild. No Metal
kernel, `.metallib`, or exact-device row changes — the change is which image is
loaded, not what it computes — so no Apple GPU proof is re-run or re-claimed.
`apple_gpu_prebuilt_skips()` now reports a rejected candidate and names the
remedy (`ninja -C build TesseraAppleRuntimeShared`), so a from-source rebuild
caused by a stale build is explainable instead of looking like "no prebuilt
existed". Note for anyone reading old logs: because the two images held
separate statics, an error set in one was invisible to a reader bound to the
other, so pre-fix runs on a host with a stale `build/` could report a clean
last-error channel that had never been consulted.


Cross-backend sync `APPLE-STUB-BINARY-OPCODES-2026-08-19` — **shared runtime
contract; Apple outcome: parity validated off-device, exact-device evidence
unchanged.** The portable-stub opcode slice fixes a silent wrong-answer class in the Apple
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
Apple impact: this is the Apple backend's own non-Darwin reference path. Metal
was always correct — `mpsg_binary_node` implements the full table — so no Metal
kernel, `.metallib`, dylib, or exact-device row changes, and no Apple GPU proof
is re-run or re-claimed. What changes is that the portable path now agrees with
Metal instead of silently disagreeing on 14 of 23 opcodes. Evidence:
`tests/unit/test_apple_gpu_binary_opcodes.py` extracts the stub's opcode switch
and error channel VERBATIM, compiles them with the host C++ compiler, and
compares every opcode against the numpy reference — 74 passing on the M1 Max,
including negative inputs that separate floor-mod from C `fmod` and int32
truncation, plus the unknown-opcode diagnosis. That compiled lane is the only
coverage that executes the stub, since a Mac loads the real Metal symbol and
never reaches it; the structural drift gates in the same file run on any host so
a newly declared opcode missing from the stub fails immediately.
Cross-backend sync `ZERO-FUNCTION-CANDIDATE-2026-08-19` — **shared frontend ABI
and diagnostics; Apple outcome: parity validated, no Metal evidence changed.**
The zero-function-candidate slice (PR #590) changes `JitFn`'s call ABI recovery
and adds one diagnostic code. A `@jit` function whose AST lowering produced no
function raised a bare `IndexError` from `_establish_tracer_authority`, and the
same absence left `_call_arg_names`/`_constraint_ir_args` empty — silently
mis-binding keyword calls and skipping call-time constraint re-checking. The ABI
is now derived from the Python signature via the shared
`graph_ir.ir_args_from_signature` (Decisions #30/#31), and the apple_gpu tracer
lane lifts foreign interpreter exceptions into the new registered
`JIT_APPLE_GPU_TRACE_FAILED` code (Decision #21). `TesseraTraceError` passes
through unwrapped.
Apple impact: `JIT_APPLE_GPU_TRACE_FAILED` is apple_gpu-specific — the tracer is
the only execution path for a body whose AST emission was deferred, which is an
apple_gpu-only route (`_trace_deferred` forces `_needs_trace`). Verified on M1
Max: the previously-crashing body now executes correctly through the tracer, and
an unmodelled `Tracer` attribute surfaces as
`TesseraJitError [JIT_APPLE_GPU_TRACE_FAILED]` quoting the decoration-time
reason. No Metal kernel, dylib, or device evidence is changed or claimed — the
change is entirely above the runtime boundary.


Cross-backend sync `SCALAR-SIDE-ORDERING-2026-08-19` — **shared Graph IR
runtime contract; Apple outcome: parity validated on Metal, plus one
follow-up required (portable stub).** The `scalar_side` slice (PR #589) makes the Graph IR lifted-scalar form carry
operand order. `graph_ir._OpExtractor._try_map_binop` lifts a literal out of
either side of a `BinOp` into the `scalar` attribute and records the side; until
now no code in `python/`, `src/`, or `tools/` read that record (Decision #29), so
`2.0 - x` and `x - 2.0` emitted indistinguishable IR and any consumer binding
`scalar` as the right operand computed `x - 2.0` for both — sign-flipped for
`sub`, reciprocal for `div`, with no diagnostic. Shared contract changed: a lone
`scalar` means the RIGHT operand, `scalar_side="left"` requests the mirrored
binding, and any other value is rejected rather than guessed (Decision #21).
Apple impact: `runtime._apple_gpu_dispatch_mpsgraph_binary` is the only consumer
that needed changing — its opcode table is mostly non-commutative (sub, div,
pow, mod, floor_div, atan2 and all six comparisons). It now swaps the operands
before the Metal/host lane split, so both lanes agree. Evidence: M1 Max, real
`tessera_apple_gpu_mpsgraph_binary_f32` symbol loaded, `tests/unit/test_binop_scalar_side.py`
123 passed across both lanes; reverting only the consumer fix fails 23.
**Follow-up required (not fixed here):** `runtime/apple_gpu_runtime_stub.cpp:2067`
— the non-Darwin portable stub implements opcodes 0-8 and its `default:` arm
assigns `out[i] = x`, so `mod`, `floor_div`, the six comparisons, and the
logical/bitwise ops silently return the LEFT operand on every non-Apple host
instead of computing or diagnosing. Pre-existing and independent of operand
ordering; surfaced by the new tests, which Darwin-gate the live-kernel lane and
name the gap in the skip reason. Needs its own change (Decision #21).


Cross-backend sync `AD-LAW-SERIES-2026-08-19` — **shared reference rules and
test infrastructure; Apple outcome: parity validated, no Metal evidence changed.** The AD-LAW series (PR #588)
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
Apple impact: forward-mode reference oracles moved for the spectral and gamma families; VJP-side values are unchanged except `dequantize_nvfp4` (previously a crash, so nothing could have depended on it) and the polygamma negative domain (previously a hang). `test_apple_normalization_vjp.py` passes eps explicitly and is unaffected. No Apple retest required and no Metal evidence is produced or claimed. The previously recorded open spectral/quantize swallow findings are
therefore CLOSED; `_OPEN_FORWARD_KEY_SWALLOWS` (42 entries from the tape
positional-routing scan) remains the open set.


Cross-backend sync `AD-LAW-1-SHARED-ORACLE-2026-08-18` — **shared test
infrastructure; Apple outcome: parity validated, no Metal evidence changed.**
AD-LAW-1 (PR #584) adds executable law oracles (adjoint `⟨Jv,u⟩ = ⟨v,Jᵀu⟩` +
canonical-forward chain check) over the shared numpy reference JVP/VJP
registries — the oracle lane Apple backward bindings differentially compare
against — plus the byte-gated `autodiff_law_audit` dashboard. Reference-rule
fixes in the same PR: `jvp_rmsnorm` eps default (1e-6 → the forward's 1e-5)
and `jvp_clamp` swallowing the canonical `min`/`max` kwargs. Apple impact:
`test_apple_normalization_vjp.py` passes `eps` explicitly and consumes
`vjp_rmsnorm` (default unchanged), so no Apple retest is required and no
Metal evidence is produced or claimed by this gate. Open follow-up shared
with all backends: 20 pinned swallowed-kwarg findings
(`test_autodiff_laws.py`) await triage. *Triage update, same key (AD-LAW-1b):* reference JVP fixes landed for `clip` alias deafness, `add`/`mul` unary-`scalar`, and fft/ifft/rfft/irfft `norm` handling (√n-wrong under `norm="ortho"`); five entries benign-classified with recorded reasons; the open set is now the stft/istft/spectral_conv and quantize families only, riding their owning family reviews. *Spec-growth update, same key (AD-LAW-1c):* law coverage roughly doubled (109 adjoint / 87 chain rows green incl. attention, spectral-complex, structural, and loss families); two more silent reference JVP defects found and fixed — `lgamma` (derivative was a dead stub returning 0) and `digamma` (whole JVP was an identity placeholder) — plus `jvp_cast` crashing on canonical dtype strings. Forward-mode reference oracles for those three ops changed; reverse-mode VJPs are untouched, so no backend backward package is affected. Reflection formulas added to the shared polygamma helpers (`_digamma_positive`/`_trigamma_positive`): the upward recurrence advanced by 1 per step, so a valid input like -1e9+0.5 spun for ~10^9 iterations — a live defect in the **reverse** path too, since `vjp_lgamma`/`vjp_digamma` already call these. Now O(1) on the whole real line, exact against the canonical forward, poles -> nan.; Apple-side
impact, if any, is limited to families whose reference oracles change when
fixed (fft/stft norm handling is the family most likely to matter here).

Cross-backend sync `W4-DYNAMIC-EFFECT-NONLINEAR-CFG-2026-08-18` — **shared
contract parity; Metal follow-up required.** Dynamic saved-slot data/shape
tapes, exact polynomial witness guards, and variadic branch CFG state are
target-neutral contracts. Effectful replay remains fail closed except for
compiler-owned extent assertions. Apple has no native region-product consumer
or Mac evidence and inherits no x86/gfx1151 proof.

Cross-backend sync `W2.4-E2E6-SYMBOLIC-2026-08-18` — **shared parity
validated; no Metal execution change.** Relational Tile legality has one staged
production authority and pure static annotations can abstract-trace before AST
compatibility capture. SM120 Lion/DeltaNet dispatch moved into family plugins;
Apple has no corresponding consumer and remains fail closed pending its own
typed package and Mac evidence.

Cross-backend sync `W1-W3-AUTHORITY-CLOSEOUT-2026-08-18` — **shared contract
parity validated; no Apple physical claim.** Bare Tile fragments are rejected,
semantic selectors including Apple KV-cache kinds are ODS-owned, WarpSpec uses
registered Schedule ancestry, and whole-program memory activity is conservative.
Loss VJP plugins have no Apple consumer and therefore fail closed; Metal family
migration remains an architecture-owned follow-up.

Cross-backend sync `W4-PRODUCT-1-RESIDUAL-CONTRACT-2026-08-17` — **shared
contract parity validated; Apple physical follow-up required.** SAVE/HYBRID
selection now crosses Graph→SCF with exact checkpoint, CFG, and residual
identity and cannot silently become recompute-all. Shared paired AD now carries
dynamic branch-local residual extents, bounded SAVE and sparse HYBRID `while`
state tapes, and bounded-dynamic counted-loop tapes as explicit SSA. The
target-neutral compiler now classifies source-CFG SCCs and structurizes bounded
pure reducible or irreducible native CFGs as a typed program-counter state
machine. Nested canonical structured bodies and mixed control/tensor state are
admitted. Saved dynamic slots require total data/shape-tape envelopes;
unbounded, unsupported-region, and unrecorded effectful forms remain fail
closed. A Metal region-product consumer remains open; the
x86/gfx1151 correctness packets do not transfer.

Cross-backend sync `E2E-REAL-6F-OPTIMIZER-VJP-2026-08-17` — **shared
optimizer lineage validated; Apple not applicable for the bounded physical
set.** SGD, Momentum/Nesterov, and Adam/AdamW now use one typed
`schedule.optimizer_vjp` contract, but this slice declares consumers only
where physical reverse packages already exist: x86/gfx1151 for SGD and
Momentum/Nesterov, and gfx1151 for Adam/AdamW. Apple remains fail closed and
inherits no sibling evidence; a future Metal package requires its own queue
item and exact-device proof.

Cross-backend sync `E2E-REAL-6E-STATEFUL-VJP-2026-08-17` — **shared
Adafactor/sequence-mixer authority validated; Apple target follow-up
required.** The shared compiler now binds factored/full Adafactor and causal
DeltaNet backward to a one-execution frontend certificate plus exact
state/workspace, Schedule, and Tile lineage. Apple has no Adafactor reverse
consumer; its existing sequence-mixer implementation is not reclassified by
x86/gfx1151 evidence. A Metal plugin must consume the shared package and carry
independent Mac correctness/performance proof.

Cross-backend sync `E2E-REAL-6D-LION-VJP-2026-08-17` — **shared flat Lion
Graph and non-reexecuting state-lineage proof validated; Apple physical
follow-up required.** Apple has no architecture-owned Lion reverse package to
attach to the new plugin, so its Target consumer remains undeclared rather
than inheriting AVX-512 or gfx1151 execution. A future Metal package must
consume `schedule.lion_vjp` → `tile.training_kernel` and provide independent
Mac correctness and performance evidence.

Cross-backend sync `E2E-REAL-6C-ATTENTION-VJP-2026-08-17` — **shared
rank-4 authority and bottom-right ragged-causal semantics validated; Apple
plugin follow-up required.** The shared family registry now owns
`flash_attn`/GQA/MQA reverse through tracer Graph →
`schedule.attention_backward` → `tile.attention_backward_kernel`. Apple’s
existing Metal backward package is unchanged and receives no x86/gfx1151
evidence. Apple must bind that architecture-owned package to the public
non-reexecuting plugin and rerun its Mac numerical packet. The rank-3
`multi_head_attention` wrapper remains outside the migrated rank-4 envelope;
active dropout also needs keyed, non-reexecuting replay proof.

Cross-backend sync `E2E-REAL-6B-SPECTRAL-VJP-2026-08-17` — **shared tracer
and package-contract parity validated; Apple physical follow-up required.**
Concrete AST specialization now resolves the same shape-derived spectral
identity as tracing, and compound spectral reverse products have one declared
Graph/Schedule/Tile/Target family-plugin boundary. Apple is deliberately not a
Target consumer in this slice: no MSL package, Metal runtime path, or Mac
evidence was added. A future Apple consumer must build its own package from the
shared contract and provide independent exact-device proof.

Cross-backend sync `GFX1151-CALIB-BAREMETAL-2026-08-16` — **shared calibration
authority parity validated; no Apple evidence transfers.** `target_perf` now
rejects explicitly provisional and WSL-hosted corpora from its measured
selector registry while exposing a non-mutating pruning reader. Apple physical
code and existing Metal packets are unchanged. Any future Apple peak corpus
must independently carry selector-eligible Metal/device timing; gfx1151 HIP and
ROCprofiler evidence is not applicable.

Cross-backend sync `REF-TIER-PHYS-2026-08-16` — **shared Schedule contract
received; Metal physical follow-up required.** Batched tridiagonal solve and
the four coalition zeta/Mobius transforms now have content-addressed
Schedule→Tile carriers. The coalition transforms share one parameterized
Yates butterfly rather than four emitters. Apple gains no MSL consumer or
device claim in this slice; any future lane must choose its own physical
solver and butterfly schedules and provide independent Apple evidence.
The shared Schedule Object now snapshots nested resource metadata before
digesting, and dynamic rearrange/GQA-fold inference preserves ranked `?`
dimensions; parity is validated without transferring a physical schedule.

Cross-backend sync `LAYOUT-SCHEDULE-OBJECT-2026-08-16` — **shared parity
validated; Apple physical follow-up deferred.** The first C++ layout-algebra
ABI and GQA-fold consumer are target-neutral, and SO-1 replaces anonymous
action tuples with one content-addressed Schedule Object. SO-2 adds symbolic
Tile roles and role-bearing mbarriers. Apple has no mbarrier consumer, so SO-2
is not applicable physically; a future Metal schedule producer may consume the
Schedule Object without inheriting NVIDIA/AMD roles. The optional role-bearing
`tile.pipeline_init` operand is shared IR but remains physically not applicable
to Metal. No MSL raster output, selector, or Apple device claim changed.

Cross-backend sync `ATTN-BWD-ARCH-2026-08-16` — **no Apple result transfers.**
ROCm's split backward program was re-audited and x86 gained deterministic
parallel query-row execution with private dK/dV partials. Apple's separately
tracked backward materializer and exact-device evidence remain unchanged.

Cross-backend sync `X86-PASS-DIALECT-DEPENDENCY-2026-08-16` — **parity
validated; no Apple physical follow-up.** The shared pass library now models
the optional hardware-free x86 Target dialect as a declared MLIR pass
dependency and fails closed when it is absent. No Apple dialect, pipeline,
Metal ABI, schedule, selector, or device evidence changed.
The same closeout removes a private permissive `schedule` dialect from the
shared transform library; all transforms now consume the canonical ODS
`TesseraScheduleIR` authority. Apple receives the build-parity fix only.

Cross-backend sync `PDE-EXACT-CONTRACT-2026-08-14` — **shared semantic parity
validated; Metal physical follow-up required.** Exact-rational PDE
classification and the first fail-closed diffusion stability certificate are
backend-neutral. No Metal stencil/boundary/halo package or evidence is claimed.

Cross-backend sync `DIST-SHARD-HVP-2026-08-14` — **shared compiler parity;
native Apple follow-up required.** Reshard plans now materialize bulk
collectives as Graph→Schedule→Tile SSA with subgroup/region identity and
deterministic all-to-all matching rounds. Exact forward-over-reverse HVP now
exists as a compiler Graph product. Apple receives the shared IR contracts but
adds no Metal collective transport or native HVP package in this slice; those
remain architecture-owned and require independent Apple execution evidence.

Cross-backend sync `E2E-REAL-6-NATIVE-VJP-2026-08-14` — **shared authority
parity assessed; Apple follow-up required.** Normalization reverse package
construction moved out of `JitFn` into a registry that declares its complete
Graph/Schedule/Tile/Target ownership. Apple has no registered native consumer
in this first slice, so eager/Metal behavior is unchanged and no x86, gfx1151,
or SM120 evidence transfers. A future Metal normalization VJP must register an
Apple Target consumer before it can use this boundary.

Cross-backend sync `AMD-ISA-DTYPE-2026-08-14` — **parity assessed; no Apple
physical change required.** The shared compiler gained an AMD-only
architecture/dtype/matrix-instruction selector and CDNA5 fragment identity.
No Graph dtype spelling, public operation, Schedule contract, Apple ABI, or
Metal capability changed; Apple must not inherit AMD low-precision legality or
evidence. The shared `OP-DTYPE-FLOW-1` generator now audits every Apple
operator/dtype row across the IR stack; policy-derived Apple dtype rows remain
`legal_only` until an Apple manifest declares a physical consumer.

Cross-backend sync `CI-LIT-BACKEND-DIALECTS-2026-08-12` — **parity validated;
no Apple change required.** The `Validate / lit` lane was dead from 2026-08-11
to 2026-08-12 (pytest collection aborted on a missing `ml_dtypes`, fixed in
#554); the first green-collection run failed 27 of 367 fixtures because
`tessera-opt` registered neither `tessera_x86` nor `tessera_rocm`. Apple was
**unaffected**: the lane already configures `-DTESSERA_BUILD_APPLE_BACKEND=ON`,
`tessera_apple` appears in the driver's available-dialect list, and no
`phase8` Apple fixture is in the failure set.

**Apple is incompatible with the lean artifact driver — do not plan around
co-enabling them.** An earlier revision of this entry claimed Apple "survives
configurations that drop core/x86" because its linkage is gated only on
`TARGET TesseraApple` (`tools/tessera-opt/CMakeLists.txt:187`) with no lean
check. The linkage observation is true but the conclusion was wrong: Apple
registers the `apple-backend` feature (`:188`), the lean permitted set is
`core-tessera-ir collectives nvidia-backend rocm-backend` (`:250`), and any
feature outside it raises `FATAL_ERROR` (`:259-273`). So a HIP-less ROCm or
CUDA-less NVIDIA configure **plus** Apple does not yield an Apple-capable lean
driver — it fails to configure at all. Any future attempt to widen the lit lane
to ROCm must therefore build a *separate* driver, not add flags to the Apple
one.

One Apple-relevant caution carried from this investigation: Apple was the sole
backend wired into the lit configure, which is why its portable Target-IR
coverage looked healthier than its siblings' for the last ten days. Read that
as "the others were dark", not as Apple parity headroom — the Apple GPU op
envelope is still runtime-delivered (see the `apple_gpu_runtime.mm` note in
`CLAUDE.md`), and nothing here changes that.
Cross-backend sync `CI-LIT-DEPS-2026-08-12` — **parity validated; no physical
follow-up required.** PR 554 made the shared opt-in MLIR lit lane install the
workflow-owned Python dependency set before `lit`/FileCheck collection. This is
backend-neutral test infrastructure, changes no compiler/runtime contract, and
requires no Apple package or device evidence.

Cross-backend sync `PDE-STENCIL-FOUNDATION-1-2026-08-12` — **shared semantic
parity validated; Metal physical follow-up required.** Neighbors now requires
explicit tap coefficients, and TPP requires scheme/order/per-axis spacing.
Unavailable Target symbols are artifact-only. This adds no MSL stencil/halo
consumer or Mac packet; Apple must bind the contract to an architecture-owned
package before any execution claim.

Cross-backend sync `BLOCK-ATTNRES-ROCM-2026-08-12` — **follow-up required.**
The shared Block AttnRes plan establishes portable balanced-partition,
epsilon-qualified numeric, VJP, and softmax-merge oracle contracts. This PR
adds Phase-1 stats/merge/finalize references, the Phase-2 stdlib recurrence,
Phase-3 typed Graph/VJP/JVP contracts, and the Phase-4 content-addressed
Schedule→Tile artifact. It adds no MSL package or Mac evidence. Apple must
later bind those contracts to
its own stats-attention/merge physical consumer and independent correctness
packet; no gfx1151 result transfers.

Cross-backend sync `MODEL-FUSED-PHYS-1-2026-08-12` — **shared MiniMax MSA
package lineage landed; Metal consumption remains follow-up required.** x86
and gfx1151 now execute digest-bound MSA artifacts without Graph redispatch.
Apple's current host-select/Metal composition remains a differential oracle,
not consumption of that package, and no Mac evidence transfers. Apple still
needs its architecture-owned package; DeepSeek MLA/DSA remain independently
open.

Cross-backend sync `MODEL-WEIGHT-PHYS-1-2026-08-12` — **shared physical-byte
weight ABI landed; Metal byte-packed consumption remains follow-up required.**
The carrier preserves INT4/FP8 checkpoint bytes and separate fp32 scales under
one digest while prohibiting full-weight materialization. Apple's existing
dequant-GEMM accepts expanded unit-grid codes, so it does not yet consume this
physical-byte ABI directly. No Mac evidence or support claim transfers from
gfx1151.

Cross-backend sync `W4-PRESBURGER-SHARD-2026-08-12` — **shared analysis
contracts landed; Metal consumption remains follow-up required.** Graph IR now
carries typed integer-affine plus exact modular/divisibility constraints into
the C++ Presburger consumer. The shared sharding layer has a fail-closed
replicated/tiled/partial-reduction fixed point and explicit reshard planner;
lowered `control_scan` owns shared recompute-all JVP/VJP products. This adds no
Metal region product, reshard lowering, native transport, or Mac packet;
x86/gfx1151 evidence does not transfer.

Cross-backend sync `W4-CFG-RESIDUAL-W5.2G-2026-08-14` — **shared compiler
carriers and scalable scheduler landed; Metal follow-up required.** The
tracer-owned structured CFG, block-wide Presburger identity, and executable
SAVE/HYBRID residual ABI change shared lineage only. The action-DAG model now
uses deterministic critical-path/list scheduling with safe lower-bound pruning
and a small-DAG exhaustive oracle. No Metal region product, physical producer
wiring, calibrated packet, or selection claim was added; architecture evidence
does not transfer.

Cross-backend sync `E2E-AUTH-DAG-2026-08-12` — **shared native-product v3 and
automatic dependency consumption landed; Metal remains follow-up required.**
Reduction and normalization now have truthful Schedule/Tile product carriers,
native JVP/VJP paths require cached tracer/AST differential proof for pure
programs, and Graph-derived R3 candidates consume compiler-generated edges.
No Metal product child, physical dependency consumer, or Mac packet was added;
x86/gfx1151 evidence does not transfer.

Cross-backend sync `E2E-AUTH-DAG-2026-08-11` — **shared frontend authority and
automatic dependence-edge contracts landed; Metal remains follow-up
required.** Pure straight-line tensor signatures now cache tracer-owned Graph
IR and can be differentially certified against the retained AST candidate.
Native-JVP plugins declare their Graph/Schedule/Tile/Target disposition and own
package construction; compatibility gaps are explicit. W2.1 facts now generate conservative Tile action-DAG
edges with reason and analysis digests. This adds no Metal family package,
edge-consuming physical pipeline, schedule promotion, or Mac evidence.

Cross-backend sync `AD-SOLVER-ISTFT-PHYSICAL-2026-08-11` — **shared product
contracts landed; Metal consumption remains open.** Graph IR now has an exact
`tessera.istft_jvp` carrier for spectrum and window tangents, and the general
solver parent binds residual plus solution/parameter JVP/VJP child identities
under a true-residual GMRES policy. The shared compiler now derives those five
children for typed pointwise, sum/mean, rank-2 matmul, bounded-dynamic/
mixed-storage, distinct parameter-space, and statically counted-region residual
Graphs. Pure scalar `if`/bounded-`while` predicates now have explicit
compare/select replay in the shared child contract. Only AVX-512 and gfx1151
have physical children and expanded-family packets. Apple needs Metal
spectral-product children, solver
children, and an independent Mac correctness/performance packet; no schedule
or evidence transfers.

Cross-backend sync `E2E-REAL-6-JVP-SOLVER-2026-08-11` — **shared frontend and
family-plugin boundary landed; Metal remains follow-up required.** Native
forward-product specialization now originates in tracer-produced canonical
Graph IR, and explicit family plugins own reduction, normalization, FFT, and
compound-spectral planning outside `JitFn`. General solver contracts bind exact
residual/JVP/VJP identities and can execute matrix-free reference products
without finite differences. This adds no Apple family plugin, native package,
or Mac evidence; the AST lane remains for unmigrated families.

Cross-backend sync `AD-FWD-DIST-3-2026-08-11` — **shared exact JVP and
structured-region products landed; Metal remains follow-up required.** Public
JVP/jacfwd now use registered tangent rules without a finite-difference
fallback, and compiler forward mode carries primal/tangent state together
through bounded SCF. Typed `collective_permute` now owns its peer map and
executes in portable and NCCL/RCCL runtimes. This adds no Metal JVP package,
subgroup communicator, native transport, or Mac evidence.

Cross-backend sync `W4-SOLVER-REGION-2026-08-11` — **shared bounded-region
adjoints and general matrix-free solver policy landed; Metal consumption is
follow-up required.** Portable tracing now emits bounded SCF, and the paired
compiler differentiates effect-safe single-block `if`, counted `for`, and
canonical bounded `while` with implicit captures. General residual execution
uses restarted GMRES/CG policy in shared IR. This adds no Metal region executor,
solver package, checkpoint packet, or performance claim.

Cross-backend sync `COMP-GRAPH-DATAFLOW-W2.1-2026-08-11` — **shared
analysis substrate landed; Metal behavior and evidence are unchanged.** Graph
IR now has one fail-closed, invalidatable shape/alias/liveness/memory-
dependence/activity analysis with C++ and Python query surfaces. Reverse AD and
await sinking consume it. This is target-independent legality infrastructure;
it transfers no schedule or performance claim to Apple. Region-aware clients
and Metal overlap proof remain separately owned.

Cross-backend sync `AD-FWD-FAMILY-2-2026-08-11` — **shared affine,
compound-spectral, solver-product, and native-collective contracts landed;
Metal consumption remains open.** Compound spectral Graph operations now own
direct tangent interfaces, including an exact ISTFT window-product carrier, and solver
artifacts distinguish JVP/non-transposed from VJP/transposed solves. No x86 or
ROCm package or evidence transfers to Apple.

Cross-backend sync `AD-FWD-NATIVE-1-2026-08-11` — **shared native-product
lineage landed; Apple consumption is follow-up required.** The parent artifact
schema binds paired-JVP IR to immutable ordered child packages and detects
child substitution. This slice has no Metal executor or Mac evidence; Apple
must add architecture-owned family products and exact-device packets before
claiming native JVP execution.

Cross-backend sync `COMP-EFFECTS-W2.2-2026-08-10` — **shared registered-effect
analysis closed; no Metal evidence claim.** Canonical Graph records now carry
effect, alias, mutation, and stochastic identity; Python and C++ consume the
same fail-closed facts and internal calls reach a fixed point. Await sinking
uses that shared query. This changes scheduling legality only; Apple still owns
native overlap execution and Mac correctness/performance packets.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R4-2026-08-10` — **shared functional
MegaMoE plan consumption landed; Metal transport/evidence remain open.** The
content-addressed plan binds chunk slices, per-expert capacity, two-live-frame
workspace limits, true-use dependencies, ordered collectives, and deterministic
combine order. R3 only prunes complete measured plan records; scalar device
latency selects. Mock multi-rank execution does not prove Metal overlap. Apple
still needs native communicator/command-buffer integration and an independent
Mac correctness/performance packet.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R3-2026-08-10` — **shared prune-only
Tile action-DAG model landed; no Metal selection claim.** R3 validates explicit
dependencies and calibration identity, uses deterministic critical-path/list
scheduling, and composes compute/memory/communication lanes with queue
serialization. Exact small DAGs and proven lower-bound losers may be pruned;
every estimate is promotion-ineligible and scalar
measured latency remains authoritative. No x86/gfx1151 calibration transfers
to Apple. Metal vectors and a Mac packet are required before an Apple DAG may
use the model; R4 remains a separate production-consumer slice.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R2-2026-08-10` — **shared measured
resource-vector schema landed; Apple evidence remains architecture-owned.**
Successful measured autotune rows may record compute time, dtype-correct bytes
moved, communication bytes, queue/resource identity, timing provenance, and
the measured-candidate digest. Analytical rows cannot claim the vector, and
scalar measured latency remains selector authority. No Metal timing or Mac
resource identity is inferred from x86 or gfx1151; Apple harnesses must populate
their own provenance before R3 composition analysis can consume it.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R1-2026-08-10` — **shared explicit
async lineage and fail-closed await sinking landed; Metal consumption remains
architecture-owned.** Python Schedule→Tile no longer emits internal
`tessera.queue.*` compatibility markers: async copies produce named tokens and
waits consume them. Collective awaits move only across operations proven
memory-effect-free; mutation, RNG, aliases/casts, regions, and ordered
collectives are barriers. No gfx1151 schedule or evidence transfers to Apple;
Metal async-copy/collective overlap needs its own executable consumer and Mac
packet.

Cross-backend sync `AD-STOCHASTIC-RNG-1-2026-08-10` — **shared stochastic JVP
contract available; Metal follow-up required.** Explicit key/counter Graph ops,
estimator provenance, dropout replay, fixed-key EGGROLL JVP, and derivative
proof obligations are target-independent. x86 and gfx1151 distribution kernels
do not transfer to Metal. Apple's existing Philox symbols need contract
alignment plus Mac compiler-JVP and exact-device proof before promotion.

Cross-backend sync `AD-FWD-PRODUCT-2-2026-08-10` — **public JVP ABI landed;
Metal execution remains follow-up required.** Forward/JVP requests now carry
mode-neutral provenance and stable `wrt_indices`, and the compiler emits only
requested tangent terms. Tanh/sigmoid add direct CPU-oracle proof. No Metal
package, selector, or Mac evidence transfers; native JVP remains fail-closed.

Cross-backend sync `AD-FWD-CORE-1-2026-08-09` — **shared compiler JVP
foundation landed; Apple physical consumption remains architecture-owned.**
The Graph dialect now exposes compiler-owned tangent rules and a paired
`--tessera-autodiff-forward` function contract. Matmul/mul has independent CPU
IR numerical proof, while unsupported active operations and regions fail
closed. The generated ledger distinguishes compiler `ir_tangent` evidence from
Python JVP registration. This changes no Metal package or Mac evidence; Apple
must lower and prove any native JVP package independently.

Cross-backend sync `X86-TYPED-FAMILY-PLUGIN-2026-08-09` — **shared schema
parity assessed; no Apple physical change.** x86 now validates a closed Tile
family and registered Target marker before selecting its prebuilt AVX-512
image. Apple may reuse the schema and fail-closed family discipline, but no
x86 ABI call, schedule, package, or Zen 5 evidence transfers to Accelerate or
Metal. The x86 backward-family allowance is narrowly one explicit forward
recompute companion, not a general multi-carrier escape hatch. Apple CPU/GPU
image production remains architecture-owned.

Cross-backend sync `EGGROLL-ES-LOWRANK-2026-08-09` — **the shared Graph,
Schedule, Tile, lineage, member-RNG-v1, and fp32 numeric-policy contract has
landed; Apple physical consumption is follow-up required.** gfx1151 owns the
first GPU exact-device rank-1 proof and Zen 5 now owns an independent AVX-512
fp32 package, but neither architecture's schedule transfers to Metal. Apple
still owns the reference MSL implementation (`simdgroup_matrix`) and its own
numerical/performance packet. The `s32` integer lane is a separate EGG track and
is **not applicable** to this float MSL path. W4 scalar-gather/member
reconstruction passes mock-mesh proof; a native Metal communicator remains
open. Contract:
`docs/audit/compiler/EGGROLL_SUPPORT_PLAN.md`.

Cross-backend sync `COLLECTIVE-RCCL-ADVANCED-LANES-2026-08-09` — **shared
fail-closed artifact vocabulary adopted; not applicable to Metal.** Copy
Engine, GIN/RMA, and gfx1250 DDA are independent RCCL lanes with distinct
target and evidence gates. Apple receives only the shared discrimination and
cannot select any AMD lane. A future Metal communicator remains separately
owned and requires Mac exact-device evidence.
The registered window and put/signal/wait Target operations remain RCCL-gated;
Metal adopts only their fail-closed IR vocabulary.
The launcher-neutral RCCL GIN executable is AMD-only and has no Metal
consumer; Apple still requires a separately owned communicator implementation.

Cross-backend sync `COLLECTIVE-NATIVE-FOUNDATION-2026-08-09` — **shared
artifact vocabulary adopted; Metal transport still open.** Target collective
artifacts now identify initiation, registration, ordering, capture policy,
backend/source version, and the capability evidence digest. The functional
NCCL/RCCL host adapters and AMD LSA/Copy Engine/DDA candidates are not
applicable to Metal. Apple still needs a native communicator/window mapping
and exact multi-device Mac evidence. Runtime/artifact capability-digest
matching is shared policy and applies to that future mapping.

Cross-backend sync `COLLECTIVE-ASYNC-UNIFY-2026-08-09` — **shared software
contract closed; Metal transport evidence open.** The legacy unregistered
`tessera.collective.*` producers are removed. Forward and adjoint insertion now
emit registered `tessera_collective` futures, explicit awaits, and real SSA
rewiring; the portable runtime rejects unknown/subgroup mesh axes rather than
executing them on the wrong communicator. Apple still needs a native Metal
multi-rank adapter and exact-device packet; no selector or performance claim
changes.

Cross-backend sync `DIST-SHARD-ALIAS-1-2026-08-09` — **shared alias mapping
available; Metal transport unchanged.** Five public reduction/broadcast names
now resolve to registered all-reduce/all-gather transport records, while the
three sharding placement/region entries remain compile-time contracts.
`collective_permute` is still a separate ordered point-to-point gap and fails
closed. Apple needs its own frontend capture, communicator mapping, and exact
multi-device packet; portable execution provides no Metal proof.

Cross-backend sync `AD-SOLVER-RESIDUAL-EVAL-2026-08-08` — **bounded x86/ROCm
pilot landed; Metal follow-up required.** The shared IFT chain now has a
content-addressed Schedule→Tile physical contract, and counted-region treeverse
can execute checkpoint replay before a row becomes eligible. Apple has no
Metal consumer for the diagonal-sqrt pilot, no general iterative solver, and
no complete-backward packet. This PR changes no Apple package, selector,
policy, or device claim.

Cross-backend sync `AD-CORE-EFFECT-CONTROL-COLLECTIVE-2026-08-08` — **shared
Graph/Tile/portable-Target contracts available; Metal follow-up required.** Compiler activity,
effects, `stop_gradient`, stochastic rejection, and fail-closed region
adjoints are target-independent. The four collectives now lower from exact
typed Tile operations into one content-addressed asynchronous Target queue and
execute through the deterministic software adapter. Apple still needs a Metal
native transport adapter and exact multi-rank execution. No Metal schedule,
selector, performance, or device evidence is claimed or transferred.

Cross-backend sync `GRAPH-VERIFY-SIGNED-1-2026-08-08` — **shared legality
parity validated; no Metal physical claim.** Graph and canonical-attention
integer bounds now use signed `IntegerAttr` values, preventing MLIR 23 unsigned
accessors from accepting negative schedules, seeds, cache windows, or control
bounds. Direct negative IR cases cover both dialects. No Apple schedule,
package, selector, runtime ABI, or Mac evidence changes.

Cross-backend sync `AD-TSOL-SPECTRAL-1-2026-08-08` — **shared Graph contract
available; Metal follow-up required.** Normalization, logical length,
packed-real/Hermitian identity, and DCT type now survive compiler autodiff, and
the core FFT/RFFT/DCT transposes have CPU numerical proof. Apple has no physical
consumer for the new compound-backward artifact; no Metal schedule, package,
selector, or device evidence is claimed or transferred.

Cross-backend sync `AD-TSOL-SPECTRAL-NATIVE-2026-08-09` — **follow-up still
required.** The bounded spectral-filter/convolution native consumers added for
AVX-512 and gfx1151 do not transfer a schedule or proof to Metal. Apple still
requires its own compound-backward package, exact-device correctness, and
performance evidence; unsupported compound kinds continue to fail closed.

Cross-backend sync `AD-CORE-LINEAR-1-2026-08-08` — **shared Graph-IR parity
validated; no Metal physical claim.** Compiler-owned linear transposition now
covers structural views, broadcast, and operand-wise matmul in both autodiff
passes with CPU numerical proof. This changes no Apple schedule, package,
selector, or exact-device evidence; later Apple backward materializers may
consume the shared backward Graph artifacts without inheriting x86/ROCm proof.

Cross-backend sync `COMPILER-DASHBOARD-PROOF-TRUTH-2026-08-08` — **proof
levels corrected; no Apple physical change.** Apple CPU reference execution
and Apple GPU exact-device proof now occupy separate op×target rows;
`fused`/`packaged` entries report implementation presence without being
promoted to execution proof. No Metal evidence or selector transfers.

Cross-backend sync `X86-BUILD-ARTIFACT-DISCOVERY-2026-08-08` — **shared
fail-closed selection assessed; no Apple package changes.** The x86 runtime
and native packager now honor the common `TESSERA_BUILD_DIR` before default
build names and reject a missing selected tree instead of falling through to a
stale image. This is applicable build-selection policy, but it transfers no
AVX-512 implementation, timing, or Mac evidence to Apple.

Cross-backend sync `STANDALONE-COVERAGE-TRUTH-2026-08-08` — **registry truth
adopted; no Metal execution claim changes.** The standalone dashboard now
generates its counts, compiler-layer rollup, exact-target manifest summary,
and open queues from the live registries. It no longer treats the strongest
sibling-backend row as universal support. Apple still owns every Mac/Metal
physical and benchmark follow-up shown by its exact-target manifest; x86 and
gfx1151 TSOL or Adafactor evidence does not transfer.

Cross-backend sync `TSOL-NATIVE-REAL-FFT-2026-08-08` — **shared artifact
follow-up required; no Metal schedule transfers.** The target-neutral FFT
contract now distinguishes logical and physical length and hashes Hermitian
layout plus packed/fallback policy. The x86 and gfx1151 N/2 packages provide
no Apple physical evidence. Apple must map this contract to its own MPSGraph or
Metal FFT consumer and prove it on a Mac before selecting packed execution.

Cross-backend sync `ROCM-BUILD-ARTIFACT-DISCOVERY-2026-08-07` — **parity
validated; no Apple physical change.** Shared compiler-test discovery now
accepts fail-closed `TESSERA_BUILD_DIR` selection while retaining explicit
`TESSERA_OPT` precedence. The migrated runtime-library and backend-tool users
are ROCm-owned; Apple Metal packages, schedules, evidence, and selectors are
unchanged.

Cross-backend sync `AUTODIFF-RELAXATION-1-2026-08-07` — **shared
Python-reference contract; Apple physical follow-up required.** `sparsemax`,
`entmax15`, `soft_top_k`, `gumbel_softmax`, and `perturbed_argmax` now have
storage-preserving reference semantics and autodiff rules, but no Metal
lowering or Mac evidence. They remain explicitly reference-only until an
Apple-owned physical package is selected and proven.

Cross-backend sync `MATH-PHYSICAL-2-2026-08-06` — **shared dtype contract
assessed; Metal follow-up required.** Physical binary math packages now require
matching input storage dtypes, but the AVX-512 scan selector and gfx1151 HIP
module cache are architecture-owned and transfer no Apple performance claim.
Apple must validate its math catalog, reduced-storage accumulation policy, and
the same difficult-domain corpus on a Mac before recording parity.

Cross-backend sync `TSOL-CONTRACT-GENERALIZE-2026-08-06` — **shared semantic
contract adopted; Metal consumer remains follow-up.** The shared TSOL layer now
validates bounded dynamic dimensions, arbitrary axes, storage policy, and
normalization before exact specialization. Zen 5 and gfx1151 now consume that
wider contract, but their native ABI and evidence do not transfer. Apple still
has no GPU FFT package,
so it cannot physically consume the compound artifact and inherits no AVX-512
or gfx1151 support claim. A Mac-owned FFT package, workspace/residency ABI,
compound consumer, and exact-device evidence remain prerequisite in that order.
Cross-backend sync `TPROF-MULTICLOCK-2026-08-06` — **shared evidence schema is
executable on ROCm/x86; Apple adoption remains follow-up.** ROCm now owns independent host-wall,
HIP-event, instrumented device-wall-clock, and profiler-activity records plus an
optional intrusive HSA/AQL provider. No HIP clock, `wall_clock64()` kernel,
`rtg_tracer` queue interception, ROCprofiler status, or gfx1151 evidence applies
to Metal. Apple should adopt the same no-substitution/provenance/validity shape
for its host wall and command-buffer timestamps when the shared schema is
implemented, while retaining its existing Mac exact-device gates and
architecture-owned counter policy.
The native-evidence extension adds content-digested provider captures,
clean-versus-instrumented image/resource comparisons, and exact-machine event
maps. These contracts are shared vocabulary only: ROCprofiler, RTG, Linux perf,
IBS, gfx1151, and Zen 5 collectors are not applicable to Metal. Apple must bind
equivalent data to its own command-buffer/counter collector on a Mac before
claiming parity.

Cross-backend sync `TSOL-ROCM-E2E-1-2026-08-06` — **shared ODS vocabulary
adopted; physical execution not applicable until Apple owns an FFT package.**
The target-neutral `schedule.spectral_program` and
`tile.spectral_program_kernel` contract is registered in production, but HIP
and AVX-512 helpers and their exact-device evidence transfer no Metal schedule
or support claim. Apple must first close its architecture-owned FFT gap, then
derive compound workspace/residency policy on a Mac.

Cross-backend sync `TSOL-GFX1151-FUSED-BATCH-2026-08-08` — **not applicable to
Metal execution.** The shared FFT artifact now names gfx1151's batched
fused-LDS residency, while the HIP build closure, AMD kernel, and WSL timing
packet transfer no Apple implementation or evidence. Apple still requires a
Mac-owned FFT package and independently measured residency policy.

Cross-backend sync `TSOL-SPECTRAL-POLICY-2026-08-08` — **shared DCT and
streaming policy adopted; Metal package remains follow-up.** DCT-I/II/III/IV
now have distinct public, autodiff, Graph, Schedule, and Tile identities. The
target-neutral chunked-STFT state binds its policy digest, retained overlap,
sample count, and emitted-frame count; centred streaming fails closed until
lookahead lineage exists. Apple receives these shared semantics and the
expanded centred/n-FFT/one-sided eager reference, but still owns no Metal
spectral package and inherits no Zen 5 or gfx1151 physical evidence. The
length-one convolution and one-sample STFT/ISTFT physical boundary repairs are
therefore not applicable to Metal until that package exists.

Cross-backend sync `ROCM-MATH-EVIDENCE-2026-08-06` — **not applicable to Apple
physical code.** ROCm's centered Welford and scalar boundary corrections do
not change MPSGraph/MSL kernels or Apple selectors. The boundary corpus is a
useful sibling checklist, but Apple requires Mac evidence before recording
parity.

Cross-backend sync `ROCM-FFT-PREBUILT-2026-08-05` — **not applicable; Apple
still has no physical spectral package.** The ROCm HIP image and persistent
device-plan policy do not alter Apple's open architecture-owned FFT work.

Cross-backend sync `FFT-PERF-2-2026-08-05` — **not applicable to Apple
execution.** Zen 5 gained an evidence-gated mixed-radix Stockham selection and
candidate Rader/Bailey implementations; gfx1151 received a timing-domain probe.
No Apple package, selector, schedule, dtype, or performance claim changes.
Apple must derive its own algorithm and residency choices from Mac evidence.

Cross-backend sync `FFT-PERF-FOUNDATION-2026-08-05` — **not applicable to
execution; shared artifact parity recorded.** The FFT artifact now hashes the
selected algorithm, radix sequence, storage/accumulation policy, workspace
policy, residency, and twiddle policy. Apple owns no GPU spectral package, so
it receives no kernel or performance claim and must define all of those fields
from Mac evidence when a lane is introduced.

Cross-backend sync `E2E-REAL-FFT-2026-08-05` — **not applicable until Apple owns a spectral package.**
The now-implemented `schedule.fft`→`tile.fft_kernel` contract separates shared
artifact lineage from physical radix selection and accepts only Zen 5/gfx1151.
Apple still has no GPU FFT target hook, so it receives no
support claim; when one lands, it must supply an Apple-owned schedule and Mac
device evidence rather than inherit ROCm's Stockham policy.

Cross-backend sync `FFT-MIXED-RADIX-BLUESTEIN-2026-08-03` — **not applicable today; inherits when a lane lands.**
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

Apple has no `spectral_fft` kernel and no `TargetHooks/Apple/` entry, so
nothing here changes its behaviour. It is listed because the composition path
is now driven off registration: the moment an Apple FFT lane registers, it
receives `rfft`/`irfft`/`stft`/`istft`/`spectral_filter` automatically, and the
shared planner means it would only need to supply butterflies, not a plan.

Nothing touches `apple_gpu_runtime.mm` or the hand-written MSL kernels.


Cross-backend sync `SHAPE-RULE-REGISTRY-2026-08-03` — **follow-up required - CPU lane supported, GPU lane rejected.**
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

**This is the Python reference lane, not generated device code.** Complex FFT is SUPPORTED on `apple_cpu` (it declares an `fft` capability entry;
vDSP/Accelerate carries the interleaved-complex form) and REJECTED on
`apple_gpu`, which declares no `fft` entry. Metal has no native complex type -
`float2` is the carrier - which matches the interleaved real-pair model this work
adopted, so the GPU gap is a capability-registration plus MSL-kernel task rather
than a type-system one.

Nothing here touches `apple_gpu_runtime.mm` or the hand-written MSL kernels. The
`install_apple_gpu_interception(ops)` ordering constraint still holds and is
still guarded by `test_apple_interception_installed.py`: the storage-dtype
enforcement wrapper must be installed AFTER it.


Cross-backend sync `SUBBYTE-STORAGE-PATH-2026-08-03` — **follow-up required, capability unverified here.**
Same multi-result quantize declaration. Whether Metal exposes native FP8/FP4
arithmetic on the M-series was NOT verified in this change, and per Decision #27
that question must be answered from on-machine SDK headers rather than
inference — which cannot be done on the Ubuntu box. Apple owns checking the
Metal/MPS capability before any sub-byte storage lowering is scoped.

Cross-backend sync `REDUCED-PRECISION-COMPUTE-2026-08-03` — **follow-up required; a real Apple regression was introduced and reverted here.**
The reduced-precision enforcement now computes at f32 and stores back at the
operand's dtype, rather than casting the result (which made dtypes right while
leaving values wrong). Apple is affected twice:
(1) **Regression, now fixed.** A refactor of
`_enforce_storage_dtype_preservation` replaced a region of `tessera/__init__.py`
that spanned `install_apple_gpu_interception(ops)` and deleted the call.
Nothing else installs it, so the eight canonical intercepted ops silently
reverted to the numpy reference and `ops.rmsnorm(..., rows=..., cols=...)` would
reject its trace-only kwargs and never return a `TraceRef`. The whole suite
stayed green because non-trace calls pass through by design and the Apple lanes
skip on this box. Restored, and guarded by
`tests/unit/test_apple_interception_installed.py`, which asserts the call site
AND the ordering (enforcement must wrap outermost or the five rebound ops lose
their dtype contract).
(2) **Ordering contract.** Enforcement runs after interception; it is a no-op
for trace calls because a `TraceRef` has no `.dtype`.
**Apple owns the follow-up** to run its device suites and confirm no encode
path depended on the previous (upcasting) dtypes. IR/reference-level only here.

Cross-backend sync `TILE-MMA-DATA-OPERANDS-2026-08-03` — **not applicable to the Tile operand contract; unrelated Apple-visible change is IR-only.**
Apple has no `tile.mma` consumer, so the data-operand correction does not reach
it. Apple IS touched by the accompanying storage-dtype work: ops whose declared
shape rule preserves the operand's storage dtype now have that enforced, and
the enforcement deliberately runs AFTER `install_apple_gpu_interception` so it
wraps outermost — the interceptor rebinds `rmsnorm`/`layer_norm`/`softmax`/
`gelu`/`bmm`, and enforcing before it left those unprotected.
**Validated at IR/reference level only** on the Ubuntu box; the Metal lanes
cannot run there. Apple retains the follow-up to confirm no encode path
depended on the previous (upcasting) dtypes.

Cross-backend sync `TARGET-IR-CONFORMANCE-2026-08-02` — **follow-up required, contract changed.**
The same W0.9 gate found Apple's emitted Target IR invalid: the function
container was an invented `tessera_apple.cpu.func` / `gpu.func` that no dialect
defines (now the standard `func.func`), and `cpu.kv_cache_read`,
`cpu.moe_solver`, `cpu.profiler_probe`, and `gpu.profiler_probe` were emitted
without declaration (now in `TesseraAppleOps.td`). Module attributes are
dialect-prefixed at render time.
**Validated at IR level only.** These changes were made and checked on the
Ubuntu/Strix Halo box; the Metal/Accelerate runtime lanes cannot be exercised
there, so Apple execution parity is unverified. Apple retains the follow-up to
re-run its device suites and confirm no runtime lane depended on the old
op names.

Cross-backend sync `CORE-ATTENTION-TRAINING-X86-2026-07-30` — **follow-up
required, no Apple contract change.** X86 adopted the shared rank-4 forward and
tensor backward attention loops and closed its Lion/Adafactor physical
adjoints. Zen 5 ABI, schedule, LSE policy, and timing do not transfer. Apple
retains its rank-4 forward, shared backward-loop/modifier, LSE-policy, and
optimizer/backward-materializer items.

## APPLE-MINMAX-1: IEEE-754-2019 min/max + NaN-suppressing-`max` audit *(closed 2026-08-23, M1 Max)*

Closes `JIT-MATH-AUDIT-2026-08-23` items (1) and (3) for Apple. Everything
below was **measured bit-exactly on the M1 Max**, not inferred; the fixes are
pinned by `tests/unit/test_apple_gpu_ieee_minmax_device.py` (7 tests,
`hardware_apple_gpu`), which was confirmed **red on the pre-fix dylib and green
on the post-fix dylib** by rebuilding both ways.

### The premise in the sync record was wrong in a useful way

The record said "the Apple **MSL** binary max/min kernels". There are none:
`tessera.maximum`/`minimum` reach the GPU through **MPSGraph**
(`mpsg_binary_node` cases 4/5), which is the single route for f32/f16/bf16 and
for the encoded device-buffer lane. The audit target was the MPSGraph op, not a
hand-written kernel — and it was non-conforming.

### Grounding (Decision #27 — on-machine headers, not recollection)

`metal_math` in the installed Metal toolchain defines `max(float,float)` and
`fmax(float,float)` as **the same `__metal_fmax` intrinsic**. MSL `max`/`min` on
floats are therefore never a `>` comparison: they are maxNum/minNum, NaN is
suppressed, and they take `__METAL_FAST_MATH__` — which is on, since
`MTLCompileOptions` defaults to the fast math mode (`MTLLibrary.h`). That is why
every NaN test in the new MSL code is on the **bit pattern**
(`(as_type<uint>(x) & 0x7fffffffu) > 0x7f800000u`) and not `x != x`: fast math
lets the compiler assume no NaNs and fold a self-comparison away.

### Finding 1 — the optimizer eps-floor pattern does not exist here *(no change)*

ROCm's Adafactor kernels and the x86 AVX-512 shim floored second-moment
statistics with a NaN-suppressing `max`. **Apple has no such floor.** The
hand-written `optimizer_f32` MSL kernel covers sgd/momentum/adam/adamw/lion and
every eps is **additive** (`sqrt(nv/b2c) + eps`); so is every norm/softmax eps
in the runtime (`1/sqrt(sumsq/N + eps)`, `1/sqrt(var + eps)`, `1/sqrt(ms + eps)`).
There is **no Apple Adafactor device kernel at all** — that op is x86/ROCm only.
Nothing to fix; recorded so the next reader does not re-audit it.

### Finding 2 — `tessera.maximum`/`minimum` violated the fleet contract *(fixed)*

MPSGraph's `maximumWithPrimaryTensor`/`minimumWithPrimaryTensor` are
maxNum/minNum. Measured over a 12-row special-value matrix, **7 rows disagreed**
with `IEEE-MINMAX-CONTRACT-2026-08-23` for each of max and min:

| a | b | device (pre-fix) | contract |
|---|---|---|---|
| `+0` | `-0` | `-0` | `+0` (max) |
| `-0` | `+0` | `+0` | `-0` (min) |
| `NaN` | `1.0` | `1.0` | `NaN` |
| `NaN` | `-inf` | `-inf` | `NaN` |
| `inf` | `NaN` | `inf` | `NaN` |

i.e. NaN suppressed in both directions, and a ±0 tie resolved to the **second
operand** — the same tie bug the x86 `vmaxps` body had. Fixed with
`mpsg_ieee_minmax` in `apple_gpu_runtime.mm`: bitwise-AND tie blend for max,
bitwise-OR for min (equal values are bit-identical unless they are ±0, so the
blend is the identity everywhere else — **the same blend the x86 AVX-512 vector
body uses, so the two backends now agree by construction**), then select the NaN
*operand* (not a fresh constant) so the payload survives, `a` winning when both
are NaN — matching `np.maximum`, which is what the shared host reference
`tessera/_ieee_minmax.py` is built on. Post-fix: **0 of 12 rows disagree**, for
both max and min, bit-level.

**Cost: none measurable.** 4.2M-element f32 lane, 30 reps: `max` 6.45 ms /
`min` 6.32 ms against `add` 6.39 ms / `mul` 6.48 ms. The lane is
transfer/bandwidth-bound, so the extra selects are free.

**Follow-up from #617 review (chatgpt-codex-connector, P2) — the C-ABI host
recovery path was aligned too.** `mpsg_ieee_minmax` fixed the *graph* node, but
`tessera_apple_gpu_mpsgraph_binary_f32` also has a CPU fallback that fires when
MPSGraph construction or Metal dispatch fails — and that any direct C-ABI caller
lands on. It still used `x > y ? x : y` / `x < y ? x : y`, which suppress a
left-operand NaN and pick the second signed zero, reintroducing exactly the
divergence on the recovery path. Fixed with a host helper
`ts_host_ieee_binop_f32` carrying the same AND/OR tie blend. The fallback switch
is now factored into an exported `tessera_apple_gpu_mpsgraph_binary_f32_host`
symbol — the void entry point calls it on `!ctx.ok` — so it is
**differentially testable without a Metal device** (a working device cannot be
forced to fail). New test `test_c_abi_host_fallback_matches_device_contract`
proven red on the bare ternary, green on the helper; the device suite is now
**9 tests**. bf16 upcasts through this f32 path so it is covered; f16 memcpy's on
fallback (Python upcasts), so it carries no host min/max.

### Finding 3 — `scatter_f32` min/max reduce laundered NaN *(fixed)*

Modes 2/3 of the scatter kernel used MSL `min`/`max`, so a NaN in either the
seed or the scattered value vanished, and a ±0 tie went to the second operand.
This reduce's **result is the output** — the same reason ROCm deliberately kept
its reduce kernel NaN-propagating — while the reference is `np.minimum.at` /
`np.maximum.at`, which propagate. Fixed with `ts_ieee_min_f32`/`ts_ieee_max_f32`
MSL helpers (bit-pattern NaN test, AND/OR tie blend).

### Finding 4 — the Cl(3,0) norm laundered NaN *(fixed)*

`sqrt(max(0.0f, s))` over the sum of squares: `max` is fmax, so a NaN component
produced a **zero norm** instead of NaN. The C++ host reference had the same
defect via `std::max(0.0f, s)` (`0 < NaN` is false → returns 0), so the two
agreed with each other and disagreed with `ga.ops.norm`, whose reference is
`np.sqrt(np.clip(<a,a>, 0, None))` — and `np.clip` propagates. Both are now a
NaN-propagating non-negative clamp. The clamp exists to absorb tiny negative
rounding in a sum of squares; NaN is not that.

### Open, recorded, deliberately NOT fixed: `relu(NaN)`

Measured: the Apple GPU unary lane returns `relu(NaN) = 0`; the reference
(`np.maximum(0.0, x)`) returns `NaN`. This is the same NaN-suppression family
but a **different op with its own contract**, and it sits on the hottest
elementwise path in the backend — wrapping it is a scope and performance
decision, not a bug fix to slip into a min/max PR. **The same question is open
on ROCm and x86**, whose relu lanes are equally unaudited for this; nothing here
establishes their behavior. Whoever picks this up should decide the `relu` NaN
contract fleet-wide first, the way `IEEE-MINMAX-CONTRACT-2026-08-23` decided
min/max, and then fix all three backends against it.

### Also open, recorded, not fixed: max/min inside the fused RL loss graphs

`mpsg_run_ppo_policy_loss_f32` and its GRPO sibling call
`maximumWithPrimaryTensor`/`minimumWithPrimaryTensor` **directly** rather than
through `mpsg_binary_node`, so they did not pick up the IEEE wrapper: the ratio
clip (`max(ratio, 1-eps)` then `min(·, 1+eps)`) and the PPO surrogate
`min(s1, s2)`. The `rl.py` references are `np.minimum`/`np.maximum`, which
propagate NaN, so a NaN-valued advantage or log-prob ratio diverges. Left alone
deliberately: these sit inside fused loss graphs with their own numeric parity
tests, and NaN inputs to a loss are a degenerate case that deserves its own
decision rather than a one-line swap smuggled into a min/max PR. The swap
itself is mechanical if that decision goes the obvious way — `mpsg_ieee_minmax`
is already in the file and takes the same operands.

### Evidence

Apple GPU device sweep, `-m "hardware_apple_gpu and not slow"`, same command
before and after the fix: **18 failed / 1135 passed → 11 failed / 1142 passed**
— the delta is exactly the 7 new tests. The 11 remaining failures are
pre-existing and untouched by this work (spectral composites, the strict retune
ledger, 8 `production_jit_phase3_while` cases, delta-erase routing). Separately,
`test_conformance_evaluator.py::test_complete_cells_are_evaluator_corroborated_on_darwin`
**segfaults in the full sweep and passes in isolation** — proven pre-existing by
reproducing it identically on the unmodified dylib, and deselected in both runs
above so the comparison is like-for-like. It is not caused by this change and is
not fixed by it.

Editing `apple_gpu_runtime.mm` changes its sha256, so the sealed E2E-SPINE-3
packet (`docs/audit/evidence/e2e_spine/apple_gpu/apple7`) had to be
**re-recorded, not hash-bumped** — `test_e2e_fleet.py` deliberately fails when
the fingerprint no longer matches the source. Re-recorded with a real device
run via `benchmarks/e2e_spine/record_apple_packet.py --lane apple_gpu`; the
packet keeps its **`device_event`** timing domain (not the degraded
`kernel_wall` fallback), i.e. a genuine measurement. After the reseal:
`test_apple_backend_roadmap.py` + `test_e2e_fleet.py` **83/83**, and
`check_generated_docs.sh` back in sync (`e2e_fleet` + `test_coverage`
regenerated). A full-Mac `pytest -m "not slow"` sweep is **32 failed / 15452
passed**; re-running exactly those 32 in isolation leaves **26**, every one
pre-existing and untouched by this work — ROCm/NVIDIA lanes that cannot execute
on a Mac, the x86-backend-absent packaging test, a `.claude/worktrees` scan
artifact, the 11 Apple failures listed above, and five perf-baseline bounds that
only trip under full-sweep contention (they pass in isolation).

## APPLE-VECTORIZE-1: the `TESSERA_JIT_VECTORIZE` lane on Darwin *(closed 2026-08-23, M1 Max)*

Closes `JIT-VECTORIZE-UNGATED-2026-08-23` and `JIT-MATH-AUDIT-2026-08-23` item
(2) for Apple. First-ever Darwin run of the vectorized lane, via the
now host-agnostic compat test.

### It aborted the process on the first compile

`test_vectorize_lane_correct_in_and_out_of_envelope` did not fail — it took the
interpreter down:

```
LLVM ERROR: checking for an interface (`mlir::SubsetInsertionOpInterface`) that
was promised by dialect 'vector' but never implemented. This is generally an
indication that the dialect extension implementing the interface was never
registered.
```

`tools/tessera-jit` registered `tensor::registerSubsetOpInterfaceExternalModels`
but not the `vector` or `linalg` ones. The vectorize lane's own output is
`vector.transfer_write` into a tensor, and `vector` promises the interface for
it. Because MLIR raises this as `report_fatal_error`, the lane's
transform-failure fallback — which is designed to catch exactly this class of
problem — **cannot** catch it. All three are now registered.

### Why the AVX-512 host said the opposite, and why that was predictable

The x86 record for `JIT-VECTORIZE-UNGATED-2026-08-23` states "the recorded
bufferization abort no longer reproduces (verified with the registration
absent)". That was true **on that host and unfalsifiable there**: the promise
check is `#ifndef NDEBUG` (`mlir/IR/OpDefinition.h`, `getInterfaceFor`), the
Ubuntu box's apt.llvm.org LLVM 23 is an NDEBUG build, and the Mac's manually
installed LLVM 23.1.0-rc1 has assertions **on**. On the NDEBUG host the
unregistered interface was silently lost (bufferization falls back to the
conservative answer — extra copies, still correct); here it is a hard abort.

This is **Decision #19's standing lesson with the polarity flipped**: there, a
host that *had* AVX-512 could not falsify a host-portability claim; here, a host
that *lacked* assertions could not falsify an unresolved-promise claim.

**Both halves are now measured, not inferred** (cross-host run 2026-08-23, with
owner-provided access to the other two boxes):

| box | `llvm-config --assertion-mode` | pre-fix `test_native_cpu_jit.py` |
|---|---|---|
| M1 Max (LLVM 23.1.0-rc1, manual install) | **ON** | aborts the interpreter |
| Strix Halo (apt.llvm.org LLVM 23.0.0) | OFF | 27/27 pass |
| The-Super-Bear (LLVM 23.1.0) | OFF | not run — see below |

The Strix Halo row is the control: same commit, clean tree, the *unmodified*
tensor-only registration, and the lane passes — the defect is genuinely
invisible there. *The Mac is the only assertions-enabled LLVM in the fleet, so
it is the only box that can falsify an MLIR contract claim.* Any "this MLIR
abort no longer reproduces" conclusion reached on either Linux box is
provisional until it is re-run here. The x86 record has been amended in place
accordingly.

**The fix was then validated on the box that owns the file.** With the patch
applied on the Strix Halo box: `tessera_jit` rebuilds clean and the same
five-suite packet is **73/73** there. An alternating A/B of the two shared
libraries at n=512, three reps each, is **indistinguishable** — 60.0 / 67.6 /
74.2 GFLOP/s pre-fix against 75.7 / 86.9 / 55.3 GFLOP/s post-fix; the host's
run-to-run variance swamps any effect, so **no performance change is claimed in
either direction** (a single-shot pair earlier looked like a 24% regression and
did not survive repetition). Registering the models removes a
conservative-copy path in principle; that did not show up above the noise.

**The-Super-Bear (CUDA sm_120) was deliberately not tested.** Its checkout is
two merges behind (`4f1ee4d`) and carries untracked work under
`benchmarks/nvidia/ptx_gemm_study/`; pulling or patching it would have put
someone else's in-progress work at risk for a CPU-lane change. Its
assertion-mode reading above is non-invasive and is the only claim taken from
it. The shared `tessera_jit` change is expected-portable there for the same
reason it is on Strix Halo, but that is unproven.

### Result

The whole shared CPU-JIT packet is green on Darwin: `test_native_cpu_jit.py`
**27/27**, and with the signature-guard, elementwise-totality,
boundary-discovery and native-required suites **73/73**. The lane genuinely
vectorizes through NEON — measured with the transform on vs. off:

| n | scalar | vectorized |
|---|---|---|
| 128 | 2.20 GFLOP/s | 27.44 GFLOP/s |
| 256 | 1.81 GFLOP/s | 53.73 GFLOP/s |
| 384 | 1.75 GFLOP/s | 59.77 GFLOP/s |

The x86 host measured 3.4 → 106.6 GFLOP/s at 256; **that number does not
transfer and is not claimed here** — different ISA, different core, different
tile fit. What transfers is the shape of the result: the lane is real on both,
and throughput still falls with size on both, so the missing cache-level
blocking above the register tiles is a shared follow-on, not an x86 one.

## APPLE-SPINE-1: reconcile retained compiler lanes after canonical selection

Cross-backend sync `EXECUTION-SPINE-2026-07-29` — **closed — host-free selector contract.**
Apple CPU and GPU already auto-promote eligible native packages through
`canonical_compile()` and retain their established
`apple_cpu_native_descriptor` / `apple_native_descriptor` runtime identities.
The NVIDIA selector cleanup changes no Apple IR, ABI, package, schedule, or
exact-device claim. Apple CPU and GPU now each own one
`native_package_kind()` / `package_native()` admission point, which the shared
driver uses for canonical selection and trace provenance. `apple_target_ir_mode`
is a closed `artifact`/`value` choice: `value` remains an explicit
compatibility/probe route and deliberately opts out of descriptor promotion;
unknown modes and `value` plus `package_native=True` are rejected rather than
silently choosing a route. Host-free decision-table tests cover both Apple
targets. This slice changes selection authority only, not emitted IR, ABI,
package contents, schedule, or exact-device evidence.

The x86 sibling has since reconciled its split by reserving canonical `x86` for
typed MLIR/native packaging and moving portable C to an `x86_c` arbiter
candidate. That naming/authority pattern is applicable to Apple, but x86
AVX-512 evidence and its shared-object loader do not transfer to Metal.

## APPLE-CALIB-1: contribute op breadth to the hardware-free score calibration

Cross-backend sync `COSTMODEL-CALIB-2026-07-29` — **superseded by the terminal
ROCm home-architecture rejection.** No Apple arbiter-score adoption work remains.

The independent Zen 5 hierarchical T1 packet now also rejects T1 for latency
ranking (median rho -0.4062, 0/3 winner matches). Its measured x86 L1D/L2/L3
inputs do not define Apple SLC semantics and transfer no selector decision;
Apple's evidence gate below remains architecture-owned.

**Historical calibration result and current subject.** The original
step-distance and bank-conflict metrics found in
production AMD code and recorded in
[`../../compiler/AMD_KERNEL_COMPILER_SURVEY.md`](../../compiler/AMD_KERNEL_COMPILER_SURVEY.md)
§3.7–3.8 were not retained after step-distance failed on its ROCm home
architecture. Apple must not revive or retune it. The current shared subject is
the T1 GEMM cache/reuse model, whose rank correlation decides how much weight the
arbiter's hardware-free pruning tier can carry, per
[`TILESIGHT_ASSESSMENT.md`](../../compiler/TILESIGHT_ASSESSMENT.md) §2.

**Apple's role.** The widest F4-verified op-family envelope in the fleet, so it
answers *does the score generalize across op kinds* — norm chains, attention with
online softmax, pointwise-reduce, gated matmul, coopmat `simdgroup_matrix`, not
just GEMM. NVIDIA and ROCm supply shape depth within GEMM/attention
(`NVIDIA-CALIB-1`, `ROCM-CALIB-1`). Both axes are required: a score fitted on one
architecture reproduces the overfit that assessment §5.2 records for NeuSight,
which led on the A100 inside its training distribution and lost that lead on
every newer part.

**Apple-specific caveat.** The production bridge passes `cache_bytes=0`
explicitly because Apple SLC is not interchangeable with a discrete-GPU L2.
Before T1 can rank Apple candidates, this item must define an Apple-owned
cache-capacity/traffic interpretation backed by a real source or measurement.
The retired LDS bank-conflict metric remains not applicable: Metal banking is
not documented to the required level.

**Missing exact-device evidence.** Rank correlation between T1 and recorded M1
Max GEMM latency after the cache semantics are defined. Apple still contributes
op breadth to later non-GEMM models, but T1 v1 is GEMM-only and must not claim
coverage of norm, attention, or pointwise/reduce families.

**Fleet outcome (2026-07-29).** ROCM-CALIB-1 re-derived the metric on the AMD
architecture it came from and reproduced 0/6 committed gfx1151 winners (median
rho -0.1381, 0% positive). Per the shared rule, the locality-as-latency-ranker
line ends without retuning; Apple no longer owes a promotion-calibration run.
Metal banking remains not applicable. Existing Apple measurements may still be
used for unrelated cache-model research, but cannot revive this rejected score.

## APPLE-RASTER-1: reconcile the MLX-inherited swizzle with the shared contract

Cross-backend sync `RASTER-CONTRACT-2026-07-28` — **closed — carried, with
row-major retained.** The owning M1 Max (apple7) now has an emitted-MSL and
exact-device decision, not merely a schedule-note claim.

**Shared contract changed.** Schedule IR gained `raster_order` (`row_major` |
`column_major` | `grouped_m` | `grouped_n`) and `raster_group` on
`schedule.tile` / `schedule.knob`, mirrored by `TuningConfig` and the tuning
cache, over the arch-neutral `compiler/tile_rasterization.py`. Rationale:
[`compiler/TILESIGHT_ASSESSMENT.md`](../../compiler/TILESIGHT_ASSESSMENT.md)
§3.2.

**Resolved divergence.** `compiler/apple_gemm_schedules.py` did carry an
MLX-derived `swizzle_log` helper, but the real Tessera MSL emitter never
consumed it; it was neither a Metal function constant nor an executable raster
route. The helper is retired. Apple now passes the shared `raster_order` /
`raster_group` directly to the MSL emitter. `row_major` retains the exact direct
`tgid.y` / `tgid.x` coordinate expressions; a non-default source specialization
uses the same shared `tile_rasterization.emit_c()` mapping as CUDA and HIP while
retaining the 2-D grid, tile storage, and 32-lane threadgroup ABI.

**Decision (2026-07-31).** The emitted `grouped_m, raster_group=2` route is
native and numerically correct, but it is not neutral-or-better across the four
matched shape buckets. Therefore `row_major` remains the selected default; the
other shared orders are carried and executable, not autotune-promoted.

**Note the shape of the win differs here.** M1 Max is unified-memory with a
48 MB SLC, not a discrete L2 — the cache tier a swizzle protects behaves
differently, so a group size ported from a discrete GPU is not evidence for this
part. Measure locally or not at all.

**Validation performed (host-free).** `tests/unit/test_tile_rasterization.py`
proves the permutation property and compiles the emitted C against the Python
reference for every block id. The emitted form is C, so it validates the ROCm and
NVIDIA lanes; Apple's MSL synthesizer would need its own emission if option (a)
is chosen — the C snippet is *not* MSL.

**Exact-device evidence.**
[`apple7_raster_2026_07_31.json`](../../../../benchmarks/baselines/apple7_raster_2026_07_31.json)
records two independent warm 15-sample Metal-timestamp rounds for fp16-storage /
fp32-accumulation 32x32x16 Tile GEMM. All 16 paired rows are native and correct.
Grouped-M 2 wins the 256-cube strongly, loses the wide bucket, improves one tall
round but not enough for a global retain rule, and is mixed on ragged. Counter
sampling is explicitly `false` on this host, so no SLC interpretation is claimed.
`benchmarks/apple_gpu/benchmark_raster_order.py` reproduces the matched matrix.

## APPLE-AOT-4: S1 probe — what an MLIR → AIR emitter would actually cost

**Status: probe complete 2026-07-28, owning host (M1 Max / apple7, Metal
toolchain 32023.883). Capability findings, executed; not a perf claim.**

S0 showed a GPU runs hand-written scalar AIR IR. The open question was whether
that extends to the path that matters — `simdgroup_matrix`, the ceiling-setter
and the reason SPIR-V was rejected. It does.

### simdgroup_matrix is ordinary code, not a special form

Dumping the real synthesized coopmat kernel with `metal -S -emit-llvm`:

| MSL | AIR IR |
|---|---|
| `simdgroup_float8x8` | `<64 x float>` — a plain LLVM vector |
| `make_filled_simdgroup_matrix` | `declare <64 x float> @air.simdgroup_matrix_8x8_init_filled.v64f32.f32(float)` |
| `simdgroup_multiply_accumulate` | `declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(...)` |
| `threadgroup_barrier` | `@air.wg.barrier(i32, i32)` |

These are **external function declarations, not intrinsics needing backend
support**. Hand-written IR calling them — plus an `addrspace(3)` threadgroup
global and a barrier — compiles, packages, loads, and **executes correctly**
(fixture `tests/data/apple/handwritten_air_simdgroup.ll`, all 64 lanes exact).
Everything here is expressible in MLIR's LLVM dialect without extension.

### The builtin surface is 11 declarations

Compiling every synthesizer family and collecting `air.*` references:

| family | IR lines | builtins |
|---|---|---|
| coopmat | 408 | 7 |
| tiled | 277 | 3 |
| attention / attention-online | 244 / 235 | 2 / 2 |
| gated-matmul | 194 | 1 |
| matmul-epilogue | 183 | 2 |
| norm-chain | 155 | 2 |
| pointwise | 78 | 1 |

**Union across all eight families: 11 distinct builtins** — four simdgroup
matrix ops, six math (`convert`, `fast_clamp`, `fast_exp`, `fast_fmax`,
`fast_rsqrt`, `fast_tanh`), one barrier. That is the entire `air.*` dependency
an emitter must know how to name.

### So what S1 costs

Not the builtins (11 declarations) and not the codegen (MLIR's LLVM dialect
already emits functions, calls, address spaces, vector types). The work is the
**metadata emitter**: `!air.kernel` naming the function, one `!air.buffer`
descriptor per argument (location index, access, address space, element
size/align/type/name), builtin descriptors for `thread_position_in_grid` and
friends, plus module flags and `!air.version` / `!air.language_version`. Roughly
five node kinds, all declarative.

On that basis it looks *week-shaped* rather than quarter-shaped — but flag
that as an estimate, not a measurement. The 11 builtins and the IR sizes are
counted; the effort figure is a judgement with no prototype behind it, and it
sits next to measured numbers where it can borrow their credibility. Treat it
as "small enough to try", not as a schedule.

### What still argues against doing it

Feasibility is no longer the constraint; **supported-ness is**. `.ll` input to
`metal` is undocumented, `-x ir` is undocumented, AIR is undocumented by
deliberate Apple choice, and there is no man page. An emitter would rest on an
input path Apple can change or remove in any toolchain update, with no contract
and no deprecation warning — while the MSL lane (`apple_gpu_air`) already
captures the whole front-end saving through the supported input.

The case for building it is therefore *architectural*: it puts Apple where ROCm
already is — device code produced by the compiler rather than by a shell-out
over synthesized source — and it is the only way Apple joins the MLIR/LLVM spine
the other three backends share. Decision #26a names exactly that condition for
revisiting. This probe supplies the missing cost and risk numbers; the call is
a judgement about risk appetite, not about difficulty.

Reproduce: `xcrun metal -S -emit-llvm <kernel>.metal` on any synthesizer output.

## APPLE-AOT-3: S0 result — a GPU executes hand-written AIR IR

**Status: PASSED 2026-07-28 on the owning host (Apple M1 Max / apple7, Metal
toolchain 32023.883). Host-verified numerics, not a perf claim.**

The four things APPLE-AOT-2 listed as unverified are now settled, three by
experiment and one against me:

| question | result |
|---|---|
| does a GPU *run* hand-written AIR IR? | **yes** — `o[i] = a[i]*3.0f` written directly as LLVM IR, no MSL front end, dispatched via `newLibraryWithURL:`; output bit-exact vs `x*3` over 1024 elements |
| is `.ll` input to `metal` supported? | **no** — it works (`-x ir` too), but Apple documents MSL as the only supported input and deliberately does not document AIR |
| is a `.metallib` portable across GPU families? | **not family-tagged** — `metal-lipo -info` reports `architecture: air64_v28`; the tag is the *AIR version*, which tracks deployment target (`-mmacos-version-min=14.0` → `air64_v26`, `15.0` → `v27`). GPU-specific compilation happens later, at pipeline creation — which is also why ~15.2 ms remains in the AOT lane. Cross-family *execution* untested: one machine. |
| does the shared-dispatch refactor shrink runtime code? | **no — my prediction was wrong.** Measured: 58 lines before, 84 after (48 shared + 18 + 18) for two lanes. Duplication would have been 116, so the *marginal* cost per lane drops 58 → 18 lines (3.2×). It grows in absolute terms; it is cheaper than duplicating. |

### What S0 changes

The AIR path needs **no reverse engineering**. The stalled LLVM `air64` RFC was
blocked because it reimplemented Apple's bitcode writer and container; emitting
IR *into* `xcrun metal` requires neither. The metadata contract is declarative
and legible — `!air.kernel` naming the function, one `!air.buffer` per argument
(location index, access, address space, element size/align/type/name), the
builtin descriptor for `thread_position_in_grid`, `addrspace(1)` device
pointers. The fixture is `tests/data/apple/handwritten_air.ll`, exercised by
`test_gpu_executes_hand_written_air_ir`.

The shape is also ordinary rather than exotic: NVIDIA is MLIR → NVVM → PTX →
**ptxas** → cubin. Apple would be MLIR → AIR IR → **metal/metallib** →
metallib. A vendor assembler in the chain is normal.

### The risk that decides it

`.ll` input is **unsupported**. It is not in any Apple documentation, there is
no man page, and AIR is undocumented by deliberate choice. So an MLIR → AIR
emitter would rest on an input path Apple can change or remove in any toolchain
update, with no contract and no deprecation warning. The MSL lane has no such
exposure — MSL is the documented, supported input, and `apple_gpu_air` already
captures the whole front-end saving through it.

That is the trade to decide, and it is now a clean one: **structure (compiler-
produced code, parity with ROCm's tessera-opt-emitted hsaco) against supported-
ness.** Not, as Decision #26a assumed, feasibility — feasibility is settled.

### C1 landed alongside

`AppleAIRRunner` is registered, closing the gap that made `apple_gpu_air` the
only registered target without a runner. It registers with `default=False` so it
cannot become the process default by import side effect — that would silently
move every F4 verification onto the AOT lane. Only `run_fused_region` has an AOT
dispatch; the other three return a `REFERENCE_EXECUTIONS` tag so the oracle
trusts the reference rather than comparing numpy against itself.

New C ABI: `tessera_apple_gpu_metallib_elementwise_f32` — a generic 1-in/1-out
metallib dispatch. Written for S0, but it is the shape most synthesized
pointwise kernels take, so it is the first of the APPLE-AOT-2 phase-B entries
rather than scaffolding.

## APPLE-AOT-2: close out the `apple_gpu_air` lane

Cross-backend sync `APPLE-AOT-METALLIB-2026-07-28` — **landing, Apple-owned.**
APPLE-AOT-1 supplies the measured AOT/JIT comparison; APPLE-AOT-3 and
APPLE-AOT-4 supply the direct-AIR feasibility probes. NVIDIA, ROCm, and x86
record their architecture-specific follow-up, parity, or not-applicable
outcomes under the same key.

**Status: landing, plan of record updated 2026-07-29.** APPLE-AOT-1 proved the
lane works and is worth ~14.5 ms per cold kernel. The runner and first
pointwise/reduction coverage tranche have landed; artifact-contract hardening,
the remaining runtime families, the arbiter, and cache maturity remain open.

### Where it stands, measured not assumed

| | emitter | compiler (`compile_fn`) | runner | dispatch symbols |
|---|---|---|---|---|
| `apple_gpu` | ✅ | deferred (`None`) | ✅ registered | 17 |
| `apple_gpu_air` | ✅ (delegates) | ✅ real `.metallib` | ✅ registered, non-default | **6 of 17** |
| `nvidia` / `rocm` / `x86` | ✅ | ✅ real `.so` | ✅ registered | n/a |

The runner gap is closed. `AppleAIRRunner` is registered with `default=False`;
its fused-region and pointwise methods reach AOT dispatch, while unsupported
attention and gated-matmul families return the explicit reference tag. Runtime
coverage is now **6 of 17** public AOT symbols: coopmat matmul-epilogue, generic
elementwise f32, and pointwise/pointwise-reduce at f32/f16. The remaining gaps
are contract hardening, family coverage, selection, and cache lifecycle.

### A1 — register an `AppleAIRRunner` *(landed 2026-07-28)*

The four `KernelRunner` methods and `register_runner(...)` are implemented.
`accuracy_atol` deliberately retains the oracle default: Apple's f16 coopmat
measured 1.2e-4 against the f32 reference, within the existing 1e-3 budget, so
ROCm's looser 0.005 WMMA budget is not inherited. Keep the non-default
registration and explicit reference tags as drift gates while coverage grows.

### A2 — make the `artifact` / `deferred` contract safe

Nothing outside `apple_air.py` and its tests currently reads
`CompiledKernel.artifact` or `.deferred`. The moment an arbiter iterates targets
and assumes `artifact` is a loadable path, it breaks on `apple_gpu` as a `None`
surprise — and the three `.so`-returning backends make that assumption easy.
Add an accessor that forces both cases to be handled, plus a guard test that
every registered target either returns a path or sets `deferred`.

### B — coverage: 6 → 17, without 17 copies

The expensive way is a hand-written AOT twin per symbol. Do not do that. The
pattern already used for coopmat is the cheap one: extract the dispatch body
(`dispatch_matmul_epilogue_coopmat`) so the JIT and AOT entries differ *only* in
how they obtain the pipeline, via `compile_msl_kernel` or `load_metallib_kernel`.
Applying it to the remaining families makes the JIT entries thinner too, so
total runtime code goes down rather than up.

Families, in the order their value lands:

1. `pointwise` + `pointwise_reduce` (f16, f32) — **landed** through shared
   dispatch bodies and matching non-Darwin stubs.
2. `matmul_epilogue` scalar / tiled (f16, f32) — completes the region the lane
   already serves.
3. `norm_chain` (f16, f32).
4. `attention` (f16, f32) — the one where cold-compile cost is felt most, since
   attention kernels are the biggest source the synthesizer emits.
5. `gated_matmul` (f16, f32).
6. `tile_simdgroup_gemm` (f16, bf16).

Each new C ABI symbol needs its non-Darwin stub; the ratchet in
`test_apple_runtime_stub_parity.py` fails the build if one is missed.

### C — the arbiter *(the actual payoff, and fleet-wide)*

Selection between `apple_gpu` and `apple_gpu_air` per
`(op, shape-bucket, dtype, target)` on measured evidence — Decision #28's
measured, accuracy-budgeted arbiter, for which this is the first backend with
two genuinely comparable candidates. It needs a persisted decision record, and
it must treat the offline build cost as amortised (~5 cold launches) rather than
per-launch.

This is where the Apple work stops being Apple-specific: the arbiter is shared
infrastructure, and ROCm/CUDA will feed it candidates too.

### D — cache maturity

Artifacts live in `$TMPDIR/tessera-apple-air` with no eviction and no sharing.
Before this is load-bearing: a durable location, an eviction policy, and a
decision on whether artifacts are fleet-shareable (they are host-ISA-specific,
so probably per-machine — but the `kernel_cache` key does *not* include host
identity today, which is the same latent collision X86-1 flags for `-march=native`).

### The real gap to ROCm and CUDA (corrected 2026-07-28)

An earlier draft of this plan claimed ROCm and CUDA "have no JIT lane at all"
and so had "nothing to catch up on". **That was wrong**, and wrong from
absence-of-evidence: it was inferred from `nvrtc`/`hiprtc` not appearing in
`emit/nvidia_cuda.py` and `emit/rocm_hip.py`. Those two files indeed have none —
but the shipping runtime lanes do. `runtime.py` documents the ROCm WMMA lane as
"HIPRTC-compiled for the device arch (gfx1151/gfx1100) **at load**" and the
NVIDIA lane as "NVRTC-compiled warp-level mma.sync". There is a dedicated
`nvrtc_jit.cpp` in the NVIDIA backend. Both vendors JIT.

What each backend actually does, counted rather than assumed:

| backend | AOT artifact | produced by | JIT path | weight |
|---|---|---|---|---|
| ROCm | **hsaco** | **`tessera-opt`** — `convert-gpu-to-rocdl` → `rocdl-attach-target` → `gpu-module-to-binary` | HIPRTC at load (WMMA lane) | hsaco dominant: 601 references |
| NVIDIA | prebuilt `.so`, kernel NVRTC'd inside at load | cmake + NVRTC | NVRTC at load | JIT-dominant; no cubin/fatbin path in `runtime.py` |
| Apple | `.metallib` | **`xcrun metal` shell-out from Python** | `newLibraryWithSource:` at launch | JIT-dominant; AOT at 1 of 17 |

Three corrections follow, and they change the plan's framing:

1. **Apple is behind ROCm on AOT, not ahead of it.** ROCm's precompiled lane is
   the dominant one and has been for a long time; Apple's is one kernel old.
2. **The gap to ROCm is architectural, not coverage.** ROCm's AOT artifact comes
   *out of the MLIR pipeline* — the compiler produces the binary. Apple's comes
   out of a Python subprocess calling a vendor CLI. Closing B gets Apple to
   ROCm's *coverage*; it does not get Apple to ROCm's *structure*.
3. **NVIDIA is the backend closest to Apple's position**, not the distant one —
   its device code is NVRTC-compiled at load and it has no precompiled lane in
   `runtime.py`. The AOT-vs-JIT question is genuinely open there, and the
   measurement method (with its cache control) transfers directly.

This also reframes the AIR deferral recorded in Decision #26a. That deferral
rests on AIR saving no more than the ~15 ms `apple_gpu_air` already captures,
which stands. But the *architectural* case is stronger than that framing
suggested: an MLIR → AIR path would put Apple's AOT artifact where ROCm's
already is — produced by the compiler rather than post-processed by a shell-out.
Revisit on that basis, which is exactly the "architecture, not performance"
condition #26a names.

### Sequencing

A1 → A2 → B1-B2 → C, with D before C ships. A1 is hours and unblocks the rest;
B is mechanical but the bulk of the work; C is the only part that needs design
discussion, and it should be designed fleet-wide rather than for Apple alone.

## APPLE-AOT-1: `.metallib` pipeline creation measured against the JIT lane

**Status: measured 2026-07-28 on the owning host (Apple M1 Max / apple7,
macOS 26.5.2, SDK 26.5, Metal toolchain 32023.883, `air64-apple-darwin25.5.0`).
Host-wall timing, not device-event evidence; not selector-eligible.**

`apple_gpu_air` (`emit/apple_air.py`) compiles synthesized MSL ahead of time —
`xcrun metal -c` → `.air` → `xcrun metallib` → `newLibraryWithURL:` — against
the default `apple_gpu` lane's `newLibraryWithSource:`. Both run the same
synthesized coopmat kernel and the same
`dispatch_matmul_epilogue_coopmat` in the runtime, verified **bit-identical**
(max |diff| exactly 0.0, both 1.2168e-4 from the f32 reference at f16 storage).

Cold pipeline creation + one dispatch, 256×256×256 f16 coopmat, n=25, a
never-before-compiled kernel per sample, device pre-warmed, lanes interleaved:

| lane | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| JIT `newLibraryWithSource:` | 28.7 | 29.3 | **29.7** | 30.0 | 30.5 |
| AOT `newLibraryWithURL:` | 14.9 | 15.1 | **15.2** | 15.4 | 15.8 |
| AOT offline build (excluded) | 72.2 | 73.0 | **73.7** | 74.4 | 77.2 |

**AOT roughly halves pipeline creation — ~14.5 ms saved, 1.95×.** The offline
build costs ~73.7 ms once per kernel per machine and repays after ~5 cold
launches. Warm steady state is a wash (0.36/0.39, 0.69/0.72, 1.61/1.53 ms at
128/512/1024 cubes) — expected, since both are then a cache lookup into the
same dispatch.

**The measurement needs a control, and the obvious one is wrong.** Metal keeps
an on-disk shader cache that survives process exit: the same kernel measured
140.8 ms in one process and 0.5 ms in the next. Timing "first launch in a fresh
process" therefore measures whether that kernel was ever built on this machine.
A first attempt controlled it with a unique *unused `constant`* — which the
Metal compiler drops as dead code, producing byte-identical metallibs, so the
AOT lane reloaded one artifact and reported 1.2 ms (a 13× win). Renaming the
kernel **entry point** per sample makes each library genuinely distinct and
moves both lanes: JIT 15.7 → 29.7, AOT 1.2 → 15.2. The *saving* was stable
across both methods (14.6 vs 14.5 ms); the *ratio* was not (13× vs 1.95×).
`test_apple_air_target.py::test_nonce_control_defeats_metals_shader_cache`
asserts the artifacts differ, so a toolchain change cannot silently restore the
flattering number.

**Strategic read for Decision #26a.** The ~15 ms AOT removes is the MSL
front end. The ~15.2 ms that remains is AIR → GPU-ISA, which *any* AIR-based
path still pays — so emitting AIR directly from LLVM IR would save the same
~15 ms and no more. The ceiling on this whole direction is about half of cold
pipeline creation, which should temper how much the undocumented-format and
legal exposure of direct AIR emission is worth.

Reproduce: `python3 benchmarks/apple_gpu/benchmark_aot_vs_jit.py --samples 25`.

**Decision (2026-07-28): ship the AOT metallib lane; defer a direct AIR
emitter.** `apple_gpu_air` is the fast path and stays on supported tooling. A
direct LLVM IR → AIR emitter is not scheduled: it would save the same ~15 ms
this lane already captures, because the residual cost is AIR → GPU-ISA which any
AIR path pays. Its case is architectural — sharing the LLVM lowering with
CUDA/ROCm/x86 — and should be reopened on that basis, with a measured need, not
on compile-time grounds. SPIR-V → SPIRV-Cross → MSL is rejected: it cannot
express `simdgroup_matrix`, so it would cap the Apple ceiling.

**Cross-backend note.** This is the fleet's fast-path shape, not an Apple
special case: a precompiled artifact behind `register_compiler(target,
compile_fn)` plus the content-addressed cache. NVIDIA, ROCm, and x86 already
return real artifacts (`.so` via nvcc / hipcc / clang); Apple was the only
`deferred` compile-on-launch lane until now. The same AOT-vs-JIT question is
expected on ROCm and CUDA as their performance work ramps — reuse this harness,
and reuse its **cache control**: a never-before-compiled kernel per sample, or
the number is the vendor's shader cache rather than the compile strategy.


Cross-backend sync `TESSERA-OPT-CAPABILITY-SKIP-2026-07-27` moves the last 43
self-resolving test files onto the shared `tests/_support/compiler_tool.py`
driver contract and folds `CompilerToolchain` onto the same resolver and
capability check, leaving one of each in the tree. Apple is **parity
validated**, not merely unaffected: the Apple compiler-tool fixtures
(`test_apple_canonical_gemm.py`, `test_apple_threadgroup_pipeline.py`) reach
the driver through `CompilerToolchain.require_tessera_opt`, whose bare-pass-name
spelling is preserved, and which now also discovers
`build-apple/tools/tessera-opt/tessera-opt` — a candidate the old
`CompilerToolchain` search order did not carry, so a `build-apple`-only tree
that previously skipped as "not built" now resolves. Selection prefers an
in-repo build but takes the first candidate registering the requested passes.
No Metal registration, MSL/MPS schedule, runtime ABI, selector, storage policy,
device evidence, or timing gate changed, and **no exact-device evidence is
claimed or required** for this host-free infrastructure change. Apple's
separately owned package and exact-device gates are untouched.

Cross-backend sync `ROCM-BF16-ATTENTION-2026-07-27` adds no Apple capability
claim. It proves exact optimized BF16 forward and deterministic five-entry
backward attention on gfx1151 for the shared ragged-GQA,
bias+softcap+causal-window+dropout contracts. AMD BF16 WMMA, LDS scheduling,
HSACO packaging, HIP launch workspace, numerical results, and resident
host-wall timing are architecture-owned and do not transfer to Metal. Apple
retains its separately owned storage policy, package, exact-device, and timing
gates while shared semantic parity remains unchanged.

Cross-backend sync `TESSERA-OPT-BUILD-CAPABILITY-2026-07-27` is **closed**.
The shared lit resolver now accepts `TESSERA_OPT_BIN`, `TESSERA_OPT_PATH`, and
`TESSERA_OPT_CPP` after the canonical `TESSERA_OPT` override, and the validation
script forwards its selected binary through that contract. Exact gfx1151
verification proves the full ROCm driver, legitimate lean ROCm artifact
driver, conflict rejection, both named streaming-attention fixtures, the
seven-fixture filter, and the complete 50-test ROCm backend lit suite. This is
shared test/build infrastructure only; no Metal registration, schedule,
runtime ABI, device evidence, or selector changes.

Cross-backend sync
`ROCM-ATTENTION-SHARED-BACKWARD-CONSUMER-2026-07-26` makes ROCm gfx1151 the
first direct physical consumer of the shared tensor-valued attention backward
phase loops. Apple remains **follow-up required** to validate the same
dQ/split-dK/dV/fixed-reduction contract and map it to a Metal-owned package.
The AMD WMMA schedule, five-entry HSACO, HIP launch workspace, gradient
evidence, and host-wall timing do not transfer. No shared IR or Apple
capability state changed in this ROCm-owned closure.

Cross-backend sync `CORE-ATTENTION-TENSOR-LOOPS-MODIFIERS-2026-07-26`
materializes the deterministic split/reduced backward contract as tensor-valued
shared `scf.for` bodies: dQ is query-head/block owned, dK/dV partials are
launch-owned `[split,B,Hkv,Sk,D]` tensors, and reduction is fixed ascending
split order. The shared forward KV recurrence now carries registered additive
bias and softcap operations in canonical
`softcap(scale*QK^T + bias)` order, including rank-4 per-head bias. Apple is
**follow-up required** to lower these shared phase operations into its Metal
backward package and direct forward schedule. The gfx1151 ABI repair, HSACO,
numerical result, and resident timing do not transfer.

Cross-backend sync `CORE-ATTENTION-BACKWARD-CONTRACT-2026-07-26` adds verified
split count, launch-owned workspace, block-loop metadata, ascending reduction
order, and canonical `softcap(scale*QK^T + bias)` semantics to the shared
carrier/oracle. Apple is **follow-up required** to map this form to Metal and
validate dropout replay; gfx1151 code, evidence, and timing do not transfer.

Cross-backend sync `ROCM-E2E-ATTENTION-BACKWARD-2026-07-26` is not applicable
to Apple physical execution. It adds a ROCm-owned five-entry HSACO and
gfx1151 split/reduced launch workspace without changing the shared launch
descriptor schema or canonical backward loop. AMD WMMA kernels, workspace
topology, exact-device gradients, timings, and selector state do not transfer.

The ROCm optimized-attention feature follow-up under
`ROCM-E2E-ATTENTION-CARRIERS-2026-07-26` adds AMD-only deterministic dropout
replay and combined bias+softcap consumption to the gfx1151 WMMA schedule,
plus a host-wall resident performance ratchet. The semantic combinations are
already represented by the shared carrier; no shared ABI or Apple capability
changed. Apple parity therefore remains unchanged and its Metal lowering,
counter implementation, numerical proof, and timing evidence do not inherit
from this ROCm result.

Cross-backend sync `SSA-STATEFUL-TRANSPORT-2026-07-26` retires every active
`#tile.buffer_ref` compatibility reader after migrating the shared
barrier-reuse and WarpSpec lifetime fixtures plus the ROCm LDS fixture to
`!tile.buffer` def-use. The deprecated attribute remains parser-visible only
for migration diagnostics and archived IR. Apple/shared IR therefore no longer
depends on name-based allocation identity; Metal threadgroup scheduling remains
Apple-owned follow-up. The same sync generalizes the proven Apple ReplaySSM
lifecycle schema to target-keyed resident ABIs and adds explicit MoE launch
workspace ownership plus optional rank/device topology binding. Apple retains
its existing session-private ring, flush/rollback, ordered submission, and
drain-before-release semantics; ROCm execution and evidence do not transfer.

Cross-backend sync `ROCM-E2E-ATTENTION-CARRIERS-2026-07-26` is a ROCm-owned
physical consumer and exact gfx1151 evidence landing for the already-shared
forward/backward attention carriers. Apple requires follow-up for any carrier
variant not already covered by its Metal execution contract; AMD wave32 WMMA,
LDS ownership, HIP descriptors, scalar recurrence, resources, timings, and
selector state are not applicable to Metal and do not transfer. The ROCm v2
benchmark's separate operation-total and resident
`hipModuleLaunchKernel`/`hipDeviceSynchronize` host-wall domains are likewise
not applicable to Metal timing or selector policy. No Apple readiness or
exact-device row changes.

Cross-backend sync `ROCM-SSA-LDS-PIPELINE-2026-07-26` lands an AMD-owned
consumer of the existing shared `!tile.buffer`, `!tile.async_token`, and
`!tile.pipeline_state` vocabulary without changing those shared definitions.
Apple parity is validated at the portable IR boundary only. AMD LDS layouts,
waitcnt/s_barrier sequencing, gfx1151 structural evidence, compiler timings,
and selectors are not applicable to Metal and do not transfer. Apple retains
its separately recorded follow-up for architecture-owned threadgroup
allocation and pipeline-state lowering.
**Resolved 2026-07-27 by APPLE-PIPE-1 (row 22).** Apple is now a real consumer
of the same vocabulary: `tessera-apple-threadgroup-pipeline` places `smem`
`tile.alloc` into one capacity-bounded Metal threadgroup arena and claims
`!tile.pipeline_state` rings as ping-pong staging, with the NVIDIA-only
TMA/mbarrier/TMEM vocabulary rejected by named diagnostic.

Cross-backend sync `PACKED-LEGALIZE-CAPABILITY-2026-07-26` makes terminal
sub-byte storage a target + operation + physical-descriptor + complete
def-use-consumer decision. The newly admitted packed load/unpack, supported
round trip, packed matmul, and explicit conversion paths are NVIDIA SM120
consumers only. Apple remains disabled for generic terminal FP4/FP6
legalization until architecture-owned Metal physical consumers and exact
device proof land; no CUDA schedule or evidence transfers.
**Rejection proven 2026-07-27 by APPLE-DTYPE-1-REJECT (row 25).** The SDK gate
is now enforced rather than incidental: `apple_gpu` stamps no
`tessera.storage_packed` where `nvidia_sm120` does, and an unrouted
cooperative-matrix descriptor is refused with `APPLE_MMA_STORAGE_UNSUPPORTED`.
Apple remains disabled for the legalization itself; this proves the block, it
does not lift it. The deprecated
`#tile.buffer_ref` attribute remains parser-only for archived IR; no
Apple/shared fixture or active pass consumes it. Apple capabilities, execution
rows, and selectors are unchanged.

Cross-backend sync `CORE-STREAMING-ATTN-2026-07-26` replaces the shared
rank-2 FlashAttention whole-KV lowering with an explicit KV-block `scf.for`
carrying the FP32 output accumulator, running maximum, normalization sum,
producer/consumer `!tile.pipeline_state` values, and absolute boundary offset.
The shared TMA-shaped seam now retains typed block coordinates and logical
source extents for ragged zero fill; NVIDIA WarpSpecialization no longer emits
name-based `#tile.buffer_ref` or annotation-only `#tile.pipeline_state`
metadata. Apple is **follow-up required** to map the same recurrence onto an
architecture-owned Metal/MPS attention schedule and threadgroup allocation
identity.
**Rank-2 resolved 2026-07-27 by APPLE-ATTN-STREAM-1 (row 24).**
`tessera-apple-streaming-attention` re-forms the rank-2 recurrence as one Metal
flash-attention dispatch carrying `causal` / `logical_sk` / `window_left/right`
/ `kv_block` read off `tessera_attn.boundary_mask` rather than re-derived.
Follow-up sync
`CORE-STREAMING-ATTN-RANK4-ROCM-2026-07-26` adds shared rank-4 batch/head
distribution and a direct ROCm consumer. Apple remains **follow-up required**
for its architecture-owned rank-4 Metal/MPS consumer — APPLE-ATTN-STREAM-1
covers rank-2 only, and row 27 owns the rank-4 gap; the gfx1151 schedule,
HSACO, resources, wall timing, and selector evidence do not transfer. Deterministic
backward workspace materialization remains open shared work; no Apple
capability or selector changes in this synchronization slice.

Cross-backend sync `CORE-GEMM-KLOOP-2026-07-25` changes the shared
Graph/Schedule→Tile GEMM contract to explicit M/N/K `scf.for`, FP32/INT32
loop-carried accumulation, zero-pad ragged guards, structured layouts, and SSA
pipeline dependencies. Apple is **follow-up required** for an
architecture-owned Metal/AMX/SIMD consumer where that loop is preferable to
Accelerate; the existing value-mode Accelerate GEMM remains intentionally
unexpanded. NVIDIA Tensor Core fragments, PTX, SM120 resource/cache/timing
evidence, and selectors do not transfer to Apple CPU or GPU. No Apple
capability, execution state, schedule, or selector changes in this slice.
**Resolved 2026-07-27 by APPLE-TILE-2 (row 23).** `tessera-apple-canonical-gemm`
recognizes the three-deep nest and re-forms it as one `simdgroup_matrix`
dispatch carrying the loop's tile decision, fp32 accumulation, and the ragged
zero-pad guarantee, with exact-device execute-and-compare on Apple7. The
incumbent rule stands: recognition is not promotion, and value-mode
Accelerate/MPS remains the production route.

Cross-backend sync `ROCM-CORE-GEMM-KLOOP-2026-07-27` is **parity validated**
for Apple. The shared Tile change is limited to preserving the canonical
ragged-zero-fill guarantee across `tessera.matmul` → `tile.mma`; APPLE-TILE-2
already consumes the same loop and guarantee. ROCm's address-space-3 LDS
schedule, barriers, gfx1151 WMMA, HSACO resources, and host-wall results do not
transfer to Metal. No Apple route, capability, execution state, or selector
changes in this slice.

Cross-backend sync `COMPILER-LIT-BACKEND-GATING-2026-07-24`: shared lit feature
hygiene now rejects undefined requirements and obsolete global GPU target
flags. The retired fixtures and decoupled CUDA/HIP instruction probes were
NVIDIA/ROCm-owned; Apple has no unsupported fixtures in the LLVM23 suite, and
no Metal IR, runtime route, schedule, evidence, or selector changed. This is
not applicable beyond parity validation of the shared test infrastructure.

Cross-backend sync `COMPILER-PYTEST-PLATFORM-SKIPS-2026-07-24`: compiler-owner
markers now skip a foreign compiler proof with its required system named in the
pytest summary (Apple, CUDA, ROCm, X86, or AVX512), including a per-system
count. Apple owns the host-free gate integration; this changes no Metal IR,
runtime route, schedule, evidence, or selector.

Cross-backend sync `STATEFUL-TRANSPORT-FOUNDATION-2026-07-19`: the shared launch
workspace schema now distinguishes per-launch scratch from session-persistent,
preserved state. ReplaySSM and MoE metadata contracts are portable, but this
NVIDIA slice changes no Metal allocation, command-buffer ordering, physical
schedule, resource claim, timing row, or selector. Apple resident ReplaySSM
must map its already-proven handle lifecycle to the shared fields in an
Apple-owned follow-up; local and distributed MoE routes retain their existing
Apple evidence and cannot inherit CUDA bandwidth.

Cross-backend sync `NVIDIA-E2E2-STATEFUL-REDUCE-2026-07-19` extends the shared
Tile surface with explicit ReplaySSM decode/flush, MoE dispatch/combine/grouped
GEMM, and `Outer/AxisExtent/Inner` reduction carriers, plus a backend-neutral
rank/device topology fingerprint. Apple does not inherit PTX images, CUDA
workspace residency, the serial/cooperative-128 reduction schedules, NCCL
submission, resources, timings, or selectors. Mapping these semantics to the
existing Metal/MPS and distributed-MoE routes is Apple-owned follow-up; FP8
epilogue execution remains SDK-gated and TF32 is not applicable to Apple.

This plan brings the proof discipline established by the CUDA and ROCm work to
the Apple backend. It complements [`APPLE_AUDIT.md`](APPLE_AUDIT.md), the
generated execution inventory, and the durable architecture under
[`docs/backends/apple/`](../../../backends/apple/). The generated execution
inventory is the authority for exact-target execution state (including
`native_gpu` versus `reference_cpu`); the durable backend documents are the
architecture authority. This file owns only the active execution order and
completion gates.

The goal is not to transplant CUDA warps or AMD waves into Metal. Apple route
selection must be measured across MPS, MPSGraph, synthesized MSL,
`simdgroup_matrix`, Metal 4 cooperative tensors/MPP, and authored package
subgraphs. Logical fixtures, ABI contracts, numerical oracles, diagnostic
rules, and benchmark schemas should be shared with CUDA and ROCm; physical
fragments, threadgroup shapes, residency strategy, and command-buffer schedules
remain Apple-owned.

## Current state and immediate risk

- APPLE-TEST-1 now has a structural inventory at
  `tests/_support/apple_inventory.py`: its current scan records **0** direct
  Apple/Darwin/Metal capability gates. Apple device, integration, compiler-tool,
  and portable simulated-host cases are classified at their actual proof
  boundary. Offline MSL
  compiler checks are now `compiler_tool` tests with a shared `metal`-tool
  boundary, rather than device-gated tests. The first
  cohorts raise `pytest -m hardware_apple_gpu` collection from **3 to 976 of
  15,374** unit tests: the MPSGraph warmup and MegaMoE measured paths, exact
  native proofs for f32 CSR/COO SpMM, SDDMM, BSMM, scatter, optimizer, local
  MoE, MoE transport, and RNG, plus gather/concat/slice/softcap/transpose,
  mixed-program residency, TopK, projections, BMM, reduction, MPSGraph
  Tier-1, composed MHA, MPSGraph-runtime/cache, control-flow stress, and
  memory-budget residency proofs, quantized matmul, TopK, complex-runtime,
  evaluator/native-required, value-executor, fusion-synthesis, GA/EBM benchmark,
  control-flow/tracing, attention, delta, LDT, MoE, and other JIT-route proofs.
  The shared pytest boundary now supplies the Darwin/Metal skip; the marked
  proofs retain their explicit `native_gpu` and `metal_runtime` assertions
  where JIT provenance is available.
  APPLE-TEST-2 binds the first cohort's
  execution-matrix row,
  generic-envelope ownership where applicable, runtime ABI symbols, marked
  native node, and explicit fallback node in one registry; the shared native
  assertion rejects a semantic `reference_cpu` result. The f32 MPS matmul,
  MPSGraph BSMM/gather, and Philox symbols used by the cohort are now
  ABI-registered, so APPLE-REG-1 rejects an unregistered replacement.
- **2026-07-18 APPLE-TEST-1 closure:** a fresh full unit-tree collection found
  **976 of 15,374** nodes behind the centralized `hardware_apple_gpu` boundary,
  while the structural inventory still found **0** direct Apple/Darwin/Metal
  capability skips. The residency, runtime, and offline Metal-compiler cohorts
  retain their marker/provenance ratchets, so a newly added inline gate or
  misclassified compiler test fails the portable inventory suite. Ongoing
  classification enforcement is maintenance, not an open implementation rung.
  **APPLE-TEST-1 is closed.** No shared marker semantics or sibling-backend
  ownership changed; NVIDIA and ROCm are not applicable.
- APPLE-CI-2 now has an executable host-free ownership gate:
  `scripts/run_apple_host_free_compiler_tests.py`. It reads the CMake backend
  declarations, probes Apple/NVIDIA/ROCm pass registration, then selects only
  `compiler_tool` tests owned by the declared compiler capability set. On the
  current Apple-only build, Apple lowering is registered while the NVIDIA and
  ROCm probes are explicitly unregistered; the selected Apple artifact lane is
  green. The gate accepts any CMake cache type for `LLVM_DIR`, resolves and
  verifies the matching MLIR runner-utils dylib, and exports that exact path to
  the selected tests. Foreign compiler tests carry `compiler_nvidia` or
  `compiler_rocm`. This closes APPLE-CI-2. NVIDIA and ROCm are not applicable:
  their compiler ownership expressions and toolchain runners are unchanged.
- Cohort ledger: **APPLE-TEST-2-C1 / APPLE-REG-1-C1** records f32 sparse
  transport (CSR/COO SpMM and SDDMM), BSMM, scatter, optimizer, local MoE,
  MoE transport, and Philox RNG. Each row binds its execution-matrix path,
  native and fallback node, and runtime ABI symbols in
  `apple_exact_device_proofs.py`; complex/conformal remains outside this cohort
  until a hardware-marked execute/compare proof replaces its fallback-capable
  portable tests.
- Cohort ledger: **APPLE-TEST-2-C2 / APPLE-REG-1-C2** records only the fused
  interleaved-f32 complex/conformal subset (`complex_mul`, `complex_exp`,
  `mobius`, and `stereographic`). The device proof requires a traced fused ABI
  route and numerical oracle; its bridge-miss negative is explicitly
  `reference_cpu`. The long-tail complex/certificate operations remain outside
  C2 because they are intentionally host structured or lack a fused ABI route.
- Cohort ledger: **APPLE-TEST-2-C3 / APPLE-REG-1-C3** records only f32
  MPSGraph `sum`, `mse_loss`, and `mae_loss` (binary subtraction plus
  multiply/absolute-value plus reduction).
  Their exact-device nodes execute and compare on Metal; a forced missing
  MPSGraph binding must return `reference_cpu` from `runtime.launch`, rather
  than retaining the execution-matrix default. Huber, smooth-L1, log-cosh, and
  the loss-family lane remain outside C3 because their middle computations are
  host structured. NVIDIA/ROCm require no plan change: their loss/reduction
  paths have separate exact-device owners and no shared ABI changed.
- **2026-07-17 C1–C3 exact-device evidence:** all 12 distinct ledger-native
  nodes passed twice on Metal from separate freshly compiled runtime images;
  the 12 corresponding fallback-injection nodes passed and asserted
  `reference_cpu`. The two C2 rows intentionally share one fused
  complex/conformal native node and one bridge-miss negative. This closes the
  first cohort's placement, oracle, fresh-runtime, and fallback-negative
  evidence only; **APPLE-TEST-2 remains landing/open** until the same proof
  ladder covers the remaining Apple families, ordering/stress, and performance
  layers.
- **2026-07-17 broader exact-device evidence:** two independent fresh-runtime
  runs collected 853 nodes and each completed **849 passed, 4 skipped, 0
  failed** (97.4 s / 99.2 s). The four skipped legacy hand-written synthesis
  comparisons have now been explicitly reclassified as retired, non-native ABI
  contracts; their live synthesized replacements carry the Metal-placement and
  numerical-oracle evidence, and a forced missing-synthesis binding must return
  the reference route. A third fresh-runtime post-reclassification run completed
  **850 passed, 0 skipped, 0 failed** (100.5 s). The LLVM/MLIR 23 migration also
  fixed the JIT engine transformer's dangling callback and bounded the
  process-wide ExecutionEngine cache, which had previously made serial device
  validation segfault after accumulated JIT compiles.
- **2026-07-17 stateful and performance ladder evidence:** a separate fresh
  runtime passed the package/session-cache, resident block-paged KV, ReplaySSM,
  command-buffer, MPSGraph-LRU, and control-flow cohort (**63 passed**), with
  the bulk-MPSGraph/control-flow ordering stress raised to 75 iterations. Two
  independent route-characterization runs (21 rows each) and two independent
  ReplaySSM runs (12 rows each) reported native dispatch and numerical
  validation for every row. The temporary artifacts are
  `/private/tmp/apple-routes-proof-{a,b}.json` and
  `/private/tmp/apple-ssm-replay-proof-{a,b}.json`; they are evidence, not a
  committed performance ratchet. The remaining proof-ledger work is to add the
  same explicit fallback-injection record to the other native family owners;
  the closure update immediately below records the final family set and
  corrected serial performance selection.
- **2026-07-17 APPLE-TEST-2 closure:** the proof ledger now includes the C1--C3
  ABI cohort, synthesized matmul/reduction replacement, paged-KV attention, and
  fused ReplaySSM. ReplaySSM's C ABI now returns an explicit dispatch bit: its
  exact-device node requires `native_gpu` and its forced missing-binding
  negative requires `reference_cpu`, so a numerically identical host reference
  can no longer earn the native rung. The final fresh-runtime correctness lane
  passed **850/850**; the serial measured lane passed **69/69**. Two simulated
  distributed-MoE wall-clock tests were removed from the Apple hardware marker
  because they use modeled communication and do not assert an Apple route; the
  JIT-bridge benchmark fixture typo was corrected. **APPLE-TEST-2 is closed.**
  The plan remains `landing` because APPLE-REG-1, TILE, retuning, paged-KV,
  ReplaySSM serving expansion, and device-keyed performance selection are
  separate owning items.
- **2026-07-17 APPLE-REG-1 closure:** the canonical Apple ABI registry,
  runtime-header ABI, target-map, exact-device proof, and Tile-envelope drift
  gates passed against the LLVM/MLIR 23 `build-apple` compiler. The Tile status
  test now honors `$TESSERA_OPT` before the stale default build path, preventing
  an ABI-incompatible LLVM dylib from masquerading as a lowering failure.
  **APPLE-REG-1 is closed.** No dtype/op/diagnostic/target state was added in
  this slice, so NVIDIA and ROCm are not applicable.
- **2026-07-17 APPLE-TILE-1 start:** the real Tile-to-Apple status/materialized
  artifact gate passes with the LLVM 23 compiler, but it is not yet an
  exact-device fragment proof: the current fixture uses `tile.mock` and asserts
  runtime status only. TILE-1 remains open until a shared logical value fixture
  selects an Apple-owned fragment/layout from target capabilities and proves
  packing, ragged store, geometry/resource record, and native execute/compare.
- **2026-07-17 APPLE-TILE-1 value/ragged evidence:** the value-preserving
  `tile.batched_gemm` path now runs both aligned `2x4x8 @ 2x8x16` and ragged
  `2x5x7 @ 2x7x9` fixtures for f32/f16/bf16. Each exact-device case asserts
  `native_gpu` + `metal_runtime` and compares against the NumPy oracle; the
  fixture supplies only logical shapes, while Apple lowering owns BMM packing
  and route selection. **8 passed.** TILE-1 remains open for an explicit
  selected physical fragment/layout and threadgroup/resource record; the MPS
  BMM value route must not be relabeled as simdgroup-fragment materialization.
- **2026-07-17 APPLE-TILE-1 fragment-materialization landing rung:** Apple7+
  Tile selection now owns an exact `simdgroup_matrix` descriptor: fp16/bf16
  storage, fp32 accumulation, an 8x8x8 MMA fragment, 32 lanes, and a
  `(32,1,1)` threadgroup. The target-selected materializer consumes that
  descriptor to emit the existing steel-shaped MSL artifact with cooperative
  packing, bounds zero-padding, partial-edge store, and double buffering.
  Its host-free structure and target limits gates passed **85 passed, 9
  compiler-tool skips**. At that point this was artifact evidence only; the
  source-backed ABI and exact-device evidence are recorded below.
- **2026-07-17 APPLE-TILE-1 resource-contract landing rung:** each selected
  simdgroup artifact now carries a target-owned record for its `(32,1,1)`
  launch geometry, 32 lanes, staged-A/B bytes, ragged-store scratch, buffering
  mode, and total threadgroup-memory demand. Materialization rejects a tile
  that exceeds the selected target's threadgroup-memory capacity (the
  double-buffered 32x32x16 fp16/bf16 case records 4,352 bytes). The focused
  fragment/emitter/feature suite passed **67 passed, 9 compiler-tool skips**.
  This completed resource planning for the artifact path; runtime evidence is
  recorded below.
- **2026-07-17 APPLE-TILE-1 native single-fragment rung:** a distinct,
  registered `tessera_apple_gpu_tile_simdgroup_gemm_f16` C ABI now accepts the
  selected steel MSL source and entry, binds fp16 A/B and fp32 output, and
  dispatches exactly one 32-lane `(32,1,1)` threadgroup per 8x8 output tile.
  It is separate from the MPS BMM ABI and rejects any other threadgroup size;
  the non-Darwin stub returns 0. A fresh runtime image compiled and ran the
  8x8 fp16 fragment on Metal with zero fp32-oracle error; the focused proof
  test also forces the ABI binding missing and observes an explicit non-native
  result. The follow-on expanded this exact-device proof to bf16 8x8x8 and
  ragged/multi-fragment fp16 `13x16 @ 16x11`; both remain native and match the
  fp32 oracle (**46 passed, 9 compiler-tool skips**). A 30-repetition warm
  end-to-end characterization retained at
  `/private/tmp/apple-tile-simdgroup-characterization.json` reports medians of
  0.310 ms (8x8x8), 0.311 ms (13x16x11), and 0.315 ms (32x16x32); it has no
  device-event timing or MPS comparison, so it is not a selector decision.
  The C++ full pipeline now selects this ABI only for strict static rank-2
  `tile.matmul`/`tile.gemm` with fp16 or bf16; rank-3 `tile.batched_gemm`
  deliberately remains on MPS BMM. The Python value executor materializes the
  selected source and rejects a non-native result rather than using MPS/NumPy.
  The compiler/runtime ABI regression passed **18 passed**. TILE-1 remains open
  for retained runtime resource/provenance telemetry and comparative device-time
  performance selection.
- **2026-07-17 APPLE-TILE-1 telemetry/first comparison rung:** every direct
  source-backed dispatch can now return a record containing the ABI symbol,
  source SHA-256, native/reference result, execution mode, selected resource
  record, and runtime MSL pipeline-cache size. Its focused regression passed
  **17 passed**. A warm 30-repetition `32x16 @ 16x32` end-to-end comparison
  retained at `/private/tmp/apple-tile-simdgroup-vs-mps.json` recorded 0.314 ms
  median for native fp16 simdgroup and 0.229 ms for the existing f32 MPS route.
  These are not equivalent dtype paths and have no device-event timing, so they
  are explicitly **not** a selector decision. Remaining work is equal-dtype MPS
  comparison plus Metal device-time/resource telemetry and a two-run stability
  gate before any production-route change.
- **2026-07-17 APPLE-TILE-1 equal-dtype stability rung:** two independent warm
  30-repetition fp16 `32x16 @ 16x32` comparisons passed their respective fp16
  numerical oracles (the MPS route uses documented `rtol=1e-2` accumulation
  tolerance). Retained evidence at
  `/private/tmp/apple-tile-simdgroup-vs-mps-f16-two-run.json` measured
  simdgroup medians of 0.336/0.293 ms versus MPS medians of 0.234/0.226 ms.
  MPS is the end-to-end winner for this one shape; no selector changed because
  the runtime at that rung exposed neither command-buffer GPU timestamps nor
  Metal counter sampling. The Tile record supplies selected static resource
  bytes and pipeline-cache state, but not measured occupancy/spills. The next
  required implementation is a dedicated runtime timing/counter ABI, followed
  by a broader shape/dtype corpus and an explicit promotion threshold.
  **Superseded — read as dated provenance, not current capability.** The two
  rungs immediately below implemented that timing/counter ABI. The runtime now
  exposes `tessera_apple_gpu_tile_last_device_time_ns` (completed
  `kernelStartTime`/`kernelEndTime`, `GPUStartTime`/`GPUEndTime` fallback),
  `tessera_apple_gpu_tile_counter_sampling_supported` +
  `tessera_apple_gpu_tile_last_counter_delta` (named
  `MTLCommonCounterSetTimestamp` set, dispatch-boundary sampling), and the
  7-bit `tessera_apple_gpu_profiling_capabilities` matrix, which additionally
  reports the statistic and stage-utilization counter sets, stage-boundary
  sampling, and the Metal 4 timestamp heap. What remains genuinely
  unavailable is narrower and is a **public-Metal limit, not missing Tessera
  work**: register count, scratch bytes, spill count, and true occupancy have
  no public query, so those fields must stay absent rather than inferred.
- **2026-07-17 APPLE-TILE-1 kernel-time rung:** the runtime now records the
  completed command buffer's `kernelStartTime`/`kernelEndTime` (falling back to
  GPU start/end only when available) through
  `tessera_apple_gpu_tile_last_device_time_ns`. The exact-device proof requires
  a positive measured value (**17 passed**). Two 30-repetition equal-dtype fp16
  kernel-time runs retained at
  `/private/tmp/apple-tile-simdgroup-vs-mps-f16-device-two-run.json` measured
  simdgroup medians 23.1/21.4 us and MPS medians 21.8/18.8 us for `32x16 @
  16x32`; MPS wins this shape in both domains. The following bounded-counter
  rung replaces the then-missing capability-gated counter path; no selector
  changed.
- **2026-07-17 APPLE-TILE-1 bounded counter/corpus rung:** the runtime now
  discovers the named `MTLCommonCounterSetTimestamp` set only when the device
  supports dispatch-boundary samples, allocates a two-sample buffer, and
  samples immediately before/after the source-backed Tile compute encoder.
  The dispatch record retains either its measured timestamp delta or explicit
  `counter_sampling_supported: false`; it never manufactures occupancy or
  spill values. This M1 Max reports the latter while retaining positive
  command-buffer kernel timing. The new
  `benchmark_tile_simdgroup.py` corpus made two independent 30-repetition
  warm runs for fp16/bf16, aligned `8x8x8`/`32x16x32`/`256x256x256`, and ragged
  `127x63x129` shapes in both end-to-end and kernel domains. All eight
  end-to-end rows retain MPS. Kernel-only microcase movement is not a
  production promotion: the selector's production domain is end-to-end, where
  MPS remains the route. The selector contract requires native placement plus
  numerical proof, retained resource/counter evidence, and a 5% win in both
  intended-domain runs; no production route changed.
- **2026-07-17 APPLE-GEMM-1 capture-telemetry landing rung:** an opt-in,
  thread-local dispatch record now spans the owned Apple command-buffer paths.
  Legacy MPS/MSL records use completed `kernelStartTime`/`kernelEndTime`
  (command-buffer time only as an explicit fallback); the shared MTL4 encoder
  uses a reusable two-entry timestamp heap and converts its raw tick delta with
  the device timestamp frequency. The same record retains the live MTL4
  threads-per-threadgroup, execution width, maximum threads, and static
  threadgroup-memory properties. Capture is disabled by default so precise
  timestamp sampling cannot perturb production dispatch. The standalone
  MPSGraph row-op path now encodes into an owned `MPSCommandBuffer`, commits its
  live `rootCommandBuffer`, and reports a whole-dispatch interval only when
  MPSGraph did not auto-flush and replace the supplied Metal command buffer;
  occupancy and spill fields remain null rather than inferred.
  `select_stable_gemm_routes.py` aggregates two or more current-schema reports
  by exact Apple GPU family and emits separate device/end-to-end decisions. A
  promotion requires native placement, numerical proof, repeated samples,
  retained resources/counters, at most 15% cross-run drift, and a 5% win in
  every run. Two fresh 30-repetition Apple7 reports at
  `/private/tmp/apple-gemm-stable-{c,d}.json` cover square, rectangular,
  ragged, fp16/f32, MPS, simdgroup, cooperative-tensor, MSL, and MPSGraph
  routes; `/private/tmp/apple-gemm-stable-ledger.json` records **0 promotions,
  13 incumbent retentions, and 9 inconclusive timing-domain rows**. MPSGraph
  device intervals are present in both reports; its three device decisions are
  inconclusive because cross-run drift exceeded the 15% bound. No production
  selector changed. NVIDIA and ROCm are not applicable: this is an Apple-only
  Metal ABI and Apple-only report extension, with no shared IR, schedule, or
  cross-backend benchmark schema change.
- **2026-07-17 APPLE-GEMM-1 paired-winner/resource evidence rung:** absolute
  cross-process latency is now diagnostic rather than a promotion veto. Each
  report runs nine alternating route blocks of 30 repetitions; a candidate
  must win at least 75% of paired blocks, clear 5% median speedup in both fresh
  processes, and keep cross-process speedup spread within five percentage
  points. The committed Apple7 ledger is
  `benchmarks/baselines/apple7_gemm_route_ledger.json`: **3 timing-domain
  promotions, 19 incumbent retentions, 0 inconclusive rows**. Only
  end-to-end winners affect production: f32 `128x257` and `256x256` softmax
  select MPSGraph instead of MSL on Apple7 (24.2--28.0% and 36.9--40.2%
  paired median wins, respectively, winning all 18 blocks). The f32
  `64x64x64` simdgroup route wins device time by 38.5--40.1% but loses end to
  end, so MPS remains production. All other measured matmul shapes retain MPS.
  The new profiling-capability ABI records what public Metal actually exposes
  on this M1 Max: compiled-pipeline limits, stage-boundary timestamp sampling,
  and the Metal 4 timestamp heap are available; statistic/stage-utilization
  counter sets and dispatch-boundary sampling are unavailable. Live MSL/MTL4
  records retain execution width, maximum threads, static threadgroup memory,
  simdgroups per threadgroup, and a clearly named threadgroup-capacity proxy.
  The runtime ABI exposes no register count, scratch bytes, spill count, or
  true occupancy metric, so those per-dispatch fields remain null rather than
  inferred from pipeline limits. A separate bounded Instruments `Metal System
  Trace` now supplies genuine compiler/spill evidence, summarized reproducibly
  by `benchmarks/apple_gpu/summarize_metal_trace.py` in
  `benchmarks/baselines/apple7_gemm_metal_trace_evidence.json`. The exact
  Apple7 process trace retained four compute-shader compile intervals (2.356 ms
  total, 1.486 ms maximum), two MTLibrary creation intervals (0.258 ms total),
  and seven named compute shaders. Exact command-buffer joins observed one
  64-byte spill event on each of ten `tessera.rowop.mpsgraph` submissions and
  zero spill events on ten f32 MPS GEMMs, ten f16 MPS GEMMs, ten explicit MSL
  softmax submissions, and twenty reusable MTL4 submissions. The MTL4 command
  buffer is intentionally reused and Instruments retains it as `Command Buffer
  0`, so its zero-event row is an aggregate MTL4 observation rather than a
  per-kernel claim. The default system-trace template recorded
  `counter-profile=0`, but the standalone `Metal GPU Counters` instrument is
  available on this M1 Max and two bounded captures enabled profile 3 with
  shader profiler 1. Its genuine `Compute Occupancy` counter (ID 24) produced
  376 command-buffer-correlated samples: f32 MPS GEMM retained 144 samples
  (one nonzero sample, 0.282% maximum), the reusable MTL4 buffer retained 12
  zero-valued samples, MPSGraph retained 216 zero-valued samples, and explicit
  MSL softmax retained four zero-valued samples. Those zeros are the measured
  counter values for this small characterization workload, not synthesized
  occupancy estimates; f16 MPS had no in-interval sample and remains null.
  The live threadgroup-capacity/concurrency proxy remains alongside the raw
  counter evidence. NVIDIA and ROCm are not applicable because no shared IR,
  schedule, or cross-backend ABI changed.
- **2026-07-17 APPLE-EPILOGUE-1 native/resource/selection rung:** synthesized
  f32, f16, and bf16 epilogues already had common-oracle coverage for bias,
  ReLU, GELU, SiLU, residual guards, ragged stores, large reductions, and a
  forced symbol-missing negative. The runtime now labels every synthesized
  command buffer and retains its live pipeline limits, actual threadgroup, and
  total pipeline-static plus encoder-requested threadgroup memory. A ragged
  `64x64x2049` tiled softmax proof records at least `2049 * sizeof(float)`
  scratch; an fp16 bias+SiLU proof records the selected cooperative-matrix
  threadgroup and both match the backend-neutral `FusedRegion` oracle.
  MPSGraph unary and binary epilogue dispatches now use an explicitly owned
  `MPSCommandBuffer` and expose status-returning ABI variants, so native
  placement is independent of numerical success. Before the later Metal 4
  envelope closure below, MPSGraph could legally call `commitAndContinue` and
  replace the supplied root command buffer, so timing remained null rather
  than reporting a partial interval.
  `benchmark_epilogue_routes.py` collected two fresh Apple7 runs with seven
  alternating trials of 15 repetitions for aligned `64x64x64`, ragged
  `65x63x67`, and `256x256x256` f32/f16 ReLU plus f32 bias+SiLU. The committed
  `benchmarks/baselines/apple7_epilogue_route_ledger.json` records a stable
  end-to-end synthesized-fusion win for all nine comparable rows (49.8--71.6%
  paired median speedup and 100% paired-block wins in both processes). Device
  decisions remain explicitly inconclusive because the unfused MPSGraph
  segments do not expose complete command-buffer intervals. Production already
  selects the synthesized fused route for these supported regions, so this
  evidence ratifies rather than changes that selector. GELU and bf16 remain
  native correctness/resource proofs but are not compared against a false
  mixed-dtype or missing-MPSGraph incumbent. NVIDIA and ROCm are not applicable:
  the new ABI and schedule evidence are Apple Metal-only and no shared IR or
  numerical contract changed.
- **2026-07-18 APPLE-TILE-1 closure:** the shared logical fixture now selects
  an Apple-owned descriptor and schedule without test-authored physical maps;
  the selected f16/bf16 fragment path has packing, ragged-edge, resource,
  provenance, native execute/compare, device-time, and counter-capability
  evidence. The two-run aligned/ragged corpus retains MPS in every end-to-end
  row. That measured non-promotion is a valid selector outcome, not unfinished
  Tile work. **APPLE-TILE-1 is closed.**
- **2026-07-18 APPLE-GEMM-1 closure:** the paired Apple7 ledger records a
  stable decision for every measured timing-domain row: three promotions and
  nineteen incumbent retentions, with no inconclusive rows. Native placement,
  numerical validation, timing-domain separation, resources, and bounded
  Instruments compiler/spill evidence are retained. New device families or
  candidate routes require a new corpus; they do not keep this Apple7 ratchet
  open. **APPLE-GEMM-1 is closed.**
- **2026-07-18 APPLE-EPILOGUE-1 closure:** every supported f32/f16/bf16
  epilogue has native placement, common-oracle, resource, ragged-store, and
  fallback-negative proof. The two-run Apple7 ledger records stable
  synthesized-fusion end-to-end wins for all nine comparable rows. Unsupported
  pairs retain an explicit non-fused route or registered diagnostic.
  **APPLE-EPILOGUE-1 is closed.**
- **2026-07-18 MPSGraph device-interval closure:** the telemetry-only Metal 4
  bracket writes a timestamp before graph execution, makes MPSGraph wait on
  that event, signals a second event at its documented completed stage, then
  writes the final timestamp after that signal. It therefore spans every
  internal `commitAndContinue` root rotation without treating a partial root
  interval as a graph interval. Unary (including the prior queue-owned
  epilogue path), binary, row-op, transpose, paged gather, and BSMM now encode
  through the owned descriptor path. The result is labeled
  `metal4_mpsgraph_envelope`, deliberately distinct from direct MTL4 encoder
  timing; without a Metal 4 timestamp heap telemetry stays unavailable rather
  than fabricated. Fresh exact-device softmax and epilogue smoke evidence has
  complete MPSGraph interval coverage. The historical Apple7 selector ledger
  is unchanged: a new two-run corpus is required before any device-domain
  selector decision can use this new timing domain. NVIDIA and ROCm are not
  applicable because this is an Apple Metal runtime telemetry path only.
- **2026-07-17 APPLE-ATTN-FWD-1 placement/resource landing rung:** the f32 and
  f16 online-softmax MSL command buffers now carry stable route labels, retain
  their actual `Sq x B`-derived threadgroup and live pipeline limits, and expose
  status-returning ABI variants. The exact-device proof covers ragged
  `B=2, Sq=17, Sk=19, D=128`, causal masking, f32/f16 storage with f32 softmax
  accumulation, positive command-buffer GPU time, and a shared NumPy oracle.
  The D=257 envelope negative returns status 0 and no device interval, so the
  legacy reference fallback cannot be mislabeled native. This is a landing
  rung, not closure: bias, softcap, window, MHA/GQA/MQA, long-context, resident
  command-buffer, cooperative-matrix, and MPSGraph candidate comparisons still
  need the full two-run measured corpus; APPLE-ATTN-BWD-1 is untouched.
- **2026-07-17 APPLE-ATTN-FWD-1 variant/selector rung:** one status-returning
  online-softmax ABI now composes additive bias, causal or sliding-window
  masking, logit soft-cap, and direct MHA/GQA/MQA KV-head indexing for native
  f32/f16 storage. It retains the actual threadgroup and pipeline limits and
  rejects invalid grouping, negative windows, and D>256 before submission. The
  exact-device matrix covers MHA, GQA, and MQA, ragged `Sq=5/Sk=37`, the
  combined bias+window+softcap contract, and MQA `Sk=1025`; every row matches
  the shared f32-accumulation oracle. The MPSGraph BSMM candidate now owns and
  labels its command buffer and returns native status. Two independent Apple7
  runs, each using seven alternating trials of 20 repetitions, compare f32/f16
  aligned `B1/H4/S64/D64`, ragged `B1/H4/Sq65/Sk67/D64`, and throughput
  `B1/H8/S128/D64` plain MHA. The retained
  `benchmarks/baselines/apple7_attention_route_ledger.json` promotes MPSGraph
  for all six end-to-end rows; production selection is exact-device,
  exact-shape, dtype, and timing-domain keyed. Device timing retains MSL for
  rows without a stable 5% MPSGraph win. The resident command-buffer candidate
  is measured separately in its device-resident input domain and retains live
  resources, but its shared-session command buffer exposes no complete device
  interval. No cooperative-matrix attention ABI exists, so that candidate is
  explicitly unavailable rather than assigned synthetic timing. This is not
  full APPLE-ATTN-FWD-1 closure: wider B/head/D and long-context matrices,
  variant-capable resident/cooperative candidates, and complete device timing
  remain open. bf16 continues to be labeled host-conversion plus f32 GPU
  compute, and APPLE-ATTN-BWD-1 remains separate. NVIDIA and ROCm are not
  applicable because the new ABI, selector, and physical schedule are
  Apple-only; shared attention semantics and numerical policy are unchanged.
- **2026-07-17 APPLE-ATTN-FWD-1 closure:** the forward lane now covers the
  remaining physical and evidence gaps without expanding into backward. The
  selector corpus spans `B=1/2`, 4/8/16 query heads, `D=64/128/256`, aligned
  and ragged lengths, and plain-MHA context through `Sk=1025`. The variant
  corpus adds MHA/GQA/MQA, bias+causal+window+softcap, `B=2`, ragged
  `Sq=65/Sk=67`, and decode-style MQA through `Sk=2049`. The resident scalar
  and one-SIMD-group-per-query-row candidates now accept the same variant ABI;
  the latter is named `cooperative_simdgroup` rather than being mislabeled a
  Metal cooperative-matrix route. No attention-specific cooperative-matrix ABI
  is available on this SDK/host, and that capability remains explicit rather
  than receiving synthetic measurements. f16 and bf16 keep native two-byte
  device storage; GPU-side casts surround f32 accumulation on the resident
  command buffer, with no host fp32 staging inside the attention ABI.
  `ts_enc_commit_wait` now publishes the completed owned-command-buffer Metal
  interval. Two independent Apple7 warm reports, each with five alternating
  trials of ten repetitions, retain 9 MSL variant rows and 18 resident versus
  cooperative rows; every row is native, matches the shared oracle, and every
  resident/cooperative row has 100% device-time coverage. Logical input/output
  bytes, residency, intermediate-storage policy, actual threadgroup/pipeline
  limits, GPU time, and end-to-end time are retained; unavailable occupancy,
  register, and spill counters remain null. The regenerated
  `benchmarks/baselines/apple7_attention_route_ledger.json` promotes MPSGraph
  for all eight plain-MHA end-to-end rows. In the distinct device-interval
  domain only f32 `B1/H16/Sq16/Sk1025/D256` has a stable two-run 5% win;
  all other device rows retain online MSL. `APPLE-ATTN-BWD-1` remains a
  separate open item and no backward implementation or policy changed.
  NVIDIA and ROCm are not applicable: this closes Apple-only runtime ABIs,
  storage handling, schedules, and evidence, with no shared IR, attention
  semantic, or numerical-policy change.
- **2026-07-17 APPLE-PAGED-KV-1 retained staged-gather rung:** the existing
  non-contiguous resident MPSGraph gather now encodes through an explicitly
  owned, labeled `MPSCommandBuffer`. `ResidentBlockPagedKVCache` retains
  `last_gather_execution` and the capture record for each gather; a framework
  pipeline that exposes no public PSO limits records the MPSGraph API and an
  explicit unavailability reason rather than synthetic resources. The
  exact-device proof interleaves two sequences to produce physical table
  `[0, 2, 4]`, gathers the correct non-identity values, and requires native
  status. Existing remap/reuse, concurrent-sequence, exhaustion, and teardown
  tests remain green. This closes provenance loss for the staged candidate but
  not APPLE-PAGED-KV-1: a direct resident page-table attention candidate,
  causal-offset/boundary stress, leak telemetry, and two-domain comparison are
  still required.
- **2026-07-17 APPLE-REPLAY-1 native block/timing landing rung:** output-only
  replay and fp32/f16 block decode now label their command buffers and retain
  live threadgroup/pipeline records. The block ABI returns native status, which
  propagates to `SSMStateHandle.last_block_execution`; N>256 returns an explicit
  reference provenance and common-oracle result. Focused rollback, forced
  binding-miss, f32/f16 block, resource, and ABI tests pass. Two independent
  Apple7 reports at 512 tokens, capacity 16, and 20 repetitions cover
  `1x128x128`, `1x256x128`, and `4x128x64`. The committed
  `benchmarks/baselines/apple7_replay_ssm_evidence.json` retains complete native,
  numerical, resource, end-to-end, and device-per-token evidence for all six
  output-only/block rows. End-to-end cross-run drift is 0.3--2.1%; device drift
  is 0.9--26.8%. The ledger deliberately makes no selector decision because the
  legacy benchmark does not interleave paired route blocks. Persistent resident
  inputs, forced flush/partial rejection/block-submit ordering, asynchronous
  ring backpressure, cleanup stress, and a paired selector corpus remain open.
  NVIDIA and ROCm are not applicable to these Apple-only runtime ABI changes;
  shared SSM state semantics and numerical policy are unchanged.
- **2026-07-18 APPLE-PAGED-KV-1 closure:** `ResidentBlockPagedKVCache` now owns
  one persistent int32 page table per live sequence. Its direct f32 MSL
  candidate forms rope-key scores and latent values by following that physical
  table in one dispatch; the staged peer performs two on-GPU non-contiguous
  gathers plus dense resident attention. Both share the same non-identity
  oracle, right-aligned or explicit causal offsets, and bounded windows. A
  failed multi-block reservation is transactional, lifecycle telemetry accounts
  for live pages/tables/calls, and teardown frees every table and pool. Thirteen
  focused tests pass on the Apple host, including exact-device direct/staged
  placement and equivalence. The committed two-run Apple7 corpus covers
  `127x64x32x1` and `512x128x64x1` with ten measured repetitions after three
  warmups. Direct wins both runs in both device and end-to-end domains and is
  promoted only for those exact f32 rows; unmeasured rows retain staged.
- **2026-07-18 APPLE-REPLAY-1 closure:** the Apple serving handle keeps scalar
  A, S0, and fixed-capacity delta/x/b/c rings in persistent `DeviceTensor`
  buffers. Block submissions encode against those buffers, commit without
  waiting, and rely on ordered Metal command-queue execution. Output slots stay
  leased until `wait()`, enforce explicit backpressure, reject flush/rollback
  while submissions are pending, and are drained during idempotent cleanup.
  Forced flush, ordered multi-block submission, rollback, partial speculative
  rejection, slot reuse, and cleanup match `SSMStateHandle` in seven new
  exact-device tests. The expanded Apple ReplaySSM and benchmark-contract
  regression set passes 52 tests. A narrow checkpoint-fold follow-up now gives
  one Metal lane to each `(batch, channel, state)` element, serially replays
  tokens without atomics, and writes resident `S0`; a second kernel clears all
  fixed-capacity rings in the same ordered command buffer. Native provenance,
  forced-boundary equivalence, repeated flush/cleanup, and a portable explicit
  fallback negative cover the
  lifecycle. The dedicated two-run Apple7 flush corpus records device and
  end-to-end timing separately at `1x128x64/T16` and `1x256x128/T16`: native
  device medians are 20.9--30.0 us and end-to-end medians are 298--318 us.
  The shared vectorized CPU fold remains faster end-to-end for these isolated
  flushes (44.6--146 us), so the native route is a residency/ordering closure,
  not a latency promotion. The paired serving two-run Apple7
  corpus compares `fused_block` with `resident_ring` at `1x128x64/T16` and
  `1x256x128/T16`, ten repetitions after three warmups. Fused block is the
  stable end-to-end winner; the smaller device-domain winner flips between
  runs and therefore earns no promotion, while the larger row stably retains
  fused block. NVIDIA and ROCm are not affected: their resident CUDA/HIP
  contexts and physical schedules remain independently proven.
- **2026-07-18 APPLE-RETUNE-1 paired-corpus foundation:**
  `benchmark_legacy_retune.py` now measures grouped GEMM, MoE SwiGLU,
  MPSGraph reduction, contiguous resident-KV reads, absorbed/explicit MLA, and
  ReplaySSM block/token-loop decode through one interleaved two-run schema.
  Every row shares a numerical oracle and records native/reference provenance,
  resource/API evidence, paired end-to-end medians, and a device interval only
  when it covers the complete route. The Apple7 corpus retains grouped fused
  GEMM and fused Replay decode, promotes single-dispatch MoE and absorbed MLA
  end-to-end on their exact small rows, and retains explicit MLA in the device
  domain. Reduction has end-to-end native evidence but no owned device interval;
  mapped KV and multi-dispatch peers remain explicitly ineligible for device
  selection. APPLE-RETUNE-1 stays active for wider shapes/dtypes, grouped
  SwiGLU/transport byte-bandwidth rows, and complete command-buffer intervals
  for the remaining composed routes.
- **2026-09-02 APPLE-RETUNE-1 promotion decisions made reproducible:** re-running
  `benchmark_legacy_retune.py --profile extended` twelve times on this M1 Max
  from one unchanged binary — same `runtime_fingerprint`, so only measurement
  noise differed — flipped two of the sixteen decisions
  (`retune_mla_decode` promoted `absorbed` in 7/12 recordings at
  `1x4x1x128x32x16x32x64` and 10/12 at `1x4x1x64x16x8x16x32`). Since every
  `apple_gpu_runtime.mm` edit forces a re-seal, a re-seal was close to a coin
  flip, and the obvious response — re-record until it passes — is selection on
  noise. **Root cause: two promotion gates that could not converge.**
  `maximum_cross_run_speedup_spread` capped `max - min` of the per-run
  speedups, and a range never shrinks as runs are added, so a route that was
  32–55% faster in all 24 runs and won 144/144 paired trials promoted 69% of
  the time at two runs and 7% at eight — *more evidence made a true winner less
  promotable*. `minimum_paired_win_fraction_each_run` was applied as
  `all(runs)` over a fraction that three trials quantise to {0, ⅓, ⅔, 1}, so
  0.75 meant "win all three" and one lost trial in fifteen flipped a route.
  Both are replaced by consistent estimators: a one-sided 95% lower confidence
  bound on the mean per-run speedup, and a win fraction pooled over every
  paired trial with a per-run floor that no run may lose on balance. Defaults
  move to `--runs 5` (a promotion needs ≥3 runs; two runs carry a t multiplier
  of 6.31) and `--trials 9`. Eight fresh recordings at the new defaults produce
  all sixteen routes identically (0/16 flips), versus 2/16 before. Decisions
  additionally carry `stability_verdict` and the new
  `retain_incumbent_unstable_candidate` status so "we could not tell" is no
  longer recorded as the same thing as "the candidate is slower".
  **Second root cause, and the more serious one — the corpus was measuring its
  own previous output.** The `moe` incumbent calls
  `rt._apple_gpu_dispatch_moe_swiglu_block`, which asks `production_route_for`
  which route to take — consulting the committed ledger this same script
  writes. So "composed" stopped meaning the composed lane the moment the ledger
  promoted `single_fused`: measured here, the incumbent runs **1995 µs with no
  ledger present and 1080 µs once the ledger promotes**, while the candidate
  holds at ~995 µs either way, swinging the recorded speedup **+50.3% → +7.7%**
  with neither kernel changing. That is a two-cycle oscillator no statistic can
  fix — a retaining ledger makes the incumbent look slow, which promotes; the
  promotion makes the incumbent *be* the candidate, which retains — and it also
  silently inverts the comparison, since a promoted row measures the candidate
  against itself. `run_report` now points the selector at a path that cannot
  exist for the duration of measurement, so every dispatcher falls back to the
  incumbent it declares and the corpus compares implementations. After the fix
  the incumbent measures ~2050–2230 µs whether or not the ledger promotes.
  **No production route changes.** All sixteen committed routes are what they
  were; the sealed ledger simply now reaches them reproducibly. One row is
  worth knowing about: with the loop fixed, `retune_moe_swiglu`
  16x32x64x32_e4 measures `single_fused` ~52% faster than the *real* composed
  lane on an otherwise-idle host and promotes it — but the committed ledger
  was sealed while a second session was running Apple GPU tests on this Mac,
  where the same recorder consistently refuses it
  (`retain_incumbent_unstable_candidate`, three consecutive recordings
  agreeing). Concurrent **GPU** work is what moves this row: eight busy CPU
  cores do not (+50.6% loaded vs +52% quiet — the paired interleaved design
  cancels CPU contention exactly as intended). So the row is a live promotion
  candidate that should be sealed deliberately on a quiet box, not folded into
  this change; refusing under contention is the safe direction and preserves
  the shipped `composed` route. NVIDIA and ROCm
  are unaffected: this is the Apple route ledger's own aggregation, and no
  parity is claimed for them. **Sibling-lane review correction:** the earlier
  warning about `flash_attn_mha` and `softmax` overstated the scope. The attention
  corpus passes `route_override`, bypassing ledger selection; the strict-route
  sealer only aggregates existing reports, and the tile/simdgroup benchmark
  binds implementations directly without ledger-based route selection. The
  demonstrated self-reference was the MoE incumbent in the legacy retune
  corpus, protected by `_measure_implementations_not_the_ledgers_choice`.
  This closes that sibling investigation, not the statistical residual below.
  **Third root cause — warmup.** One warmup call per route left a process's
  *first* run cold: the fused MoE candidate measured ~2250 µs for all of run 0
  and ~990 µs in every run after, dragging that row's interval down in 2 of 8
  recordings. The paired design cannot cancel this, because the two routes
  reach steady state at different rates. `_WARMUP_CALLS = 12`, interleaved.
  **Residual, and deliberately not papered over:** after all three fixes, 10
  fresh recordings give 15/16 rows identical; `retune_moe_swiglu`
  16x32x64x32_e4 refused once, when a single run hit an external stall
  (2538 µs against ~1000 µs either side) on a host carrying background load.
  The cross-run bound is a mean ± t·sd/√n and is not robust to one outlier run, so
  a transient stall reads as instability and the row is refused. That is the
  safe direction — it errs toward the incumbent, never toward a promotion — and
  the row is recorded as `retain_incumbent_unstable_candidate` rather than as
  a settled loss. Making the cross-run aggregate robust (a trimmed or
  median-based bound) would trade that safety for stability and is a separate
  call, not made here.
  **Three review findings on PR #701, all fail-open, all fixed.** (1) Masking
  `TESSERA_APPLE_ROUTE_LEDGER` did not neutralise the incumbent, because
  `TESSERA_APPLE_MOE_FUSED=1` is read as `... == "1" or selected_route ==
  "single_fused"` and bypasses the selector outright — an inherited one would
  have put the `composed` row on the fused kernel and measured it against
  itself, reproducing the exact corruption the block exists to prevent. The
  recorder now clears a named `_ROUTE_FORCING_ENV` set and restores it.
  (2) Each threshold in the confidence-bound rule set guarded its own check, so
  a ledger declaring the bound but omitting `minimum_promotion_runs` or
  `minimum_pooled_paired_win_fraction` skipped those checks while still reading
  as complete — a truncated two-report promotion would have been admitted. They
  are now mandatory whenever the bound is declared. (3) The aggregator admits a
  run on `fraction > 0.5` while the re-derivation rejected only `< 0.5`, so a
  run sitting exactly on 0.5 was refused by the producer and admitted by the
  loader; the comparison is now carried in the rules as
  `paired_win_fraction_each_run_is_strict`, absent-means-non-strict so the
  pre-pooling 0.75 ledgers verify unchanged. Each has a regression test that
  fails against the pre-fix source.
- **2026-07-27 APPLE-RETUNE-1 ledger invalidated by a runtime-source edit:**
  `benchmarks/baselines/apple_strict_route_ledger.json` pins
  `context.runtime_fingerprint = sha256:74eb6e95…`, a whole-file hash of
  `apple_gpu_runtime.mm`. Adding the APPLE-PLACEMENT-ABI-1 status twins changed
  that hash, so `load_strict_route_ledger` now rejects every retained decision
  with `context_mismatch:runtime_fingerprint` and the exact-host admission test
  fails on a Metal host. **This is the guard working, not a regression** — the
  retained decisions were measured against a different runtime source, and the
  fingerprint cannot know that the new symbols touch no measured path. The
  ledger must be **re-measured**, not re-stamped: editing the fingerprint in
  place would assert that old measurements describe new code. (The test was
  already failing before this change for the same reason, so this is a renewal
  that was already owed.) The test is `hardware_apple_gpu`-gated, so CI is
  unaffected. **Design note for the renewal:** a whole-file hash over a
  ~24k-line runtime means any edit anywhere — including adding an unrelated
  symbol — invalidates every retained route decision. Consider fingerprinting
  the per-route kernel source instead, so evidence survives edits that provably
  cannot reach the measured path.
- **2026-07-21 APPLE-RETUNE-1 extended exact-host renewal:** the owned fresh
  dylib produced two committed-strength (`5 reps`, `3` interleaved trials) runs
  over both the original and 2x geometry rows: 48 route rows across grouped
  GEMM, MoE SwiGLU, reduction, resident-KV, MLA, and Replay decode. The fresh
  strict-v2 ledger admits 16 exact Apple7 f32 decisions and retains eight
  negative rows as explicit ineligible evidence: mapped resident-KV in both
  domains plus composed MoE and MPSGraph reduction in the device domain. The
  corpus does not claim f16/bf16 because the committed rows predate their
  owned same-storage ABI/oracle pair. Low-precision retuning remains separate
  until fresh committed-strength rows measure that pair; it must not convert
  inputs or borrow another route's proof.
- **2026-07-21 APPLE-RETUNE-1 transport/low-precision ratchet:** every renewed
  row now retains logical host-visible input/output bytes and its end-to-end
  logical bandwidth, explicitly labeled as distinct from device bandwidth. The
  grouped-SwiGLU small and 2x rows carry 102,432 and 409,632 logical bytes per
  call. The historic f16/bf16 rejection rows remain sealed evidence for the
  then-f32-only corpus. MoE now owns raw-storage f16/bf16 ABI symbols with an
  exact-device oracle and one complete command-buffer interval; it needs fresh
  committed-strength retune rows before selector admission. Grouped GEMM keeps
  its same-storage composed C-ABI route but has no fused low-precision package.
  Complete device intervals for composed SwiGLU, mapped resident-KV, and
  MPSGraph reduction remain absent: they need an owned complete-route ABI, not
  telemetry summation.
- **2026-07-21 APPLE-RETUNE-1 low-precision MoE admission:** two fresh exact
  Apple7 reports at five repetitions and three interleaved trials seal
  `apple7_lowp_moe_retune_two_run.json` and its strict-v2 sibling ledger. Raw
  f16/bf16 MoE storage at the base and 2x shapes is numerically valid, native,
  and has complete command-buffer intervals in every sample; all eight
  shape/dtype/timing-domain decisions retain `single_fused_lowp`. The remaining
  low-precision gap is grouped GEMM: its C ABI remains composed-per-expert, not
  an owned fused low-precision package.
- **2026-07-18 APPLE-ROUTE-1 strict-ingestion foundation:** production lookup
  no longer reads a literal exact-row table. The v2 ledger gate matches the live
  Apple family and physical-device model, OS, SDK, configured LLVM/compiler
  digest, runtime-source digest, expiry window, native provenance, correctness,
  and requested timing domain; admitted decisions expose their exact ledger-row
  citation. The fresh Apple7 retune ledger admits eight decisions in a clean
  host process. Older v1 GEMM/attention/backward/paged-KV/Replay ledgers lack
  this envelope and are rejected, so those operations conservatively retain
  their incumbents until fresh strict ledgers are recorded. APPLE-ROUTE-1 stays
  active until each completed family is migrated and package-subgraph selection
  is separated into its own strict ledger namespace. NVIDIA and ROCm are not
  applicable to this Apple-only corpus/selector change; their physical-device
  probes, retained ledgers, and production selectors are unchanged.
- **2026-07-21 APPLE-ROUTE-1 remeasurement rail:** every owning benchmark now
  captures the live exact-device context in its raw report, and
  `seal_strict_route_ledger.py` accepts only two independently produced reports
  to create a `runtime_route` v2 ledger. Sealing retains SHA-256 source-report
  digests and places rows without a selectable full-domain result in
  `ineligible_decisions`, outside selector-visible `decisions`. The renewed
  paired corpus plus GEMM/softmax, forward-attention, backward-attention, and
  epilogue owner lanes are migrated. Each historical schema-v1 file remains an
  inventory only; its sibling v2 ledger is the sole selector evidence.
- A fallback result can prove semantics, but it cannot prove `native_gpu`, GPU
  residency, Metal ordering, resource lifetime, or performance. Device tests
  must assert their execution state and provenance explicitly.
- Apple already has broad MPS/MPSGraph/MSL execution, Metal 4 probes,
  `simdgroup_matrix` and cooperative-matrix candidates, fused GELU/SiLU
  epilogues, online-softmax attention, resident block-paged KV, ReplaySSM,
  command-buffer batching, route characterization, and a hot-path baseline.
  The work below strengthens, compares, and retunes these paths rather than
  reimplementing them blindly.
- The committed Apple hot-path ratchet is predominantly f32 and end-to-end
  wall-clock. It does not yet provide the square/rectangular/ragged/dtype matrix
  or per-candidate GPU-counter/resource evidence now required for CUDA/ROCm.
- Attention backward now has an Apple-owned native proof and stable route
  ledger; its physical schedules remain independent of CUDA and ROCm.
- **2026-07-18 APPLE-ATTN-BWD-1 native-candidate foundation:** the Apple
  runtime now exposes a status-only f32 MHA backward ABI. Two MSL encoders on
  one labeled command buffer recompute the softmax and produce dQ, dK, and dV
  with f32 accumulation; each output element owns its reduction, so the route
  is deterministic and has zero workspace/atomic traffic. Exact-device ragged
  and causal oracle tests verify all three gradients and repeated launches are
  bit-identical. The same ABI now owns a zero-workspace atomic dK/dV candidate
  using relaxed compare/exchange f32 accumulation and a deterministic two-way
  split candidate using exactly one additional f32 dK+dV partial plus a
  fixed-order reduction. The policy rejects deterministic atomic requests and
  insufficient split workspace before dispatch. Exact-device tests cover all
  three routes on ragged, batched, causal, and noncausal shapes against the same
  oracle; serial and split repeats are bit-identical, while atomic repeats are
  validated numerically under its explicitly nondeterministic contract.
  `benchmark_attention_backward.py` produces paired route rows with warmup
  separation, per-trial GPU/end-to-end medians, resources, workspace policy,
  and per-gradient error. Two Apple7 smoke collections each have twelve native,
  numerically valid rows and complete device-time coverage. Atomic wins every
  end-to-end row on this small foundation matrix; device-interval winners vary
  by row and run, so no timing domain is collapsed into another and `auto`
  remains on serial recompute. This is not yet a selector corpus: GQA/MQA,
  bias, softcap/window, f16/bf16 storage, workspace caps, wider and long-context
  shapes, and a committed stable selection corpus remain active. NVIDIA
  and ROCm are not applicable: the shared derivative semantics are unchanged
  and no CUDA/ROCm schedule is transferred.
- **2026-07-18 APPLE-ATTN-BWD-1 closure:** all three candidates now use
  query-streaming softmax/dP work rather than recomputing one softmax per output
  element. Atomic work owns one query row and confines contention to final
  dK/dV updates; serial gives one deterministic owner each KV head; split gives
  two deterministic owners one exact additional f32 dK+dV footprint and then
  reduces in fixed order. The status ABI shares forward's flattened-head
  MHA/GQA/MQA mapping, right-aligned causal and sliding-window masks, additive
  bias, and correctly differentiated logit softcap. Legacy rectangular causal
  callers retain their original zero-offset triangle. Native f16 and bf16
  inputs are read directly from two-byte Metal storage; dQ/dK/dV accumulate and
  return f32. Exact-device tests cover every route and dtype, batched/ragged
  MHA, GQA, MQA, bias, causal/noncausal windows, softcap, invalid-route
  rejection, deterministic repeats, and workspace limits.
  Two independent Apple7 reports contain 18 native, numerically valid, fully
  device-timed rows each. The committed
  `benchmarks/baselines/apple7_attention_backward_route_ledger.json` contains
  twelve timing-domain decisions. End-to-end selection promotes split-reduce
  for four rows, including causal `Sk=1025`, and atomic for two rows; paired
  median wins range from 27.8% to 67.3%, with 100% trial wins in both reports.
  Every device-interval row retains serial recompute. The legacy ledger records
  exact-device/shape/dtype/domain decisions, but strict v2 production ingestion
  now retains serial until those rows are re-recorded with current context;
  determinism and split-workspace policy remain enforced independently.
  **APPLE-ATTN-BWD-1 is closed.** NVIDIA and
  ROCm are not applicable to the Apple ABI, storage readers, schedules, or
  selector rows; shared derivative semantics remain unchanged.
- FP8/FP4/MX execution remains gated by the macOS 27 SDK/runtime surface. The
  compiler-side scale-layout and multi-plane contracts already exist; do not
  claim hardware execution until the public Metal tensor path runs natively.
- Cross-backend sync `NVIDIA-TEST5-2026-07-16`: the shared autotune corpus now
  carries additive compiler/resource, cold/warm, cache, and two-run stability
  evidence. Existing v1/v2 rows migrate without changing Apple selection, and
  no CUDA schedule or selector is transferred to Metal. Apple follow-up is to
  populate the same logical evidence fields from Metal-native counters during
  its own performance work; current Apple plan state is otherwise unaffected.
- Cross-backend sync `LLVM23-NVIDIA-2026-07-16`: not applicable to Apple
  execution. The fixes are confined to Ubuntu apt.llvm.org discovery,
  CUDA/NVVM lowering, and Linux NVIDIA/ROCm lit shell selection. No Apple IR,
  ABI, Metal schedule, numerical policy, or exact-device evidence changed.

## Completion definition

This plan reaches `closed` only when all of the following are true:

1. Host-free and compiler-artifact tests remain portable lanes. Apple promotion
   is owned by one local Metal 4 exact-device gate with two fresh-process
   correctness runs, an isolated paired-performance corpus, and a sealed packet
   pushed to the coordinating PR. Registered GitHub self-hosted runners are not
   used. Metal 3 is a non-blocking compatibility lane.
2. Every device test proves `native_gpu` placement on the intended route. A
   non-Darwin stub, NumPy fallback, symbol-presence check, or reference
   recomputation cannot earn a device pass.
3. Dtype, op, target, diagnostic, runtime-symbol, execution-state, and generated
   documentation registries are drift-gated. Every newly emitted diagnostic is
   registered and every live plan uses `open`, `landing`, or `closed`.
4. Portable Tile fixtures execute without test-authored physical fragments and
   select an Apple-owned layout/schedule from observed device capabilities.
5. Performance records use repeated medians after warmup, separate GPU/kernel
   time from end-to-end time where Metal counters permit it, and retain route,
   compiler, OS/SDK, device, residency, and resource evidence.
6. Paged KV and ReplaySSM pass the same non-identity, rollback, ordering, stress,
   and lifecycle closure used on CUDA/ROCm.
7. Production route changes consume only matching native-and-correct evidence;
   stale reports, reference rows, or records from another Apple GPU family
   cannot change selection.
8. The complete exact-device correctness lane passes twice from a fresh runtime
   image, and the isolated performance lane produces stable winner decisions.

## Apple-host preflight

Run decisive tests outside a sandbox in a fresh process. Record the exact host
before interpreting a skip or timing change:

```bash
sw_vers
system_profiler SPDisplaysDataType
xcodebuild -version
xcrun --sdk macosx --show-sdk-version
xcrun --find metal
python3 --version
git rev-parse HEAD
```

Also record Apple GPU family/capability probe output, macOS deployment target,
Metal language version, power mode, thermal state, and whether another process
is using the GPU. Metal 4 promotion requires a named Metal 4 host. Metal 3
coverage is compatibility-only and cannot promote a Metal 4 route; never
generalize a winner across Apple GPU families without a matching record.

### Use the dedicated LLVM/MLIR 23 prefix

**Updated 2026-08-28 — the accepted prefix is now Homebrew's `llvm` keg.**
Homebrew shipped the production LLVM/MLIR 23.1.0 release (RTTI ON, assertions
**OFF**), and by owner decision it replaces the manual pre-release
`/opt/homebrew/llvm-23.1.0-rc1` build, which has been removed from the
machine. The RTTI requirement stands: the toolchain must be built with
`LLVM_ENABLE_RTTI=ON`, or Tessera's pass and dialect typeinfo cannot link
(the brew keg satisfies this). Note the keg is an NDEBUG build — no fleet box
now has an assertions-enabled LLVM, so MLIR promise/contract regressions
(APPLE-VECTORIZE-1 class) are invisible everywhere until checked against an
assertions build; the source-build recipe below is preserved for recreating
one on demand. Before configuring or testing, set and validate the prefix:

```bash
export TESSERA_LLVM23_PREFIX=/opt/homebrew/opt/llvm
test -x "$TESSERA_LLVM23_PREFIX/bin/llvm-config"
test -d "$TESSERA_LLVM23_PREFIX/lib/cmake/mlir"
export PATH="$TESSERA_LLVM23_PREFIX/bin:$PATH"
export CMAKE_PREFIX_PATH="$TESSERA_LLVM23_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

"$TESSERA_LLVM23_PREFIX/bin/llvm-config" --version
"$TESSERA_LLVM23_PREFIX/bin/mlir-opt" --version
"$TESSERA_LLVM23_PREFIX/bin/mlir-tblgen" --version
```

All three version commands must begin with `23.`. If either path check fails,
stop rather than falling back to AppleClang's system libraries. To recreate a
dedicated **assertions-ON** toolchain (for falsifying MLIR promise/contract
claims — none is resident anywhere in the fleet since 2026-08-28), install the
Xcode Command Line Tools first, then build it into a distinct prefix and point
`TESSERA_LLVM23_PREFIX` at it:

```bash
xcode-select --install                    # omit if already installed
brew update
brew install cmake ninja lit
git clone --depth 1 --branch release/23.x https://github.com/llvm/llvm-project.git /private/tmp/llvm-project-23
cmake -S /private/tmp/llvm-project-23/llvm -B /private/tmp/llvm-23-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/homebrew/llvm-23-asserts \
  -DLLVM_ENABLE_PROJECTS='mlir;clang;lld' \
  -DLLVM_TARGETS_TO_BUILD='AArch64;AMDGPU;NVPTX;X86' \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_LINK_LLVM_DYLIB=ON
cmake --build /private/tmp/llvm-23-build --target install --parallel 8

export TESSERA_LLVM23_PREFIX=/opt/homebrew/llvm-23-asserts
export PATH="$TESSERA_LLVM23_PREFIX/bin:$PATH"
export CMAKE_PREFIX_PATH="$TESSERA_LLVM23_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
"$(brew --prefix lit)/bin/lit" --version
```

Do not use AppleClang's system LLVM libraries or mix the stable LLVM 23 keg
with this LLVM/MLIR 23 prefix. Record the upstream commit plus
`LLVM_ENABLE_RTTI=ON` in the build evidence.

For compiler artifacts, build the Apple backend and portable MLIR tools:

```bash
export TESSERA_LLVM23_PREFIX=/opt/homebrew/opt/llvm
cmake -S . -B build-apple -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$TESSERA_LLVM23_PREFIX/bin/clang" \
  -DCMAKE_CXX_COMPILER="$TESSERA_LLVM23_PREFIX/bin/clang++" \
  -DLLVM_DIR="$TESSERA_LLVM23_PREFIX/lib/cmake/llvm" \
  -DMLIR_DIR="$TESSERA_LLVM23_PREFIX/lib/cmake/mlir" \
  -DLLVM_EXTERNAL_LIT="$(brew --prefix lit)/bin/lit" \
  -DTESSERA_BUILD_APPLE_BACKEND=ON \
  -DTESSERA_BUILD_EXAMPLES=ON
cmake --build build-apple --target tessera-opt tessera-translate-mlir \
  TesseraAppleRuntime
export TESSERA_OPT="$PWD/build-apple/tools/tessera-opt/tessera-opt"
export PYTHONPATH="$PWD/python:$PWD"
```

Use the actual Ninja output path if the local LLVM/MLIR build lays out
`tessera-opt` differently. Build or load one fresh Apple runtime image for the
device lane; duplicate or stale dylibs invalidate symbol and placement proof.

The 2026-07-16 shared compiler migration raises the project floor to matched
LLVM/MLIR 23 and removes the obsolete Apple dialect property switch. The
portable Apple sources are assessed by the shared-source migration, but the
current WSL host cannot build or execute the Darwin/Metal runtime; Apple
LLVM/MLIR 23 build and exact-device parity are **follow-up required** on the
named Apple hosts.

## Ordered work

This is a live queue, not a historical checklist. `closed` means the item's
stated gate is met; `landing` means the principal implementation and evidence
landed but a deliberately narrower follow-up remains; `active` is the next
implementation/proof work; `blocked` names an external prerequisite.

| Order | ID | Status | Current state and next action |
|---:|---|---|---|
| 1 | APPLE-CALIB-1 | **active — cache semantics + exact-device analysis** | Define an evidence-backed Apple SLC/traffic interpretation, then correlate T1 with the committed Apple7 GEMM corpus. Keep the retired bank-conflict metric not applicable and do not claim non-GEMM coverage from T1 v1. |
| 2 | APPLE-RASTER-1 | **closed — row-major retained** | Retired the unconsumed MLX `swizzle_log` helper; the MSL Tile emitter consumes shared raster order/group under the unchanged 2-D launch ABI. Two 15-sample Apple7 warm pairs prove native correctness but mixed latency and unavailable counters, so grouped-M 2 is carried rather than selected. |
| 3 | APPLE-AOT-2 | **landing** | Runner registration and pointwise/reduction AOT coverage are landed. Harden the artifact/deferred contract, complete the remaining runtime families, then land cache maturity before the shared arbiter ships. APPLE-AOT-1/3/4 are completed evidence under the same sync key. |
| 4 | APPLE-TEST-1 | **closed** | The centralized hardware boundary collects 976 of 15,374 unit nodes, the structural scan finds zero inline Apple capability gates, and portable marker/provenance ratchets reject classification drift. |
| 5 | APPLE-CI-2 | **closed** | The host-free compiler ownership gate is executable and green for the declared Apple capability set, and now validates the exact LLVM/MLIR runner-utils path for every CMake cache type. |
| 6 | APPLE-TEST-2 | **closed** | Fresh-runtime correctness (**850/850**), fallback-injection negatives, ordering/stress, and the serial measured lane are complete. |
| 7 | APPLE-REG-1 | **closed** | ABI/target-map/exact-device/Tile drift gates are registered and passing. |
| 8 | APPLE-TILE-1 | **closed** | The selected f16/bf16 simdgroup fragment and its two-run corpus meet the completion gate. MPS retaining every measured end-to-end row is the valid production decision. |
| 9 | APPLE-GEMM-1 | **closed** | The Apple7 paired ledger has stable decisions for every measured row: three promotions and nineteen incumbent retentions. New devices/routes require a new corpus. |
| 10 | APPLE-EPILOGUE-1 | **closed** | Supported f32/f16/bf16 fusions have native-oracle/resource proof and stable end-to-end selection; MPSGraph now has an explicitly labeled Metal 4 whole-graph envelope, pending a fresh two-run device-domain corpus. |
| 11 | APPLE-ATTN-FWD-1 | **closed** | Native forward variants, resident/cooperative candidates, full stated corpus, two-run route ledger, and timing-domain selection are complete. Do not reopen it for backward work. |
| 12 | APPLE-ATTN-BWD-1 | **closed** | Native f32/f16/bf16 MHA/GQA/MQA serial, atomic, and split-reduce routes share one oracle and explicit workspace/determinism policy. The stable two-run Apple7 ledger selects end-to-end routes per exact row and retains serial for every device-domain row. |
| 13 | APPLE-PAGED-KV-1 | **closed** | Direct resident page-table MLA attention and the staged peer share a non-identity oracle, causal/window boundary proof, transactional exhaustion/leak telemetry, and a paired two-domain Apple7 corpus. The legacy corpus records direct wins; strict production ingestion retains staged until those rows are re-recorded with the v2 context envelope. |
| 14 | APPLE-REPLAY-1 | **closed** | Resident inputs, ordered asynchronous ring submissions, native deterministic checkpoint folding plus same-command-buffer ring clearing, forced flush/rollback/partial-rejection ordering, backpressure/cleanup stress, and paired selector evidence are complete. Unstable device-domain evidence retains the fused-block incumbent. |
| 15 | APPLE-RETUNE-1 | **active** | Fresh Apple7 f32 corpus has 16 selector-admissible decisions and eight explicit partial-domain negatives; the separate two-run f16/bf16 MoE corpus adds eight `single_fused_lowp` retain decisions with complete command-buffer proof. Grouped-SwiGLU logical byte/bandwidth accounting is retained. The remaining low-precision route gap is a fused grouped-GEMM package; complete-route ABIs remain required for composed/mapped device intervals. |
| 16 | APPLE-ROUTE-1 | **active** | Strict v2 sealing now binds producer context and source-report digests and retains only explicit negative rows outside selector decisions. The paired corpus and every legacy runtime-route owner (GEMM/softmax, forward/backward attention, epilogue) have fresh sibling v2 evidence; schema-v1 files are inventories only. Package subgraphs remain a separate namespace. |
| 17 | APPLE-DTYPE-1 | **blocked — SDK** | FP8/FP4/MX native execution awaits the public macOS 27 Metal tensor path. Keep older-host int4/int8/f16/bf16 regression coverage. |
| 18 | APPLE-CI-1 | **closed** | The local Metal 4 release gate serializes the physical Mac without registering a GitHub runner, builds fresh LLVM/MLIR 23 compiler/JIT/runtime artifacts, records power/thermal/GPU-contention availability, rejects incomplete or skipped evidence, runs correctness twice, and seals paired device/end-to-end evidence. The retained `docs/audit/evidence/apple/metal4/20260718-b1ee875/` packet proves two clean 11-test Apple7 runs under Xcode 26.6, two 8-row route reports with four Metal 4 rows each, and an 8-decision two-domain ledger against commit `b1ee87591ec701dd06a156cad8449f6498ae0891`. Portable CI validates its hashes and contents. Metal 3 remains non-blocking compatibility coverage. |
| 19 | APPLE-E2E-1 | **closed / bounded Level C** | Static, exact-device-oracle-backed GPU ABI families are closed: rank-2 f32 softmax/transpose; f32/f16/bf16 rank-3 batched-GEMM; strict and side-tensor PPO; EBM energy/Langevin/refinement/partition; cl30 Clifford geometric product; and static/batched Cholesky, Cholesky-solve, and triangular-solve. Every family has package, owned-fresh-dylib execute/compare, and repeated-launch cleanup proof on the exact device. Composite/package-subgraph, dynamic, stateful, unsupported, and multi-result GPU contracts, plus fleet/second-device proof, are retained follow-on work in APPLE-NATIVE-E2E-2. Metal-owned schedules, placement policy, and selectors are unchanged. NVIDIA and ROCm are not applicable: this changed no shared IR, ABI, schedule, or evidence claim. |
| 20 | APPLE-CPU-E2E-1 | **closed / bounded Level C** | Static f32 rank-2 matmul/gemm and rank-3 BMM; single-result Cholesky, triangular-solve, and Cholesky-solve; and tuple-output LU/QR/SVD Accelerate/LAPACK descriptors have exact-host execute/compare and repeated-launch cleanup proof through the owned rebuilt dylib. Dynamic shapes, other dtypes, and non-linalg contracts remain retained/reference and belong to APPLE-NATIVE-E2E-2. |
| 21 | APPLE-NATIVE-E2E-2 | **landing / fleet packets sealed; second-device proof hardware-gated** | The bounded local descriptor program is complete on the exact Apple7 Metal-4 device. Existing CPU f16/bf16 matmul/gemm and descriptor-state-registered, exact-host-replayed f32 row-softmax are complete. GPU static/dynamic f32/f16/bf16 GELU descriptors carry explicit fp32-accumulation provenance, two-byte low-precision bindings, storage-rounding oracles, and rank/dtype/result-shape/scalar rejection ratchets. Ordered reduced SVD and ReplaySSM lifecycle packages are joined by dynamic rank-1 i32 popcount (`Elements`), rank-2 f32 last-axis count-nonzero (`Outer/AxisExtent`), and rank-2 f32 row-softmax (`Rows/Columns`). Ordered top-k now uses a dedicated status-returning Metal ABI with numeric-descending, NaN-last, lower-index-tie semantics, ordered `(values,indices)` output bindings, `Rows/Columns/K` verification, and exact-device execute/compare/replay/rejection proof. The Metal-4 composite package remains sealed by tree digest, reflected positional externals, a private intermediates heap, and replay-safe cache identity. Its paired strict-v2 `package_subgraph` evidence comprises two independent 50-repetition by 5-trial reports: package promoted at `64x64x64`, live retained at `256x256x256`, and device-domain rows explicitly ineligible because the lane exposes only comparable complete-call timing. The local CPU ABI audit still finds no further owned static non-linalg candidate; speculative wrappers remain forbidden. **Both Apple fleet packets are now recorded and sealed (2026-07-27).** `benchmarks/e2e_spine/record_apple_packet.py --lane {apple_gpu,apple_cpu}` produces the two independent identities the registry keys on — `apple_gpu`/`apple7` and `apple_cpu`/`apple_m1_max` — into `docs/audit/evidence/e2e_spine/apple_gpu/apple7/` and `.../apple_cpu/apple_m1_max/`. Evidence never transfers between them, so each lane seals its own `report`/`resources`/`manifest` triple. Both validate independently with **max absolute error 0.0** against the shared oracle: `apple_gpu` proves `matmul` + `softmax` (Level A/B/C, 4 benchmark rows), `apple_cpu` proves `matmul` (2 rows). `matmul` on the GPU is proven as a batch-1 BMM because the Apple GPU GEMM contract is batched-only; the route name `apple_gpu_bmm_f32_batch1` keeps that visible rather than implying a 2-D kernel. `apple_cpu` scope is `matmul` alone because the registration declares only `("matmul","linalg")` — the lane executes softmax, but claiming it would be an undeclared family. **Both lanes report `kernel_wall`, not `device_event`**, because the MPSGraph matmul route has no device timer while the MSL softmax route does (row 32 — the first diagnosis of this was wrong and is corrected there). **GPU placement is proven independently of the oracle**: the void `..._f32` ABIs fall through to a numerically-identical CPU reference, so the recorder requires a positive `..._f32_status` result at both the fixture and timing shapes before sealing (row 33). Host identity is pinned per lane — CPU brand for `apple_cpu`, live Metal GPU family for `apple_gpu`, so an M3/M4 host cannot seal an `apple7` packet. `linalg`/`ppo`/`ebm`/`clifford` (GPU) and `linalg` (CPU) remain `packet_pending` — "Family absent from active packet" — and need corpus fixtures before they can be claimed. Only *second-device* proof (a non-Apple7 family) remains genuinely hardware-gated.  Any additional retained/reference family requires a separately owned future ABI item with a shape/dtype contract and exact-device oracle. NVIDIA and ROCm are not applicable: these Apple-private descriptor metadata and proof ratchets change no shared dtype, Graph-IR spelling, sibling ABI, or schedule. |

### APPLE-PIPE-1 landing evidence (2026-07-26)

`src/compiler/codegen/Tessera_Apple_Backend/lib/Target/Apple/Lowering/
ThreadgroupPipelineToApple.cpp` is Apple's architecture-owned consumer of the
shared Tile physical-allocation and staged-pipeline SSA contract — the Metal
sibling of `ROCMWaveLdsPipeline.cpp`. It follows the AMD precedent of claiming
the shared ops in place with architecture-owned physical decisions rather than
translating them into a private handle type.

What Apple now decides, and where the decision is enforced:

- **Placement.** Each `smem` `tile.alloc` gets `tessera_apple.address_space`,
  `_threadgroup_offset` (16-byte aligned, packed in declaration order),
  `_threadgroup_bytes`, and `_threadgroup_capacity_bytes`; the enclosing
  function carries `tessera_apple.threadgroup_arena_bytes`. A function with no
  Tile allocation carries no arena attribute — the pass states a demand it
  measured, never a zero it assumed.
- **Buffering mode.** A depth-2 ring is named `ping_pong` and depth-1 `single`,
  so the emitter reads the mode instead of re-deriving it from `depth`.
- **Capability boundary.** Nine registered diagnostics
  (`APPLE_THREADGROUP_*`, `APPLE_STAGE_*`, `APPLE_TILE_UNSUPPORTED_VOCABULARY`)
  reject `tmem`/`gmem` placement, over-capacity arenas, rings deeper than the
  Metal ping-pong pair, name-based `#tile.buffer_ref` identity, and the
  NVIDIA-only TMA / mbarrier / TMEM / TCGen05 vocabulary. Per Decision #21
  each names the operation and the target; none silently no-ops. Silently
  narrowing a depth-4 ring to 2 was explicitly rejected as a design option —
  it would change the program's synchronization structure behind the author.

Evidence, all host-free on this Mac against the LLVM/MLIR 23 `build-apple`
compiler:

- `tests/tessera-ir/phase8/apple_threadgroup_pipeline.mlir` — SSA identity end
  to end, checked twice: driven directly *and* through the real
  `tessera-lower-to-apple_gpu` pipeline, so placement is decided once and
  survives to Target IR.
- `tests/tessera-ir/phase8/apple_threadgroup_pipeline_invalid.mlir` — eight
  split-input rejection cases. The harness was sanity-checked by mutating one
  expected code and confirming the fixture then fails.
- `tests/unit/test_apple_threadgroup_pipeline.py` (5 tests,
  `compiler_tool` + `compiler_apple`) — the cross-owner drift gate binding the
  C++ placer to `msl_gemm_emit.materialize_apple_simdgroup_tile_msl`: same
  capacity as the `AppleGPUTargetProfile`, same ping-pong rule as the emitted
  `As[2]`/`Bs[2]` staging, and an arena that reproduces the artifact's
  `total_threadgroup_bytes` at the expected offsets. Both owners reject the
  same over-capacity tile, each with its own code.
- Lanes: phase8 lit **72 passed / 2 unsupported / 0 failed**; the Apple
  host-free compiler ownership gate selects the new tests and passes
  (**6 passed, 46 foreign-compiler skips**); diagnostic-registry and
  pass-metadata drift gates green; `mypy` clean.

**PR #467 review fixes (2026-07-26).** Three real defects were found in review
and are now regression-locked in `tests/unit/test_apple_threadgroup_pipeline.py`:

1. **Loop-carried state was reported unrooted.** A ring threaded by `scf.for`
   reaches its advance as a *region iter_arg*, not the `pipeline_init` result —
   which is exactly how the canonical GEMM and streaming attention emit it. The
   membership test rejected that with `APPLE_STAGE_UNROOTED_ADVANCE`, so the
   pass refused the very shared contract it exists to consume. Block arguments
   are now resolved back to the matching loop init operand.
2. **The MMA gate could be bypassed.** It read only `numeric_policy.storage`,
   but the op verifier requires only `mma`, so an int4/FP8 descriptor with no
   policy — or with a laundering fp16 one — passed. The descriptor's declared
   A/B/accumulator types are now authoritative, and an unreadable descriptor is
   treated as "cannot prove a route exists" rather than as permission.
3. **Advances contradicted their own ring.** Every advance was stamped
   `ping_pong`, so a depth-1 pipeline handed the emitter two different physical
   schedules. Advances now carry the rooted initializer's mode.

NVIDIA and ROCm are not applicable to the Apple pass, attributes, or
diagnostics. The one shared-ground change is the `LseSaveOp` `Pure` correction
recorded below.

### APPLE-TILE-2 landing evidence (2026-07-26)

`CanonicalGemmToAppleGPU.cpp` recognizes the shared canonical reduction and
re-forms it as one Apple dispatch. The architectural choice, stated so it is
reviewable rather than implicit: **the canonical loop is a semantic contract,
not a schedule Metal must reproduce statement by statement.** It says "reduce
over K in FP32, zero-padding the ragged tail"; the Apple steel GEMM already
implements exactly that reduction with its own cooperative staging and
edge-masked stores. Emitting a literal three-loop MSL nest would be slower and
no more correct. NVIDIA and ROCm consume the same loop differently — per-K-step
Tensor Core / MFMA fragment issue — which is the intended per-target freedom.

- **Recognition is guarded, not greedy.** The claim requires the
  `tessera.canonical_k_step` marker, the three-deep `scf.for` structure, a K
  loop carrying `!tile.pipeline_state`, and two loop-invariant sliced operands.
  A user-written loop containing a matmul is left untouched.
- **Four registered diagnostics** (`APPLE_CANONICAL_GEMM_{UNRECOGNIZED,
  SHAPE,DTYPE,ACCUM}_*`) own the envelope. The f32 rejection is the one that
  keeps the incumbent honest: f32 GEMM stays on Accelerate/MPS instead of being
  quietly rerouted to a `simdgroup_matrix` path that has no f32 operand form.
- **Logical tile vs physical block.** The loop's `tessera.tile_*` values are
  logical steps the shared tiler clamps to the extent (a `13x16x11` GEMM yields
  steps of 13/11/16). An Apple threadgroup block must be a multiple of the
  8x8x8 fragment, so `msl_gemm_emit.apple_block_for_canonical_tile` rounds *up*
  — which is precisely what the contract's `ragged_zero_pad` guarantee licenses.
  This distinction is now explicit instead of an accidental shape agreement.

Evidence:

- `tests/tessera-ir/phase8/apple_canonical_gemm.mlir` — a plain Graph-IR matmul
  driven through the **shared** `--tessera-tiling` and then the Apple pass in
  one run, so the fixture consumes the real canonical form rather than a
  hand-written imitation. Checks the nest is consumed (`CHECK-NOT: scf.for`),
  the dispatch is singular, and fp16/bf16 select storage-matched symbols.
- `tests/unit/test_apple_canonical_gemm.py` — 7 tests. Four host-free
  (descriptor contents, bf16 symbol, f32 rejection, no-misclaim), and three
  `hardware_apple_gpu` **execute-and-compare rows that ran on this Apple7
  Metal device**: `16x16x16`, `32x16x32`, and the ragged `13x16x11`, each
  driven by the compiler-produced descriptor, each asserting `native is True`
  with a positive device time and matching the fp32 NumPy oracle.
- Lanes: phase8 lit **74 passed / 2 unsupported / 0 failed**; Apple unit sweep
  **3068 passed / 10 skipped / 1 failed**, the single failure being the
  pre-existing `test_strict_retune_ledger_admits_on_its_exact_live_apple_host`
  (see the note below); registry/metadata drift gates and `mypy` green.

**Pre-existing failure worth its own fix (not caused by this work).**
`test_strict_retune_ledger_admits_on_its_exact_live_apple_host` fails with
`context_mismatch:runtime_fingerprint`, and reproduces on a clean `HEAD` with
all of this work stashed. `apple_route_selector._runtime_source_fingerprint`
hashes `apple_gpu_runtime.mm`, whose current content no longer matches the
digest the committed strict-v2 retune ledger was sealed against. The practical
consequence is that the strict ledger currently admits **no** decisions on this
host, so APPLE-RETUNE-1's selector evidence is inert until the corpus is
re-recorded and re-sealed against the current runtime source.

### APPLE-ATTN-STREAM-1: LSE checkpoint migration (updated 2026-07-27)

The 2026-07-26 investigation below is retained as historical diagnosis and is
**superseded**. Cross-backend sync
`LSE-CHECKPOINT-CONTRACT-2026-07-27` removed destination-less emission and
replaced the declarations with explicit memref source/destination, SSA row
offset, identity, global-memory space, lifetime scope, cache policy, and
`MemWrite`/`MemRead` effects. The Apple pass no longer erases `lse.save`; any
live LSE remains a real unsupported ABI request and is rejected. Inference-only
forward sees no save. ROCm measured and selected its own gfx1151 128+ policy;
that threshold and AMD schedule do not transfer. Apple follow-up is required
only if a Metal training package elects to persist LSE.

#### Historical diagnosis (superseded)

This item first landed **blocked**, and the investigation is worth keeping
because the blocker turned out to be a bug in the shared contract rather than a
capability gap.

**The apparent blocker.** The shared lowering always terminates the recurrence
with `tessera_attn.lse_accumulate` -> `tessera_attn.lse.save`. `LseSaveOp`
carried no `Pure` trait, so MLIR had to treat it as side-effecting, while
Apple's fused ABI (`tessera_apple_gpu_flash_attn_*`) returns the attention
output only. Re-forming would either leave the whole recurrence alive to
recompute the LSE — attention computed twice — or silently drop a checkpoint
backward appeared to depend on. The pass refused.

**What the code actually shows.** There is no checkpoint to protect:

- the emission site (`TileIRLoweringPass.cpp:374`) **discards the result**;
- the result type is scalar `f32`, not the per-row `[tile_q]` LSE it names;
- `LseLoadOp` takes **no operands**, so no SSA edge, symbol, or handle links a
  load to a save — a backward lowering could not express *which* save its load
  reads, even if one wanted to;
- the only `lse.load` in the tree is a v1.3 example fixture.

This is the same name-free global-state modeling the 2026-07-26 wave already
ruled against twice: `#tile.buffer_ref` became `!tile.buffer`, annotation-only
`#tile.pipeline_state` became threaded SSA, and `TilePipelineLegality` now
*rejects* the annotation-only form. The LSE pair is the next unmigrated
instance, not an unlucky edge case.

**No backend consumes it, and all three that have an attention backward chose
recompute** — for the same reason, since a saved LSE reintroduces the workspace
their determinism contracts exist to eliminate:

| Backend | Attention backward | LSE source |
|---|---|---|
| ROCm gfx1151 | `GenerateWMMAFlashAttnBwdKernel.cpp` | recomputes `L[q] = logsumexp_k(scale*QK^T)` in a `_pre` pass; header states the backward "needs nothing saved from the forward" |
| NVIDIA sm_120 | `sm120_attention_backward_kernel.mlir` | `workspace_bytes = 0`, `workspace_owner = "output_element"` |
| Apple Apple7 | `flash_attn_bwd_*` | `bwd_query_stats` recomputes m/l per query; ABI takes no LSE buffer |
| x86 AVX-512 | none (forward only) | n/a |

**Resolution — and a rejected first attempt.** Marking `LseSaveOp` `Pure` was
tried first and **backed out**. Testing showed it changes emitted IR on every
backend (`tests/tessera-ir/phase3/flash_attn_full.mlir` asserts the op's
presence), and it would leave a trap for whoever implements the real FA-2
checkpoint, since a store must be non-`Pure` with `MemWrite`. An earlier claim
here that the trait "changes nothing emitted on any target" was wrong.

What landed instead touches no shared declaration: the Apple pass erases only a
`lse.save` whose own result is unused, as part of re-forming the recurrence it
is already rewriting. Both the Apple fixture and `flash_attn_full.mlir` pass
with the shared op unchanged. A `retain_lse` flag was also considered and
rejected — it would gate an op that cannot perform its stated function.

The vocabulary is deliberately **kept**, and the save-versus-recompute question
is now owned by the backends with the memory systems to settle it:
[`NVIDIA-LSE-1`](../nvidia/todo.md) and [`ROCM-LSE-1`](../rocm/todo.md), with
the contract, the FA-2 design, and the preferred source-level fix documented in
[`../../compiler/LSE_CHECKPOINT_CONTRACT.md`](../../compiler/LSE_CHECKPOINT_CONTRACT.md).

The Apple consumer erases only a `lse.save` whose own result is unused; anything
genuinely reading the LSE still refuses with
`APPLE_STREAMING_ATTN_LSE_UNSUPPORTED`. It also erases the now-dead staging
(Q copy, ring init/advance), because a leftover depth-3 `tile.pipeline_init`
would otherwise fail APPLE-PIPE-1 for a schedule the program no longer has.

NVIDIA and ROCm review note: nothing in this slice changes a shared op,
verifier, or emitted lowering. The one shared-ground item is the *documented*
source-level fix — stop emitting a destination-less save — which is filed in
both queues rather than landed from here.

### Attention-backward and stateful-transport rows (opened 2026-07-27)

A second wave of shared contracts landed while rows 22-26 were in flight, all
clustered on attention backward and stateful transport. They arrived with
Apple follow-ups recorded in the sync notes above but no owning rows, which is
the same gap rows 22-26 were opened to close for the previous wave. Rows 27-31
own them.

Two of these are close to work Apple already has. APPLE-ATTN-BWD-1 is closed
with proven serial / atomic / split-reduce Metal routes, and the ReplaySSM
lifecycle is closed with session-private ring and ordering semantics — so rows
25 and 28 are mostly *contract adoption*: deciding whether the shared phase
loops and generalized resident schema describe what Apple already runs, and
saying so explicitly either way.

### Shared-Tile-contract consumer rows (opened 2026-07-26)

The 2026-07-25/26 core wave (`CORE-GEMM-KLOOP`, `CORE-STREAMING-ATTN`,
`ROCM-SSA-LDS-PIPELINE`, `PACKED-LEGALIZE-CAPABILITY`) landed shared Tile
contracts whose Apple follow-ups were recorded in the sync notes above but had
no owning row. These rows own them. The audited starting position, verified in
source rather than prose:

- The Apple GPU pipeline runs `createTilingPass(valueMode=true)`, which
  deliberately does **not** tile to `scf.for` (`TilingPass.cpp:952`), so the
  canonical M/N/K K-loop has no Apple consumer.
- `TileToApple` matches whole-tensor `tile.matmul` / `tile.gemm` /
  `tile.batched_gemm` by name (`TileToApple.cpp:677`).
- `FlashAttnToAppleGPU` rewrites `tessera.flash_attn` **directly from Graph
  IR** to a monolithic runtime ABI call (`FlashAttnToAppleGPU.cpp:77`); Apple
  attention never crosses the Tile layer.
- Nothing under the Apple backend references `!tile.buffer`,
  `!tile.async_token`, or `!tile.pipeline_state`; the only consumers are
  `ROCMWaveLdsPipeline.cpp` and `WarpSpecializationPass.cpp`.

Apple is therefore the only actively developed backend whose GEMM *and*
attention paths bypass the shared Tile contracts. That is defensible on
measured grounds — the committed ledgers retain MPS/Accelerate for every
measured row — but it must be a *declared* architecture decision with a
capability-rejection or consumer proof, not undeclared divergence.

| Order | ID | Status | Current state and next action |
|---:|---|---|---|
| 22 | APPLE-PIPE-1 | **closed (2026-07-31)** | The `tessera-apple-threadgroup-pipeline` pass consumes the shared SSA vocabulary: `!tile.buffer` allocations are placed 16-byte-aligned into a capacity-bounded per-function arena and `!tile.pipeline_state` rings are claimed as `ping_pong` / `single` Metal staging. Canonical GEMM now carries that physical decision forward as one `canonical_tile_ir` staging contract on its Apple descriptor: fragment-rounded tile dimensions, stage depth, staged-A/B bytes, edge scratch, arena total, and capacity. The runtime materializer consumes those bytes for the emitted MSL declarations and rejects any disagreement by name; it no longer substitutes its old `32x32x16` default for compiler-produced descriptors. Host compiler tests prove descriptor arithmetic and mismatch rejection; the existing Apple7 exact-device canonical-GEMM oracle now materializes from that descriptor contract. Sibling outcome: no shared Tile IR or sibling schedule changed. |
| 23 | APPLE-TILE-2 | **closed (2026-07-31, incumbent retained by strict-v2 evidence)** | `tessera-apple-canonical-gemm` recognizes the shared three-deep M/N/K nest and re-forms it as one `simdgroup_matrix` dispatch carrying the loop's tile decision, `accumulate = "fp32"`, and the `ragged_zero_pad` guarantee. Exact-device execute-and-compare passes on Apple7 Metal for aligned and ragged rows, driven by the compiler-produced descriptor. The Tile-owned producer now writes the shared strict-v2 source-report schema: two fresh Apple7 processes each interleave MPS with the exact f16/bf16 source-backed Tile ABI across `8x8x8`, `32x16x32`, ragged `127x63x129`, and `256x256x256`, retaining native placement, oracle correctness, owned-command-buffer timing, and resource/counter records. [`apple7_tile_strict_v2_route_ledger.json`](../../../../benchmarks/baselines/apple7_tile_strict_v2_route_ledger.json) seals those reports and admits all 16 exact shape/dtype/domain rows against the live context. Every row retains MPS under the paired stable-win rule, so the MPS/Accelerate incumbent selector is intentionally unchanged; the ledger makes that retention measured rather than defaulted. Sibling outcome: no shared Tile IR, ABI, or NVIDIA/ROCm/x86 schedule changed. |
| 24 | APPLE-ATTN-STREAM-1 | **landing** | `tessera-apple-streaming-attention` recognizes the shared KV-block recurrence and re-forms it as one Apple flash-attention dispatch, carrying `causal` / `logical_sk` / `window_left/right` / `kv_block` **read off `tessera_attn.boundary_mask`** instead of re-derived — the ownership fix this row exists for. It runs first in `tessera-lower-to-apple_gpu`, ahead of APPLE-PIPE-1, because the shared depth-3 KV ring must be re-formed before the threadgroup pass judges a schedule the program is about to stop having. Unblocked without changing the shared `LseSaveOp` declaration (see below). **Narrower follow-up:** the descriptor targets the same ABI family as the incumbent, so numerical parity is proven structurally plus an on-device oracle check — not yet a full APPLE-ATTN-FWD-1 corpus re-run, and no selector changed. |
| 25 | APPLE-DTYPE-1-REJECT | **closed** | The macOS-27 SDK gate is enforced, not incidental. `tests/tessera-ir/phase8/apple_lowprecision_capability_gate.mlir` runs the same module through `--tessera-storage-legalize` twice: the `apple_gpu` target stamps no `tessera.storage_packed`/`_container` on either a block-scaled NVFP4 decode or a packed int4 contraction, while the `nvidia_sm120` contrast run stamps both — so the negative cannot pass merely because the pass did nothing. A block-scaled or otherwise unrouted cooperative-matrix descriptor is separately rejected with `APPLE_MMA_STORAGE_UNSUPPORTED`, and `tests/unit/test_apple_threadgroup_pipeline.py` binds that gate to `select_apple_simdgroup_fragment`: fp16/bf16 accepted by both owners, nvfp4/fp4/fp6/fp8/int4 refused by both. APPLE-DTYPE-1 itself stays **blocked — SDK**; this row proves the block, it does not lift it. |
| 26 | APPLE-COUNTER-1 | **landing** | `compiler/apple_counter_evidence.py` maps Metal telemetry onto the shared autotune-evidence fields with an explicit four-state reason on every field: `measured`, `not_measured` (device can, this run did not), `unsupported_by_device` (this GPU family cannot), `no_public_api` (Metal exposes no query — register count, scratch bytes, spill count, achieved occupancy). Supplying a value the capability bits do not support raises rather than silently downgrading, so a corpus cannot claim evidence the device cannot produce. Bit positions are drift-gated against the runtime's own documented matrix. **Narrower follow-up:** the benchmark writers do not yet emit these fields into a committed corpus, so this is the vocabulary and its guards, not a recorded two-run corpus. |
| 27 | APPLE-ATTN-STREAM-2 | **closed (2026-07-31, narrow f32 GQA contract)** | `tessera-apple-streaming-attention` now recognizes the marked rank-2 KV loop only when it is enclosed by the shared `query_head` then `batch` distributions, and replaces the **batch-loop result** with one `flash_attn_gqa` descriptor — never the rank-2 inner slice. The descriptor carries static `B/Hq/Hkv/Sq/Sk/D`, GQA group size, `scale`, causal/logical-KV semantics, and the shared KV block, and binds `tessera_apple_gpu_flash_attn_gqa_f32` through the Apple value-artifact executor. Structural tests prove the enclosing loops/staging disappear; an Apple7 exact-device repeat-KV oracle proves that descriptor ABI. Scope is deliberately static f32, no live LSE, no dropout, and causal/scale only; f16/bf16 output policy and window/bias/softcap coverage remain owned by APPLE-ATTN-MODIFIERS-1. Sibling outcome: shared IR was unchanged, so ROCm/NVIDIA/x86 schedules were not altered or reclassified. |
| 28 | APPLE-ATTN-BWD-2 | **closed (2026-08-16)** | Consume the shared tensor-valued attention **backward** phase loops. `ROCM-ATTENTION-SHARED-BACKWARD-CONSUMER-2026-07-26` made gfx1151 the first direct physical consumer; Apple must validate the same dQ / split-dK/dV / fixed-reduction contract and map it to a Metal-owned package. APPLE-ATTN-BWD-1 already owns proven serial / atomic / split-reduce Metal routes, so this is contract adoption, not new kernels — the question is whether the shared phase loops describe the schedules Apple already runs. The AMD WMMA schedule, five-entry HSACO, HIP workspace, and host-wall timing do not transfer. |
| 29 | APPLE-ATTN-BWD-3 | **closed (2026-08-16)** | `CORE-ATTENTION-BACKWARD-CONTRACT-2026-07-26` adds verified shared backward contracts; confirm Apple's backward satisfies them or record the divergence. The shared LSE checkpoint contract is now real and conditional; Apple retains recompute until an exact Metal package and benchmark justify a saved checkpoint. |
| 30 | APPLE-ATTN-MODIFIERS-1 | **active — narrowed to windows + named diagnostics** | `CORE-ATTENTION-TENSOR-LOOPS-MODIFIERS-2026-07-26` lands shared tensor-valued attention loop modifiers. Apple owns validating that its causal / sliding-window / softcap / bias / GQA-MQA envelope still expresses every admitted modifier after the shared change, and rejecting the rest by name rather than silently narrowing. |
| 31 | APPLE-STATEFUL-TRANSPORT-1 | **active — unchanged, no Apple consumer** | `SSA-STATEFUL-TRANSPORT-2026-07-26` retired the `#tile.buffer_ref` compatibility reader and generalized the proven Apple ReplaySSM lifecycle schema to target-keyed resident ABIs, adding MoE launch-workspace ownership and optional rank/device topology binding. Apple keeps its session-private ring, flush/rollback, ordered submission, and drain-before-release semantics; the open item is Metal threadgroup scheduling against the generalized schema. APPLE-PIPE-1 already rejects name-based `#tile.buffer_ref` identity, so Apple is aligned with the retirement. |
| 32 | APPLE-DEVICE-EVENT-1 | **closed (2026-08-16)** | *Corrected 2026-07-27: the first diagnosis of this row was wrong.* The device timer is **not** broken on the descriptor lane. `tessera_apple_gpu_last_dispatch_device_time_ns()` reads `-1` only because dispatch telemetry is **opt-in and off by default**; after `tessera_apple_gpu_dispatch_telemetry_set_enabled(1)` the MSL softmax route reports a real `device_time_ns` with `timing_source=1` and a full threadgroup/execution-width resource record. The genuine gap is narrower: the **matmul route runs through MPSGraph**, which populates neither the command-buffer device timer nor the MSL dispatch record. Because `required_timing_domains` is report-wide and every family in scope must supply both domains, one family without a device timer forces the whole `apple_gpu` packet onto `kernel_wall`. Closing this means giving the MPSGraph route a device timer (or moving matmul to an MSL/`simdgroup_matrix` route that already has one), then re-recording with `required_timing_domains = ["device_event", "end_to_end"]`. Independent of `CAP_DISPATCH_BOUNDARY_SAMPLING` (bit 4), which this M1 Max does not report — the command-buffer interval needs no counter sampling. **Closed 2026-08-16:** `mpsg_run_bmm` now encodes into an owned `MPSCommandBuffer` under the shared `MPSGraphTimingBracket`, the recorder probes device-interval availability per family instead of hard-coding it, and the re-recorded packet seals `required_timing_domains = ["device_event", "end_to_end"]`. |
| 33 | APPLE-PLACEMENT-ABI-1 | **landed (2026-07-27), extension open** | `tessera_apple_gpu_softmax_f32` and `tessera_apple_gpu_bmm_f32` are `void` ABIs that fall through to a numerically-identical CPU reference when Metal is unavailable or a pipeline/allocation/command fails. Nothing in a numerical proof distinguishes the two paths, so an oracle-matching fixture could have been sealed as Level-C GPU evidence while running on the host. Both now have status-bearing twins (`..._f32_status`) following the documented TILE-1 precedent at `tessera_apple_gpu_mps_matmul_f16_status`, and the fleet recorder refuses to seal a fixture or benchmark whose placement is not positively proven — at the fixture shape *and* the timing shape, since a dispatch can succeed at one and fail at the other. The MSL dispatch record is captured where the route populates it (softmax) and reported absent, never inferred as CPU, where it does not (MPSGraph matmul). **Open:** the other ~130 `void` Apple GPU entry points have the same latent hazard; any that a benchmark or packet records must gain a status twin before its result is admitted as GPU evidence. |

## Canonical validation lanes

After APPLE-TEST-1 establishes complete marker coverage, the Apple host should
run these as independent commands:

```bash
# Host-free compiler, selector, validation, rejection, and fallback contracts.
python3 -m pytest tests/unit -q \
  -m "not hardware_apple_gpu and not performance"

# Apple compiler artifacts; this lane does not claim device execution. It
# reports foreign compiler proofs as explicit per-platform skips.
python3 scripts/run_apple_host_free_compiler_tests.py \
  --build-dir build-apple \
  --tool build-apple/tools/tessera-opt/tessera-opt

# Native Metal correctness, twice from the same fresh build/runtime image.
python3 -m pytest tests/unit -q \
  -m "hardware_apple_gpu and not performance" --durations=100 \
  --junitxml=/tmp/apple-device-correctness.xml

# Measured lane: serial execution only.
python3 -m pytest tests/unit -q -n 0 \
  -m "hardware_apple_gpu and performance" --durations=0 \
  --junitxml=/tmp/apple-performance.xml

# Metal 4 promotion runs locally on the named Mac, never through a registered
# GitHub runner. Push the sealed packet into the coordinating PR; portable CI
# rejects zero selected tests, skips, hash drift, unknown GPU families,
# reference rows, missing device intervals, and incomplete two-domain ledgers.
bash scripts/run_apple_metal4_release_gate.sh \
  --publish-dir docs/audit/evidence/apple/metal4/<run-id>

```

The first focused parity and characterization loop is:

```bash
python3 -m pytest -q \
  tests/unit/test_apple_gemm_schedules.py \
  tests/unit/test_apple_sdpa_schedules.py \
  tests/unit/test_apple_gpu_metal4.py \
  tests/unit/test_apple_gpu_mpsgraph_lane.py \
  tests/unit/test_apple_gpu_resident_block_paged.py \
  tests/unit/test_ssm_apple_gpu_fused.py

python3 benchmarks/apple_gpu/benchmark_route_characterization.py \
  --matmul-shapes 64x64x64 128x256x64 257x129x65 256x256x256 \
  --softmax-shapes 64x64 128x257 256x256 \
  --reps 30 --output /tmp/apple-routes.json

python3 benchmarks/apple_gpu/benchmark_ssm_replay.py \
  --shapes 1x128x128 1x256x128 4x128x64 \
  --tokens 512 --capacity 16 --reps 20 \
  --output /tmp/apple-ssm-replay.json

python3 benchmarks/apple_gpu/record_hot_path_baseline.py --reps 20 --margin 2.0
```

Focused tests are edit-loop aids, not substitutes for the full marker lanes.
Files under `/tmp` are review artifacts only. Update a committed baseline or
route corpus only after two stable runs, explicit native-placement review, and
before/after resource inspection.

## Failure and benchmark evidence contract

For each failure or candidate record retain:

- test node, proof layer, Apple GPU family, macOS/SDK/compiler, dtype, shape,
  seed, selected route, and observed placement;
- fresh-runtime identity and whether the result reproduces alone, serially, and
  on the second clean run;
- named diagnostic or runtime error kind, compiler output, and Metal validation
  messages;
- maximum absolute/relative error, first failing index, non-finite policy, and
  the exact shared oracle;
- GPU/kernel time versus end-to-end time, warmup/repetition policy, cold compile
  or package-authoring cost, and command-buffer/dispatch count;
- residency and traffic bytes, threadgroup memory, occupancy/concurrency proxy,
  compiler statistics, and spill evidence available from the Metal toolchain;
- disposition: product defect, test-state defect, stale route/baseline, duplicate
  proof, unsupported capability, or exact external environment blocker.

Do not widen numerical tolerances or latency caps solely to turn the lane green.
Derive numerical policy from storage/accumulation semantics and performance
policy from stable repeated-median evidence.

## Next update

Cross-backend sync `NVFP4-TILE-SCALES-2026-07-16`: shared typed Tile IR now
permits logical `scale_a`/`scale_b` fragments only on NVFP4 MMA descriptors.
Apple has no enabled NVFP4 cooperative-matrix route, so this is follow-up
required at capability rejection only; no NVIDIA nibble, lane, scale-selector,
or OMMA mapping applies to Metal.

Cross-backend sync `EPILOGUE-CONTRACT-2026-07-16`: the shared `FusedRegion`
oracle now names bias/activation/residual order and emits registered
`E_FUSED_EPILOGUE_*` rejection diagnostics. Apple retains its architecture-owned
MSL/Metal 4 schedules. NVIDIA validates the complete 43-case supported
execution matrix; Apple independently validated its supported semantic order,
dtype matrix, residual guards, and diagnostics on the exact Metal host before
closing APPLE-EPILOGUE-1. The schedules and exact-device claims remain
architecture-specific.

Cross-backend sync `PR420-REVIEW-2026-07-17`: not applicable to Apple compiler
or runtime behavior. The scale-origin repair and canonical `fp16` alias are
confined to the SM120 NVIDIA fragment materializer/selector, and the bootstrap
ordering repair is confined to Ubuntu apt.llvm.org setup. No Apple IR, Metal
layout, dtype support, ABI, schedule, or exact-device claim changes.

Cross-backend sync `NVIDIA-SM120-LOWP-2026-07-18`: not applicable to Apple
runtime execution. The change adds a CUDA-owned packed NVFP4 ABI, SM120
HMMA/QMMA/OMMA kernels, CUDA-event evidence, and device-keyed NVIDIA selector
rows. It changes no portable dtype spelling, ScaleLayout, epilogue order, or
autotune schema. Apple remains SDK-gated for FP8/FP4 tensor execution and does
not inherit CUDA fragments, resource values, timings, or promotions.

Cross-backend sync `E2E-SPINE-2026-07-18`: Apple participates in the shared
native-image and launch-descriptor contract through **APPLE-E2E-1** and
**APPLE-CPU-E2E-1**. The shared work may select and package an existing typed
Apple pipeline, but it does not transfer CUDA/ROCm schedules, change Metal
placement, promote a route, or convert host-free compilation into exact-device
proof. Existing runtime and artifact routes remain available until their
canonical replacements meet all four proof layers on the named Apple host.
The behavior-neutral E2E-SPINE-0 foundation is complete: Apple CPU/GPU exact
targets now have total declared-pipeline ownership and truthful partial-B/
absent-C inventory rows; Apple runtime selection is unchanged. E2E-SPINE-1 is
also complete: Apple will consume the shared image/descriptor identity,
bindings, generic geometry, workspace, ordering, and diagnostics, while Metal
threadgroup schedules and placement remain Apple-owned. No Apple route or
exact-device status changes until APPLE-E2E-1.
E2E-SPINE-2 completes the shared typed carriers, stage ledger, cache join, and
descriptor-first exact-target launcher registry. It registers no Metal hook and
does not change value-mode classification, MPSGraph/Metal placement, pipeline
cache policy, or selectors; APPLE-E2E-1 still owns native package production,
Apple registration/submission, comparison, cleanup, and Level-C proof.
E2E-SPINE-3 is applicable as a family-granular proof envelope around bounded
Apple GPU/CPU Level-C scope. It standardizes shared fixture identity, cache
replay fields, benchmark metadata, sealed attachment hashes, and generated
fleet truth without changing Metal/Accelerate ABIs, schedules, placement, or
selectors. The existing Apple7 packet remains exact-device evidence for its
declared scope only; second-device/fleet proof remains APPLE-NATIVE-E2E-2 and
cannot be inferred from Apple7.
Fleet packet identity is now `(target, architecture)`, and Apple CPU plus
Apple7 GPU packets remain assigned to the M1 Max host. The NR2 WSL
`x86_64_base`/`sm_120a` slice and Strix Halo `x86_64_avx512`/`gfx1151` slice
transfer no Metal/Accelerate ABI, schedule, resource, timing, or readiness
claim. Apple packet recording remains a Mac-host follow-up.
The post-merge NR2 WSL packets now hash-seal base-x86 and bounded SM120
softmax/reduction evidence against source commit
`9f3757ef2dda2dd61ff94f1aefe0244f1b80f064`. Their generated-dashboard rows
do not change the Apple disposition: Apple CPU and Apple7 remain
`packet_pending` until independently recorded on the assigned M1 Max.
The NVIDIA-E2E-1 f16 landing slice was assessed as NVIDIA-only: it adds an
SM120 PTX package producer and exact CUDA submission hook, with no Metal hook,
Apple ABI, dtype registration, schedule, placement, or selector change.
The completed NVIDIA-E2E-1 NVFP4 slice extends the shared `tile.matmul_kernel`
verifier with an explicit packed-A/packed-B/scale-A/scale-B/output/M/N/K form.
Apple has no enabled NVFP4 cooperative-matrix execution route, so this is not
applicable to Metal lowering and requires no Apple ABI or selector change.
Apple inherits only the shared verifier rejection contract, not CUDA scale-word
packing, warp geometry, resource values, timings, or exact-device claims.

The first NVIDIA-E2E-2 slice changes the shared Graph→Tile async contract so a
copy produces `!tile.async_token`, its wait retires that token, and a matrix
consumer carries the dependency. Apple has no consumer for CUDA TMA/WGMMA
physical scheduling; its Metal and CPU pipelines, ABI, placement, selectors,
and execution claims are unchanged. The additive pipeline-registry
driver-source field and `tessera_nvidia` dialect manifest row are NVIDIA
bookkeeping. Exact SM builders are not applicable to Apple and transfer no
CUDA layout or schedule.

The NVIDIA-E2E-2 softmax slice adds the shared semantic
`tile.softmax_kernel(X, O, Rows, K)` envelope with explicit storage,
accumulation, and last-axis fields; the envelope now accepts f16/f32 storage
with f32 accumulation. It is not applicable to the current Apple
value/Metal/MPS compilation path, which already owns different typed calls and
physical reduction schedules. Apple does not inherit the SM120 thread-per-row
schedule, `nvvm.ex2`, PTX ABI, resources, timings, placement, or selector; no
Apple execution state changes.

The NVIDIA-E2E-2 dtype-totality slice changes the shared MMA selector contract
so fp32 Tensor Core selection requires an explicit TF32 math mode and bare
`fp4_e2m1` cannot alias NVIDIA NVFP4. Apple has no TF32 or NVFP4 cooperative
matrix route, so this is semantic parity only: it receives no CUDA scalar type,
fragment packing, MX/NV scale layout, PTX ABI, execution, or selector claim.
APPLE-DTYPE-1 remains SDK-gated for its own FP8/FP4 tensor formats.

The follow-on SM120 dtype slice adds a backend-private
`tessera_nvidia.mx_block_scale_mma` Target IR op and ptxas-backed FP6/MXFP4
register contracts. This is not applicable to Apple code generation: it adds
no shared storage dtype, Metal op, SIMD-group layout, scale ABI, runtime route,
or selector state. Apple FP8/FP4 proof remains owned by APPLE-DTYPE-1.

The NVIDIA-E2E-2 reduction slice adds a shared launch-level
`tile.reduce_kernel` semantic carrier. It is not applicable to Apple's current
value/Metal/MPS compilation path, which owns different typed reduction calls,
placement, and SIMD-group schedules. Apple inherits no SM120 launch ABI,
resources, timings, execution state, or selector change.

The NVIDIA-E2E-2 epilogue slice tightens only the shared Tile launch verifier
for explicit residual operands and order. Apple's existing typed Metal/MPS
epilogue contracts remain architecture-owned and inherit no CUDA ABI, layout,
resources, timings, execution state, or selector change.

The NVIDIA-E2E-2 attention slice adds a shared launch-level semantic carrier
for Q/K/V/O dimensions, storage/accumulation, scale, and causal behavior. It is
not applicable to Apple's existing MPSGraph/Metal attention executors and
transfers no CUDA schedule, pointer ABI, resources, timing, readiness, or
selector state. Any Apple adoption requires its own Metal materializer and
exact-device proof.

The NVIDIA paged-KV slice adds a shared logical-page gather carrier with
explicit f32 page storage, i32 page table, dimensions, range, and direct-route
semantics. Apple's resident Metal page-table attention remains architecture
owned and inherits no PTX ABI, CUDA schedule, evidence, or selector state.

The NVIDIA backward-attention slice adds a shared launch-level VJP carrier with
explicit determinism, mask/softcap, route, and workspace semantics. It is not
applicable to Apple's existing Metal/MPSGraph backward executor without an
Apple-owned materializer; no CUDA single-owner schedule, pointer ABI,
atomic/split resources, timing, readiness, or selector state transfers.

Cross-backend sync `E2E-DEVICE-LIBS-2026-07-19` adds logical name, content
digest, and link mode for LLVM-stage device libraries to the shared native-image
schema. It is not applicable to the current Metal/MSL/metallib path, which does
not link CUDA libdevice or ROCm OCML/OCKL/OCLC bitcode. Apple records no device
library and inherits no CUDA/ROCm discovery paths, symbols, cache keys, or
linker choices.

Cross-backend sync `CUDA-MATH-CONTRACT-2026-07-19` adds backend-neutral
`exp_mode` and `ftz` semantics to the shared Tile softmax envelope. The current
Apple paths do not consume that launch-level op, so the SM120 mapping to PTX
`ex2.approx.f32`, its 2-ULP bound, and CUDA cache-policy version are not
applicable. A future Apple lowering must select and prove its own Metal precise
or fast-math exponential route rather than inherit the CUDA approximation.

Cross-backend sync `CUDA-INTRINSIC-SURFACE-2026-07-19` adds shared canonical
toward-positive and toward-negative rounding names without changing the
existing default tuning sweep. CUDA's RN/RD/RU/RZ cast suffixes, integer packed
dots, and 2x16/4x8 SIMD functions are not Metal execution evidence. Apple must
map directed conversions and packed operations to its own MSL/Metal semantics
and device proof; no Apple ABI, route, or selector changes in this landing.

Cross-backend sync `PTX-TYPE-MEMORY-TRUTH-2026-07-19` is NVIDIA-private
physical truth: PTX bit registers, fragment operands, scopes, proxies, and
packed-access ordering do not transfer to Metal. Apple retains its own MSL
storage, SIMD-group/cooperative-tensor formats, address spaces, barriers, and
memory-order proof. The shared architectural conclusion is only that a language
dtype wrapper is not evidence of a native register or matrix execution route.

Cross-backend sync `NVIDIA-E2E-DTYPE-EXEC-2026-07-19` adds f64 to the portable
Tile epilogue output vocabulary for NVIDIA's compiler-owned DMMA path. This is
not applicable to Metal GPU execution: Apple GPU profiles expose no native
fp64, so no MSL type, SIMD-group matrix route, ABI, timing, or selector state
changes. Apple CPU fp64 remains independently owned.

Cross-backend sync `ROCM-E2E1-SOFTMAX-2026-07-19` is ROCm-owned. It maps the
already-shared `tile.softmax_kernel` envelope to `tessera_rocm.softmax`, adds an
HSACO package producer, and registers a gfx1151 HIP descriptor consumer. Apple
inherits no AMD exponential implementation, wave/LDS schedule, HSACO ABI,
resource value, timing, execution state, or selector change. The only shared
surface remains the previously assessed semantic Tile envelope and portable
native-image/launch schema. ROCm's content-addressed OCML/OCKL/OCLC population
is not applicable to Metal; Apple inherits no device-library record or cache
change.

Cross-backend sync `ROCM-DTYPE-TOTALITY-2026-07-19` is ROCm-owned and not
applicable to Apple target state. It introduces no shared dtype spelling or
alias and transfers no RDNA scalar, packed-dot, WMMA, accumulator, storage,
runtime, or selector claim to Metal or Apple CPU.

Cross-backend sync `ROCM-DTYPE1-CLOSE-2026-07-21` promotes signed `int4` and
alias `i4` into the shared canonical/Graph-IR vocabulary and adds signedness to
the shared packed-storage descriptor. Apple parity is validated at the logical
signed-int4 boundary; existing Metal packed-weight ABIs remain backend-owned.
No Apple target capability, physical schedule, runtime route, or selector is
promoted by the gfx1151 proof, and unsigned packed-4 remains unregistered.

Cross-backend sync `E2E-FROZEN-IDENTITY-CACHE-2026-07-19`: ROCM-E2E-1 memoizes
deterministic hashes for frozen runtime artifacts, native images, and launch
descriptors. Serialized identity values and required launch validation are
unchanged, so Metal schema parity is validated; no Apple ABI, schedule,
runtime route, performance claim, or selector changes.

Cross-backend sync `ROCM-E2E2-REDUCE-2026-07-19` is ROCm-owned. It consumes the
already-shared `tile.reduce_kernel` carrier and widens only its portable storage
verifier to admit bf16; the op registry and `Outer/AxisExtent/Inner` schema are
unchanged. Apple mappings are unchanged, and Metal/MPS reduction ABIs,
threadgroup schedules, exact-device evidence, runtime routes, and selectors are
unchanged; the ROCm five-argument HSACO ABI transfers no Apple claim.

Cross-backend sync `ROCM-E2E2-PAGED-KV-2026-07-19` is ROCm-owned. It consumes
the existing shared paged-KV carrier without changing its verifier or public op
schema. The ROCm directive, 256-thread gather, HSACO ABI, page-table validation,
and gfx1151 evidence transfer no Metal/MPS schedule, ABI, readiness, timing, or
selector claim; Apple's paged-cache routes remain independently owned.

Cross-backend sync `ROCM-E2E2-MOE-DISPATCH-2026-07-19` is ROCm-owned. It
consumes the existing shared MoE dispatch carrier and public operation without
changing their verifier or dtype registry. The AMD direct-gather schedule,
HSACO ABI, index validation, and gfx1151 evidence are not applicable to Metal;
Apple retains its independent MoE transport implementation and selector.

Cross-backend sync `X86-E2E1-NATIVE-CPU-2026-07-19` classifies shared native
descriptor results for host x86 targets as `native_cpu` with CPU-wall timing.
Apple GPU remains `native_gpu`, Apple CPU retains its independently owned
runtime routes, and no Metal/MPS ABI, schedule, device evidence, timing,
readiness, or selector state transfers. The x86 pilot consumes existing Tile
softmax/reduction carriers without changing their shared dtype or operation
registration.

Cross-backend sync `X86-E2E1-BREADTH-2026-07-19` consumes the existing shared
matmul and attention carriers for f32 AVX-512 descriptors. Apple inherits no
x86 ABI, host vector schedule, timing, readiness, or selector state. Metal/MPS
matmul and attention remain independently selected, and x86's equal-head and
zero-dropout descriptor restrictions change no Apple capability or verifier.

Cross-backend sync `E2E-SPINE-2026-07-18` records the 2026-07-20 scoped x86
selector retirement: eligible static X86-E2E-1 modules now use their canonical
descriptor by default. Apple parity is not applicable; no Apple pipeline, ABI,
schedule, capability, or selector changes. X86-E2E-2 subsequently closed the
remaining inventory and reassessed Apple at each shared-contract boundary.

Cross-backend sync `X86-E2E2-ELEMENTWISE-2026-07-20` adds the internal shared
`tile.elementwise_kernel` semantic carrier for f32 unary/binary and f32-to-bool
predicate requests. Apple parity is assessed at the carrier boundary only;
the AVX-512 ABIs, host-vector schedule, CPU-wall evidence, 16K binary selector
threshold, and runtime readiness do not transfer to Metal or Accelerate. No
Apple target, dtype, operation, ABI, execution, or selector row changes.

Cross-backend sync `X86-E2E2-TYPED-LOGIC-2026-07-20` widens that internal
carrier with compare, logical, and bitwise semantics plus explicit f32/i8/i32
physical storage. The only capability change is x86-owned bool/int32 truth for
already-shipped AVX-512 ABIs. Apple inherits no host-vector ABI, null-operand
convention, 32K selector threshold, CPU timing, or execution claim; Metal and
Accelerate rows remain unchanged.

Cross-backend sync `X86-E2E2-FLAT-FOLLOWON-2026-07-20` extends the shared
elementwise carrier with where, transcendental, and binary-math semantics.
Apple parity is assessed at the carrier boundary: no AVX-512 polynomial,
CPU-wall threshold, C ABI, Metal schedule, execution row, or selector transfers.
Existing Apple operations and routes remain independently owned.

Cross-backend sync `X86-E2E2-DTYPE-2026-07-20` adds an x86-only datatype/CPUID
contract and BF16, VNNI U8/S8, and FP64 descriptor ABIs. Apple parity is not
applicable: no Accelerate/Metal dtype capability, ABI, schedule, evidence, or
selector changes, and future ACE planning transfers no Apple execution claim.

Cross-backend sync `ATTN-DIALECT-MLIR23-2026-07-20` corrects the internal MLIR
attention dialect namespace from the nested `tessera.attn` spelling to the
MLIR-23-compatible `tessera_attn` spelling. Public Graph IR operation names,
attention semantics, Apple target capabilities, Metal/MPS ABIs, schedules, and
selector state are unchanged; the Apple outcome is parity validated by the
shared attention lit coverage.

Cross-backend sync `LLVM23-BACKBONE-2026-07-20` makes LLVM/MLIR 23.x the sole
accepted compiler build environment. Top-level and standalone CMake entry
points reject every other major and mixed installations. The Apple Metal
evidence lane uses the pinned prefix described above — since 2026-08-28 that
is Homebrew's production `llvm` keg 23.1.0 (`/opt/homebrew/opt/llvm`),
replacing the removed pre-release `llvm-23.1.0-rc1` build.
Apple target semantics and Metal/MPS runtime contracts are unchanged, and the
LLVM 23 compiler/lit build validates parity.

Consumer plan `SEQUENCE-MIXER-2026-07-17`: the compiler-direction Sequence Mixer
track ([`../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md`](../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md))
now consumes items **8, 9, 10, 11, 13, 14** as its Apple execution vehicle — it
adds candidates/state-types under existing items rather than opening new ones,
and **inherits this plan's evidence contract unchanged** (native `native_gpu`
placement, separate GPU/end-to-end timing-domain keys, two-run + ≥5% promotion,
forced binding-miss → `reference_cpu`). Concretely: channel-wise KDA/GDN decode →
**APPLE-REPLAY-1** (extend ReplaySSM / `SSMStateHandle` / `DeltaNetStateHandle`);
`sliding_window`/full mixer forward has closed its current **APPLE-ATTN-FWD-1**
scope; any new Sequence Mixer forward candidate requires a separately scoped
follow-up rather than silently reopening that item. `windowed_kv` +
uniform-block planner → **APPLE-PAGED-KV-1**;
chunkwise-scan inner GEMMs → **APPLE-RETUNE-1**; mixer arbiter → **APPLE-ROUTE-1**;
low precision → **APPLE-DTYPE-1** (stays SDK-gated — no NVFP4 cooperative-matrix on
Apple, so the executing FP4 proof is on NR2 Pro sm_120); mixer backward →
**APPLE-ATTN-BWD-1**. This is a direction pointer; it changes no Apple gate,
route, or exact-device claim here.

On a subsequent Apple-host collection, refresh the recorded marker totals and
append any new failure table by execution family and device generation without
discarding the current exact-device evidence. This plan is already in
`landing`; move it to the Apple archive only after every completion gate is
met.

Cross-backend sync `X86-E2E2-COHORT2-2026-07-20` adds shared typed Tile
carriers for argreduce, inclusive scan, unweighted row normalization,
interleaved-pair RoPE, and ALiBi. Apple parity is assessed at the semantic
carrier boundary only. AVX-512 ABIs, CPU schedules, Ryzen timing, and route
disposition transfer no Metal/MPS implementation, device evidence, or selector.

Cross-backend sync `X86-E2E2-BREADTH-2026-07-20` adds an explicitly x86-owned
`tile.x86_abi_kernel` and cohort-3/4 C-ABI registry. It changes no portable
semantic Tile carrier, Apple ABI, Metal/MPS schedule, dtype capability,
execution row, or selector. Apple parity is therefore not applicable; public
composite semantics continue to be assessed by Apple-owned typed routes.
X86-E2E-2 is now closed with measured x86-only selector thresholds; this does
not change the Apple not-applicable disposition or transfer device proof.

Cross-backend sync `LLVM23-LOCAL-CLEANUP-2026-07-20` repairs the host build and
sanitizer lanes after the LLVM/MLIR 23 migration. The shared capability audit
also corrects the existing Apple GPU matmul row to admit the already-shipped
f32/f16/bf16 MPS and Tile-simdgroup value ABIs. This is parity repair for an
existing Apple contract, not a new Metal schedule or exact-device claim.

Cross-backend sync `E2E-SPINE-2026-07-18` extends the shared launch-level Tile
carrier inventory with deterministic f16/f32 attention-backward dropout replay,
an explicit fused paged-attention causal-offset descriptor, and typed
f16/bf16/f32 MoE storage. The NVIDIA materializers, PTX ABIs, SM120 schedules,
and exact-device evidence do not transfer. Apple already owns separate Metal
attention/paged-cache and low-precision dispatch contracts, so no Apple runtime
mapping is required by this NVIDIA slice; future use of the new portable
carrier spellings must be proven through an Apple-owned lowering and exact
Apple device evidence.

Cross-backend sync `ROCM-E2E-SPINE3-TEST1-2026-07-21` adds shared paged-KV and
MoE fixture identities to the E2E-SPINE-3 corpus and correctly marks nine
Metal-only compiler nodes as `compiler_apple`. Apple fixture-schema parity and
compiler ownership are validated; the gfx1151 HSACO, HIP launch contract,
resource fingerprints, timing, and exact-device packet do not transfer to
Metal. No Apple capability, schedule, execution row, or selector changes.

Cross-backend sync `CORE-COMPILER-1-2026-07-22` lands the Apple-owned
declarative fusion table/generic rewrite and declarative value-envelope shape
constraints, and closes 11 shared dialect verifier holes. The shared MMA
selection is now recorded in Apple manifest rows and is available as an
equal-tier arbiter cost tie-break. Existing Metal/MPS ABIs and physical
schedules are unchanged; the LLVM 23 build validates compiler parity, while
exact-device performance evidence remains Apple-owned.

Cross-backend sync `CORE-COMPILER-2-2026-07-22` adds an executable physical
layout contract to the generic emitter/cache and lands the first guarded
dynamic-shape execution route on x86. Apple is **follow-up required**: its
shape-materialized MSL candidates remain bucketed and no x86 row-major
materializer, CPU guard, or dtype default transfers to Metal/MPS. Apple keeps
its existing dtype and physical-layout ownership until an Apple-specific
materializer and exact-device evidence land.

Cross-backend sync `CORE-COMPILER-NEXT-2026-07-22` tightens shared Graph layout
propagation through agreed-layout pointwise chains and last-axis reductions,
preserves packed-storage attributes, and records source-layout provenance on
inserted casts. Apple remains **follow-up required** for an architecture-owned
Graph-cast materializer; the pass stays opt-in and transfers no Metal layout,
schedule, selector, or device proof. The x86 dynamic last-axis reduction guard
is not applicable to Apple’s bucketed MSL routes. Shared add/multiply/static-
broadcast adjoints change Graph IR only; no Apple backward runtime or exact-
device promotion is claimed.

Cross-backend sync `CORE-COMPILER-FOLLOWON-2026-07-22` adds shared kind-aware
sum/mean, GELU/SiLU, and softmax Graph adjoints with host CPU oracle proof.
Dynamic mean, max/min, ReLU, and normalization remain explicit fallbacks for
the documented Graph-contract reasons. Guarded dynamic softmax, attention, and
growing KV-cache execution are x86-only and are not applicable to Apple's
bucketed MSL routes; no Metal ABI, schedule, selector, backward runtime, or
exact-device claim transfers. Apple's architecture-owned Graph-cast consumer
is host-validated: it accepts row-major/BHSD/NHWC before runtime fusion/per-op
lowering and rejects unsupported column-major bindings. This changes binding
metadata only and claims no exact-device proof.

Cross-backend sync `CORE-COMPILER-ADJOINTS-2026-07-22` registers shared
tensor-to-i1 comparison contracts plus internal scalar-threshold,
rank-reduced normalization-statistics, and explicit broadcast-in-dimension
Graph carriers. ReLU and unweighted RMSNorm/LayerNorm paired adjoints are
static/dynamic Graph-native and CPU-IR oracle-proven; the static shared path
lowers through linalg. Apple is **follow-up required** for backward execution:
no Metal/MPS ABI, affine gamma/beta contract, schedule, selector, runtime
binding, performance result, or exact-device proof is added here. Dynamic
statistics remain Graph IR until an Apple-owned materializer is implemented.

Cross-backend sync `CORE-COMPILER-NORM-AFFINE-2026-07-22` makes integer
comparison signedness explicit in shared Graph IR and adds dynamic-dimension
carriers plus channel-affine RMSNorm/LayerNorm adjoints. Apple is **follow-up
required** for an architecture-owned dynamic affine normalization materializer
and backward runtime: the gfx1151 HSACO and AVX-512 ABIs, schedules, timing,
and exact-device evidence do not transfer to Metal/MPS. Shared static/dynamic
linalg and CPU-oracle proof validate the Graph contract only; no Apple
selector, execution row, or device claim changes.

Cross-backend sync `CORE-COMPILER-NORM-BWD-DETERMINISM-2026-07-22` changes only
the ROCm architecture-owned backward schedule and temporary-buffer ABI. The
shared affine adjoint and f32 accumulation contract are unchanged. Apple still
requires its own Metal/MPS backward materializer and exact-device proof; the
gfx1151 two-kernel schedule, bitwise evidence, and timing do not transfer.

Cross-backend sync `CORE-COMPILER-NORM-BWD-2026-07-22` adds family-specific
RMSNorm/LayerNorm backward execution rows and public JIT binding for ROCm and
x86. Apple remains **follow-up required**: neither the gfx1151 HSACO ABI nor
the AVX-512 f32 ABI, schedule, dtype-accumulation contract, timing, or device
evidence transfers to Metal/MPS. The shared Graph adjoint and dynamic Linalg
contract remain parity validated; no Apple execution row or selector changes.

Cross-backend sync `CORE-COMPILER-LAYOUT-AUTODIFF-MEMORY-2026-07-23` completes
the shared transpose/packed epilogue/reduction layout envelope and adds native
guarded-dynamic broadcast, runtime-extent mean, and equal-share max/min Graph
adjoints. Apple parity is host-validated at Graph/linalg level, and
function-budgeted liveness-aware rematerialization is shared. The
address-space-3 Tile arena pipeline change is not applicable to Metal, whose
threadgroup allocation is architecture-owned; no MSL allocation, backward
runtime, performance, selector, or exact-device claim changes. An Apple
threadgroup-arena materializer remains follow-up required.

Cross-backend sync `CORE-COMPILER-TRAINING-SPINE-2026-07-23` registers
`tessera.loss.mse` and its paired backward carrier as verifier-checked shared
Graph IR, with dynamic none/sum/mean Linalg lowering and FP32 compute for
FP16/BF16 storage. Shape-preserving MSE participates in shared layout
propagation, and post-autodiff rematerialization now distinguishes saved
forward activations from backward temporaries. Apple parity is validated at
the shared IR boundary only. The gfx1151 HIP composition, module cache,
timings, and AVX-512 execution do not transfer to Metal/MPS; an Apple-owned
compiled MSE backward launch and exact-device evidence remain follow-up
required.

Cross-backend sync `CORE-COMPILER-DEEPENING-2026-07-23` adds shared
runtime-sized arena planning, cost-aware rematerialization metadata, and the
x86-owned default Graph-layout bridge. The new MSE backward execution proof is
ROCm gfx1151-only. Apple still requires its own compiled MSE VJP and
threadgroup-arena materializer; no HIP schedule, address-space-3 allocation,
x86 binding default, selector, or device claim transfers to Metal/MPS.

Cross-backend sync `CORE-COMPILER-TRAINING-BREADTH-2026-07-23` adds shared
Graph-native MAE, Huber, SmoothL1, and SGD adjoints with dynamic Linalg and CPU
oracle proof. Apple is **follow-up required** for an architecture-owned
Metal/MPS backward materializer and exact-device evidence. The gfx1151 HIP
kernel, AVX-512 C ABI, module cache, timing, and selector state do not transfer.

Cross-backend sync `CORE-COMPILER-TRAINING-SERIES-2026-07-23` adds shared
Graph-native stable BCE-with-logits, class-index/label-smoothed cross entropy,
KL/JS, explicit Momentum/Nesterov state, and explicit Adam/AdamW moment-state
adjoints. Dynamic shared Linalg contracts are live for BCE, Momentum/Nesterov,
and Adam/AdamW. Apple is **follow-up required** for Metal/MPS backward
materializers and exact-device evidence; the gfx1151 and AVX-512 loss and
optimizer ABIs do not transfer. No Apple selector or support claim changes.

Cross-backend sync `CORE-COMPILER-TRAINING-FUSION-2026-07-23` adds shared
single-use loss-backward to SGD/AdamW fusion carriers and one-loop dynamic
Linalg lowering for MSE, MAE, Huber, SmoothL1, and BCE-with-logits. Apple parity
is validated only at the shared Graph/Linalg contract. Apple remains
**follow-up required** for an architecture-owned Metal/MPS fused training
materializer and exact-device evidence; gfx1151 HIP and AVX-512 ABIs, cache
identities, timings, and selector decisions do not transfer.

Cross-backend sync `CORE-COMPILER-MEMORY-LAYOUT-CLOSEOUT-2026-07-23` replaces
the shared static address-space-3 alloca with a workgroup global and supports
dominance-scoped dynamic arena cohorts. This is not an Apple Metal allocation
claim: Apple still needs its architecture-owned threadgroup materializer and
exact-device evidence. The measured rematerialization corpus has gfx1151 and
AVX-512 rows only; no cost, selector, layout-default, or occupancy result
transfers to Apple.

Cross-backend sync `CORE-COMPILER-HONEST-BOUNDARIES-2026-07-23` broadens the
shared measured-rematerialization schema to exact consumer chains and
64/128/192 matmul shapes with ReLU/GELU/SiLU. Apple remains **follow-up
required** for Metal measurements and policy selection. ROCm dynamic
normalization epilogues, HIP launch-sized LDS materialization, and packed IU4
WMMA are architecture-owned and transfer no MSL threadgroup-allocation,
packed consumer, performance, or selector claim. Apple's threadgroup arena
and physical packed consumers remain architecture-owned follow-ups.

Cross-backend sync `CORE-COMPILER-HONEST-BOUNDARIES-2-2026-07-24` extends the
shared rematerialization corpus schema with softmax, RMSNorm, and MSE producer
families plus measured workload-budget decisions. Apple remains **follow-up
required** for Metal measurements and policy selection. ROCm's packed
multi-arena LDS ABI, GELU normalization epilogue, and terminal-pack
dequant-GEMM consumer are architecture-owned; no MSL threadgroup allocation,
packed consumer, timing, selector, or support claim transfers. Apple's
threadgroup path-max contract and physical packed consumers remain open.

Cross-backend sync `CORE-COMPILER-HONEST-BOUNDARIES-3-2026-07-24` extends the
shared rematerialization evidence schema to a measured four-layer workload with
softmax, RMSNorm, MSE, Huber, SmoothL1, and BCE instances. Apple remains
**follow-up required** for Metal measurements and policy selection. ROCm's
branch-path dynamic-LDS expression, binary normalization epilogues, and packed
elementwise/sparse/cache ABIs are architecture-owned; no MSL threadgroup
expression, packed ABI, timing, selector, or support claim transfers.

Cross-backend sync `CORE-COMPILER-CFG-MEMORY-BUDGETS-2026-07-24` adds a shared
model/device-derived rematerialization budget contract with explicit override
precedence and bounded dynamic parameters. Apple is **follow-up required** to
inject exact device capacity/reserve policy and validate model-level selection
with Metal measurements. ROCm's alias-aware nested/loop LDS slots and
40,208-byte gfx1151 packet are architecture-owned; no MSL threadgroup-memory
expression, occupancy, execution, or selector claim transfers.

Cross-backend sync `E2E-SPINE3-SM120-MEMORY-2026-07-24` extends the shared
fleet fixture corpus with bounded epilogue, attention, and ReplaySSM identities
and seals the six formerly pending NVIDIA SM120 family rows. Apple can reuse
only fixture identity and proof-schema structure. CUDA image descriptors,
NVPTX address-space-3 materialization, ptxas accounting, SM120 resources,
timings, and release readiness do not transfer to Metal. Apple CPU/Apple7
packet scope and the architecture-owned threadgroup-arena follow-up are
unchanged.

Cross-backend sync `CUDA-TRAINING-MEMORY-FOUNDATION-2026-07-24` is
NVIDIA-owned. It changes no shared Graph/Linalg mathematics and no Apple
execution row, selector, Metal ABI, or threadgroup-memory policy. The CUDA PTX
image/descriptor, CUDA-driver launch-v2 entry points, NVPTX external shared
symbol, ptxas/driver resources, and SM120 timings do not transfer to Apple
CPU/GPU. Apple retains its architecture-owned training and dynamic
threadgroup-arena follow-ups.

Cross-backend sync `CUDA-TRAINING-MEMORY-BREADTH-2026-07-24` adds only the
portable Graph IR carriers for model-parameter marking and bounded dynamic
parameter storage. NVIDIA owns the CUDA-context capacity/free-memory query,
FP16/BF16 PTX training ABIs, serialized dynamic-shared launch expressions, and
SM120 measurements. None transfers to Metal or closes Apple training,
threadgroup-memory, capacity-policy, or exact-device evidence.

Cross-backend sync `NVIDIA-BF16-CANONICAL-BREADTH-2026-07-25` widens the
shared Tile softmax and attention verifier envelope to BF16 storage with FP32
accumulation and preserves the already-shared BF16 reduction contract,
including min. Apple parity is validated at that semantic boundary: the
existing Apple capability and execution records already own independent BF16
softmax, attention, and reduction routes. NVIDIA's typed PTX descriptors,
serial/cooperative-128 schedules, CUDA-driver ABI, ptxas resources, SM120
numerics, and WSL timings do not transfer to Metal or Apple CPU. No Apple
manifest, schedule, execution state, or selector changes are required.

The NVIDIA continuation adds an architecture-owned compiler/PTX normalization
image and consumes the already-shared `tessera.storage_pack` descriptor in
scale-bearing NVFP4/MXFP4/FP6 CUDA materializers. Apple semantic parity is
unchanged: its independent BF16 normalization routes remain authoritative,
and CUDA packing factors, scale ABIs, PTX byte loads, rejection fixtures,
resources, and SM120 evidence do not transfer to Metal. No Apple storage
legalization default, packed dtype, capability, or selector changes.

Cross-backend sync `NVIDIA-PACKED-MATH-2026-07-25` adds a CUDA-owned signed
INT4 descriptor consumer and a typed internal Tile carrier for a bounded CUDA
integer/cast/packed-SIMD subset. The shared `tessera.storage_pack` schema is
unchanged. NVIDIA's nibble layout, PTX instructions, CUDA launch ABI,
resources, cache evidence, and SM120 numerics do not transfer to Metal or
Apple CPU. Apple retains its own packed INT4 and math execution contracts; no
Apple capability, storage-legalization default, schedule, or selector changes.

Cross-backend sync `NVIDIA-PACKED-SSA-FOUNDATION-2026-07-25` changes the shared
pack descriptor from an unstructured dictionary to portable
`#tile.packed_format`/`#tile.packed_view`/`#tile.scale_layout` attributes and
adds generic packed load/store plus SSA buffer/pipeline vocabulary. Apple is
**follow-up required** for architecture-owned Metal packed physical consumers
and threadgroup allocation/pipeline threading. NVIDIA scale indexing, PTX,
CUDA Math target operations, SM120 resources, and device evidence do not
transfer; no Apple support or selector state changes.

The same synchronization point now adds shared SSA TMA descriptor, mbarrier,
mbarrier-token, TMEM, and TCGen05 vocabulary and makes NVIDIA WarpSpec consume
the shared allocation/pipeline identity. These operation definitions are
portable compiler structure, not Apple execution support: Metal has no TMA,
TMEM, or TCGen05 consumer, so those operations are **not applicable** to Apple
with that architecture-specific reason. Apple threadgroup allocation and
pipeline-state threading remain **follow-up required** on its own lowering;
no NVIDIA resource, runtime, or exact-device claim transfers.

Cross-backend sync `ROCM-TRAINING-MEMORY-FUSION-2026-07-27` adds ROCm-owned
Adam/AdamW and KL/JS physical backward execution plus a ROCm normalization
softcap epilogue; no HIP kernel, gfx1151 timing, or selector evidence
transfers to Metal/MPS. Apple remains follow-up required for its
architecture-owned training backward materializers. The shared change is the
target-neutral, serializable dynamic-local-memory expression field on
`LaunchDescriptor`; Apple has no threadgroup-memory consumer of that field in
this change and retains its separately owned threadgroup materialization gap.

Cross-backend sync `ROCM-LION-BACKWARD-2026-07-27` adds only the ROCm-owned
physical consumer of the already-shared Lion stop-sign VJP policy and extends
the ROCm operation-total benchmark packet. HIP code objects, gfx1151 numerics,
and WSL timings do not transfer to Metal. Apple remains follow-up required for
an architecture-owned compiled Lion backward materializer; no Apple
capability, execution row, selector, or threadgroup-memory contract changes.

Cross-backend sync `CORE-SCHEDULE-1F1B-MATERIALIZE-2026-07-27` emits a shared
unique-clock warmup/steady/cooldown dependency order after pipeline legality.
At this synchronization point Metal/runtime consumption and collective overlap
remained Apple-owned follow-up; the immediately following
`CORE-COMPILER-RUNTIME-CLOSEOUT-2026-07-27` record supersedes the portable
runtime-consumption gap. A real multi-rank Metal transport packet remains
Apple-owned. The structural carrier changes no Apple capability, selector, or
exact-device claim.

Cross-backend sync `CORE-COMPILER-RUNTIME-CLOSEOUT-2026-07-27` adds a shared
runtime consumer for emitted 1F1B steps, including an independent collective
transport executor; measured schedule records now alter physical Schedule/Tile
attributes after target and evidence validation; and DeltaNet-family reverse
mode is an analytic carried-state recurrence with explicit forward/backward
schedule metadata. Apple can consume these shared contracts, but this change
contains no multi-rank Metal transport packet, Apple capacity injection,
measured Metal selector result, or sequence-mixer backward kernel. Those remain
Apple-owned exact-device follow-ups.

NVIDIA layout assignment now defaults on only because its named pipeline has an
immediate physical Graph-cast consumer. Apple already owns a separate
row-major/BHSD/NHWC Graph-layout materializer; no NVIDIA layout or execution
claim transfers. ROCm's factored Adafactor HSACO and gfx1151 timing likewise do
not transfer to Metal. An Apple factored optimizer implementation remains
follow-up required.

Cross-backend sync `CORE-PRODUCTION-EVIDENCE-2026-07-27` makes emitted pipeline
steps own serializable collective descriptors and adds a shared
replicated/rank-local OptimizerShard state machine. The runtime integration is
portable, but this continuation has no Metal multi-device transport
implementation or Apple exact-device packet. ROCm's physical Adafactor adjoint
and reverse-chunk DeltaNet HSACO are AMD-specific and do not transfer. Apple
sequence-mixer backward packaging and refreshed measured selector evidence
remain architecture-owned follow-ups.

Cross-backend sync `CORE-SEQUENCE-MIXER-PHYSICAL-BACKWARD-2026-07-28` adds the
exact modified-Delta normalization VJP to physical ROCm and AVX-512 backward
paths and proves affine parallel chunk composition for `erase=false`. This
changes shared algorithm evidence, not Metal execution. The five-entry gfx1151
HSACO, AVX-512 ABI, and their resident timing packets do not transfer to Apple.
Metal sequence-mixer backward packaging, nonlinear/erase chunk scheduling, and
a refreshed exact-Apple-host selector packet remain architecture-owned.

## Cross-backend sync `TILE-FRAGMENT-TYPE-PARAM-2026-08-03` — `!tile.fragment` parameterized (W1.1 step 1)

Shared Tile IR type changed: `!tile.fragment` gained `(m, n, k, elem, acc, role, layout, family)` and a domain verifier. **No behaviour changes in this PR** — the bare `!tile.fragment` still parses AND still prints bare, so every existing producer and fixture is unaffected. All 7 C++ `FragmentType` uses are `isa<>` checks, so there were no construction sites to migrate.

**Outcome: not applicable — architecture-specific reason.** Zero files under `Tessera_Apple_Backend/` reference `FragmentType` or `!tile.fragment` (measured 2026-08-03). The Apple GPU lane does not consume the cooperative-matrix fragment contract: its Tile→Target lowering emits `func.call` to hand-written runtime symbols, and the compiler-synthesized `simdgroup_matrix` path lives in the Python synthesizer (`emit/apple_msl.py`), not in this Tile type. See CLAUDE.md's Apple seam note.

**This is worth revisiting when that seam closes.** `simdgroup_matrix` IS a cooperative-matrix fragment in every meaningful sense, so if the MLIR lane ever synthesizes it, Apple should acquire a `family` value here rather than growing a parallel fragment concept — which would be the Decision #31 duplication this project keeps finding.

## Cross-backend sync `TILE-FRAGMENT-KLOOP-ACCUM-2026-08-03` — typed `tile.mma` K-loop (W1.1 step 2)

Shared Tile IR contract changed: `MMAOp::verify()` (and the `fragment_pack` / `fragment_zero` producers) now read the operand contract from the fragment TYPE when it is parameterized, falling back to producer-chasing for the bare form. `#tile.mma_desc` is optional on the typed path and cross-checked when present. **The canonical K-loop now verifies.** No lowering changed in this PR, and no existing IR is affected — the bare form keeps its old path.

**Outcome: not applicable — architecture-specific reason.** Unchanged from `TILE-FRAGMENT-TYPE-PARAM-2026-08-03`: zero files under this backend consume `!tile.fragment`, so the typed `tile.mma` contract and the K-loop accumulator question do not reach the Apple lane. The `simdgroup_matrix` note recorded under that key still stands as the thing to revisit when the MLIR/synthesizer seam closes.

## Cross-backend sync `NVWGMMA-ACCUMULATOR-GUARD-2026-08-03` — WGMMA accumulator drop (W1.1 step 2b guard)

A `tile.mma` carrying an accumulator was lowered by `NVWGMMALoweringPass` to a **two-operand** WGMMA call: the accumulator was discarded, the shape hardcoded `m64n64k16`, and the dtype inferred through `dyn_cast<ShapedType>` (which a `!tile.fragment` is not, so it defaulted to bf16) — with **rc=0 and no diagnostic**. A K-loop recomputed A×B from nothing each step and returned the last partial product.

Measured on merged main, this was **not** specific to the typed fragment form: a legacy bare `tile.mma(A, B, C)` — what `LowerKReductionAddToTileMMA` emits for the canonical K-step — was dropped identically. **No fixture in the tree covered either case**, which is how it survived. The guard therefore keys on *has an accumulator*, not *is typed*.

**Outcome: not applicable — architecture-specific reason.** Probed: `--tessera-lower-to-apple_gpu` leaves `tile.mma` unlowered, so there is no path on which an accumulator could be silently dropped. Consistent with this backend having zero `!tile.fragment` consumers (`TILE-FRAGMENT-TYPE-PARAM-2026-08-03`).

## Cross-backend sync `ROCM-COMPILED-STRICT-DISPATCH-2026-08-04` — compiled-lane failures stop masquerading

Runtime dispatch contract changed. A compiled-ROCm **failure** (tessera-opt ran and serialized no kernel, or emitted a non-ELF blob) now routes through the existing `_note_dispatch_fallback` funnel, so `TESSERA_STRICT_DISPATCH=1` raises instead of degrading. **Envelope limits** (no libamdhip64, hipInit failed, tessera-opt not built, dtype/rank/arch out of range) are unchanged and still degrade silently — making those raise would break strict runs on every CPU-only host.

Measured before the fix: a deliberately broken pass pipeline returned `ok=True, compiler_path="rocm_compiled", execution_kind="native_gpu"` with correct numbers. Strict-mode suite results are identical before and after (18 fail both ways, all pre-existing), so this adds no new failures.

**Outcome: parity validated — no change required.** Apple already routes failure-class dispatch through `_note_dispatch_fallback`; those sites are among the 18 pre-existing strict-mode failures measured on main (`tessera.matmul: MPS matmul symbol unavailable for dtype float32`). That is the funnel working as designed on this backend, and it is the precedent this change follows rather than a parallel mechanism (Decision #31).

## Cross-backend sync `ROCM-PIPELINE-TILE-LOWERING-2026-08-04` — the compiled pipeline can lower `tile.mma`

Both ROCm compiled pipelines (plain and canonical) now run `lower-tile-to-rocm{arch=<chip>}` after `generate-wmma-gemm-kernel`. Verified byte-identical hsaco with and without the pass on the default path, so the production lane is unchanged.

**Outcome: not applicable — architecture-specific reason.** No `!tile.fragment` or `tile.mma` consumers on this backend (`TILE-FRAGMENT-TYPE-PARAM-2026-08-03`), and no ROCm pipeline is involved.

## Cross-backend sync `TILE-VIEW-BOUNDED-CONTRACT-2026-08-04` — bounded `tile.view` is a shared contract

`ViewOp::verify` now defines the pointer-backed operand contract: exactly 3 `(base, rowOrigin, colOrigin)` or 5 with `(rowBound, colBound)`. It previously accepted any count >= 3, so a 4-operand view was legal and meaningless and the bounded form's validity was decided by whichever backend looked.

**Outcome: not applicable — architecture-specific reason.** No `!tile.fragment` or `tile.view` consumers on this backend (`TILE-FRAGMENT-TYPE-PARAM-2026-08-03`).

## Cross-backend sync `TILE-VIEW-LINEAR-BASE-2026-08-05` — should `tile.view` carry a precomputed linear base?

ROCm W1.1 step 3 (`W1_1_TYPING_DESIGN.md` §4.7) established that isolated
fragment address derivation could not express the direct lane's shared row
offset. Measurement selected an optional precomputed `linear_base` operand on
`tile.view`; logical row/column origins remain present for bounds.

ROCm implemented explicit `tile.view` linear-base sharing. Its new same-run
final rebuilt measurement improves typed/direct from 0.685x to 0.711x, but does not close the
gap; load scheduling/wait overhead remains the ROCm-owned follow-up.

**Outcome for Apple: NOT APPLICABLE.** This backend consumes neither
`tile.view` nor `tile.fragment_pack` (0 files). The MLIR lane lowers to
`func.call` on hand-written runtime symbols, and the MSL synthesizer is a
separate Python path (`compiler/emit/apple_msl.py`); neither consumes Tile
fragment ops, so there is no address form to hoist. Re-open under this key if
the synthesizer/MLIR seam closes onto Tile fragments.

## Cross-backend sync `TILE-DYNAMIC-LEADING-DIM-2026-08-04` — generic typed fragment addresses

Shared `tile.view` / `tile.store` can now carry an SSA leading dimension when
`#tile.memory_layout` states zero. **Outcome for Apple: NOT APPLICABLE.** Apple
has no Tile-fragment or pointer-backed `tile.view` consumer; neither its MLIR
runtime-call lane nor its MSL synthesizer changed.

## Cross-backend sync `E2E-REAL-LINEAGE-SCHEDULE-2026-08-05`

Shared compiler orchestration now records explicit artifact ancestry and
production `tessera-opt` registers the generated Schedule dialect. **Apple
outcome: follow-up required for compiler-spine parity, with no Apple-host claim
in this slice.** Apple CPU/GPU packages still consume `GraphIRModule`, so their
package-owned Tile artifacts record that Graph parent and remain
lineage-incomplete. No MSL, AIR, metallib, descriptor ABI, or selector changed.
Apple package consumption follows the x86/ROCm vertical proof and must be
revalidated on the owning Mac.

## Cross-backend sync `E2E-REAL-SCHEDULED-MATMUL-2026-08-05`

Shared Graph→Schedule→launch-Tile lowering is now real for the initial x86-f32
and ROCm-f16/f32 matmul instances. **Apple outcome: follow-up required, not
applicable to a physical Apple package in this slice.** Apple target selection
fails closed rather than borrowing either schedule, and existing CPU/GPU
packages still enter from Graph IR. A later Apple-owned vertical slice must
define its own numeric/schedule contract, consume the canonical Tile artifact,
and be revalidated on the Mac; no MSL, metallib, Accelerate, or MPS claim is
made here.

## Cross-backend sync `E2E-REAL-PHYSICAL-CONSUMERS-2026-08-05`

The shared package boundary is now concrete: a validated
`ScheduledMatmulArtifact` carries exact Graph, Schedule, and launch-Tile text
plus content identities into x86 and ROCm physical consumers. **Apple outcome:
follow-up required on the owning Mac.** No Apple code, schedule, MSL,
metallib, Accelerate, MPS, or selector changed. A later Apple slice must define
its own bounded dtype/schedule instance, consume the same boundary, and produce
Apple-host numerical and performance evidence; x86/ROCm evidence does not
transfer.

## Cross-backend sync `E2E-REAL-PERFORMANCE-2026-08-05`

The shared scheduled-matmul contract now separates instruction and macro tile
extents and binds the latter into artifact and launch provenance. **Apple
outcome: follow-up required on the owning Mac.** No MSL/MPS/Accelerate schedule
or selector changed, and Apple inherits neither gfx1151's 32x64 macro tile nor
Zen 5 evidence. Its later scheduled consumer must select and measure an
Apple-owned macro tile before promotion.

## Cross-backend sync `E2E-REAL-SEMANTIC-KERNELS-2026-08-05`

The shared spine now has content-addressed `schedule.softmax` and
`schedule.reduce` SSA edges and atomically lowers the bounded canonical f32
contracts to launch-level Tile artifacts. x86 and gfx1151 packages consume the
exact emitted artifact and have owning-host numerical proof. **Apple outcome:
follow-up required on the owning Mac.** Apple CPU/GPU packages still consume
`GraphIRModule`; no MSL, MPSGraph, Accelerate, metallib, descriptor, schedule,
or selector changed here. Apple must define its own Schedule policy and consume
this artifact boundary before its existing softmax/reduction kernels can earn
lineage-complete evidence. The x86/gfx1151 workgroup choices and device results
do not transfer. Canonical Graph reduction currently excludes mixed-output and
keepdims forms, so those need a shared Graph-contract extension before any
backend can claim them through this spine.

## Cross-backend sync `E2E-REAL-ATTENTION-2026-08-05`

The shared spine now defines a content-addressed `schedule.attention` edge and
one launch-level Tile artifact for the bounded x86/gfx1151 instances. **Apple
outcome: follow-up required on the owning Mac.** Apple still lacks the shared
rank-4 forward consumer and retains its evidence-driven recompute policy; no
MSL, metallib, MPSGraph, LSE policy, selector, or device claim changes here.
Apple must define an architecture-owned schedule instance, consume the exact
artifact, and validate modifiers and numerical/performance behavior on Apple
hardware. x86 and gfx1151 schedules and evidence do not transfer.

## Cross-backend sync `E2E-REAL-ATTENTION-BACKWARD-2026-08-05`

The shared spine now defines a content-addressed three-result
`schedule.attention_backward` program carrying dQ, split-dK/dV, fixed reduction,
workspace, and LSE checkpoint identity. **Apple outcome: follow-up required on
the owning Mac.** No Apple backward-loop consumer, MSL/metallib package, or LSE
policy changed. Apple must select its own saved/recompute identity, consume the
exact program artifact, and validate MHA/GQA/MQA plus the modifier/ragged
envelope on Apple hardware; Zen 5 and gfx1151 evidence does not transfer.

## Cross-backend sync `E2E-REAL-5C-STATE-LINEAGE-2026-08-05`

The shared training spine now defines content-addressed logical-buffer lineage,
mutation identity, and typed Schedule→Tile contracts for Lion VJP,
factored/full Adafactor VJP, and sequence-mixer backward. **Apple outcome:
follow-up required.** This changes no Metal package. Apple must bind its own
buffers to those exact artifacts and validate on the owning Mac; x86 and
gfx1151 evidence does not transfer.

## Cross-backend sync `ROCM-TYPED-EXECUTABLE-PIPELINE-2026-08-07`

The shared orchestration direction now has a concrete typed configuration for
family ownership and the Tile-producer→Target-IR-consumer→backend-codegen
boundary. **Apple outcome: follow-up required on the owning Mac.** Apple must
define Metal/CPU family plugins around its own `tessera_apple` and native
packaging contracts; ROCm async-copy, waitcnt encoding, schedules, and gfx1151
evidence are not applicable. No MSL, AIR, metallib, MPS, selector, or Apple
device claim changes in this slice. Apple accepts the shared strict native-image
boundary policy; enforcement in Metal/CPU family plugins remains Apple-owned
follow-up.
ROCm has now retired its final generic runtime pass-name helper. **Apple
outcome: follow-up required on the owning Mac:** Metal/CPU plugin configuration
should adopt the same closed semantic-family rule. No Apple binary, schedule,
selector, or device evidence changes here.

## Cross-backend sync `TSOL-PACKED-FUSION-2026-08-08`

The shared `schedule.spectral_program` contract now hashes packed-real fusion
topology and N/2 child identity. **Apple outcome: follow-up required on the
owning Mac.** No MSL/metallib or Accelerate consumer changed. Apple must choose
an architecture-owned real-transform plan and consume the exact v5 artifact;
x86/gfx1151 physical choices and evidence do not transfer.

## Cross-backend sync `TILE-SYNC-RECONCILE-2026-08-10`

`tile.async_copy`/`tile.wait_async` now have one declared contract (ODS dual
form, `TileOps.td`); the Python spine's `stage`/`vector` checks are now
optional-when-present (`tile_ir.py`). New shared diagnostic
`TILE_ASYNC_STAGE_NEGATIVE`. **Apple outcome: parity validated on the owning
Mac.** Apple lanes consume the Python tile IR spine and the runtime
`func.call` lowering, not the C++ stage-model verifier that changed; the
spine relaxation is covered by the on-box unit suite (`test_tile_ir.py`
no-key-verifies-clean case). No MSL/MPS consumer changed.

## TILE-SYNC-TYPED-2026-08-15 — shared Tile sync ABI assessment (PR #566)

**Parity validated at lit level; no ABI consumer.** The Apple lane consumes
no `tile.mbarrier.wait` (the changed operand-segment ABI), and its
threadgroup pipeline fixtures that touch shared TMA vocabulary
(`phase8/apple_threadgroup_pipeline{,_invalid}.mlir`, incl.
`tile.tma.descriptor`) pass unchanged in the full 324/0 lit run on the
changed tree. The new `tile.tma.descriptor` source-type constraint
(buffer/memref/ranked-tensor) admits every Apple fixture form. No Metal/MPS
runtime surface changed; no on-Mac evidence is claimed or required.

## REF-TIER-OPS-2026-08-15 — reference-tier op registration assessment (PR #568)

PR #568 registered ten new public operations through the canonical op catalog
and the primitive coverage registry — `tridiagonal_solve` (Thomas recurrence,
PDE plan §III.1 / TSOL-A1) and the nine-op coalition-lattice family
(`game_subset_zeta`, `game_subset_mobius`, `game_superset_zeta`,
`game_superset_mobius`, `game_coalition_marginal`, `game_semivalue`,
`game_boltzmann_value`, `game_coalition_excess`, `game_mex`). Op registration
is a shared contract, so this queue records the outcome per AGENTS.md
"Cross-backend work coordination"; PR #568 itself landed without these records.

**Follow-up required — no Apple lane exists for any of the ten.** Neither
family appears in `apple_gpu_envelope.runtime_ops`, the MSL synthesizer
(`compiler/emit/apple_msl.py`), or the Accelerate CPU shim; the declared tier
is the Python reference. Per the parity methodology, absence from
`runtime_ops` — not the backend manifest — is the accurate Apple-GPU gap
signal here. `game_boltzmann_value` is the one op with a plausible near-term
Apple shape when G5 opens, since it rides the existing online-softmax emitter;
the remaining eight need the G1b butterfly lowering and the solver needs the
PDE plan's phase. No on-Mac evidence is claimed or required — nothing here
changes a Metal/MPS surface.

## E2E-REAL-3-APPLE-GPU-MATMUL-2026-08-15 — Apple GPU joins the shared scheduled-matmul boundary

**Landed — compiler-boundary + on-Mac correctness; device-time promotion still
gated.** This closes the Apple-GPU half of the 2026-08-05
`E2E-REAL-SCHEDULED-MATMUL` / `E2E-REAL-PHYSICAL-CONSUMERS` follow-ups: the
Apple GPU backend now **consumes** the shared `Graph → Schedule → launch-Tile`
matmul artifact instead of re-classifying the `GraphIRModule`.

What landed (WS-1 first slice of the Apple compiler-foundation integration):

- **C++** `getMatmulSchedule` (`PMPasses.cpp`) admits `apple_gpu` static rank-2
  f32→f32 (arch `apple7`, logical 16×16 macro-tile). One decision function
  unlocks both `--tessera-graph-to-schedule` and `--tessera-schedule-to-tile`;
  the emitted `tile.matmul_kernel` uses `family = "auto"`. Fail-closed tamper
  fixtures unchanged.
- **Python** `apple_native.package_scheduled_matmul` consumes `artifact.tile_ir`
  verbatim (no Graph re-entry) and binds the proven
  `tessera_apple_gpu_bmm_f32` **batch-1** route (`apple_gpu_bmm_f32_batch1`,
  sealed by APPLE-NATIVE-E2E-2). Apple GPU has no rank-2 f32 cooperative-matrix
  GEMM, so the shared macro-tile/mma decision is a **named dropped decision**
  (`dropped_reason = "delegated_to_mps_bmm"`) per E2E §0.2 point 5, not a silent
  loss. `scheduled_matmul._graph_contract` gained the matching apple_gpu branch;
  the driver gained an isolated apple_gpu scheduled block + dispatch (zero risk
  to x86/ROCm); the BMM runtime submit accepts rank-2 operands as batch-1
  (`runtime.py` only — no `apple_gpu_runtime.mm` fingerprint impact).

Evidence, on the owning Mac (M1 Max / apple7, LLVM/MLIR 23 `build-apple`):

- Lit: `tests/tessera-ir/phase2/e2e_matmul_graph_schedule_tile.mlir` gains an
  Apple typed instance beside x86/ROCm — 4/4 e2e_matmul fixtures pass; the three
  fail-closed fixtures still reject.
- Unit: `tests/unit/test_scheduled_matmul_consumers.py` — package consumption,
  non-apple-contract rejection, and driver lineage adjacency
  (`test_driver_records_adjacent_scheduled_matmul_lineage[apple_gpu]`) pass
  host-free.
- **Exact-device**: `test_apple_gpu_scheduled_matmul_executes_exact_artifact`
  ran on Metal for `16×16×16` and `17×19×23`, asserting `native_gpu` placement
  and matching the NumPy oracle — not a `reference_cpu` fallback.

NVIDIA and ROCm are not applicable: no shared IR, sibling ABI, or schedule
changed; the C++ branch is target-guarded to `apple_gpu`.

### Update 2026-08-15b — softmax (E2E-REAL-5) and f16 simdgroup matmul landed

Two more WS-1 families now consume the shared scheduled boundary on Apple GPU,
each with on-Mac exact-device proof (M1 Max / apple7):

- **Softmax (E2E-REAL-5).** `apple_native.package_scheduled_kernel` consumes the
  shared `schedule.softmax → tile.softmax_kernel` artifact and binds the native
  MSL `tessera_apple_gpu_softmax_f32` route (which *does* expose a device timer,
  so it is not DEVICE-EVENT-1 gated). The C++ `getSemanticKernelSchedule` admits
  `apple_gpu` f32 softmax and **fails closed on reduction** (no Apple GPU
  scheduled reduce consumer). Rank-2 f32 softmax now migrates to this boundary
  by default; `test_apple_gpu_package_trace_uses_descriptor_provenance` was
  updated (work_item `APPLE-E2E-1 → E2E-REAL-5`), and the direct descriptor
  route stays covered by `test_apple_softmax_package_hashes_dylib_and_names_abi`.
  Exact-device: `test_apple_gpu_scheduled_softmax_executes_exact_artifact`.
- **f16→f32 simdgroup matmul.** `package_scheduled_matmul` now dispatches by
  dtype: f32 → batch-1 MPS BMM (above), **f16 → the compiler-emitted
  `simdgroup_matrix` MSL GEMM** (`tessera_apple_gpu_tile_simdgroup_gemm_f16`,
  APPLE-TILE-1). The C++ `getMatmulSchedule` gained an apple7 f16→f32 branch
  (32×32 macro tile); `_submit_apple_gpu_native` gained a simdgroup descriptor
  branch that reuses the proven TILE-1 materializer + dispatch. **This route
  honors the scheduled tile and has a device timer, so it is NOT gated by
  APPLE-DEVICE-EVENT-1** (`device_time_promotion = "eligible"`) — it is the
  intended lift of that gate for matmul. Exact-device:
  `test_apple_gpu_scheduled_simdgroup_f16_executes_exact_artifact` (16³ and
  48×32×80 on Metal).

Consolidated: 121 shared/apple scheduled + e2e-spine + fleet + lineage tests
pass, 9 skipped (x86/rocm device lanes), 0 failed. Lit: the shared
`e2e_matmul_graph_schedule_tile.mlir` and `e2e_semantic_kernel_graph_schedule_tile.mlir`
each carry Apple typed instances and pass 100%.

### Update 2026-08-15c — rank-4 forward attention (E2E-REAL-5A) landed

Apple GPU now consumes the shared `schedule.attention → tile.attention_kernel`
artifact, closing the last *forward* WS-1 family. This is the item the
2026-08-05 `E2E-REAL-ATTENTION` sync recorded as "Apple must define an
architecture-owned schedule instance … x86 and gfx1151 schedules and evidence do
not transfer."

- **Apple owns its LSE identity.** `schedule.attention` gained a third declared
  backward-LSE policy, `apple7_recompute` (`ScheduleDialect.cpp`), because
  Apple's backward recomputes m/l per query row and its ABI takes no LSE buffer
  (APPLE-ATTN-STREAM-1). The verifier allowlist stays closed by design — Apple
  declares its own identity rather than inheriting x86's `save_lse` or the
  gfx1151 threshold.
- **Consumer:** `apple_native.package_scheduled_attention` binds
  `tessera_apple_gpu_flash_attn_variant_f32_status` — the **status-returning**
  GQA MSL route from APPLE-ATTN-FWD-1. Because it reports placement, a
  numerically-correct CPU fallback cannot be sealed as GPU evidence
  (APPLE-PLACEMENT-ABI-1).
- **Two ABI facts were read off the MSL kernel, not assumed** (both had to be
  corrected against a first wrong guess, and both are now commented at the call
  site): its `B` operand is the **flattened `batch × q_heads`** extent, and
  `window_size` is active only when `> 0`, so the canonical "no window" request
  (`-1`) is passed as `0`.
- **Envelope, enforced in both owners (C++ gate + Python recognizer), never
  narrowed:** f32 storage, MHA/GQA/MQA with whole group size, shared
  head/value dim, `D <= 256`, **no live window**, no dropout. The window
  exclusion is deliberate: the MSL non-causal window is a *symmetric
  half-window* (`window_size/2` per side), which is **not** the shared
  `window_left`/`window_right` semantics — admitting it would silently compute a
  different mask than the program requested. Windowed Apple attention needs its
  own contract and oracle.

Evidence (M1 Max / apple7): new shared fixture
`tests/tessera-ir/phase2/e2e_attention_graph_schedule_tile.mlir` carries x86 and
Apple typed instances and passes; four **exact-device** Metal configurations
(GQA, MHA batch-2, MQA, causal) execute with `native_gpu` placement and match the
NumPy oracle at f32 epsilon (~1e-7 max abs error); a five-way envelope test
proves each out-of-envelope request fails closed. Consolidated: **230 passed /
11 skipped / 0 failed** across the scheduled, e2e-spine, fleet, lineage, and
Apple attention/MLA/MPSGraph suites.

**Pre-existing lit failures, proven not mine.** `lit tests/tessera-ir/phase2/`
reports 14 failures in this Apple-only `build-apple` (x86/ROCm fixtures whose
dialects are not configured). The failing set is **byte-identical** with this
work stashed and applied, so no regression was introduced; the honest reading is
that this host cannot evaluate those lanes.

Still open on Apple WS-1: **attention backward** (E2E-REAL-5B) has no Apple
scheduled consumer and is gated separately; scheduled **reduction** has no Apple
GPU consumer; and the f32 BMM matmul route remains DEVICE-EVENT-1 gated for
device-time promotion (correctness/boundary only). Windowed attention and
f16/bf16 attention storage are out of the admitted envelope.

#### PR #570 review findings — two real defects, both regression-locked

Automated review of PR #570 found two P1 defects that the original tests did not
catch. Both are recorded here because the *shape* of each is a recurring hazard
in this codebase, not because the fix was large.

1. **A live attention window was silently erased (fail-open).** The Apple
   recognizer read only the `window` alias, so a mask expressed in the canonical
   `window_left`/`window_right` pair defaulted to `-1` and was accepted as
   *unwindowed* — and `lower_scheduled_attention` then wrote that `-1` back over
   the operation's own attributes (`scheduled_attention.py:147`), erasing the
   window and computing full attention instead of the requested mask. This is
   precisely the failure the slice's own design note claimed to prevent, arriving
   through a spelling the recognizer did not read. The recognizer now reads
   **every** spelling present, requires them to agree, and fails closed on any
   live, asymmetric, or self-disagreeing window. Locked by
   `test_apple_gpu_attention_reads_every_window_spelling` (7 cases).
2. **The softmax descriptor hard-coded rank 2, and the test hid it.** The shared
   contract admits any positive rank and flattens leading dims into `rows`, but
   the Apple bindings declared rank 2, so launching a rank-3 graph with its own
   declared shape failed `E_LAUNCH_BINDING_MISMATCH` before submission — a
   regression, since such a module previously just declined to package natively.
   The original exact-device test masked this by passing a pre-flattened `(6,5)`
   buffer for its logical `(2,3,5)` graph. The descriptor now binds the
   **logical** rank and shape (matching the x86 consumer), and the flatten to
   `(rows, columns)` happens at submission, where it is exact because softmax
   rows are independent and both operands are already required C-contiguous (so
   the reshapes are views, not copies). The test now launches with the logical
   shape and is parametrized over rank 2/3/4, asserting the binding rank itself.

The general lesson, worth carrying: **a test that constructs its input to match
the implementation cannot falsify the implementation.** The rank bug was
invisible for exactly as long as the test pre-flattened its own buffer.

3. **A runtime-only Apple install hard-failed (found by CI, not by the local
   suite).** The shared scheduled artifact is produced by *running* production
   `tessera-opt`. The driver entered the Apple scheduled lane on the strength of
   `supports_scheduled_*` alone, so on a host with no built compiler an ordinary
   rank-2 f32 softmax raised `RuntimeError: scheduled softmax/reduction lowering
   requires production tessera-opt` instead of using its descriptor route — a
   regression, since that module previously just declined to package natively.
   The `package_native` auto-enable had the same defect for rank-2 f32 matmul.
   Both now consult `driver._apple_scheduled_boundary_available()`, and Apple
   falls back to the independently proven descriptor route when the tool is
   absent. Locked by
   `test_apple_scheduled_boundary_falls_back_without_tessera_opt` and by
   parametrizing the trace-provenance test over both worlds.

   **This was invisible on the Mac** because the dev box always has a built
   `tessera-opt`. The reproduction that matters is
   `env -u TESSERA_OPT TESSERA_BUILD_DIR=/nonexistent python -m pytest …`,
   which reproduces the CI host locally in seconds; the two host-free lineage
   proofs now pin the predicate rather than inheriting whatever the runner has.

### Update 2026-08-16 — rank-4 attention BACKWARD (E2E-REAL-5B) landed

Apple GPU now consumes the shared `schedule.attention_backward →
tile.attention_backward_kernel` artifact, closing the last WS-1 attention
family. `AttentionBackwardOp`'s LSE allowlist gains `apple7_recompute` on the
same closed-by-design basis as the forward op.

- **Consumer:** `apple_native.package_scheduled_attention_backward` binds the
  status-returning MSL VJP proven by APPLE-ATTN-BWD-1, for f32/f16/bf16 storage;
  dQ/dK/dV are f32 for every input storage, matching the shared gradient
  identity. Envelope (enforced in the C++ gate *and* the Python contract):
  shared head/value dim, `D <= 256`, whole GQA group, **no live window**, no
  dropout, and `saved` LSE explicitly refused rather than silently recomputed
  under a `saved` label.
- **Route selection is contract-driven, not speed-driven.** The MSL ABI offers
  serial (0), atomic (1), and split (2) dK/dV routes. The shared contract
  declares a two-way split with **ascending fixed-order reduction**, so only the
  split route is a faithful mapping — atomic is explicitly nondeterministic and
  would violate the declared order. Measured cost of that correctness, same
  shape (`B1 Hq4 Hkv2 S64 D64`, 10 reps, median): **atomic 65 ms, split 193 ms,
  serial 373 ms**. Split is ~3× slower than atomic; it is still the right
  selection, and the number is recorded so the tradeoff is visible rather than
  implied.
- **A third ordering fact read from source, not assumed:** the shared artifact
  orders its inputs **(dO, Q, K, V)**, not (Q, K, V, dO). Confirmed against the
  x86 consumer before wiring.

Evidence (M1 Max / apple7): four exact-device configurations (GQA, MHA, MQA,
causal) match an **independently derived float64 analytic VJP** — dQ, dK and dV
each to ~1e-6, with GQA gradients reduced over the query-head group (the
property that catches a kernel accumulating into the wrong KV head). Repeats are
**bit-identical**, which the atomic route would fail. 17 of 18 tests in the
backward consumer suite pass; the one failure
(`test_gfx1151_..._packages_exact_tile_program`) is **proven pre-existing** —
it needs AMD clang, absent on this host, and fails identically with this work
stashed.

#### Benchmark — `benchmarks/apple_gpu/benchmark_scheduled_boundary.py`

New characterization over all four scheduled families. Every row re-checks its
numerical oracle, so a fast wrong answer cannot be recorded as a win, and every
row asserts `native_gpu`. Apple7, 20 reps after 5 warmups, end-to-end host wall
(the common denominator, since the MPS matmul route has no device timer):

| case | route | median ms |
|---|---|---|
| matmul.fp32 64³ / 256³ | `apple_gpu_bmm_f32_batch1` | 0.95 / 1.03 |
| matmul.fp16 64³ / 256³ | `apple_gpu_simdgroup_gemm_f16` | 0.48 / 0.82 |
| softmax rank-2 / rank-3 | `apple_softmax_native_library` | 1.14 / 0.99 |
| attention fwd / causal | `apple_gpu_flash_attn_variant_f32` | 1.64 / 2.81 |
| attention bwd / causal | `apple_gpu_flash_attn_bwd_split_f32_grads` | 195 / 396 |

The report sets **`selector_eligible: false`** and writes no committed ledger:
one process, one timing domain, no paired interleaving and no retained counter
evidence is none of what the Apple promotion contract requires.

**Open finding worth its own item: backward is ~120× forward** (195 ms vs
1.64 ms at the same shape). That is a property of the existing hand-written MSL
backward kernels, not of this boundary work, and it is recorded here rather than
quietly optimized — a kernel rewrite needs its own slice and its own paired
corpus. **Closed 2026-08-16 — see APPLE-ATTN-BWD-PERF-1 below.**

### Update 2026-08-16b — scheduled reduction landed (first synthesized family)

Apple GPU now consumes the shared `schedule.reduce → tile.reduce_kernel`
artifact, closing the last WS-1 family. **This one is different in kind from its
siblings:** matmul f32 delegates to MPS and softmax binds a hand-written MSL
kernel, but reduction is genuinely **compiler-emitted** — the Decision #28
tier-1 synthesizer (`emit/apple_msl.py::synthesize_pointwise_reduce_msl`)
produces the kernel and the source-carrying
`tessera_apple_gpu_synth_pointwise_reduce_f32` ABI runs it. It is the first
Apple family where the MLIR-side boundary and the Python synthesizer meet on one
artifact, which is the seam `CLAUDE.md` names as the real Apple gap.

- **Content-addressed source.** The package records the synthesized kernel's
  SHA-256 and carries the MSL itself as the Target IR (`tessera_apple.gpu.
  msl_kernel`, not a call stub). The runtime **re-derives the source and
  verifies the digest before dispatch**, so a descriptor cannot execute a kernel
  other than the one it was built from. Synthesis is deterministic in the reduce
  kind alone, which is what makes that check meaningful rather than decorative.
- **Placement is asserted, not inferred.** `run_pointwise_reduce` silently falls
  back to a NumPy reference on any dispatch failure, so the submit path requires
  a `metal_runtime` provenance and raises otherwise — a correct answer alone
  would not distinguish GPU from CPU (APPLE-PLACEMENT-ABI-1).
- **Envelope:** f32, rank-reducing, **last axis only** (`inner == 1`), kinds
  `sum`/`mean`/`max` (mapped to the synthesizer's `amax`). The kernel gives one
  thread per row and folds the trailing extent, so an interior axis fails closed
  in both owners rather than being reordered. gfx1151 keeps its arbitrary-axis
  support — the Apple bound must not leak to siblings, and a test asserts that.

Evidence: six exact-device configurations (sum/mean/max × rank-2/rank-3) execute
on Metal with `native_gpu` placement and match NumPy; 21 tests in the
scheduled-kernel suite pass.

### Update 2026-08-16c — APPLE-DEVICE-EVENT-1 closed

Row 32 is closed. The GPU matmul route ran through `MPSGraph
runWithMTLCommandQueue:`, which **owns and commits its own command buffer**, so
no object existed on which to observe a device interval and the whole `apple_gpu`
packet was forced onto `kernel_wall`.

`mpsg_run_bmm` now encodes into an explicitly owned `MPSCommandBuffer` under the
**already-existing** shared `MPSGraphTimingBracket` — the same one the sibling
MPSGraph paths (gather, transpose, row-op, BSMM) use, so this reuses one
implementation rather than adding a second (Decision #31).

- **Measured, not asserted.** `record_apple_packet.py` no longer hard-codes
  `device_event_available: False` with a now-false reason. It probes the device
  interval per family and claims the `device_event` domain **only when every
  family in scope published one** — one family without it would make a
  report-wide domain dishonest, so the packet falls back to `kernel_wall` for
  all families rather than mixing domains.
- **The sealed packet is re-recorded, not re-stamped.** Editing
  `apple_gpu_runtime.mm` correctly tripped the packet's source fingerprint;
  the fix is a fresh measurement. The packet now seals
  `required_timing_domains: ["device_event", "end_to_end"]` with both families
  stable at 31×80 sampling (the 15×50 default drifted 5.2% > the 4% bar for the
  device cohort — device intervals are noisier than wall time at these shapes,
  so the sampling was raised rather than the bar lowered).

**What the stronger domain immediately revealed:** at the packet's timing
shapes, matmul is **0.595 ms device** inside **1.810 ms** end-to-end, and
softmax is **0.0235 ms device** inside **1.014 ms**. Softmax spends ~23 µs on
the GPU inside a millisecond of wall time — i.e. these lanes are host-overhead
dominated, which `kernel_wall` could not have shown. That is a materially better
question to optimize than any of the kernel timings above.

**Cost check, since a `.mm` edit is not free:** the strict route ledger's
fingerprint was **already** invalid before this change (pinned `74eb6e95…` vs
live `9b12af9a…`, inert since 2026-07-27), so no live selector evidence was
destroyed. End-to-end matmul timing is unchanged (0.95 → 0.89–0.96 ms across
runs, inside p90 noise), so the device timer costs nothing measurable.

#### Benchmark, all four WS-1 families (Apple7, 20 reps, end-to-end)

| case | route | median ms |
|---|---|---|
| matmul.fp32 64³ / 256³ | `apple_gpu_bmm_f32_batch1` | 0.90 / 0.95 |
| matmul.fp16 64³ / 256³ | `apple_gpu_simdgroup_gemm_f16` | 0.49 / 0.82 |
| softmax rank-2 / rank-3 | `apple_softmax_native_library` | 1.14 / 0.98 |
| **reduce sum/mean/max** | `apple_gpu_synth_pointwise_reduce_f32` | **0.37** |
| attention fwd / causal | `apple_gpu_flash_attn_variant_f32` | 1.64 / 2.78 |
| attention bwd / causal | `apple_gpu_flash_attn_bwd_split_f32_grads` | 195 / 389 |

Still `selector_eligible: false` — one process, no paired interleaving.

#### PR #571 review finding — `max` violated its own declared NaN contract

Review found a real semantic-contract violation in the reduce slice, confirmed
on device before fixing. The shared synthesizer emitted `max(acc, v)`; Metal's
`max`/`min` are IEEE maxNum/minNum-style and **suppress** a NaN operand, so:

| row | GPU (before) | numpy reference |
|---|---|---|
| `[1, NaN, 3]` | `3` | `NaN` |
| `[NaN, NaN, NaN]` | **`-inf`** | `NaN` |

The reduce Schedule artifact literally declares `nan_mode = "propagate"`, and
the synthesizer's *own* numpy reference propagates — so the kernel disagreed
with both. The all-NaN case is the one that matters: missing data silently
became a finite extreme, which a downstream `argmax` or clamp would act on.

Fixed at the single source (`fusion_core.py::_PW_REDUCE_KINDS`, whose only
kernel consumer is `emit/apple_msl.py`) so extrema propagate explicitly; `sum`
and `mean` needed no change because IEEE addition already propagates. All four
kinds now match numpy exactly, including ±inf rows, and
`test_apple_gpu_scheduled_reduce_propagates_nan` locks it on device — asserting
both the numeric equality and that an all-NaN row does not become `-inf`. The
sibling queues record this as not-applicable-with-reason (no other kernel
consumer), with the caution that CUDA `fmaxf` and x86 `maxps` suppress NaN the
same way, so a future port must propagate explicitly.

#### Operational note that cost real debugging time

After rebuilding `libTesseraAppleRuntime.dylib`, ~20 device tests failed with
`Apple runtime is missing <symbol>` even though `nm` showed the symbol present.
The runtime **publishes the dylib into a temp cache**
(`$TMPDIR/tessera_apple_gpu_runtime/libtessera_apple_gpu_runtime.<n>.dylib`),
and that cache was stale until the next process republished it. It is not a
regression and not a build failure — but it is indistinguishable from one at
first glance. After rebuilding the dylib, run any Apple test once to force
republication before trusting a sweep.

## APPLE-ATTN-BWD-PERF-1 — the backward gap was an unimplemented workspace stage

**Closed 2026-08-16.** The `~120x forward` finding recorded in the E2E-REAL-5B
slice is resolved: attention backward went from **195 ms to 6.0 ms** (non-causal)
and **389 ms to 10.9 ms** (causal) at the benchmark shape, ~32-36x, with the
numerical result unchanged (analytic-VJP max error 3.4e-06 before and after) and
bit-identical repeats preserved.

### The cause was a declaration with no consumer

`plan_attention_backward_workspace` already declares a **`row_prepass`** stage
producing `row_lse` and `row_delta`, consumed by `dkdv_split` and `dq`. Apple
declared that workspace and never used it: every kernel recomputed the row
statistics inline via `bwd_query_stats`. That cost twice — the O(Sk*D) statistics
were computed once in the dQ pass and again in the dK/dV pass — but the
expensive consequence was structural. The statistics are a reduction over the
whole key axis, so a kernel that recomputes them inline **must own a whole query
stream**, which pinned the dK/dV split kernel at one thread per (partial, KV
batch): for `B1 Hq4 Hkv2 S64` that is **4 threads** for the entire dK/dV
computation, on a GPU with thousands of lanes. The dQ kernel, which owns one
query row, already ran 256.

This is Decision #29 in its costly form: the declaration read as a closed
contract in review while carrying nothing, and the missing consumer was worth a
30x.

### What changed

- A `flash_attn_bwd_row_prepass_f32` kernel computes `row_lse = m + log(l)` and
  `row_delta` once per query row. A fully masked row stores `+INFINITY`, so
  `exp(score - lse)` is exactly 0 and the old early-out becomes branchless.
- `flash_attn_bwd_split_stream_f32` is now **key-parallel**: one thread per
  (partial, KV batch, key row), raising parallelism from `2 * kv_outer` to
  `2 * kv_outer * Sk` — 4 threads to 256 at the benchmark shape. Each key row is
  written by exactly one thread, so the route stays single-owner deterministic:
  no atomics, no zeroing pass, bit-identical repeats.
- **The declared schedule did not change.** Still a two-way split over the query
  stream (`split_count = 2`) reduced in ascending partial order
  (`reduction_order = (0, 1)`). Only the thread mapping changed, which is
  architecture-owned.

### The determinism/speed tradeoff recorded last slice is now withdrawn

The E2E-REAL-5B entry recorded atomic 65 ms vs split 193 ms and argued the split
route was worth ~3x for its determinism. That tradeoff no longer exists — the
deterministic route is now the fastest by a wide margin, and the margin grows
with size:

| shape | atomic (nondeterministic) | split (deterministic) | split speedup |
|---|---|---|---|
| `B1 Hq4/Hkv2 S64 D64` | 65.9 ms | **6.0 ms** | 11.0x |
| `B1 Hq8/Hkv8 S128 D64` | 169.3 ms | **10.0 ms** | 17.0x |
| `B2 Hq8/Hkv2 S256 D64` | 545.1 ms | **32.5 ms** | 16.8x |

All three routes still agree numerically (max pairwise difference 1.9e-06, f32
accumulation order). Backward is now ~3.6x forward rather than ~120x, which is
the ordinary range for attention backward.

### Deliberately not changed

The `serial` (route 0) and `atomic` (route 1) candidates still recompute their
statistics inline and keep their old thread mappings. They are alternative
candidates, not the contract route, and the contract route is now the fastest on
every measured shape; rewriting them would widen the blast radius for no
selection benefit. The prepass now runs on their dispatches too, and the cost is
noise (serial 373 -> 379 ms, atomic 65 -> 66 ms), which is measured rather than
assumed.

Evidence: the four exact-device analytic-VJP configurations (GQA/MHA/MQA/causal)
still match an independently derived float64 VJP to ~1e-6 on all three
gradients, repeats remain bit-identical, and 164 tests pass across the Apple
scheduled, e2e-spine and attention suites. Timing above is single-process
characterization, not a selector corpus.

## APPLE-NORM-VJP-1 — Apple joins the native-VJP boundary (E2E-REAL-6 / WS-2)

**Landed 2026-08-16.** Apple GPU is now a registered Target consumer in the
native-VJP registry for all three normalization ops (`rmsnorm`, `rmsnorm_safe`,
`layer_norm`), with exact-device proof. This is the Apple follow-up the
`E2E-REAL-6-NATIVE-VJP-2026-08-14` sync recorded as required.

### The scope was not what "registry wiring" implied

Investigation first, per the plan: Apple had **no** normalization-backward ABI
(zero `*_bwd` / `*_vjp` norm symbols), no `_execute_apple_compiled_norm_backward`
executor, and the synthesizer's `NormChainRegion` is forward-only. Adding an
`apple_gpu` entry pointing at nothing would have been precisely the unconsumed
declaration Decision #29 rejects — the same defect that cost a 30x in
APPLE-ATTN-BWD-PERF-1. The registry entry required writing the kernel first.

### What landed

- **Two MSL passes.** A row pass (one thread per row) computes `dX` and
  publishes per-row `mean`/`inv`; a column pass (one thread per channel)
  reduces `dGamma`/`dBeta` over rows in **fixed ascending order**, so the affine
  gradients are deterministic and repeats are bit-identical. The column pass
  rebuilds `y_hat` from the published statistics rather than re-reducing the row.
- **One row routine serves both norms.** `mean` is 0 for RMSNorm, so
  `is_layer_norm` selects only whether the row is centred — and RMSNorm
  correspondingly drops the `sum(dy)` term, which is the asymmetry easiest to get
  wrong. The non-affine case passes a unit scale vector rather than branching, so
  the two paths cannot drift apart.
- **Status-returning ABI**, so a numerically identical CPU fallback cannot be
  mistaken for GPU execution (APPLE-PLACEMENT-ABI-1); a zero status raises.
- Registered in `native_vjp_plugins` (`apple.metal_normalization`, evidence
  target `apple7`), in the runtime executor table, and as two
  `execution_matrix` rows behind one `apple_gpu_norm_bwd_compiled` executor.

### Evidence

`tests/unit/test_apple_normalization_vjp.py` — 12 tests. Every registered op,
affine, across rank-2, **rank-3**, and a **ragged width (13)**, each against an
independently derived float64 VJP; all gradients match to ~1e-6 with
`native_gpu` placement asserted. Affine gradients are proven bit-identical
across repeats.

**A test bug the rank-3 case caught.** The first reference computed `dGamma`
with `.sum(0)`, which reduces only the first axis. That is correct at rank 2 and
silently wrong at rank 3, where the kernel reduces over `prod(shape[:-1])` rows.
The kernel was right and the oracle was wrong — which is the argument for
carrying a rank-3 and a ragged shape rather than a tidy power-of-two pair.

Still open for this family: f16/bf16 storage (the ABI is f32-only today) and a
paired performance corpus; neither is required for the registry claim, which is
a correctness/consumer claim.

## APPLE-LAUNCH-OVERHEAD-1 — the integrity check cost 25x the computation

**Landed 2026-08-16.** Following the APPLE-DEVICE-EVENT-1 finding that these
lanes are host-overhead dominated (softmax: ~23 us of GPU work inside ~1014 us
of wall time), the launch path was profiled rather than guessed at. Row softmax
`64x256` end to end: **1368 us -> 478 us (2.9x)**.

### Where the time was

| stage | per launch |
|---|---|
| `_split_native_arguments` | 11 us |
| `descriptor.validate_invocation` | 2 us |
| **dylib SHA-256 + file read** | **~890 us** |
| native ABI call (incl. submit/sync) | ~478 us |

`_submit_apple_gpu_native` re-read the whole ~1 MB Apple dylib from disk and
SHA-256'd it **on every launch** to confirm the loaded runtime still matched the
compiler-produced image. The check is genuinely load-bearing — it is what stops a
stale or substituted dylib masquerading as the image a descriptor was built
against, which this backend gets wrong easily (a rebuild republishes under a new
cache path; see the operational note under APPLE-DEVICE-EVENT-1). But it cost
roughly 25x the GPU work it was guarding.

### What changed, and what it trades

The digest is memoized on the file's `(path, mtime_ns, size)` identity, so a
launch pays one `stat` (~26 us including path resolution) instead of a 1 MB
re-hash. Content is still byte-verified whenever the file identity changes — a
rebuild, a republish, any size or mtime change re-hashes. **What is given up:**
an in-place edit preserving both mtime and size would no longer be caught. That
is a deliberate swap rather than the staleness this guard exists to catch, and it
is the same assumption every mtime-based build cache already makes. Stated here
rather than left implicit.

### A near-miss worth recording

The first version of the cache called `_runtime_library_path()` at module scope,
where it is not bound. `launch` swallowed the resulting `NameError` into
`ok: False`, and the change appeared to deliver a **62x speedup** — which was
the cost of failing fast with an all-zero output. It was caught only because the
result was checked against its oracle rather than timed alone. **A speed
assertion without a correctness assertion beside it can report a broken path as
a win**; `test_launch_still_produces_a_correct_native_result` now pins that,
repeating the launch so the cached path and not just the first uncached one is
proven.

### What remains, and why it is not in this slice

The residual ~478 us is inside the native ABI: command-buffer creation,
buffer acquire/copy, submission, and the completion wait — against ~23 us of GPU
work. That is Metal submission latency for a single small dispatch, and the
runtime already owns the mechanism that addresses it (the encode-session API,
`ts_enc_commit_wait` and the `*_dev_*_enc` symbols, which batch several ops into
one command buffer). Routing the descriptor path through session batching is a
design change to the dispatch model with its own correctness surface, so it is
recorded as the next candidate rather than attempted here.

Regression: `tests/unit/test_apple_runtime_identity_cache.py` (4 tests) pins that
the guard still accepts the real digest, still rejects a wrong one *on the cached
path*, re-hashes when file identity changes, stays bounded, and that a launch
remains correct and `native_gpu` across repeats. 198 tests pass across the Apple
scheduled, e2e-spine, fleet and MLA suites.

## APPLE-ROWS-28-31-ASSESSMENT-2026-08-16 — shared-contract adoption, checked in source

Rows 28-31 were opened as contract-adoption items and had sat `active` since
2026-07-27. Each is now assessed against source rather than left open by default.
Two are satisfied by work that has since landed; two are not, and are narrowed
rather than closed.

### Row 28 — APPLE-ATTN-BWD-2: **closed**

The row asked Apple to consume the shared tensor-valued backward phase loops and
map the dQ / split-dK-dV / fixed-reduction contract to a Metal-owned package.
E2E-REAL-5B does exactly that: `apple_native.package_scheduled_attention_backward`
consumes the shared `ScheduledAttentionBackwardArtifact`, **verifies**
`split_count == 2` and `reduction_order == (0, 1)` rather than assuming them, and
selects the MSL split route because it is the only faithful mapping of an
ascending fixed-order reduction. Four exact-device configurations match an
independent float64 VJP and repeats are bit-identical. The AMD schedule, HSACO
and HIP workspace did not transfer, as the row required.

### Row 29 — APPLE-ATTN-BWD-3: **closed**

The row asked Apple to satisfy the shared backward contracts or record the
divergence. Apple declares its own `apple7_recompute` LSE identity, and that
declaration is **enforced by the ODS verifier** on both `schedule.attention` and
`schedule.attention_backward` — Apple cannot silently inherit x86's `save_lse` or
the gfx1151 threshold. The row's condition for retaining recompute ("until an
exact Metal package and benchmark justify a saved checkpoint") is now materially
better informed: APPLE-ATTN-BWD-PERF-1 shows the row statistics cost is real, and
answers it by computing them **once per launch** in the declared `row_prepass`
rather than by saving them across the forward/backward boundary. A saved
checkpoint remains unjustified, now on measured grounds rather than absent ones.

### Row 30 — APPLE-ATTN-MODIFIERS-1: **active, narrowed**

Apple's envelope expresses **causal, softcap, additive bias, and MHA/GQA/MQA**,
and rejects the remainder closed (live window, dropout, `D > 256`, non-whole GQA
group, mismatched head/value dim). So the row's substance — validate what is
expressed, reject the rest rather than silently narrowing — is met for four of
five modifiers. Two things keep it open, both specific:

1. **Windows are excluded, not expressed.** The MSL non-causal window is a
   *symmetric half-window* (`window_size/2` per side), which is not the shared
   `window_left`/`window_right` semantics; admitting it would compute a different
   mask than requested. Expressing windows needs its own kernel contract and
   oracle.
2. **Rejections name their reason in the message, but are not registered
   diagnostics.** Decision #21 asks for a stable named diagnostic per refused
   lowering. These are `ValueError`s with specific text, which is honest but not
   the registered form the sibling passes use.

### Row 31 — APPLE-STATEFUL-TRANSPORT-1: **active, unchanged**

No Apple consumer of the generalized target-keyed resident ABI schema exists —
searched, not assumed. Apple retains its session-private ReplaySSM ring with
flush/rollback, ordered submission and drain-before-release (APPLE-REPLAY-1,
closed), and APPLE-PIPE-1 already rejects name-based `#tile.buffer_ref`, so Apple
is aligned with the retirement half of the shared change. The open half — Metal
threadgroup scheduling against the generalized schema, including MoE
launch-workspace ownership and optional rank/device topology binding — is
untouched by any work to date and stays open with its original scope.

## APPLE-NORM-VJP-2 — f16/bf16 storage for the normalization VJP

**Landed 2026-08-16.** Closes the storage gap APPLE-NORM-VJP-1 named: the Apple
normalization VJP was f32-only. It now accepts **f32, f16 and bf16** storage for
`rmsnorm`, `rmsnorm_safe` and `layer_norm`, affine and non-affine.

### Policy: accumulate f32, store at the operand's dtype

Not an upcast. Operands and **gradients** are two-byte for f16/bf16, while every
accumulation inside the kernel stays f32 — the project's reduced-precision rule
(`REDUCED-PRECISION-COMPUTE-2026-08-03`). An f16 graph therefore keeps f16
gradients rather than acquiring f32 ones the caller must cast back.

Implementation reuses the substitution idiom `flash_attn_bwd_source` already
uses: one MSL template with a `%%NORM_STORAGE%%` declaration supplying
`norm_storage_t` plus `norm_load`/`norm_store`, rather than three copies of the
kernel. bf16 stores **round-to-nearest-even**
(`(bits + 0x7fff + ((bits >> 16) & 1)) >> 16`), matching the convention already
in this runtime — truncating would bias every gradient in one direction.

### The measured error says the accumulator is real

Relative error against a float64 analytic VJP **fed the same rounded inputs the
kernel received** (comparing against full-precision inputs would charge the
kernel for the caller's own input rounding):

| storage | observed | ~ulp of that format |
|---|---|---|
| f32 | 7e-08 | — |
| f16 | 3.2e-04 | ~0.65 ulp |
| bf16 | 2.1e-03 | ~0.54 ulp |

Both low-precision formats land at roughly **half an ulp of their own storage
epsilon** — i.e. the error is one final rounding on store, not accumulation. A
kernel that accumulated in the storage format would sit near `sqrt(cols)` ulps
instead, and `test_low_precision_is_far_better_than_low_precision_accumulation`
asserts against that level specifically, so a future change that quietly dropped
the f32 accumulator is caught rather than absorbed by a loose tolerance.

Tolerances in the suite are **derived from each format's epsilon**, not picked.

### Evidence

`tests/unit/test_apple_normalization_vjp_lowp.py` (6 tests) proves both norms in
both low-precision formats execute with `native_gpu` placement, return gradients
**in the operand dtype** (asserted, not assumed), and stay inside 8 ulps. 112
tests pass across the Apple norm-VJP, identity-cache, e2e-spine and native-VJP
suites. The non-Darwin stub gained the four new entry points returning 0, so an
off-Darwin build still cannot look like a Metal dispatch.

**Worktree note:** this was developed in an isolated git worktree while another
agent worked the shared tree. Five e2e-spine failures during the sweep were
traced to `TESSERA_OPT` pointing at a `tessera-opt` the worktree had never built —
an environment artifact, not a regression; the same tests passed against main
throughout, and passed here once the worktree's compiler was built.

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

**Apple-specific note.** `Accelerate`/`LAPACK` already backs the numpy
reference here, so an "Apple lane" for this family would mean routing through
the existing CPU shim rather than writing MSL — the GPU question is separate and
not opened by this PR.

## Cross-backend sync `NVIDIA-BARRIER-AT-BIRTH-2026-08-21`

**Not applicable to Apple physical lowering.** The shared role-bearing Tile
barrier birth contract was exercised by NVIDIA WarpSpecialization/TMA. Apple
does not consume TMA/mbarrier or CUDA schedule mechanics; no Metal schedule or
device-evidence state changes.

## Cross-backend sync `TARGET-IR-CONFORMANCE-2026-08-21`

**NVIDIA-only conformance closure.** NVIDIA tightened its Target IR value
envelope and exercised its registered dialect. Apple Target IR, Metal ABI, and
exact-device evidence are unchanged.


## Cross-backend sync `NVIDIA-SPECTRAL-PHILOX-JVP-2026-08-22`

**Outcome: not applicable; NVIDIA-only physical packages.** The shared spectral
and compiler-autodiff semantics are unchanged. CUDA added its own cuFFT
C2C/R2C/C2R ABI, compound consumers, spectral VJP executor, and seeded Philox
JVP child binding. Apple retains its architecture-owned FFT/RNG implementations
and evidence; no Metal schedule or device state transfers.


Cross-backend sync `NUMPOL-CARRIER-1-SCHEMA-AND-REDUCTION-2026-08-25` — **the
policy gets a schema and the reduction family carries its accumulator;
Apple outcome: shared contract assessed; no Metal change.** Two measured defects closed, both shared,
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

Portable Graph-level contracts. Zen 5 execution transfers no Metal or Apple-device claim; a Metal consumer of the carried accumulator must bind its own device proof.

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

**Outcome for this backend: `not applicable` — Apple has no delta-family
backward lane to be wrong.** Apple declares three delta *forward* runtime ops
(`tessera.gated_deltanet`, `tessera.kimi_delta_attention`,
`tessera.modified_delta_attention` — `apple_runtime_ops.inc:101-104`) and no
backward or VJP symbol for any of them; `apple_gpu_runtime.mm` has no
`*_deltanet_backward` export, and the bounded `erase=True` forward is itself a
documented numpy fallback ("`modified` (Kimi bounded) variant has no fused
kernel"). The defect lives strictly in the reverse correction term, so no Metal
kernel, MSL source, or Apple schedule is touched, and none is implied.

**What an Apple lane would have to match if one is ever built.** Divide the
correction by `n`, never by `max(n, 1)` — and note that the clamp is not a
defensive guard being removed: it is wrong in the *ordinary* regime
(L2-normalised keys give `n < 1` almost always) and unnecessary in the singular
one, since the `O(n²)` numerator makes the quotient vanish at `n → 0` and the
`norm > 0` select covers `n == 0` exactly. Bind Apple's own M-series device
proof; the gfx1151 and AVX-512 results under this key transfer nothing.

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

**Outcome for this backend: `follow-up required` — Apple believes a device
timestamp with no independent witness, which is the gap this key exists to
close.** `ts_record_tile_gpu_elapsed` and `ts_record_dispatch_gpu_elapsed`
(`apple_gpu_runtime.mm`) take `cb.kernelStartTime`/`kernelEndTime`, fall back to
`GPUStartTime`/`GPUEndTime`, and validate only `end >= start && end > 0.0`.
That is a sanity check, not a cross-check: it rejects a clock that is obviously
broken and accepts any value that is merely wrong.

**The drain half is genuinely not applicable here, and for a real reason** —
Metal reports a completed command buffer's own kernel interval rather than the
span between two counters the caller places, so there is no queue for a start
marker to sit behind. That makes the ordering constraint above moot for Apple
and leaves the *witness* half fully owed: a `CACurrentMediaTime()` bracket
around the committed-and-waited region, band-checked the same way, plus a
recorded timing source so a caller can tell a device number from a fallback
(both NVIDIA and ROCm now expose one; Apple's `g_last_dispatch_timing_source`
distinguishes *which Metal timestamp* was used, not whether it was believed).

**Owed on the M1 Max.** No Apple code changed here and no Metal evidence is
claimed; the sm_120 numbers above say nothing about Apple's timestamps.

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

**Outcome for this backend: `follow-up required`, and it compounds with
`NVIDIA-TIMER-DRAIN-2026-08-31`.** The contract applies unchanged; no Apple row
declares a separation yet.

**The two open Apple items are the same gap seen from both ends.** That key
recorded that `ts_record_tile_gpu_elapsed` believes `cb.kernelStartTime`/
`kernelEndTime` on an `end >= start && end > 0` check alone — no independent
witness. This key adds that a verdict needs a *noise floor*. A timer with
neither a witness nor a dispersion produces a number that is both unvalidated
and unqualified, and the arbiter will now record `separated: True` off it
whenever two Apple lanes differ at all, because a single unvalidated sample has
a spread of zero by construction.

**So the ordering matters here specifically:** `relative_spread` returning
`0.0` for a single sample is right (there is no dispersion in one number) but it
means a backend that reports one timestamp per candidate gets a free
separation. Apple should land the repeated-measurement half — the equivalent of
`device_repeats` reaching the Metal path — *with* the wall witness, not after
it, or the separation field will read as validated on evidence that is neither.

**Owed on the M1 Max.** No Apple code changed here and no Metal evidence is
claimed.

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

**Outcome for this backend: `parity validated` — owed here, done here, on the
M1 Max.** `tessera_apple_gpu_last_dispatch_wall_time_ns` is captured at **all
six** sites that publish a device time, via an RAII `DispatchWallWitness` so the
default is "witnessed" and a new path must opt out rather than silently forget.

**Review caught that the first version covered two of the six** — the four it
missed reported a null wall, which the rule reads as "no witness" and passes
through unchecked. The lesson is the enumeration: closing over the *commit*
sites is the wrong set; the right one is the sites that call a telemetry
**recorder**, because those are what publish a number. A related trap the
containment bound then caught by itself: two witnesses were live on the
MPSGraph path and the narrower destructed last, overwriting the wider span, so
a cold dispatch reported a device interval **9.18× its own wall**.

`device_time_ns` now carries the band-checked value, with the raw one under
`device_time_raw_ns`. Publishing the verdict in a new key instead would have
left every consumer using the rejected number exactly as before — the check
would exist and change nothing. It is captured in
`commit_and_wait_with_timeout`, `MPSGraphTimingBracket`,
`commit_mpsgraph_and_wait_with_timeout`, `mtl4_encode_and_wait`, and both
resident-session paths;
`accept_apple_device_ns` owns the band and the measurement behind it. The
export is in `_SENTINEL_SYMBOLS`, because a null wall *disables* the check
rather than failing it — a stale dylib would otherwise restore the unwitnessed
behaviour with every test still passing.

**Correcting this plan's own `AUTOTUNE-SEPARATION-2026-08-31` entry.** That
entry said the repeated-measurement half must land *with* the witness, or a
single unvalidated timestamp would earn a free separation (spread of zero by
construction). The risk is real but **not yet reachable**: `emit/apple_msl.py`
declares **no `measure_device_latency` at all**, so no Apple candidate enters
the device-timed arbiter path and there is no verdict to earn. The two are
therefore *ordered*, not simultaneous — and the ordering is the other way
round: the witness first, then a device timer, then repetition. Wiring an Apple
`measure_device_latency` before the repeat count exists is what would create
the free separation, so that is the pairing to enforce.

**Still open here, and it is the stronger form of the floor.** Apple exposes
**two independent device clocks** — `metal4_timestamp_heap` (MTL4 counter heap,
`writeTimestampWithGranularity:`) and `metal_kernel_interval`
(`cb.kernelStartTime`/`kernelEndTime`). Bounding one against the other beats
bounding either against a host clock measuring a different span, which is the
conclusion ROCm already reached (`_select_rocm_latency_ms` lower-bounds its
device clock against the *event* clock, not the wall). Today the two overwrite
each other in `g_last_dispatch_device_time_ns`; capturing both per dispatch is
the follow-up.

**Hardware note worth keeping.** This M1 Max reports **no dispatch-boundary
counter sampling** (`MTLCounterSamplingPointAtDispatchBoundary` absent, so
`tessera_apple_gpu_tile_counter_sampling_supported()` returns 0) but **does**
support MTL4 counter heaps. The legacy `MTLCounterSampleBuffer` route is not
available on this hardware; the MTL4 heap is the only independent device clock
here, which is why the follow-up above depends on Metal 4 rather than being
implementable on the classic API.

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

**Outcome for this backend: `parity validated` — defect found and fixed here,
on the M1 Max.** `ts_gpu_interval` prefers `GPUStartTime`/`GPUEndTime` and
forces the properties to publish; the undocumented pair remains only as a
labelled fallback (`timing_source` distinguishes them, so a consumer that cares
about fidelity can check). On the resident-attention lane, KV 19 → 1216 now
reports **864,000 → 1,505,333 ns (1.74×)** against a wall of 1.53×, where
before it reported **58,166 → 28,875 ns (0.50×)** against a wall of 1.55×.

**What this corrects in `APPLE-TIMER-WITNESS-2026-08-31`.** That entry justified
the one-sided band partly on resident-session ratios of 0.037–0.101 being "a
25 µs kernel inside a 265 µs window, entirely correct". **They were not
correct — they were this defect.** The one-sided *conclusion* stands (a wall
floor still cannot separate a small kernel from an under-read), but that stated
reason is withdrawn.

**Recorded so it is not rediscovered.** This M1 Max supports
`MTLCounterSamplingPointAtStageBoundary` but **not**
`...AtDispatchBoundary` — so `tessera_apple_gpu_tile_counter_sampling_supported()`,
which checks the dispatch point, reports 0 and makes the classic counter route
look unavailable when a usable one exists. And the two counter domains differ:
`sampleTimestamps:gpuTimestamp:` shows the classic sample-buffer domain is
**1 GHz (ticks are already nanoseconds)**, while `queryTimestampFrequency`
returns **24 MHz** and belongs to the MTL4 heap. Converting one with the other's
frequency scales by 41.7×.

**Still open here.** The stage-boundary clock is *verified good* on this
hardware but is not wired in — it needs a `MTLComputePassDescriptor` on each
encoder, and none of the 120 owned encoders use one today. It is the right
second clock if a cross-clock bound is ever wanted; `GPUStartTime` agreeing with
it to the nanosecond is what makes the cheaper fix defensible in the meantime.

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

**Outcome for this backend: `parity validated` — the hole was mine and is
closed here.** `record_apple_packet.py` now refuses to seal when a fingerprinted
source differs from HEAD, and the packet is re-recorded against its own commit
(`tested_commit` 8dd79bbc, runtime at that commit hashing to the packet's
`2653a004…`).

The check is scoped to the files the packet actually fingerprints rather than
the whole tree: a dirty README does not invalidate a device measurement, a dirty
`apple_gpu_runtime.mm` does, because the fingerprint is taken from working-tree
bytes. Widening it to the whole tree would make the recorder unusable during
ordinary work and would train people to bypass it.

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

**Outcome for this backend: `parity validated` — found here, on the M1 Max.**
`_apple_gpu_dispatch_spectral` now calls the shared check. The corrected test
passes a rank-matched `(1, 5)` kernel, and its comparison now actually runs:
Apple dispatch vs host reference agree to **1.2e-06**, and both match
`np.convolve` full-mode ground truth to **2.4e-07**.

Two tests added: one asserting the refusal (so a lane cannot quietly go
permissive again) and one pinning the result against **numpy** rather than the
reference — because "both Tessera lanes agree" and "both Tessera lanes are
wrong the same way" are indistinguishable without an outside oracle, and this
op's two lanes agreeing is exactly what hid the defect.

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

**Outcome for this backend: `not applicable` to the code, but Apple is the host
that EXHIBITED it and the one that should keep the lesson.** No Apple code
changed; the 70.6 ms was ROCm's build path failing on an Apple host.

**The transferable part is the diagnosis, not the fix.** This was found by
asking why a "flaky" test failed — and the answer was a 677× overhead that a
hand-picked constant had been sized to accept. Apple has the fleet's most
timing-sensitive assertions after this session's work
(`APPLE-TIMER-WITNESS`, `APPLE-DEVICE-CLOCK`), and every one of them carries a
constant. Each should be able to answer the question this key answers: **would
it fail on the defect it exists to catch?** The 2.0 ms floor can (71.3 vs 2.0);
the 75.0 it replaced could not.

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

**Outcome for this backend: `parity validated` — done here on the M1 Max.**
Four invariants are pinned and mutation-verified: no unbounded wait in the read
path, the semaphore wait is bounded (never `DISPATCH_TIME_FOREVER`), the direct
read precedes the `kernelStartTime` fallback, and the slot is not
`thread_local`.

**Measured cost of the handler: +5.9% median dispatch latency** (1.3647 →
1.4446 ms at 512²), with *lower* variance (sd 12.78% → 11.41%). That is a real
price for a bounded, correct timestamp path, and it is stated rather than
buried.

**A gate defect found on the way.** `test_only_documented_waituntilcompleted_sites_remain`
greps raw source text and was tallying a mention of `[cb waitUntilCompleted]`
**inside an explanatory comment** as a third call site — it expected 3, found
3, and passed on a wrong basis. It now strips comments and expects 2, the real
count. A text-grep gate that counts prose is the same hollow-check pattern this
plan has been recording all session, in the gate itself.

**Still open here:** the packet recorder's 4% stability gate refused twice
(11.1%, 20.0%) before sealing. Verified by control that this is the host, not
the change — WindowServer at 46% and a desktop GPU process at 27% contend for
the same GPU. Any Apple exact-device evidence recorded on this machine is
subject to that, and the honest response is to re-measure, never to widen the
gate.

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

**Outcome for this backend: `not applicable` to the fix, but Apple is where all
three surfaced and the one with the most to lose from the pattern.** No Apple
test changed here.

**Apple's exposure is structural, not incidental.** Its execution tests are
gated three different ways — a file-level allowlist in `conftest`
(`_APPLE_GPU_EXECUTION_TESTS`), per-test `@pytest.mark.hardware_apple_gpu` in
the **154 mixed-mode files** that are deliberately not on it, and inline
`require_apple_metal()` calls. Three mechanisms for one property is three
places to forget one, and this session forgot the second (`#666`, caught by
CI's Linux lane).

A static gate for that was considered and **rejected on evidence**: scanning
all 154 files yields 97 candidate tests, and a static check cannot separate
"executes and raises" from "falls back to numpy". CI's Linux lane already
distinguishes them dynamically and is green on all 97, so a static gate would
be ~97 false positives. The dynamic gate is the right one and it works.

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

**Outcome for this backend: `not applicable` — Apple has no rows in this
corpus, and cannot have any yet.** `emit/apple_msl.py` declares no
`measure_device_latency`, so no Apple candidate enters the device-timed
arbiter at all.

**That is the ordering already recorded under `APPLE-TIMER-WITNESS`, and this
run supplies the missing motivation for its last step.** Witness → device timer
→ repetition. The NVIDIA measurement shows why the third step is not optional:
with `device_repeats=3` the reported noise floor is **2.5× the truth**
(48.31% against a 19.34% thirty-sample estimate). An Apple lane that landed a
device timer and left repetition at one sample would report a spread of
**zero** — and earn `separated: True` on every row it touched, automatically.
The free-separation risk is not theoretical; it is what a single sample
mechanically produces.

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

**Outcome for this backend: `not applicable` — no rows in this corpus.** The
tightening cannot affect a backend with nothing recorded.

**What it changes is the bar for entry.** A row that ranks two or more
candidates must now arrive with a separation verdict or it will never be
served. For a backend building its first device-timed rows, that makes the
sample-count decision load-bearing at the point of writing rather than
something to fix later: a single measurement yields a spread of zero and earns
`separated: True` automatically — which is exactly the defect
`record_paged_kv_corpus` had, discovered only because its rows were re-raced.

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

**Outcome for this backend: `not applicable` — Apple's structure already
threads the shape, for an architecture-specific reason.** Apple synthesizes MSL
per shape, so its eligibility predicates take the dimension as an explicit
argument rather than reading it off a region: `coopmat_reduce_eligible(region,
N)` gates on `N % 8 == 0` (the simdgroup 8×8 stores) and the threadgroup-memory
cap, both at synthesis time. There is no Apple `Candidate` that accepts a
region and then declines a shape inside `run`, which is the defect. The shared
seams still apply if one is added — `applies_to_inputs` and the `_measure`
backstop are backend-agnostic.

The one Apple-side gap is unchanged by this PR and still owed:
`measure_device_latency` has no Apple implementation, so Apple has no
device-timed lane to bias in the first place. When it lands it must land with
repeated measurement, or one sample earns `separated: True` free.

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

**The red lane that started this.**
`test_strict_retune_ledger_admits_on_its_exact_live_apple_host` had been
failing on this Mac and was reported as "pre-existing" more than once. It was
right to fail: the ledger was rejected on **four independent axes** —
`os_version` (26.5.2 → 26.6.2), `compiler_fingerprint`, `runtime_fingerprint`
(this session edited `apple_gpu_runtime.mm`), and `stale_evidence` (expired
2026-08-20, twelve days before). The gate was doing its job; the evidence was
genuinely dead.

**Re-measured, not re-stamped**, per the standing rule for this artifact:
`benchmark_legacy_retune.py --profile extended` (2 runs × 24 rows), then sealed
through the existing `seal_strict_route_ledger`. Zero rejections against the
live context; 16 decisions and 8 ineligible, **identical key sets to the
July ledger** — nothing lost, nothing invented.

*Near-miss worth recording:* the first re-run used `--profile core`, which
admitted cleanly with **8** decisions instead of 16. `rejected == ()` would
have read as success while silently halving the Apple route evidence — the
same shape as the corpus recorder that once produced a corpus with *less*
evidence than it started with. Diffing the decision key sets against the
committed ledger is what caught it, not the admission result.

**One real route change, and its cause is not the obvious one.**
`retune_mla_decode` end-to-end flips `explicit` → `absorbed` at both shapes.
`absorbed` was never slower: in July it was **54–57% faster with 100% paired
wins in both runs**, and was held back solely because its cross-run speedup
spread (5.97%) missed the ledger's own 5% stability cap by ~1 point. The re-run
measures 37–39% faster, again 100% paired wins, spread **2.3%** — so it
promotes. The flip is "the measurement became consistent enough to conclude",
not "the kernel got faster".

**Diagnostic, deliberately not a route claim** (`promotion_rules` sets
`absolute_time_drift_is_diagnostic_only: true`): absolute times moved the wrong
way. `absorbed` went 518µs → 836µs at the 128-token shape and 482µs → 557µs at
64, while `explicit` held (1.15× / 0.96×). The larger shape's regression is
outside this harness's own run drift; the smaller one is not. Cause is
confounded — OS 26.5.2 → 26.6.2, runtime `.mm` edits, and a new compiler
fingerprint all landed between the two measurements — and is **not attributed
here**. It is worth a dedicated look.

**The structural finding: `promotion_rules` was a declaration with no
consumer.** `aggregate_stable_route_reports` computes the thresholds, applies
them, and writes them into the ledger; `seal_strict_route_ledger` copies them
forward for audit; **twelve committed ledgers carry the block and nothing ever
read it back**. `load_strict_route_ledger` checks provenance exhaustively —
schema, scope, exact context, freshness, source-report digests, native
provenance, correctness, timing domain, device, duplicates — and the promotion
criteria not at all, so `status: "promote_candidate"` was self-certifying. A
row naming a route that lost every paired trial would have been admitted and
served. Decision #29, in the evidence layer rather than the IR.

`promotion_rule_violations()` now re-derives each promotion from the evidence
the ledger retained, and the **loader rejects a decision its own ledger's rules
refuse** — a production gate, not an audit script. Applied to all 12 committed
strict ledgers: **59 promotions checked, 0 violations**, so this confirms the
aggregator rather than accusing it. Mutation-verified against seven forged
rows (lost every trial / speedup under minimum / spread over maximum / no
numerical proof / no resource evidence / evidence deleted / rules block
removed) — each is caught, and a missing rules block fails **closed**
(`promotion_rules_incomplete`), because without a threshold there is nothing to
hold the promotion to and the honest verdict is "unverifiable", not "fine".

**Outcome for this backend: `parity validated` — measured on the Mac (M1 Max,
macOS 26.6.2), which is the only host that can produce this evidence.**

## APPLE-DISPATCH-WEDGE-1: a timed-out MPSGraph dispatch has no circuit breaker *(open, investigated 2026-09-01, M1 Max)*

**Trigger.** An Apple `-k apple` sweep sat for **70 minutes** at 0.0% CPU
(1:42 CPU time), stack-sampled entirely inside

```
mpsg_run_binary
 -> commit_mpsgraph_and_wait_with_timeout(ctx, mps_cb, metal_cb, 30000, "mpsgraph_binary", &timing)
   -> -[IOSurfaceSharedEvent waitUntilSignaledValue:timeoutMS:]  (in IOSurface) + 72
     -> iokit_user_client_trap  (in IOKit) + 8
```

**Correction, recorded because the first reading was wrong.** I initially wrote
that the function's namesake guarantee "doesn't hold" — that
`waitUntilSignaledValue:timeoutMS:` was ignoring its timeout. **That is false,
and it was checked rather than assumed.** A standalone probe
(`tools/apple_probes/mtl_shared_event_timeout_probe.mm`, built against the
on-machine SDK per Decision #27) measures the API honouring its deadline
precisely:

| case | timeout | returned | elapsed |
|---|---|---|---|
| never signalled | 250 ms | `NO` | 251.0 ms |
| never signalled | 1000 ms | `NO` | 1001.0 ms |
| never signalled | 3000 ms | `NO` | 3000.2 ms |
| CPU signal at 100 ms | 5000 ms | `YES` | 104.2 ms |
| GPU-encoded signal | 5000 ms | `YES` | 0.1 ms |
| **committed buffer, unreachable value** | 1000 ms | `NO` | **1001.0 ms** |

The last row is the exact shape of the failure — work committed, awaited value
never arriving — and it times out correctly.

**And the probe provably exercises the same code path**, which is the step that
makes the above admissible rather than merely suggestive: sampling the probe
mid-wait yields the identical frame *at the identical address* as the hung
process — `-[IOSurfaceSharedEvent waitUntilSignaledValue:timeoutMS:] + 72
[0x1988de184]` -> `iokit_user_client_trap + 8 [0x190604ae0]`. (`[dev
newSharedEvent]` returns `_MTLSharedEvent`, whose wait is implemented by
`IOSurfaceSharedEvent`; the two class names are one path, not two.)

**What is NOT established.** Whether the 70 minutes was *one* uninterruptible
wait — a driver wedge defeating the kernel-side deadline — or **~140 sequential
30-second timeouts** (70 min / 30 s = 140, a suspiciously round fit). A `sample`
aggregates by stack, so it cannot separate one 70-minute wait from 140
consecutive ones; and the process was killed before forward progress could be
checked. No GPU fault appeared in `log show --last 4h` and no hang report was
written, so a driver wedge has **no positive evidence** either. Both remain
open. It has not reproduced: the same sweep re-ran clean in **4:03**.

**What IS established, and is a defect under either hypothesis: there is no
circuit breaker.** Seven call sites, six at 30 000 ms and one at 60 000 ms. On
timeout the caller correctly falls back to the host recovery path — but nothing
records that the *device* is unusable, so every subsequent dispatch pays the
full timeout again. Worse, `commit_mpsgraph_and_wait_with_timeout` calls
`ts_clear_dispatch_telemetry()` **on entry**, so the previous timeout's evidence
is erased before the next attempt. `g_last_gpu_error_kind` is a `thread_local`
*last-error* for reporting, not accumulating state, and nothing consults it to
short-circuit. So a device that wedges early in a suite turns a 4-minute run
into an unbounded one, and by design leaves no accumulated trace of why.

30 s is a defensible production timeout and a poor test one; the missing piece
is not a smaller number but a state that says *stop asking*.

**Proposed fix (not yet made, and it carries a sequencing constraint).** A
process-wide sticky counter: after N consecutive dispatch timeouts, stop
attempting GPU dispatch, go straight to the host path, and emit one stable
diagnostic naming the op and the count (Decision #21 — never a silent no-op).
The telemetry clear must move so it cannot erase the signal it is counting.

**Sequencing.** `AppleRouteContext.runtime_fingerprint` is
`sha256(apple_gpu_runtime.mm)` (`apple_route_selector.py::_runtime_source_fingerprint`),
so **any edit to that file invalidates the strict route ledger** and puts
`test_strict_retune_ledger_admits_on_its_exact_live_apple_host` back to red.
The fix therefore has to land together with a benchmark re-run and re-seal
(`ROUTE-LEDGER-RULES-UNCONSUMED-2026-09-01`), not before or after it.
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

**Outcome for this backend: `parity validated`, no change needed.** Apple is
where this rule landed first (`ROUTE-LEDGER-RULES-UNCONSUMED-2026-09-01`): its
`promotion_rules` block is a real threshold set and `load_strict_route_ledger`
now refuses a decision those thresholds reject, in production rather than in a
test. The NVIDIA and ROCm work above is the same question asked of their
evidence artifacts; nothing here changes.

Worth noting for contrast: Apple is the only one of the three whose gate is
**machine-readable in the artifact itself**. NVIDIA carries a bare number
(`noise_fraction` / `noise_policy`) whose meaning lives only in the recorder,
and ROCm carries a sentence. When an evidence format is next designed, Apple's
shape is the one to copy.

## APPLE-DISPATCH-WEDGE-1: caller-side circuit breaker *(mitigated 2026-09-01; runtime-side fix still open)*

A device that stopped answering kept being asked. `commit_mpsgraph_and_wait_with_timeout`
waits 30 s (60 s at one of seven call sites) and, on expiry, reports the timeout
and returns — correctly, and **with no memory that it happened**, so every later
dispatch paid the full timeout again. An Apple sweep was observed stalled for
**70 minutes** where the healthy run takes 4. Nothing accumulated by design: the
runtime's `g_last_gpu_error_kind` is a `thread_local` *last*-error for
reporting, and the wait helper clears the dispatch telemetry **on entry**,
erasing the previous timeout before the next attempt.

**Mitigated at the caller layer, which is where the repeated cost is paid.**
`runtime._apple_gpu_run_checked` already arms and consumes that error channel
for eight Apple GPU lanes — including `_apple_gpu_dispatch_mpsgraph_binary`,
the lane that hung. After three consecutive **kind-1** (timeout) errors it stops
dispatching and goes straight to the host path, with a stable diagnostic naming
the op and the streak (Decision #21).

Design points that are decisions, not details:

* **Three, not one.** A single timeout can be a slow dispatch under load; three
  in a row is a device that stopped answering. Tripping on the first turns a
  hiccup into a process-wide GPU shutdown.
* **Kind 1 only.** Both `ts_set_last_gpu_error(1, …)` sites are the
  hung/timed-out path; kinds 2–4 are per-op failures (bad buffer, unsupported
  shape) and say nothing about whether the device answers. Counting those would
  open the breaker on a workload that merely uses an unsupported op a few times
  in a row — and an interleaved op-failure resets the streak, so it can neither
  mask a real one nor manufacture one.
* **The asymmetry is deliberate.** Tripping wrongly costs a host-computed
  result, which is *correct*, just slower. Not tripping costs unbounded wall
  time and produces no evidence of why.
* **No self-healing timer,** because there is no cheap probe for "is it
  answering again" that does not cost another full timeout.
  `reset_apple_gpu_dispatch_breaker()` is for a caller that knows better, and
  `TESSERA_APPLE_GPU_NO_DISPATCH_BREAKER` restores the old behaviour — a
  breaker nobody can turn off is a new way to lose a working device.

**Scope, stated honestly: this is a mitigation, not the complete fix.** It
covers the eight lanes routed through `_apple_gpu_run_checked`. Other MPSGraph
C-ABI entry points (`…_reduce_f32`, `…_argreduce_f32`, `…_scan_f32`,
`…_topk_f32`, `…_bsmm_*`) have not been verified to route through it, and
nothing outside Python is covered at all. The complete fix is runtime-side, in
`apple_gpu_runtime.mm`.

**And that is exactly why this landed at the caller layer first.**
`AppleRouteContext.runtime_fingerprint` is `sha256(apple_gpu_runtime.mm)`, so
editing that file invalidates the strict route ledger and forces a benchmark
re-run plus re-seal in the same change. This branch leaves the `.mm`
byte-identical (verified against `origin/main`), so the mitigation ships now
and the runtime-side fix stays sequenced behind
`ROUTE-LEDGER-RULES-UNCONSUMED-2026-09-01` without blocking on it.

**Still unresolved** (unchanged from the investigation above): whether the 70
minutes was one uninterruptible wait or ~140 sequential 30 s timeouts. The
breaker addresses the second directly and bounds the damage under either.
Host-free coverage: `tests/unit/test_apple_gpu_dispatch_breaker.py` (10 tests),
mutation-verified — breaker never opens, every kind counts as a timeout, and
success fails to reset the streak are each caught by a different test.

### Second instance: the device-resident paths never went through the breaker *(fixed 2026-09-03, M1 Max)*

**Trigger.** A `pytest tests/unit -q -m "not slow"` sweep on the Mac wedged
with `tessera_apple_gpu_gather_blocks_dev_f32` at the top of the stack. The
mitigation above covered only the numpy-in / numpy-out lanes that call
`_apple_gpu_run_checked`. Every `DeviceTensor` path — `gather_blocks_dev`,
`paged_latent_attention_dev`, `dense_latent_attention_dev`, `rowop_dev`,
`bmm_dev`, `mtl4_matmul2d_dev`, `ts_dev_cast`, the MTL4 MLP session's
`run_dev` — called its C symbol directly and returned `None` on any `rc != 1`.
Three consequences: an **open** breaker did not stop them asking the device
(one full timeout each); their timeouts did not count toward the streak; and a
reported timeout became a silent `None` with no diagnostic (Decision #21).

**Fix.** `runtime._apple_gpu_device_call_checked(op_name, call) -> bool` wraps
the raw `rc` call in `_apple_gpu_run_checked`, so the eight resident paths now
share the same streak, the same open-breaker short-circuit, and the same
named fallback log entry. An `rc == 0` with no error kind set is a C-side
validation decline and resets the streak, as on the numpy lanes.
`tests/unit/test_apple_gpu_dispatch_breaker.py` drives each path host-free
(open breaker never dispatches and frees the output; timeouts count; a
validation decline does not; success closes the streak; the fallback is named)
and adds an AST drift gate: any function that *calls* a name bound from a
`*_dev_sym()` / `*_dev_f32()` accessor or a `getattr` on a `..._dev` /
`ts_dev_cast` symbol must route it through the helper. The `.mm` is untouched,
so the route ledger fingerprint is unchanged and no re-seal was needed.

**Reproduction attempts — it did not reproduce, and that is recorded rather
than explained away.** All unsandboxed on the M1 Max, with the dylib newer
than the source (`libTesseraAppleRuntime.dylib` built 10:23, `.mm` last
modified 22:24 the previous day), with a hard external `kill -9` deadline
(macOS has no `timeout`; SIGALRM does not interrupt the IOKit trap):

| arm | result |
|---|---|
| `test_apple_gpu_resident_block_paged.py` alone, idle, x3 | 12/12, 0.39 s each |
| same file x3 while the whole Apple lane ran concurrently on the device | 12/12, 0.44-0.58 s |
| whole Apple GPU lane in one process (`tests/unit/test_apple_gpu*.py`), x2 before the fix | 1610 passed / 1 skipped, 77-79 s |
| same lane after the fix (+ resident/mtl4 files) | 1649 passed / 1 skipped, 78 s |

So the wedge that was observed depends on a condition these arms did not
recreate. The dylib's age *at the time of the wedge* is not known, and
`APPLE-MLPKG-HANG-1` above is the same shape with a stale dylib as the cause —
treat that as the leading hypothesis, not a finding. A "the tool sandbox
cannot reach Metal" hypothesis was also raised and is **unproven**: the
intended sandboxed control turned out not to be sandboxed at all (a `$HOME`
write succeeded), so there is no control either way. Do not cite either as
the root cause.

**Review found the fix was uniform where the runtime is not (2026-09-03).**
The first version routed all eight resident paths and implied they were now
equally covered. Tracing each entry point to its wait helper says otherwise,
and the three classes need different work:

| resident path | wait helper | bounded? | reports timeout? |
|---|---|---|---|
| `gather_blocks_dev` | `commit_mpsgraph_and_wait_with_timeout` | 30 s | yes, kind 1 |
| `paged_latent_attention_dev` | `commit_and_wait_with_timeout` | 30/60 s | yes, kind 1 |
| `dense_latent_attention_dev` | `commit_and_wait_with_timeout` | 30/60 s | yes, kind 1 |
| `ts_dev_cast` | `commit_and_wait_with_timeout` | 30/60 s | yes, kind 1 |
| `mtl4_matmul2d_dev` | `mtl4_encode_and_wait` | 10 s | **no** |
| MTL4 MLP session `run_dev` | `mtl4_encode_and_wait` | 10 s | **no** |
| `rowop_dev` | `[MPSGraph runWithMTLCommandQueue:]` | **no** | no |
| `bmm_dev` | `[MPSGraph runWithMTLCommandQueue:]` | **no** | no |

* **The MTL4 pair returns `false` after 10 s with the error channel
  untouched** (`mtl4_encode_and_wait`, `apple_gpu_runtime.mm:4348`; there is no
  `ts_set_last_gpu_error` anywhere in the MTL4 region). By return value alone
  that is indistinguishable from a shape decline, so the streak reset and no
  diagnostic was emitted — the exact wedge, inside the fix for it. Handled with
  `silent_failure_timeout_s`: a failure that reported nothing and took ≥ 5 s is
  counted as a timeout. Duration is the only signal the C side leaves, and the
  two classes differ by four orders of magnitude (a decline is microseconds).
  The proper fix is one line in the `.mm` and is deferred only because that
  file's hash is the route-ledger fingerprint.
* **`rowop_dev` and `bmm_dev` cannot be protected from Python at all.** They
  call MPSGraph's own `runWithMTLCommandQueue:`, which is **unbounded** — a
  device that stops answering never returns, so there is no repeated cost to
  cut and no fallback to take. One permanent hang is worse than N bounded
  timeouts, and it is invisible to `pytest --timeout` for the same reason
  `APPLE-MLPKG-HANG-1` was. **38 call sites in the runtime use that form.**
  Runtime-side fix, filed here, not attempted in this change.

**Class size, stated honestly.** With this change the breaker covers 9 numpy
lanes + 8 resident paths. An enumeration of `rc = <symbol>(...)` dispatches in
`runtime.py` finds **34 further Apple call sites** that call a C symbol
directly and are not routed: linalg (`chol`/`solve`/`tri` 2-D, batched
cholesky / tri-solve, `svd`), `random`, GQA x2, batched attention x2,
`cf_scan` / `cf_serial_draft` / `cf_while_generate`, Metal 4 `caps` /
`tensor_roundtrip` / `mtl4_scan` / `matmul_sg` / `matmul2d_{f16,bf16,epilogue}`
/ `mtl4_conv2d`, `msl_spec_accept`, the MLP session's host-input `run`, and
the seven `*_value_available` probes + seven `_dispatch_gpu_*` lanes
(PPO / EBM / Clifford). Every one reaches `commit_and_wait_with_timeout`, which
has **113 callers — 38 at 30 s and 72 at 60 s** (the "60 s at one of seven
call sites" above counted only the MPSGraph helper), so each can still re-ask
a device that stopped answering. Follow-up: route them through the same
helper; the AST gate covers the `_dev` class by construction and will not
catch these.

**The drift gate is per-dispatch, after review.** Its first form asked whether
a function containing a resident dispatch *also mentioned* the helper, which a
function with one wrapped and one bare dispatch satisfies — the exact way this
regresses. It now checks each call to a bound resident symbol for a
`_apple_gpu_device_call_checked` among its own AST ancestors, and a companion
test pins that the per-dispatch check catches the mixed case a function-wide
flag calls clean.

**Cross-backend assessment** is recorded in all four queues under
`DISPATCH-BREAKER-RESIDENT-2026-09-03`: NVIDIA and ROCm *not applicable* for
the breaker (both use blocking `*StreamSynchronize` with no timeout, so a wait
never returns to be retried) with a *follow-up* for having no bounded device
wait at all; x86 *not applicable* structurally (synchronous host calls, no
device).

**Corrected 2026-09-03 — that enumeration was a regex, and the count is wrong
in both directions.** See the third instance below: parsing `runtime.py` and
classifying each symbol against the `.mm` finds 85 dispatching functions, not
34 further ones, and several sites the regex named never reach a command
buffer. Read the count there, not here.

### Third instance: the direct `rc = sym(...)` sites *(fixed 2026-09-03, M1 Max)*

**The second instance closed with a count taken from a regex, and the count was
wrong in both directions.** It reported "34 further Apple call sites"; a
re-enumeration that parses `runtime.py` with `ast` and classifies each symbol
against `apple_gpu_runtime.mm` finds **85 functions** that dispatch a
`tessera_apple_gpu_*` symbol directly, of which the regex list named about a
third — and some of what it did name never reaches a command buffer at all.
Both errors came from the same place: a regex can see `rc = sym(...)`, but it
cannot see which `sym` that is or whether the C function behind it waits.

**What the enumeration does instead.** An accessor is any function that
`getattr`s an Apple symbol *and returns it*; a name bound from an accessor call
(with constant arguments resolved through it) and then called is a dispatch.
Each resolved symbol is then looked up in the `.mm`, and its body plus
everything it transitively calls is searched for a device wait — the two timed
choke points, MPSGraph's synchronous `runWithMTLCommandQueue`, and the Metal 4
shared-event wait. A symbol reaching none of those cannot hang, whatever the
Python looks like.

**Result: 62 of the 85 are now routed** (up from 17). Of the 23 that are not,
**20 are exempt by that classification, not by assertion** — the Metal 4 and
SIMD capability probes, the last-error channel itself, the memory statistics
and cache calls, the archive enable/flush, and the encode-only `_enc` entries
whose command buffer belongs to the caller's session. Three remain, each named
with its reason in the gate's own `KNOWN_UNROUTED` table:
`_apple_gpu_raw_handle` (its symbol is a parameter, so it reads as dynamic and
fails closed; both callers pass a pointer getter), and the `gumbel` / `rowop`
encode-session methods (their shared C helper contains both an encode and a run
branch, and the `_enc` entry always takes the encode one).

**Closed 2026-09-03 — the allowlist is now empty, and the three were
reclassified rather than routed.** Both reasons above describe a *limit of the
classifier*, not a property of the code, so each became a rule:

* **A wait can belong to the call rather than to the helper.**
  `encode_or_run_rowop_dev` and its gumbel sibling take a command buffer and
  branch on it — `if (cb) encodeToCommandBuffer … else runWithMTLCommandQueue`
  — so the synchronous entry point (passing `nil`) waits and the `_enc` entry
  (passing the session's buffer) does not. Classifying the helper either way
  is wrong for half its callers, so the classifier propagates the argument,
  and the fixture pins the split on the one pair that shares a helper. The
  guard must be a **command-buffer** parameter: an earlier form accepted any
  parameter and wrongly exempted three int4 matmul lanes, whose `tiled` flag
  also guards an `if`/`else` with a wait in one arm. That was caught by
  diffing every symbol's classification before and after the rule, which is
  the check that forced the narrower version. **Review of #708 tightened it
  twice more:** the arms are now bounded exactly (brace block or statement)
  instead of by a fixed character window, and **every** wait in the function
  must lie inside the else arm rather than merely be absent from that window —
  otherwise a wait added before or after the branch would be exempted along
  with it, and the `_enc` entry would read as non-blocking while it now
  blocks. Bounding the arms exactly also found a third helper the windowed
  version had truncated past, `mpsg_run_gather_blocks`.
* **A one-symbol wrapper is resolved from its callers.**
  `_apple_gpu_raw_handle` takes its symbol as a parameter, so its own body
  says `<dynamic>` and fails closed; both callers pass a literal, so what it
  can reach is known exactly, and neither symbol takes a command buffer. A
  caller passing anything unreadable keeps it `<dynamic>`, hence closed.

Both rules are mutation-checked: widening the guard past command buffers, and
accepting a non-literal caller, each fail the suite. The `KNOWN_UNROUTED`
mechanism stays, so a future exception must still be named, and a line there
that stops being an offender still fails the gate.

**One runtime-side gap this could not close, because the `.mm` is untouched
— closed 2026-09-03, see the entry below.** (Editing it invalidates
`AppleRouteContext.runtime_fingerprint` and forces a ledger re-seal — the same
sequencing as the second instance.) **The MPSGraph
`runWithMTLCommandQueue` lanes waited with no timeout at all.** `cf_scan`,
`cf_serial_draft`, the reduce / argreduce / scan and conv families,
`gumbel_argmax`, `mla_decode`, `ppo_policy_loss` and the rest of that family
block until the graph completes, so a wedged device never returns to them —
there is no repeated cost for a caller-side breaker to cut, only one permanent
hang. Routing them still buys the open-breaker short-circuit and the named
fallback. The runtime-side fix is filed with the second instance's own finding
about the same 38 call sites.

**The Metal 4 half of that gap is closed by the second instance's duration
heuristic, not left open.** `mtl4_encode_and_wait` bounds its wait at 10 s and
sets no error kind, so an MTL4 stall is indistinguishable from a shape decline
by return value; `silent_failure_timeout_s` classifies a silent failure that
took ≥ 5 s as a timeout instead. Every Metal 4 lane routed here — `mtl4_scan`,
`matmul_sg`, `matmul2d_{f16,bf16,epilogue}`, `mtl4_conv2d` and the MLP
session's host-input `run` — passes that opt-in; the lanes whose C side reports
honestly deliberately do not, so an ordinary decline stays a decline.

**One Python-side gap left deliberately open — closed 2026-09-03.**
`apple_gpu_batched.batched_session` commits through `ts_enc_commit_wait`, a
30 s timed wait, in a `finally`. It could not be *routed*, because an open
breaker must not skip it: everything encoded into that session would go
uncomputed and the caller would read unwritten `DeviceTensor` output as if it
were a result — a bounded stall traded for silent wrong answers, the opposite
of the trade the breaker makes everywhere else.

`_apple_gpu_commit_accounted` is the accounting-only variant: it **always**
dispatches and only observes. All six commit sites go through it — the
batched-session context manager, the encode session's `commit` and its
auto-`_flush`, and the three SSM-replay commits.

**Superseded in part, same day: `ts_enc_commit_wait` now reports.** This
paragraph originally read that the symbol "is `void` and, on expiry, prints to
stderr without touching the error channel, so duration is the only signal it
leaves." That was true when the accounting variant was written and PR #711
(`fix/apple-commit-wait-reports-timeout`) made it false: both expiry branches
now call `ts_set_last_gpu_error(1, "enc_commit_wait", …)` like every other
bounded wait in the runtime, so a stalled commit is **read** rather than
inferred. `silent_failure_timeout_s` stays on that path as a fallback for an
older dylib — the Python package and the runtime are versioned separately, so a
prebuilt library from before that change still only prints — and a reported
timeout is counted once, not twice. The two classes remain milliseconds versus
30 s where the inference is still needed. A
stalled commit therefore counts toward the streak and lands in the fallback
log, and the *next* numpy or resident lane finds the breaker open.

The streak rule itself now lives in one place (`_apple_gpu_record_dispatch_outcome`)
because two callers move that counter and must not drift: the breaker, which
may skip a dispatch, and this, which may not. A drift gate fails any bare
`ts_enc_commit_wait` call, and three mutations are each caught — making the
commit short-circuit on an open breaker, dropping the duration classification,
and bypassing the helper at a call site.

The exact fix is still runtime-side: one `ts_set_last_gpu_error(1, …)` beside
that 30 s wait would make this reported rather than inferred. It is filed with
the `runWithMTLCommandQueue` work below, since both need the same re-seal.

**Behaviour that deliberately changed.** A reported dispatch failure on a
routed lane now returns the host value instead of the kernel's untouched output
(Decision #21), counts toward the streak, and raises under
`TESSERA_STRICT_DISPATCH`. The `*_value_available` probes wrap their dispatch in
`except Exception`, which would have swallowed that strict-dispatch error, so
they now re-raise it; they also refuse to run — and, more importantly, refuse to
**cache** — while the breaker is open, since a `False` cached then would outlive
`reset_apple_gpu_dispatch_breaker()` and disable the lane for the rest of the
process. That probe gate reads `TESSERA_APPLE_GPU_NO_DISPATCH_BREAKER` as well
(review of #707): the opt-out is consulted per call, so setting it after a
streak has opened the breaker makes every other lane dispatch again, and a
probe keyed on the raw open bit alone would have stayed unavailable until an
explicit reset. The MPS single-matrix linalg lanes report success as `rc == 0`, which
the helper carries as an explicit `ok_rc` rather than normalizing.

**The gate is per-dispatch, and finding that out cost a bug.** It inherits the
second instance's per-dispatch rule, which needs two wrapper shapes a plain
ancestor walk does not recognise: a nested `def _run(): sym(...)` and an
assigned `call = lambda: symbol(...)`, both handed to a helper by name. A
mutation check — add a bare second dispatch beside a wrapped one — then failed
to fail, because accessors that bind by plain attribute
(`runtime.tessera_apple_gpu_rope_f32`) rather than `getattr` were invisible to
the enumeration entirely. Teaching it that form both caught the mutation and
surfaced one genuinely unrouted lane the first pass had missed,
`_apple_gpu_dispatch_swiglu`, now routed.

**Coverage.** `tests/unit/test_apple_gpu_dispatch_breaker.py` drives one
representative per family host-free (linalg both rc conventions, random, GQA,
batched attention, MPSGraph control flow, Metal 4 matmul/conv/MLP session,
`msl_spec_accept`, a value probe, a value-lane executor, and a void-symbol
executor) through four properties each: an open breaker never dispatches,
timeouts count toward the streak, a validation decline does not, and success
closes it. The drift gate is the enumeration above, so a new dispatching lane
that copies the old shape fails it by construction, and a `KNOWN_UNROUTED` line
that stops being an offender fails it too.

The value-probe cases stub `_apple_value_compile_pipeline_available`. That gate
runs a real canonical compile: it is **False on a host without the Apple value
pipeline**, so the probe short-circuits before reaching the symbol under test —
which is why the first push passed on this Mac and failed five cases on the
Linux CI runner, the reverse of the usual host asymmetry. Stubbing it also
drops the file from 34 s to 3 s, since that compile ran once per process.

**Evidence, M1 Max, unsandboxed, dylib built 10:23 the same day.**

| run | result |
|---|---|
| `tests/unit/test_apple_gpu*.py` before | 1 failed, 1627 passed, 1 skipped (76 s) |
| `tests/unit/test_apple_gpu*.py` after | 1 failed, 1687 passed, 2 skipped (112 s) |
| `tests/unit/test_apple*.py` after | **2625 passed**, 2 skipped (186 s) |
| `tests/unit/test_apple*.py` on the stashed tree | 1 failed, 2564 passed (183 s) |

Both failures are flaky and both landed on **unmodified** code: a `cumscan`
case that passes in isolation, and `test_warmup_then_production_call_is_faster_than_cold`,
a timing ratio. `mypy python/tessera/runtime.py` is clean, and
`scripts/check_generated_docs.sh --write` rewrote no dashboard — this change
adds no ops.

### Fourth instance: the unbounded MPSGraph wait *(fixed 2026-09-03, M1 Max)*

**The one a caller-side breaker could never reach.** The breaker cuts the
*repeated* cost of asking a device that stopped answering. `runWithMTLCommandQueue:`
submits and waits with no timeout, so a wedged device never returns from it —
one permanent hang, and nothing to cut. 35 call sites used that form.

Each now encodes into an explicitly owned `MPSCommandBuffer` and waits through
`commit_mpsgraph_and_wait_with_timeout`, the helper this file already used
elsewhere: bounded at 30 s, and on expiry it sets error kind 1, which is
exactly what the Python breaker counts. The conversion follows the
APPLE-DEVICE-EVENT-1 precedent already in the runtime, which moved the `bmm`
route off the same call for a related reason.

**Two shapes, and one that would have been a silent bug.** The
`resultsDictionary` form maps directly; the returning `targetTensors` form
becomes its encode analog, whose results are valid once the wait completes —
which is what the wait now guarantees. But the two `encode_or_run_*` helpers
hold their call in a **braceless `else`**, where a multi-statement replacement
binds only its first statement and leaves the wait running unconditionally.
Those two are braced, and the transform detects that case rather than assuming
it.

**Three gates.** No `runWithMTLCommandQueue` call may return (comments still
name it, so the gate counts calls, not mentions); the bounded helper must still
report kind 1 on expiry; and every call site must pass a finite timeout, since
a zero would restore the old behaviour under a bounded name.

The PPO source assertions required the old spelling to prove those lanes were
MPSGraph-backed rather than host loops. They now require the encode plus the
bounded wait, and the *absence* of the unbounded spelling — asserting the old
name would have pinned the very hang this fixes.

**The re-seal, measured rather than re-stamped.** The `.mm` changed, so
`runtime_fingerprint` changed (`sha256:5c4941d8…` → `sha256:b12d2e92…`). Both
sealed artifacts were re-recorded on this M1 Max after committing the runtime
(the packet recorder refuses a dirty fingerprinted source):

* **Fleet packet** — re-recorded and re-sealed. Medians moved rather than being
  restamped: matmul end-to-end 1.134 ms → 1.071 ms, softmax 0.437 ms → 0.381 ms.
* **Route ledger** — re-recorded with `--profile extended`, **not** the default.
  The default `core` profile produced 9 decisions against the committed 16, and
  reported `rejected == ()` while doing it. That is the same near-miss this
  queue already records; diffing decision **key sets** against the committed
  ledger is what caught it, and it is the check to keep running.

**The ledger legitimately moved, and the reason is the change itself.** Sixteen
routes became eighteen and three flipped, all from one cause: a route that let
MPSGraph own its command buffer offered no object on which to observe a device
interval.

| row | before | after |
|---|---|---|
| `retune_reduce_sum` (both shapes), device | ineligible — "incumbent paired evidence is incomplete" | eligible, `mpsgraph` retained |
| `retune_mla_decode` (both shapes), device | `explicit` | `absorbed` (joining end-to-end, which already read `absorbed`) |
| `retune_moe_swiglu 16x32x64x32_e4`, end-to-end | `composed` | `single_fused` |

The last row is the one recorded here as load-sensitive, so it was held to the
standard the original pin set: **eight independent five-run recordings on an
idle box, all eighteen decisions unanimous**, no row disagreeing across
recordings.

**Evidence, M1 Max, unsandboxed, against a dylib built from this source.**
`tests/unit/test_apple*.py` — **2638 passed**, 2 skipped. Every test that reads
the ledger or the packet passes (41). The e2e_fleet dashboards were regenerated
with the packet.

**Two P1 review findings, both latent in the timed helpers before this work.**
Converting 35 call sites onto them is what made the blast radius wide enough to
matter, so they are fixed here rather than filed:

* **The shared event could satisfy the wrong waiter.** Both helpers reserved an
  increasing value on the context-wide event under a lock they released
  *before* encoding and committing, so two concurrent dispatches can submit out
  of reservation order. If the value-2 command buffer signals first, the
  value-1 waiter is already satisfied and reads its results while its own
  command buffer is still running — stale or half-written output, with no error
  anywhere. Each wait now creates its own event and waits for value 1: one
  waiter, one signal. The shared counter survives only as the fallback for a
  failed creation. Fixed in **both** helpers, so the 113 pre-existing callers
  of the MSL one benefit too, not just the converted MPSGraph paths.
* **A timed-out dispatch recycled its buffers.** The helper's own comment says
  the command buffer may still be in flight, and the caller's guards then
  returned pooled buffers to the shared pool on the way out — so the next
  dispatch could be handed storage the stalled command still reads or writes.
  Every timed wait now bumps a timeout epoch on expiry, and a guard whose
  acquire predates the bump drops its buffer instead of pooling it. Dropping is
  safe and is the point: Metal retains a command buffer's resources until it
  completes, so the memory outlives the stalled command and merely stops being
  handed out. Buffers acquired after the bump pool normally, so one timeout
  does not disable the pool.

Re-sealed again for the second fingerprint (`sha256:b12d2e92…` →
`sha256:5b95836e…`). The ledger was unchanged by these fixes: the same 18
decisions with the same routes across eight fresh recordings, so only the
fingerprint moved.

**The stderr-only expiry is closed too (2026-09-03).** `ts_enc_commit_wait`
now sets timeout kind 1 on the error channel, like every other bounded wait
here, so a stalled session commit is **read** rather than guessed. The duration
rule in `_apple_gpu_commit_accounted` stays as a fallback for an older dylib —
the package and the runtime ship separately, so a prebuilt library from before
this change still only prints — and a reported timeout is counted once, not
twice, because the inference only fires when the runtime reported nothing.

The test that distinguishes reading from inferring is the one worth keeping: a
commit that returns with **no time elapsed** but reports kind 1 must still
count. Under the duration-only rule that was invisible. Removing the C-side
report is mutation-checked and caught.

**Cross-backend sync.** The three runtime-side contract changes on this key —
publishing timeout kind 1 from `ts_enc_commit_wait`, a per-dispatch event in
both bounded waits, and quarantining pooled buffers after a timeout — are
assessed in all four queues under `DISPATCH-BREAKER-RESIDENT-2026-09-03`. All
three are **not applicable** to NVIDIA, ROCm and x86, each for a reason checked
against those sources rather than inherited from the earlier note: neither CUDA
nor HIP has a timeout-bearing wait to report an expiry from, both create one
event per event object so no concurrent dispatch can satisfy another's wait,
and the recycling buffer pool exists only in `apple_gpu_runtime.mm`. The
standing follow-up for those two backends — that they have no bounded device
wait at all — is unchanged by this work.

**That last gap is closed too (2026-09-03).** Four fallbacks used the untimed
`waitUntilCompleted` — in `commit_and_wait_with_timeout`,
`commit_mpsgraph_and_wait_with_timeout`, `ts_enc_commit_wait` and
`ts_enc_wait_destroy` — each described in the source as "no timeout
protection, but at least correct". It is not correct: `newSharedEvent` fails on
a device that is **already in trouble**, so the one path taken *because* the
device was unhealthy was the only one that could hang forever.

Metal offers no timed `waitUntilCompleted`, so `ts_wait_for_completion_with_timeout`
polls `status` against a deadline. Polling rather than a completion handler is
deliberate and worth recording: a handler must be attached **before** commit,
and two of these sites reach their fallback after committing while a third is
handed an already-submitted buffer. Polling is uniform across all four, needs
no ordering, and keeps nothing alive past the wait; the 1 ms tick is a bounded
number of wakeups on a path that is rare by construction. On expiry each
reports kind 1 and quarantines its pooled buffers, so the breaker counts them
like any other timeout.

**One `waitUntilCompleted` remains, and is meant to:** the completed-buffer
telemetry read, gated on `prefer_command_buffer` and reached only after the
event has signalled. The buffer is finished by then, so it returns at once —
bounding it would put a deadline on a no-op. The existing wait-pattern
migration gate documented all five sites and **caught the removal**; it now
documents the one, asserts zero session fallbacks, and checks the survivor is
still the telemetry read rather than a fallback that crept back under the same
spelling.

## APPLE-MLPKG-RACE-1: the packaged ML dispatch raced its own signal *(fixed 2026-09-04, M1 Max)*

**How it was found is the useful part.** It was not reported as a bug. It
surfaced while attributing a pre-existing flake during unrelated work: the same
command failed **2, then 21, then 62, then 28** tests across four runs, and
every failure carried one signature — `dispatch returned False;
last_error_kind=0`. A varying failure count with a constant signature is a race,
not twenty-one bugs.

**Two defects, and the first hid behind the second.**

*The race.* `mtl4_shared_queue` is shared by every packaged dispatch, and this
lane deliberately does not take `mtl4_dispatch_mu` (fresh allocator + command
buffer per call). But `commit:` and `signalEvent:value:` are **two separate
queue operations**. Unlocked, thread B interleaves between A's commit and A's
signal, and A's signal no longer denotes A's buffer.

*Why nobody saw it.* The context-wide `mlpkg_event` made any thread's signal
satisfy any thread's wait. A waiter returned on **another dispatch's**
completion and read its outputs while its own command buffer was still
running — a silent wrong answer whose only visible trace was an occasional
False. Switching to a private event turned that into a wait that never
completes, which is the *truthful* symptom and the evidence the signal was
genuinely not arriving. Both halves are needed: a per-dispatch event, and
commit+signal under one lock. The wait stays outside the lock, so dispatches
still overlap on the GPU.

This is the correction `7079e95a` already applied to `commit_and_wait_with_timeout`
and its MPSGraph sibling, with the reasoning recorded there. **The packaged lane
was simply missed** — worth remembering when a fix is applied "to the waits":
enumerate them.

*The silence.* The function returned 0 from **ten** places and set the
last-error channel from **none**. Every failure reached the caller as
`last_error_kind=0` — true, useless, and indistinguishable between "no ML
encoder on this SDK" and "the device stopped answering". Each path now names
itself: kind 2 for an ordinary per-op failure, kind 1 for the timeout — the
kind that feeds the Python dispatch breaker, so a wedged device now stops being
asked instead of paying this timeout once per packaged dispatch. The timeout
path also quarantines pooled buffers, as the sibling waits do.

**Evidence** (M1 Max, unsandboxed, fresh process per trial, hang counted as a
failure via an external deadline). The test is the one written for this race —
`test_apple_mlpkg_concurrency`, 4 threads x 8 dispatches:

| runtime | trials | pass | fail |
|---|---:|---:|---:|
| unmodified | 20 | 18 | 2 |
| fixed | 60 | 60 | 0 |

At the baseline's 10% rate, 60 clean trials has probability ~0.0018. The eight
files that failed in the original sweep run **101 passed / 0 skipped**, and the
reproducer's failure count stopped varying between runs.

**Two process traps hit while verifying this, both worth avoiding.**

1. **A `build` symlink into another checkout's build tree silently compiles
   THAT checkout's source.** CMake stores absolute paths, so
   `ninja -C build TesseraAppleRuntimeShared` from a worktree rebuilt the main
   tree's `.mm` and produced a dylib without the fix. The tell is the compiler
   warning paths naming a directory you are not editing. Configure a build in
   the worktree instead.
2. **A minimal build turns the tests you are trying to verify into skips.** The
   first "all green" family run had 21 skips reading *"integration requires the
   Apple GPU runtime and libtessera_jit ABI"* — and those skips were exactly
   five of the failures under investigation. Build `libtessera_jit` and
   `tessera-opt`, then re-run: `-rs` and a skip count are the check that a green
   result evaluated anything ([[hollow_green_signal_pattern]]).

**Cross-backend:** not applicable to NVIDIA, ROCm or x86 under
`DISPATCH-BREAKER-RESIDENT-2026-09-03`'s reasoning — this is a Metal 4 queue
API pairing with no sibling. The *general* lesson does transfer and is worth
stating: any backend that submits work and signals completion as two separate
calls on a shared queue has this bug shape, and a shared completion counter
will hide it as an occasional wrong answer rather than a failure.

## APPLE-MOE-ROUTE-1: the low-precision MoE path jumped the arbiter *(fixed 2026-09-04, M1 Max)*

**The MoE SwiGLU composite had three implementations and no single place that
chose between them.** `single_fused`, `lowp` and `composed` were three
fall-through blocks, each ending in `except Exception: pass`, so which one ran
was not predictable from the inputs and a failure between them was invisible.

**The defect is narrower and worse than "untidy".** `lowp` sat **ahead of the
arbiter** and preempted it unconditionally for any uniform f16/bf16 operand, so
`production_route_for` never got to decide for the common low-precision
inference shape. Measured here (best of 5 after warm-up, ms):

| (T,K,H,N,E) | dtype | single_fused | lowp | composed |
|---|---|---:|---:|---:|
| 64,128,256,128,4 | f32 | 13.23 | — | **1.27** |
| 64,128,256,128,4 | f16 | — | 15.04 | **1.26** |
| 256,256,256,256,8 | f32 | 32.00 | — | **1.89** |
| 256,256,256,256,8 | f16 | — | 35.66 | **1.84** |
| 1024,512,512,512,8 | f16 | — | 1571.06 | **26.31** |

`composed` wins every case and the gap widens with size (10x → 17x → 60x). It
is also the more accurate: **6.3e-8** relative error against an fp32 reference
where `lowp` is **2.6e-4**, because `composed` accumulates in fp32. So the
low-precision default was ~12-60x slower *and* ~4000x less accurate than the
route it displaced — and `lowp` **has no ledger row at all**: it was preferred
by default while never having been measured into the arbiter.

**The fix restores the arbiter rather than removing it.** The first attempt
deleted the `production_route_for` call, which the suite caught
(`test_moe_dispatch_consumes_the_strict_exact_row`) and was right to: that
would have made the ledger row a declaration nothing reads (Decision #29) and
"fixed" a slow route by deleting the mechanism whose job is to choose between
routes (Decision #28). Checking the ledger directly settled it — on the
committed ledger it answers `composed`, correctly. It was never the problem.

`_apple_moe_select_route` is now the one selection point, returning
`(route, reason)`. The arbiter decides between the routes it has evidence
about; `lowp` is opt-in (`TESSERA_APPLE_MOE_LOWP=1`) until it earns a ledger
row; `quant` outranks everything, because per-GEMM quantization is the one
thing the single-kernel paths cannot express. Choosing a route measured slower
than the default lands in the dispatch fallback log under the op's name
(Decision #21), so a machine running a slow lane can be found rather than
guessed at. `tests/unit/test_moe_route_selection.py` (10) pins the matrix,
including an unreadable ledger and a ledger choice the shape cannot honour.

**Do we need all three? On this evidence, no — but keep them reachable.**
Neither alternative earns a default. Both are waiting on rewrites that could
change that: the fused kernel is one-thread-per-token and needs a
threadgroup-cooperative version, and `lowp`'s single command buffer may win on
a resident streaming lane where the composed path's three dispatches dominate.
The rule this entry sets is that such a path must be **measured into the
arbiter**, not wired ahead of it.

**Cross-backend:** not applicable to NVIDIA, ROCm or x86 — this is Apple
dispatch-selection code with no sibling, and it changes no dtype, Graph IR
spelling, ABI or numeric contract. The *general* lesson transfers and is worth
stating: an implementation that selects itself ahead of the arbiter is outside
the measured-arbiter model of Decision #28 no matter how fast it is, and this
one was neither fast nor accurate.

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

**Outcome for this backend: `follow-up required` — the same gap exists here and
is not yet closed.** `coopmat_reduce_eligible` gates the
`simdgroup_matrix` reduce lane on `N % 8 == 0`, and the emitted comment says
why: *"N is a multiple of 8 by eligibility, so each 8x8 store stays in bounds."*
That is the store-suppression half of the rule above, currently solved by
refusing the shape instead of guarding the store. A ragged N therefore falls to
the scalar path.

Apple is the best-placed of the three to fix it, for a structural reason worth
noting: it synthesizes MSL **per shape**, so `N` is already an explicit argument
to the eligibility predicate rather than something the kernel must discover.
The tail bound is compile-time there, as it is for ROCm's `head_dim` — which is
what made the ROCm side cost no runtime predicate on the contraction axis.

Not attempted in this change: it needs its own device proof on the Mac, and
bundling a third backend into one PR would put three device claims behind one
review.

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

**Outcome for this backend: `follow-up required` — Apple was missing from the
map entirely, and now says so.** Neither Apple target can be classified by
`bootstrap_prune_audit` today:

* `apple_cpu` **is** in its `_BACKEND_MODULES`, but
  `apple_cpu_native.native_package_kind` returns a computed expression
  (`op.op_name.removeprefix(...)`) rather than string literals, so the AST
  walker derives no families;
* `apple_gpu` is **not in that audit at all**, although `driver.py` has a live
  scheduled dispatch for it (`target_kind == "apple_gpu"` at the
  `package_native` branch).

Both now render as `❔ unclassified` with the reason, not as `—`. That
distinction is the point: a dash means measured and nothing serves it, and
absent columns read as unserved — the `unmeasured` failure in a new place.

Two follow-ups, in order: add `apple_gpu` to `_BACKEND_MODULES`, and give
`apple_cpu_native.native_package_kind` literal family names (or an explicit
family table) so the walker can see them. Until then Apple's real routes are
invisible to the prune plan, which is a coverage gap in the plan rather than in
the backend.

## APPLE-MLPKG-HANG-1 — a stale runtime dylib hangs the dispatch instead of failing (P1, opened 2026-09-01)

> **Corrected 2026-09-01, same day it was filed. The original root cause was
> wrong and the severity is reduced from P0.** This was opened as "the ML-package
> dispatch wait never returns", blaming `tessera_apple_gpu_mlpkg_dispatch` for
> ignoring its own timeout. It was not the dispatch. The Mac's
> `libTesseraAppleRuntime.dylib` was **two days stale** — the rebuild took 320
> targets — and against a fresh dylib the same selection is
> `115 passed in 32.03 s`, where it had previously hung twice, for 16 and 14
> minutes, at 0.0% CPU. The entry's own "next steps" listed checking for a stale
> dylib as step 1, and it was filed without doing it.
>
> **What that invalidates.** The "Linux control" argument below concluded
> *"the sweep is healthy and finite; it is this host's Apple GPU dispatch that
> does not return."* The Linux measurement was real, but the Mac half of that
> comparison was a stale binary, so the comparison did not show what it claimed.
> Do not cite it as evidence of an Apple-specific dispatch defect.
>
> **What survives, and why this stays open.** A stale dylib must produce a clean
> load-or-ABI error, not an **uninterruptible** wait. The hang is real and its
> properties are unchanged: `pytest --timeout --timeout-method=thread` cannot
> preempt it, because the block is a C call that never re-enters the
> interpreter, so the lane cannot fail — it can only stop, and CI sees an
> unattributed job timeout. A mismatched runtime is an ordinary developer state;
> reaching it should cost a diagnostic, not the whole gate.

**Reproduction (the corrected one).** With a `libTesseraAppleRuntime.dylib`
older than the Python/ABI surface that calls it, `pytest tests/unit/ -k mlpkg`
blocks indefinitely in:

```
tessera_apple_gpu_mlpkg_dispatch  (libTesseraAppleRuntime.dylib) + 2416
  -[IOSurfaceSharedEvent waitUntilSignaledValue:timeoutMS:]  (IOSurface) + 72
    iokit_user_client_trap  (IOKit) + 8
```

The GPU never signals the shared event, and the wait does not honour
`timeout_ms` from that state. The timeout plumbing itself is sound:
`MLPackagePipeline.dispatch` defaults to `timeout_ms=30_000`, the composite path
rejects a non-positive timeout, and `apple_gpu_runtime.mm` prints a named
diagnostic on the not-signalled branch — none of which is reached.

**Open work.** (1) Make the dispatch bound its own wait so a non-signalling GPU
returns a diagnostic rather than blocking — the caller's `timeout_ms` should be
enforced on the Tessera side, not delegated to an IOKit trap that can outlive
it. (2) Give the runtime a version/ABI stamp the Python side checks at load, so
a stale dylib is refused up front; the standing rule to rebuild before an Apple
sweep is a convention, and this entry is what a convention costs when it is
missed. (3) Consider `--timeout-method=signal` for the Apple lanes as a backstop.

**Confirmed end-to-end (2026-09-01).** With the rebuilt dylib the FULL sweep now
finishes on this Mac: `16199 passed, 3235 skipped, 0 failed in 12 m 44 s`
(`pytest tests/unit/ -m "not slow"`). That is the run which previously hung
twice without completing, and it closes the correction: there is no
Apple-specific dispatch defect blocking the gate, and the Mac's clean sweep is
now the baseline this host never had.

**A second, distinct phenomenon — do not confuse them (measured 2026-09-01).**
With the fresh dylib the full sweep is still very slow, and sampling shows the
parent at 0.0% CPU again — but blocked in `select_poll_poll` → `poll`, waiting
on a **child process**, not in `iokit_user_client_trap`. The child is
`clang++ -shared -x objective-c++ … apple_gpu_runtime.mm` building a
**test-local** copy of the ~27k-line Apple runtime into a pytest tmpdir
(`test_ebt_sweep_emits_pair_per_0/libtessera_apple_gpu_runtime.dylib`). That is
progress, not a hang, and it is why "0.0% CPU" alone is not a hang diagnosis —
sample the stack and look for the child before concluding. Whether recompiling
the whole runtime inside a test is the right cost is a separate question worth
asking, but it is not this entry's defect.

**Standing lesson (the reason to keep this entry at all).** A stale build is not
a benign starting state — it produced an unfalsifiable hang here, and the same
staleness was fleet-wide when checked: the Mac 2 days behind (320 targets), The
Super-Bear 2-3 days, Princess-Luna 7 hours. Measure the build's age before
trusting any sweep, and rebuild before treating a hang or a failure as a code
defect.

## LAUNCH-OVERHEAD-BOUND-1 — cross-backend assessment (recorded 2026-09-02)

`tests/_support/launch_overhead.py` (PR #686) is shared test infrastructure: it
bounds `rt.launch` overhead against the `execution_kind` the launch reports.
A device dispatch gets a flat ceiling; every other lane keeps the
self-calibrating `max(2.0, direct_ms*4)` against the oracle arm. Review on #686
asked for a per-backend verdict, and the four are NOT the same.

**Apple — follow-up required before any Metal row adopts this.** Same shape as
the NVIDIA entry: Apple GPU launches report `native_gpu` and would inherit a
gfx1151-derived ceiling. Apple's dispatch profile is its own -- unified memory
removes the host-to-device copies that are 35% of the measured ROCm per-launch
cost, so the honest expectation is that Apple's floor is LOWER and a 20 ms
ceiling would be slack rather than tight. No Apple test uses the helper today.
Measure on the M1 Max before adopting, and note that APPLE-MLPKG-HANG-1 must be
settled first or a stale-runtime hang will be measured as dispatch cost.

## MSW-4A-CODIFF-SIGN-1 — cross-backend assessment (recorded 2026-09-02)

`ga.calculus.codiff` changed from the unsigned `⋆d⋆` composition to the true
codifferential `δ = (-1)^(n(k+1)+1) ⋆d⋆` (PR #688), and its `clifford_codiff`
VJP changed with it. That is a shared numerical contract, so each backend gets
an explicit verdict.

**Apple — PARITY VALIDATED, and this backend carried the only native work.**
`tessera_apple_gpu_clifford_codiff_cl30_f32` is an exported ABI named `codiff`,
so it now applies the sign at the C boundary rather than returning `⋆d⋆` —
otherwise a caller binding the symbol directly (or reaching it through the
manifest's `symbol_prefix`) would get `+div` on a vector field where δ is
`-div`. The MSL shader and the C++ reference are unchanged and still compute
the unsigned composition, which is what their names say; the sign is applied
once, in the exported entry point, over whichever path ran. The Python wrapper
therefore signs ONLY the numpy composition — re-signing the Metal result would
double-apply it.

Measured after the change: Metal and numpy agree to **0.000e+00** on the
interior; the Metal lane returns `-div` on a vector field (9.0e-08, fp32
noise). The parity test compares the raw symbol against `ga.calculus.codiff`
with **no correction**, restoring a real conformance check — an earlier
revision post-multiplied the native output by the sign table, which made the
assertion true by construction and tested nothing about the kernel. 56 GA/MSL
tests and 134 benchmark tests pass. Requires a rebuilt
`TesseraAppleRuntimeShared`.


## Cross-backend sync `FRONTIER-MSW-2026-09-04`

Owners: `FRONTEND-IR-MEDIUM-1`, `APPLE-DISPATCH-WEDGE-1`,
`APPLE-MOE-ROUTE-1`, and `MSW-5` through `MSW-9`.

Shared changes: pending metadata snapshots are verified before a new record and
the preceding boundary's drop declarations are retired; einsum execution and
AD/trace recording consume one alpha-normal equation; sampled reference fields
carry orthogonal coordinate contracts with registered diagnostic
`FIELD_COORDINATE_CONTRACT`. Coordinate laws extend the existing field-calculus
registry. MSW-6/MSW-8 are host-free examples; MSW-9 is a design spike, with the
native program-pair evaluator/fusion gate still open. No new ODS operation,
backend candidate, physical schedule or native metric ABI is introduced.

**apple outcome:** Follow-up required for native validation: private-event allocation failures now use bounded per-buffer polling in the legacy helpers and encode sessions. Sessions keep a separate commit counter. MoE selection checks all four operands and queries the actual uniform dtype; mixed storage uses composed execution. WSL validates the Python and source-contract tests but cannot compile or execute Metal. The local macOS TesseraAppleRuntimeShared build succeeded after these edits. Owning-host concurrency/fault-injection execution remains necessary; compilation alone is not device proof.

Validation records live in `MATH_SOURCE_WORKSTREAM.md` and the focused review
fixtures. The metadata pass was rebuilt on WSL; the positive/negative lifecycle
fixtures pass. Reference proof does not transfer exact-device status.

## Cross-backend sync `FRONTEND-RECIPE-2026-09-04`

Owner: `FRONTEND-IR-MEDIUM-1`, staged rank/prune acceptance. Shared contracts:
AST location fidelity and location-sensitive capture caching; argument-local
symbolic dimension consumption; opt-in native parametric CSE with one recipe
digest across constraint-checked buckets; strict source-loop matmul candidate
raising. Existing passes and operations are reused; execution selection is
unchanged. The validation spine now binds checkout fingerprints, and coverage
conflicts are regenerated after authored inputs are resolved.

Apple outcome: follow-up required for any future recipe-based native selection. No Metal execution or performance evidence is claimed.

## Engineering sync `EVAL-TELEMETRY-IKF-2026-09-04`

Owners: MSW-9, APPLE-DISPATCH-WEDGE-1 and IKF-1. Program-pair evaluator
comparison keeps both native provenance gates; ANN composition and ReLU
identity checks are separately registered reference laws. IKF-1 is now bound
in the integrated plan: P0 timing first, P2 after validated clocks and
Schedule-Object region identity, P1 host schema/math independently.

Apple: follow-up required. Two device-clock tests leaked the process-wide
capture toggle; scoped capture now restores it and an opt-in pytest order
tracer attributes transitions after teardown. The reported latch is not yet
root-caused. Metal 4 direct wait stamps timeout kind/message; device re-seal
and exact triggering order remain outstanding. No lowp ledger promotion,
cooperative-kernel performance claim, or change to mean ± t·sd is made.


## Cross-backend sync `EVIDENCE-POLICY-20260904`

Decision #26 coverage snapshots move to revision-bound CI artifacts with source
commit/tree fingerprints and output hashes. The canonical renderer and semantic
coverage checks remain shared; this changes evidence delivery, not backend
capability or execution status. Host-free validation applies to this contract.
No device measurements or schedule parity are inferred.

### APPLE-ROUTE-1: explicit robust cross-run policy

The sealer now accepts `--cross-run-estimator median_order_statistic`. It uses
exact binomial order-statistic bounds for the population median of independent
run medians, each at least 95% one-sided coverage. This is a different estimand
from the existing mean, not a trimmed mean relabeled as the same proof. Fewer
than five runs yield no finite bound; eight runs can tolerate one extreme
value in the bound. Every run must still clear the existing speedup and win
floors, and placement, numerical, resource, paired-trial and provenance checks
remain mandatory. A losing run cannot be discarded as a stall. The consumer
recomputes the median bound and rejects unknown estimator tags.

Default selection remains `mean_student_t`; existing ledgers are unchanged.
Use a fixed, predeclared run count when evaluating the opt-in policy; do not
keep recording until a confidence bound passes. Owning-device comparison and
any default-policy migration remain follow-up work, separate from CI re-seals.
Method reference: [NIST median confidence limits](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/mediancl.htm).

IKF-1 admission guard: the shared D2 cache and persisted-record consumer refuse
L2/L3 intra-kernel timings (`evidence.instr_level`) and malformed levels as
dispatch evidence. L0/L1 and existing pre-instrumentation records retain their
semantics. This is a host-contract check for this backend, not a device-clock
or instrumentation implementation claim.

### APPLE-ROUTE-1 follow-up: fixed-count policy experiment

`benchmarks/apple_gpu/compare_cross_run_policy.py` prepares eight fresh processes
using the extended retune corpus, seeds 1701--1708, five repetitions and nine
paired trials per run. Each process warms both routes. The plan and source/runtime
hashes precede measurement; failed runs are retained without replacement, and a
changed source or dylib invalidates collection. Use a fresh, explicit
`TESSERA_APPLE_GPU_RUNTIME_LIB` and an unused output directory:

```sh
PYTHONPATH=python python benchmarks/apple_gpu/compare_cross_run_policy.py --output /tmp/apple-policy-eight-run
```

The observed reports feed both estimators unchanged. Separate synthetic scenarios
multiply every candidate timing in run zero by 1.5 and 3.0, retaining all per-run
floors. These are sensitivity calculations, not measurements of a physical stall.
The output is analysis-only and cannot install a production ledger. Eight fresh
processes reduce shared runtime state; they do not establish statistical
independence from thermal or OS effects. Default migration requires reviewing the
owning-device results; no migration or new Metal evidence is claimed here.

Owning-Mac measurement completed under the user's explicit exception to
AGENTS.md's WSL-only validation rule. A fresh runtime was built from isolated
commit `fe3c59ed`, then the declared eight processes ran once. The retained
packet is `benchmarks/baselines/apple7_cross_run_policy_20260904/`: raw reports,
pre-measurement plan/input hashes, source-report hashes and a replayable summary.
Each run has 18 native-GPU rows and six explicitly reference-CPU rows; all 24
pass numerical checks. Both policies produce five candidate wins, 13 incumbent
retentions and six insufficient-evidence decisions. Neither the observed cohort
nor the two synthetic slowdown scenarios produces a policy disagreement.

Decision: retain `mean_student_t` as the default. No ledger was installed, no
physical-stall robustness claim is made, and this packet does not admit lowp
MoE. Logic and packet replay tests run on Princess-Luna WSL.
Sync: `APPLE-POLICY-COMPARE-20260904`.


## Compiler foundation sync `IR-NATIVE-FOUNDATION-1` — 2026-09-04

Both GPU value packages and Apple CPU Graph-owned descriptors are migration targets. Extend scheduled consumers and native IR-derived call/ABI records; compiler-owned MSL/Metal materialization is the GPU endpoint, LLVM is the general CPU endpoint. Follow-up required on M1 Max per family. No direct LLVM-to-Metal backend or new execution proof is assumed.

Sequencing and acceptance are owned by
[`INTEGRATED_COMPILER_PLAN.md`](../../compiler/INTEGRATED_COMPILER_PLAN.md#mlirllvm-native-foundation-program--2026-09-04).
Shared change in this slice: architectural migration plan only; runtime, ABI,
selector and physical schedules are unchanged. Historical routes have explicit
replacement and deletion gates, not permanent compatibility exemptions.

## Compiler archive handoff — 2026-09-04

The [August review](../../compiler/archive/CODE_REVIEW_2026-08-29.md)
and [historical typing census](../../compiler/archive/W1_1_TYPING_INVENTORY.md)
are archived. Their [reconciliation](../../compiler/COMPILER_AUDIT.md#archive-reconciliation--2026-09-04)
retains unresolved work in live owners; archival does not close this backend's
`P2-REVIEW-SHARED-PASSES-2026-08-29` proof obligations. This is a documentation
and diagnostic-specification link change; runtime parity testing is not
applicable, and no new device evidence is claimed.

No backend execution contract changes in this archive handoff; existing
shared-pass and native-foundation follow-ups retain their current status.

## Foundation F1 — `IR-NATIVE-FOUNDATION-1` — 2026-09-04

NVIDIA scheduled matmul packaging now consumes its scheduled artifact without a
`GraphIRModule` argument or a base-package compilation. Descriptor fields are
checked against the durable Schedule record and Tile entry signature; the driver
records adjacent Graph → Schedule → Tile → Target → PTX ancestry. Runtime ABI,
physical kernels and numerical policy are unchanged.

Sibling outcome: not applicable to execution parity. Apple scheduled packagers and compiler-owned MSL/Metal paths are unchanged.
The driver change is confined to the NVIDIA branch; no device evidence transfers
to this backend. Its F2 migration obligations remain open.
