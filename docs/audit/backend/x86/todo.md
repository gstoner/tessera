---
last_updated: 2026-09-05
audit_role: plan
plan_state: open
owner: x86 backend
target: x86_avx512
scope: x86 AVX-512 implementation/proof; AMX retired (superseded by ACE)
---

# x86 backend TODO

## Current integrated-plan handoff

[The integrated compiler plan](../../compiler/INTEGRATED_COMPILER_PLAN.md) owns
sequencing; [the backend audit map](../README.md) defines document authority.
This queue owns architecture-specific execution and evidence. Start with the
[2026-09-05 native checkpoint, packed/state boundaries and ownership audit](#native-checkpoint-packedstate-boundaries-and-ownership-audit--2026-09-05)
entry below (sync `IR-NATIVE-FOUNDATION-1`, E2E-REAL-5 / W2.4 / W2.4a).

That entry assesses the bounded saved-LSE, INT4 and paged-read migrations and
ownership spike. Allocation-scoped release, control-flow lifetime, remaining
Graph-owned constructors and architecture-specific performance proof stay open.
Earlier synchronization notes retain their original scope and date; statements
such as “no follow-up owed” apply to that increment, not the whole backend.


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

**x86 outcome: not applicable as written; the underlying rules still bind.**
The AVX-512 route consumes Tile IR and its own package ABI, so neither the
preflight nor the `loc` suffix reaches x86 codegen, and the static annotation
render (`tessera.bf16[16, 32]` → `tensor<16x32xbf16>`) is byte-unchanged —
deliberately, so no x86 golden moves. The dtype refusals (`tf32`,
planned/gated) are frontend-side and precede any x86 lowering. Zen 5 exact
lanes were not run for this PR and no x86 claim is made. **No follow-up owed.**

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

**x86 outcome: not applicable as written; the underlying rule still binds.**

There is no device clock, no event API and no stream on a CPU backend, so
`_select_rocm_latency_ms` has nothing to rank — an x86 "device latency" is a
host timer around resident work. Neither x86 candidate implements
`measure_device_latency` at all yet (`x86_generic_c` T1, `x86_aocl_dlp` T3), so
nothing here regresses.

What does carry over is the reason the ranking exists: **a timer must exclude
the marshalling that differs between candidates.** On NVIDIA, end-to-end wall
time was ~91% numpy conversion and ranked two kernels backwards. An x86 timer
that wraps the numpy round-trip would reproduce exactly that, on a backend
where the host path is the largest term. Time the resident call, not the call
plus its conversions.

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

**x86 outcome: follow-up required — no committed rows today, but the same gap.**

x86 has no rows in the corpus, so nothing is refused and no AVX-512 re-proof is
owed for the change itself. The contract still applies the moment x86 records
one, and the precondition is unmet: **neither x86 candidate implements
`measure_device_latency`** (`x86_generic_c` T1, `x86_aocl_dlp` T3). Any x86
device-timed row recorded today would skip both and cache an empty race — which
the new rule now catches (an all-untimeable race caches nothing) instead of
storing an arbitrary registration-order pick as a verdict.

For a CPU backend "device latency" means a host-side timer that excludes the
numpy marshalling, which is the same distinction that made the NVIDIA numbers
meaningful: end-to-end there was 91% host overhead and ranked the lanes
backwards. Owed on Princess-Luna (Zen 5, AVX-512) — **not gated on
`hardware_amx`**, which would skip on the only host that can produce it.

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

**x86 outcome: follow-up required — exact-device proof owed on AVX-512.**

`_execute_x86_compiled_deltanet` (and its backward peer) read the same three
flags and document the same operand order, so x86 is affected identically to
ROCm: the compiled lane accepted only the three-operand form and now becomes
reachable with `gate`/`beta`/`decay`.

Owed on Princess-Luna (Zen 5), which is the fleet's only AVX-512 host. **AMX is
not a Tessera target and no fleet box has it** — do not gate this evidence on
`hardware_amx`, or it will skip on the one machine that can produce it.

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

**x86 outcome: follow-up required.**

Change (1) is behaviour-identical for both x86 candidates (`x86_generic_c` T1,
`x86_aocl_dlp` T3): neither overrides the hook. No AVX-512 re-proof is owed.

Change (2) is unused, and x86 has the clearest case for it in the fleet.
**`x86_aocl_dlp` is a Tier-3 lane into AMD's AOCL-DLP — an actual vendor
library, so it is the canonical `provenance="vendor_library"` delegate**, the
one arm of `DelegateContract` that no delegate exercises yet (NVIDIA's is
`handwritten_kernel`). Declaring it would put a versioned third-party artifact
behind the contract, which is also the case Decision #11's amendment cares
about: a delegate's ABI/library version belongs in the autotune key, or an
AOCL upgrade silently invalidates stored measurements without invalidating the
cache.

Do not reuse the sm_120 tolerance numbers. AOCL-DLP is a CPU library with its
own accumulation order, and the delegate's budget must be measured on
Princess-Luna (Zen 5, AVX-512). **AMX is not a Tessera target and no fleet box
has it**, so x86 proof here means AVX-512 — never gate this on `hardware_amx`.

## Standing: AMX is retired, not merely unavailable (project direction, 2026-08-30)

**AMX is a dead end and is not a Tessera target.** It was Intel-only, and it
is superseded by **ACE (AI Compute Extensions)**, the matrix-instruction spec
agreed jointly by AMD and Intel. Read every "AMX not applicable (no fleet
hardware)" row below in that light: those rows are **closed by direction**,
not parked pending a hardware purchase. Do not scope AMX enablement,
benchmarking, or box acquisition, and do not treat an AMX row as owed work.

Consequences that bite in practice:

* **x86 native execution proof means AVX-512.** Zen 5 (Princess-Luna) has
  AVX-512 and no AMX; the NR2 Pro's Core Ultra 7 265F has neither.
* **Never gate a test on `hardware_amx` to mean "x86 hardware".** It skips the
  test on the only box that could run it. The two files noted in the sync
  block below were nearly marked that way.
* The compiler-side position is unchanged and already recorded in CLAUDE.md's
  Decision #19 discussion: the `tessera_x86` AMX ops stay an **IR-level
  contract with no `amx.*` lowering**, and only the `x86vector.*` / AVX-512
  half is live follow-on work. When a matrix lane is next needed here, the
  target is ACE.

Cross-backend sync `HOLLOW-GREEN-GATES-2026-08-30` — **shared test infra
changed; this backend has the one open gap.**
A pytest session ledger (`tests/_support/device_accounting.py`) now tallies
executed-vs-skipped per hardware family and fails the session when a family
skipped everything on a host that plausibly has the device.

Cross-backend sync `AVX512-MARKER-AND-AMX-CONSUMER-2026-08-30` — **this
backend owns the change.**

*x86 outcome: **closed 2026-08-30** (was follow-up required).*
`hardware_avx512` now exists across every registry that owns a marker —
`policy.MARKERS`, the PR expression and its four verbatim copies
(`validate.yml`, `validate.sh`, `setup_ubuntu.sh`, `tests_manifest.py`),
`pyproject.toml`, and a device family with a `/proc/cpuinfo` `avx512f` probe.
The x86 native lane is no longer invisible to the ledger. Measured:
Princess-Luna `avx512=True, amx=False` and `tests/device/x86/` **executes**
there (its one failure is identical on `main`, so pre-existing); the Mac
skips all three honestly.

**Only `test_native_vjp_execution_certificates.py` carries the marker, and
that boundary is the point.** It drives the AVX-512 elementwise runtime
directly (`device_arch == "x86_avx512"`, and it skips when either production
`tessera-opt` or the runtime is absent). `test_mpi_rank_collectives.py`
deliberately does **not** carry it: its schedule functions are `target="cpu"`
and it exercises MPI collectives, not the AVX-512 runtime. Marking it was a
review finding on PR #646 and was reverted — because
`DeviceLedger.hollow_lanes()` clears a family as soon as *any* of its tests
executes, a generic CPU test in the avx512 family would clear the lane on a
two-rank MPI run that never touched AVX-512 at all. That is precisely the
hollow state the ledger exists to detect, reintroduced through the marker.
**Rule this establishes: a test may only carry a hardware marker if it
exercises that hardware's runtime**; needing the same *host* is not enough.
The MPI packet keeps its honest `importorskip("mpi4py")` gate and remains
untracked by the ledger; giving it a marker of its own is a separate change.

Also corrected here: the same review found that the scripted edit which added
these markers had **duplicated the body of both files** (352 vs 182 and 121
vs 61 lines). CI did not catch it — the tests skip without `mpi4py`, and
duplicate `def`s merely shadow. Both files were restored from `main` and the
one legitimate marker re-applied by hand.

That work also fixed a live false red it turned up: `test_amx_int8_gemm.py`
is marked `hardware_amx`, but **nothing consumed that marker**, so on arm64
it FAILED at the AMX compile step instead of skipping. `conftest` now
consumes `hardware_avx512` and `hardware_amx` centrally, mirroring the
`hardware_nvidia` boundary — `tests/device/x86/` on the Mac goes
1 failed / 2 skipped → 3 skipped.

*Historical note on why the two files were left unmarked in #645:*

Two files under `tests/device/x86/`
(`test_mpi_rank_collectives.py`, `test_native_vjp_execution_certificates.py`)
were deliberately **left unmarked** rather than given `hardware_amx`. They
need AVX-512 and MPI, not AMX — and per the standing section above AMX is a
**retired** target, not a pending one, so `hardware_amx` will never be
satisfiable on any fleet box. Marking them that way would make them skip on
the only box that can actually run them, permanently. That would have
manufactured exactly the failure this work exists to prevent. Both
already skip honestly today (verified on the Mac: "production tessera-opt
and AVX-512 runtime are required"), so the gap is coverage of the ledger,
not a live false green.

Closing it means adding a `hardware_avx512` marker plus a `/proc/cpuinfo`
probe and a family entry, then marking those two files — a small change,
but one that alters selection for existing `compiler_avx512` tests and so
wants its own PR and its own before/after count on Princess-Luna.
`tests/unit/test_device_accounting.py::test_every_hardware_marker_belongs_to_a_family`
is the drift gate that will fail if a marker is added without a family.

*x86 execution evidence in this batch (Princess-Luna, Zen 5 AVX-512):*
`tests/unit/test_x86_optimizer_compiled.py` is **10 passed / 0 skipped**, and
re-run under `-W error::RuntimeWarning` after the Adafactor fixtures declared
`v_representation` it is 32 passed alongside the ROCm file — so the marked
representation now executes on this backend rather than only the legacy one.

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

*x86 outcome: parity validated 2026-08-30 (was follow-up required).*
`tessera_x86_avx512_adafactor_*` takes `beta2` as a scalar and is unchanged;
`tests/unit/test_x86_optimizer_compiled.py` was migrated to pass `step`.
**The owed exact-device run is done.** On Princess-Luna (Zen 5, AVX-512, the
box that owns x86 execution) that file is **10 passed, 0 skipped**, including
`test_adafactor_factored_forward_and_backward` and
`test_adafactor_full_forward_and_backward`, which assert
`execution_kind == "native_cpu"` against the numpy reference. As on ROCm, the
hand-built state was unmarked and took the legacy `v_representation` branch;
the fixtures now declare the marker.

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

*What this queue must run.* `test_x86_plugin.py::test_x86_kernel_runs_and_matches_numpy`
including the new `FusedRegion(epilogue=("bias","relu"), prologue=("gelu",))`
chain added to `_CHAINS` — those lanes skip honestly on the Mac. The fused
scalar body now emits k-outer with a zeroed row accumulator whenever a prologue
is present, so the prologue activation runs K times per row rather than N*K;
the epilogue-only path is byte-for-byte unchanged. Host-compiled with clang on
the Mac and checked against `FusedRegion.reference` across 9 region shapes
(<=2.4e-7), which is a numerics check, not an AVX-512 one.

Cross-backend sync `LINUX-BASELINE-2604-LLVM231-2026-08-29` — **this is the migrated host; AVX-512 execution unchanged.**
The Linux baseline moves to **Ubuntu 26.04 LTS** and the compiler-backbone pin
tightens from "LLVM/MLIR 23.x" to **23.1.x exactly**; `scripts/setup_ubuntu.sh`
now FAILS on any other Ubuntu release rather than warning. `CLAUDE.md`'s host
record moved in the same change, because leaving it at 24.04 pointed this
project's own instructions at a bootstrap command that exits immediately.

Measured on the migrated box (`Princess-Luna`): Ubuntu 26.04.1, LLVM/MLIR
23.1.0 (assertions OFF), ROCm 10 series (HIP 7.15), repo at
`~/programming/tessera`, ssh on the default port.
*x86 outcome.* The x86 backend shares this box, so the same migration
applies. Native AVX-512 kernels build and the full `lit` suite is 425/425 there
post-migration, so nothing in the x86 lane depends on the 24.04 baseline.
Cross-backend sync `DEVICE-PROOF-DELIVERED-2026-08-29` — **the native
half this queue owed is DONE.**

Run on Princess-Luna (Zen 5, AVX-512, Ubuntu 26.04.1, LLVM/MLIR 23.1.0), from a
clean worktree at `f65f9b3b` configured with `TESSERA_BUILD_X86_BACKEND=ON` —
i.e. with the native AVX-512/AMX kernels actually compiled, which arm64 cannot
do.

**`lit tests/tessera-ir/` is 425/425.** That closes the open question on the two
x86 P0s from PR #635 (f16 refused rather than decoded as bf16; `fused_epilogue`
refusing rather than dropping the activation). Both are *refusals*, so the risk
was never that they fail — it was that they narrow a configuration which
previously lowered AND executed correctly. On the host that builds the real
kernels, nothing regressed.

Also confirmed here: the `CMAKE_SYSTEM_PROCESSOR` gate added for arm64 still
selects the native kernels on an x86 host — the build compiled them and no
"x86 native kernels skipped" line appeared.

Cross-backend sync `SHARED-CONTRACTS-P1-REVIEW-2026-08-29` — **follow-up required on the native half.**
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
*x86 outcome.* Three of the four are shared-IR only, but the composed-layout
fix in this PR is x86-specific and needs this box: `materializeX86ComposedLayouts`
applied rem/div to EVERY basis leaf, wrapping coordinates at the declared
extent so two distinct coordinates aliased one address. The slowest leaf now
keeps the whole remaining quotient, matching what `TileToROCM.cpp` has always
emitted for the same shared carrier. Verified by lit on arm64; **the native
AVX-512 lane still owes an execution check** that no address aliasing survives
in a real kernel.

Cross-backend sync `FOUNDATION-LLVM231-REVIEW-P0-2026-08-29` — **action
required: two lowering P0s landed and a new host-architecture build gate needs
confirmation on a real x86 host.**

*Foundation (all backends).* The LLVM/MLIR major pin is unchanged at **23**;
the Mac moved from a manual pre-release `23.1.0git` prefix to Homebrew's
production `llvm` keg **23.1.0** (the old prefix is deleted). This box stays on
apt.llvm.org `/usr/lib/llvm-23`. **No fleet box has an assertions-enabled LLVM
any more** — which matters here specifically: the open `TileToX86Pass`
in-pass `getOrLoadDialect` item below is an assertions-only hard error and is
now invisible on *every* machine, so its "does not reproduce" status is
provisional (Decision #19). Python/deps: the `numpy<2.2` venv cap **can be
dropped** — `pyproject.toml` now skips numpy/scipy stubs
(`follow_imports = "skip"` **plus** `follow_imports_for_stubs = true`), keeping
`python_version = "3.10"` while making the mypy ratchet independent of the
installed numpy. Also: `check-{clifford,ebm,spectral}` no longer hardcode
`llvm-lit`, and driver discovery now requires a candidate binary to **start**,
not merely exist (a tree from an uninstalled toolchain otherwise wins).

*The build gate that needs this box to confirm it.* `src/CMakeLists.txt` now
adds the native `tessera_x86_backend` kernel subdirectory only when
`CMAKE_SYSTEM_PROCESSOR` matches `x86_64|AMD64|amd64|i[3-6]86`; the
hardware-free `tessera_x86` Target IR dialect (a separate subdirectory) is
always built. On arm64 this makes `ninja -C build` complete with
`TESSERA_BUILD_X86_BACKEND=ON` and takes `lit tests/tessera-ir/` from **414/425
to 425/425** — the 11 phase2 x86 fixtures now run on the one fleet host
*without* AVX-512, which per Decision #19's standing lesson is the only host
whose green result on them is evidence of portability. **Confirm on this box
that the regex still selects the kernels**: a gate that silently skipped them
here would disable the AVX-512 kernel build on the only machine that has the
ISA. Check `cmake` prints no "x86 native kernels skipped" line, and that
`build/src/compiler/codegen/tessera_x86_backend/` still produces its library.

*The two P0s this box owns.* Both in `src/transforms/lib/TileToX86Pass.cpp`,
both previously **silent wrong answers**, now fail closed via a completeness
walk after pattern application (an `emitError` inside a pattern that returns
`failure()` prints but does not fail the pass — the driver exited 0):
1. **f16 routed to the bf16 kernel.** The patterns accepted `isF16()` but
   called `tessera_x86_{amx,avx512}_gemm_bf16`, whose ABI carries no dtype
   selector and whose kernel decodes each `uint16` as bf16. f16 `1.0`
   (`0x3C00`) is read as bf16 `≈0.0117` — every element wrong, with all IR
   types self-consistent so no verifier objected. Now rejected with a named
   diagnostic; operand element types must also match.
2. **`fused_epilogue` dropped the activation.** The epilogue was emitted only
   inside `if (hasBias)` and only for a static bias, and RELU was mapped to
   `tessera_x86_epilogue_bias_fp32`, which applies **no activation**
   (`kernels/epilogue.cpp`). Unsupported configurations replaced the op with
   the bare GEMM result. Now diagnosed and refused (Decision #21).
   Only `none` and `gelu` have kernels; adding a RELU kernel would widen what
   lowers and is the natural follow-up.

Verified so far by lit on arm64 only
(`tests/tessera-ir/phase2/x86_dtype_and_epilogue_fail_closed.mlir`, plus the
existing 11 fixtures). **This box owes the native half**: `ninja -C build` with
the kernels actually compiled, `lit tests/tessera-ir/`, and the AVX-512 device
lane, to confirm the guards did not narrow a configuration that previously
lowered and executed correctly.

Cross-backend sync `IKF-INTRA-KERNEL-CONTRACT-2026-08-27` — **deferred with
reason; AVX-512 execution unchanged.** The IKF-1 intra-kernel measurement
plan (`docs/audit/compiler/INTRA_KERNEL_FEEDBACK_PLAN.md`, PR #634) targets
GPU-style pipelined kernels first (ROCm gfx1151). On x86, intra-kernel
visibility is already owned by the TPROF-X86 lanes (independent
host/raw/TSC/perf clocks, permission- and multiplex-aware `perf_event_open`,
ASLR-safe IBS symbol correlation) with claim-disciplined artifacts.
Indexed-slot instrumentation of tiled CPU kernels is coherent in principle —
invariant TSC via `rdtscp` is the constant-rate clock analog — but has no
consumer until a measured need appears; adopting it without one would violate
Decision #29. Revisit when a CPU cost-model coefficient needs per-region
labels that perf/IBS sampling cannot provide.

Cross-backend sync `X86-AVX512-IMAGE-ADMISSION-2026-08-27` — **load safety and
post-change Zen 5 numerical closure complete.** The monolithic
AVX-512 image no longer constructs four `__m512i` FFT permutation vectors at
ELF load time: their bit-identical lane maps are constant-initialized scalar
tables and are loaded only inside admitted AVX-512 entry points. The legacy
runtime loader now consults the canonical complete-feature authority before
`ctypes.CDLL`, matching content-addressed native-image admission. On the local
AVX2 Threadripper the rebuilt image has no FFT translation-unit initializer,
direct `dlopen` returns normally, runtime admission declines the image, and the
formerly crashing solver test collection completes without SIGILL. A clean
Ryzen AI Max+ 395 native rebuild produced digest
`5aee2d5a98c7abc899c765436a7b505c53b9c544f60d92da07697b0c7dab287c`;
its symbol audit has no FFT static initializer, direct `dlopen` succeeds, and
the pinned-image packet passes all 41 FFT plus 38 solver cases (79/79). This is
exact post-change evidence, not a corollary from the prior image or source lane
maps.

Cross-backend sync `CUDA-SOLVER-KRYLOV-SCALE-2026-08-27` — **NVIDIA-owned
physical expansion; AVX-512 execution unchanged.** The additive shared
packager contract is target-locked to `nvidia_sm120`; cooperative CUDA grids,
resident Arnoldi state, CTA reductions, and native f16/bf16 `mma.sync` matmul
have no x86 runtime or ISA consumer. Existing host matrix-free solver evidence
does not inherit the dense CUDA performance result, and CUDA evidence does not
promote an AVX-512 CG/GMRES row. A future x86 resident solver requires its own
threading/reduction schedule, numerical packet, and performance ratchet.

Cross-backend sync `CUDA-SOLVER-FAMILY-2026-08-27` — **NVIDIA-owned physical
expansion; AVX-512 execution unchanged.** Shared residual-product packaging
now retains authored matmul numeric policy, and a generic `linear_solver="cg"`
contract no longer silently executes GMRES. CUDA's unary/reduction/predicate/
where/IEEE-matmul children and single-launch diagonal-SPD CG state are not x86
evidence and change no host ABI, AVX-512 kernel, or performance row. A future
x86 CG promotion requires an x86-owned package and runnable host evidence.

Cross-backend sync `CUDA-SOLVER-IFT-PILOT-2026-08-27` — **NVIDIA-owned
physical corollary; AVX-512 solver evidence unchanged.** The shared
diagonal-sqrt and general solver contracts gained explicit SM120 admission,
CUDA binary residual replay, and architecture-owned CUDA packages. No x86 ABI, AVX-512 kernel, selector, or
performance row changed, and the CUDA packet is not evidence for the x86
general solver.

Cross-backend sync `CUDA-BINARY-SPECTRAL-JVP-2026-08-27` — **NVIDIA-owned
physical closure assessed; AVX-512 parity unchanged.** The new SM120 binary
package and CUDA spectral tangent accumulator change no x86 ABI or schedule.
x86 retains its independently proven binary arithmetic and compound spectral
JVP routes; CUDA device numerics and reduced-storage evidence transfer no host
claim.

Cross-backend sync `CUDA-SPECTRAL-JVP-NUMPOL-2026-08-26` — **CUDA closure
assessed; AVX-512 parity remains independent.** NVIDIA's public digest-bound
JVP child, cuFFT Schedule→Tile profile, and f16/bf16 CUDA ABIs change no x86
kernel or schedule. Existing AVX-512 spectral evidence remains authoritative;
no SM120 result is counted as host proof.

Cross-backend sync `CUDA-OPTIMIZER-VJP-2026-08-26` — **shared plugin/lineage
carrier updated; x86 disposition unchanged.** NVIDIA is now an exact owner for
the shared optimizer and Adafactor reverse families. AVX-512 retains its own
SGD, Momentum/Nesterov, and full/factored Adafactor packages; Adam/AdamW remain
explicitly fail closed on x86 because no physical reverse consumer exists.
CUDA PTX and `sm_120` certificates transfer no host evidence.

Cross-backend sync `TSOL-CUDA-POLICY-V1-2026-08-26` — **NVIDIA-owned physical
package assessed; AVX-512 parity remains independently validated.** The CUDA
spectral ABI and SM120 logical-complex capability row do not alter the x86 v8
ABI, schedules, dtype policy, or host certificates. The CUDA host cannot load
or execute this workspace's AVX-512 entry points, so no x86 numerical rerun or
evidence is claimed here; after `X86-AVX512-IMAGE-ADMISSION` it safely declines
the image before `dlopen` instead of taking an illegal instruction. The
existing target-owned packet remains the x86 proof. CUDA JVP/Schedule admission was subsequently closed under
`CUDA-SPECTRAL-JVP-NUMPOL` without changing the x86 row.

Cross-backend sync `TSOL-POLICY-PHYS-1-8C8G-2026-08-26` — **Order 8c–8g
AVX-512 bounded package landed and is independently host-proven.** The v8 x86
package consumes rank/shape/element-stride descriptors without Python
`ascontiguousarray`, including negative and interior-axis strides. Forward and
reverse support explicit `n_fft >= window` (including n=20/window=15), odd
lengths, and one-sided or full-complex spectra. Independent NumPy/Python-VJP
packets cover non-contiguous axis-1/axis-2 operands and full-spectrum adjoint
weighting; the reverse Schedule→Tile digest binds
`native_runtime_stride_descriptor_v1`. Streaming state now chains policy,
window, tail, counters, and parent-state digests and rejects lineage drift, but
that target-neutral oracle is not an AVX-512 streaming execution claim.
Per-batch windows now right-align and broadcast across transform batches in
forward, inverse, and both native adjoints; `dwindow` reduces exactly over the
broadcast axes and that identity is Schedule→Tile digest-bound. The physical
streaming entry consumes original chunk strides plus a canonical tail and
emits frames/next-tail without Python framing or concatenation. Runtime-origin
state certificates bind the artifact and parent lineage. Independent forward,
Python-VJP, and chunk/monolithic packets pass on this AVX-512 host. No x86
evidence transfers to gfx1151, CUDA, or Metal. The separately executed
gfx1151 expanded reverse packet is architecture-owned and does not replace or
augment these AVX-512 certificates.

Cross-backend sync `TSOL-POLICY-PHYS-1-8B-2026-08-26` — **Order 8b parity
validated on the AVX-512 host.** Centered STFT and centered/cropped ISTFT now
execute arbitrary normalized logical axes for C-contiguous tensors. Native
package packing uses Schedule→Tile-bound `outer`/`inner` dimensions rather than
a Python transpose, and both forward and reverse pass the nontrivial
`(outer, inner)=(2,3)` NumPy/Python-VJP packet for f32/f16/bf16; f32 also
satisfies the native forward/adjoint identity. Negative-axis normalization and
extent-overflow ABI regressions are pinned. Non-contiguous views still fail
closed and are not claimed as physical stride support. Full spectrum, broader
lengths, broadcasting, true strides, and streaming remain open; gfx1151
evidence was not transferred.

Cross-backend sync `TSOL-POLICY-PHYS-1-8A-2026-08-26` — **Order 8a parity
validated on the AVX-512 host.** The native compound package now owns centered
constant/reflect padding and centered or explicitly cropped ISTFT output for
the bounded n=16/n=18 final-axis envelope. f32, f16, and bf16 storage execute
with f32 accumulation; the native reverse matches the independent Python VJP
and centered-reflect STFT satisfies the forward/adjoint inner-product identity.
The Schedule→Tile digest binds center, pad mode, crop, and output length.
Non-contiguous storage, unsupported lengths, and altered identities fail
closed. Full-spectrum, broader explicit transform lengths, broadcasting, true
strides, and streaming state remain open in Order 8. No x86 evidence transfers
to ROCm, CUDA, or Metal.

Cross-backend sync `AD-TSOL-STFT-GFX1151-2026-08-26` — **gfx1151 Order 7
closure assessed; x86 parity remains independently proven.** The AMD direct-DFT
kernel and its certificates transfer no AVX-512 evidence. The existing x86
packed-C2R n=16/n=18 f32/f16/bf16 package retains its own host proofs. The
shared compound-spectral Schedule→Tile carrier now binds
`numeric_policy={storage,accum=fp32}` separately from reduction order; x86
metadata regeneration and Tile identity validate it. Order 8 owns broader
policy on both architectures.

Cross-backend sync `E2E-REAL-6F-EXACT-CERT-2026-08-26` — **all 10 declared x86
native-VJP families exact-host certified on AVX-512.** The all-family packet
runs independent numerical oracles and requires runtime-origin
`x86_avx512` attestations before exact set equality with the live plugin
registry can pass. This includes factored Adafactor, SGD/Momentum, Lion,
attention, normalization, losses, sequence mixer, and spectral reverse.
Test-double launches remain `runtime_unattested`. No ROCm, SM120, or Apple
evidence is inferred; those targets retain their own packets and hardware
requirements.

Cross-backend sync `BOUNDED-GATE-RELAXATION-2026-08-26` — **AVX-512
STFT/ISTFT backward expanded through stored-bin vectorization and low-precision
envelopes; MPI rank transport started.** Content-addressed native packages execute the exact transpose
of uncentered, one-sided, last-axis f32 STFT/ISTFT with explicit hop,
`n_fft == window`, and uncropped ISTFT length. Signal/spectrum and window
cotangents match the independent Python VJP oracle. The STFT frame adjoint now
uses packed C2R with proven interior-bin half weights and real DC/Nyquist
projection; n=18 mixed-radix ragged batches and explicit f16/bf16 two-byte
storage ABIs pass with f32 accumulation. Factored Adafactor also executes via
the public family plugin and emits a tamper-evident execution certificate
binding its topology and compiler spine. Unsupported policy,
shape, dtype, axis, target, or altered digest fails before launch. The shared
bounded symbol-body `control_scan` reverse also lowers and passes its direct
compiler regression. DIST-NATIVE-1 now lowers all five explicit collective SSA
forms through Schedule→Tile into a content-addressed rank-local MPI artifact.
The artifact binds Schedule hash, communicator, ordered subgroup, reshard plan,
issue ordinal, topology, dtype, and shape. A project-local bundled MPICH
4.1.2/`mpi4py` environment executed the checked-in two-process numerical packet:
all five collectives and a reordered two-rank derived communicator pass, while
order, subgroup, artifact/topology digest, dtype, and shape mismatches fail in
shared admission before transport. This is exact x86 host MPI evidence, not an
NCCL/RCCL or mock promotion; a >2-rank proper-subgroup packet remains open. The composed-layout
review changed only the duplicated ROCm/CUDA physical emitters; x86 already
uses the canonical quotient-retaining layout algebra, so no x86 schedule or
ABI change was required.

Cross-backend sync `W4-EFFECTS-1-E5-2026-08-25` — **one physical family carrying an admissible effect, end to end; x86 outcome: **parity VALIDATED (AVX-512 host)**.** Same family through `x86_rng_compiled` with `execution_kind=native_cpu` asserted exactly. Replay from the recorded product is bit-identical, and the cross-target row shows this host and gfx1151 produce the same bits from the same product.


Cross-backend sync `W4-EFFECTS-1-E4-2026-08-25` — **ordered-collective
recorded products (identity only); x86 outcome: parity validated (host-side analysis only).** The product
binds communicator, issue order, reduction algorithm and topology; the
verifier rejects a permuted order and a changed tree under an identical
order. Order evidence comes from the deterministic mock-mesh executor.
Nothing in the AVX-512 lane changes: this slice is host-side recording and verification with no kernel or numerical effect. The mock-mesh order evidence runs here.


Cross-backend sync `W4-EFFECTS-1-E3-2026-08-25` — **shared state-lineage
identity change; x86 outcome: parity validated, no behaviour change.** Same
change as the rocm entry, and the same evidence: the f32 default keeps every
existing lineage id byte-stable, so the AVX-512 stateful packages
(Lion/Adafactor/sequence-mixer) keep their digests. The E3 content-digest
binding is additive.


Cross-backend sync `W4-EFFECTS-1-E2-2026-08-25` — **shared autodiff gate
change (AutodiffPairedPass); x86 outcome: parity validated, no behaviour
change on this backend.** Same gate split as the rocm entry; it is a
diagnostic refinement over a fail-closed check, not a relaxation. Validated
on the AVX-512 host with the same lit and autodiff suites. The Python-side
call-form classifier that decides which draws may carry a product lives here
too (`stochastic_product_for_call`), and its verdicts are pinned against the
op's MEASURED behaviour rather than a convention.


Cross-backend sync `W4-EFFECTS-1-2026-08-25` — **UPDATED 2026-08-25 (slice E1
landed): shared recorded-product carrier + verifier implemented in Python;
x86 outcome: still follow-up required, no x86 surface consumes it yet.** Same
carrier as the rocm entry. E1 is host-neutral Python with no target coupling,
so nothing in the AVX-512 lane compiles through it and no evidence is claimed.
x86 remains the natural home for the E2 keyed-RNG slice, whose acceptance bar
is bit-identical replay of a recorded dropout region; E1 already demonstrates
that property against the live S4 generator at the library level.

Cross-backend sync `SPECTRAL-PAYLOAD-CHAIN-2026-08-25` — **shared
Schedule->Tile spectral identity contract + pipeline carrier ordering; x86
outcome: parity validated (AVX-512 host).** Same contract change as the rocm
entry; x86 is the producer target exercised end to end here (the spectral
suite runs `target="x86"`, arch `zen5-avx512`). Validated: spectral suite 12/12
with `tessera-opt` live, including the new co-edit and payload-swap rejections;
x86 spectral compiled lane green; no numerical change (identity verification
only).


Cross-backend sync `SCHEDULE-AUTHORITY-RESHARD-2026-08-24` — **SO-3 exact Zen 5 regression and shared W5.4 mock boundary closed; native multi-rank remains open.** Compound spectral producers now infer their fused-stage action DAG, bind roles/resources into one Schedule Object, and stamp its digest through Schedule→Tile; the complete compiled AVX-512 spectral suite remains numerically unchanged on the Ryzen AI Max+ 395. Placement emits exact mesh-sized local-shard/collective SSA and executes every movement form on the deterministic mock mesh. Mock execution transfers no MPI/OFI/SHMEM proof and cannot satisfy DIST-NATIVE-1. `NUMPOL-CARRIER-1` (queue row 3b) owns the generalized S5 carrier; no x86 Target policy change is implied.


Cross-backend sync `SO3-INFER-EDGES-2026-08-24` — **shared W2.1/W5.2e
dependence-inference semantics + MegaMoE R3 producer; x86 outcome: parity
validated (AVX-512 host).** Same shared-analysis correction as the rocm
entry. x86 hosts the R3/composition analysis itself, so this backend is the
regression gate for the change: the composition-cost, graph-dataflow, and
MegaMoE suites pass on this host against the corrected semantics, with no
generated code or numerical change (analysis is prune/rank-only).


Cross-backend sync `NUMPOL-CARRIER-1-2026-08-24` — **shared Schedule→Tile
`numeric_policy` carrier contract (integrated-plan queue row 3b); x86
outcome: follow-up required.** Newly owned row, nothing implemented yet.
x86 has no fragment type, so it has NO carrier today — the accumulator
contract stated at Graph IR does not exist by the time the AVX-512 emitters
pick an instruction (Decision #32's original defect, on this backend in its
purest form). The row's acceptance names bit-identical existing x86 outputs,
so this backend is both a consumer of the contract and a regression gate for
it. Clean Zen 5 evidence required for any realizability verdict (FORGE §1.3).


Cross-backend sync `LAYOUT-ALG-APPLE-PHYSICAL-2026-08-24` — **shared ABI
assessed; no x86 physical change.** Apple now exports the existing C++ rank-2
plan through the native layout ABI for its MSL emitters and owns fresh M1 Max
proof. The four x86 core GEMMs continue to include the header authority
directly; Metal source templates, simdgroup scheduling, and Apple evidence
transfer no CPU ISA, host-admission, or performance claim.

Cross-backend sync `LAYOUT-ALG-L5-X86-2026-08-24` — **x86 composed-layout
consumer and exact CPU proof closed for the shared materializable set.**
`tessera-tile-to-x86` now re-runs the native C++ layout proof, expands tuple
codomains only as products of independently proven scalar maps, and emits exact
i64 mixed-radix `div/rem` arithmetic for static, bounded-dynamic, nested, and
static tuple-product layouts. CPU-owned guards reject negative or out-of-range
coordinates, nonpositive dynamic extents, and negative dynamic strides before address arithmetic. The generated
LLVM returns the independent expected values on both the local AVX2
Threadripper and the AVX-512 Ryzen AI Max+ 395 host; four malformed-runtime
controls fail closed on each. Non-affine and non-separable tuple codomains stay
outside the shared admissible set rather than being flattened.

Cross-backend sync `LAYOUT-ALG-L4-X86-2026-08-24` — **core AVX-512 GEMM index
authority and exact-host proof closed.** The f32, f64, bf16, and u8s8/VNNI
microkernels now derive every A/B/C rank-2 offset through the same header-only
C++ authority used by Tile materialization; their architecture-owned loop
nests and intrinsics are unchanged. Odd dimensions and vector tails pass an
independent scalar oracle on the AVX-512 Ryzen AI Max+ 395 host. The local
Threadripper AVX2 host now rejects AVX-512 images before `dlopen` instead of
risking `SIGILL`; deterministic admission tests cover both outcomes. This
closes the x86 core-matmul part of L4, not a composed-layout Target consumer or
an AMX claim.

Cross-backend sync `LAYOUT-ALG-L3-L5-DYNAMIC-2026-08-24` — **shared proof,
carrier parity, and x86 physical materialization are closed for the shared
admissible set.**
Factorization/residency and Schedule Object v2 proof fields are portable. The
new tuple-product Tile carrier is registered and the x86 target pass now
materializes it through exact scalar CPU arithmetic. CUDA/ROCm device results
and GPU schedules do not transfer. The core AVX-512 GEMM templates consume the
shared rank-2 authority independently.

Cross-backend sync `DYNAMIC-COMPOSED-SM120-2026-08-24` — **shared carrier
parity and x86 scalar-affine materialization are closed.** Nested outer
shape/stride trees with dynamic scalar-affine leaves now verify in Tile IR and
the x86 target pass consumes that canonical operand order with runtime guards.
The CUDA strided ABI and RTX proof still transfer no AVX-512 schedule or host
evidence. Non-affine carriers remain fail closed.

Cross-backend sync `SCHEDULED-MATMUL-TAIL-EPILOGUE-LDS-2026-08-24` — **shared
Graph lineage assessed; GPU physical work not applicable.** Optional matmul
bias and residual are now ordered Graph SSA operands, but the scheduled x86
consumer rejects the fused form until an AVX-512-owned descriptor/epilogue is
implemented. CUDA K-tail staging, reduced f16 stores, ROCm LDS ownership, and
RTX/Radeon evidence change no CPU loop nest or selector. Dynamic/nested
composed-layout materialization remains carrier-only for x86 too.

Cross-backend sync `SM120-MACRO-CTA-2026-08-24` — **not applicable to x86
physical lowering.** CUDA CTA/warp ownership, f16/bf16 two-stage `cp.async`
shared A/B staging, M/N zero-fill tails, barriers, launch geometry, and RTX
5070 profiling change no CPU loop nest, cache blocking, AVX-512/AMX ABI, or
exact-host evidence. The shared composed-layout carrier itself is unchanged.

Cross-backend sync `SM120-SCHEDULED-LICM-2026-08-24` — **not applicable to x86
codegen.** This is a private NVIDIA MLIR-to-PTX pipeline optimization with RTX
5070 evidence. x86 retains its CPU tiling/vectorization authority; CUDA warp,
CTA, shared-staging, and selector conclusions do not transfer.

Cross-backend sync `SM120-BLOCK-COORDINATE-2026-08-24` — **not applicable.**
The target-owned CUDA CTA coordinate operation changes no x86 loop nest,
AVX-512/AMX address calculation, runtime ABI, or exact-host evidence.


Cross-backend sync `CUTE-LAYOUT-MATERIALIZE-1-2026-08-23` — **shared static
affine coordinate ABI; x86 outcome: not applicable to physical lowering.**
The linear-base producer changes no AVX-512 address form, host ABI, or Zen 5
evidence. NVIDIA's SM120 view mapping and RTX 5070 proof are not an x86 layout
consumer or host proof.

Cross-backend sync `ROCM-CI-HSACO-SERIALIZE-2026-08-23` — **ROCm-owned host-free
CI serialization lane; x86 outcome: not applicable.**
x86 emits host code through the LLVM/JIT path rather than a separate GPU code
object, so there is no hsaco-equivalent artifact to serialize and no analogous
blind spot: the x86 Target IR is already exercised in CI by the lit lane
(`TESSERA_BUILD_X86_BACKEND=ON`). No AMX/AVX-512 evidence is involved.

Cross-backend sync `CI-BACKEND-CAPABILITY-SKIP-2026-08-23` — **Apple-owned
pytest capability gate; x86 outcome: not applicable by design.**
x86 deliberately cannot use a `--help` probe: the x86 executable pass is
**always** registered so it fails closed with a rebuild diagnostic, which is
exactly why `tests/tessera-ir/lit.cfg.py:137` probes the dialect directly with a
`!tessera_x86.tile` load-time fixture instead. That mechanism also guards the
assertions-enabled registration regression behind Decision #19, so it is
intentionally kept, not replaced. No AMX/AVX-512 evidence is involved.


Cross-backend sync `NVIDIA-AOT-PACKAGE-V1-HARDEN-2026-08-22` — **NVIDIA-owned
fatbin/cubin runtime admission; x86 outcome: not applicable.** CUDA image
metadata, driver admission, and NVRTC fallback transfer no x86 shared object,
cache identity, physical schedule, AVX-512/AMX ABI, or exact-host evidence.

Cross-backend sync `NVIDIA-FFT-WORKSPACE-1-2026-08-22` — **NVIDIA cuFFT
foundation landed; x86 outcome: parity validated/no physical change.** The CUDA
package's reusable-plan and explicit-workspace contract transfers no cuFFT
handle, device workspace, schedule, or RTX evidence to AVX-512. x86 retains its
architecture-owned FFT package and bounded digest-keyed workspace.

Cross-backend sync `NVIDIA-RNG-PHILOX-CORE-2026-08-21` — **NVIDIA parity
exact-device validated; x86 outcome: parity validated/no physical change.** The CUDA backend's
new typed four-mode directive consumes the shared explicit Philox4x32-10
key/counter identity. x86 retains its architecture-owned distribution package,
runtime ABI, and exact-host evidence; no AVX-512 image, schedule, or evidence
transfers to sm_120.

Cross-backend sync `JIT-CACHE-BLOCK-2026-08-23` — **vectorize-lane
cache blocking landed: flat ~139-161 GFLOP/s across n=512-2048 (was
77/45/43 decaying), plus two structural fixes the loop surfaced.**
Sweep result (Strix Halo/Zen 5, env `TESSERA_JIT_CACHE_TILES`): tiling
M or N at the cache level made things WORSE (18-35 GFLOP/s — strided
cache-tile views defeat the inner kernel's vector loads); K-only
chunking won, and shrinking KC to the register k-tile (16) won
outright — 161/144/139 at 512/1024/2048. That config is really a loop
interchange: the k-chunk loop hoists outermost so a (16 x N) B
row-panel stays cache-resident across the whole (i,j) sweep. Pinned as
the default `{0,0,16}`. Structural fixes: **(1)**
`ConvertVectorToSCF` staged transfers through `memref.alloca`s INSIDE
the loop nest (scf.for is no AutomaticAllocationScope) — per-iteration
`llvm.alloca` overflowed the stack at invoke under two-level tiling
(~65k inner iterations at n=512) and was silently costing the
single-level lane too; switched to `full-unroll`, which alone lifted
single-level n=512 from 69 to 94.5 GFLOP/s. **(2)** The vectorizer
unconditionally rewrites the cache-level `tensor.insert_slice` into a
whole-tile transfer pair (64 KB vector SSA values); a
`demoteLargeFullTileTransfers` rewrite restores insert_slice form for
pairs ≥8 KB so bufferization makes them in-place. **(3)** Acceptance
check: `vectorize_children` "succeeds" having vectorized nothing (a
non-dividing KC left the matmul as scalar loops at ~1 GFLOP/s); a
candidate now counts only if no `linalg.matmul` survived, else the
chain falls back (two-level → single-level → scalar). Correctness:
odd/rectangular shapes (130x257x64, 257³, 1000³) verified; full packet
green with the lane on and off; HIP dlopen coexistence retained.


Cross-backend sync `JIT-VECTORIZE-UNGATED-2026-08-23` — **the
`TESSERA_JIT_VECTORIZE` GEMM lane now actually vectorizes on LLVM 23:
3.4 → 106.6 GFLOP/s at n=256 (AVX-512 host; 69.0 at 512, 44.4 at 1024,
correctness within the reassoc GEMM tolerance).** Deep-loop findings,
each with a control: **(1) the LLVM-23 scalar carve-out was closed on a
wrong root cause and had never been re-checked** — every fleet box is
LLVM 23 and the compat test was Darwin-gated, so the gate itself kept
the vectorized path unexercised everywhere (the Decision #19
standing-lesson pattern); the recorded bufferization abort no longer
reproduces (verified with the registration absent — the gate removal,
not the registration, unblocked the lane). The tensor
SubsetOpInterface external models are now registered at engine setup
regardless (their absence is MLIR's abort-not-failure path when
bufferization queries an insert_slice).

> **Corrected 2026-08-23 (M1 Max, apple plan `APPLE-VECTORIZE-1`).** The
> abort *does* still reproduce, and the tensor-only registration was
> incomplete. The lane emits `vector.transfer_write` into a tensor, and
> the `vector` dialect promises `SubsetInsertionOpInterface` for it;
> `linalg` carries the same promise. On Darwin the first vectorized
> compile died with `LLVM ERROR: ... promised by dialect 'vector' but
> never implemented`. **This host could not have falsified the claim:**
> the promise check is `#ifndef NDEBUG` (`mlir/IR/OpDefinition.h`,
> `getInterfaceFor`) and the apt.llvm.org LLVM 23 used here is an NDEBUG
> build, so it silently loses the interface (conservative extra copies,
> still correct) instead of aborting. The Mac's LLVM 23.1.0-rc1 has
> assertions on and is currently the only box in the fleet that can
> falsify an MLIR contract claim of this kind — treat any "this MLIR
> abort no longer reproduces" conclusion reached here as provisional
> until it is re-run there. All three registrations (`tensor`, `linalg`,
> `vector`) are now in `tools/tessera-jit`.
>
> **Re-taken on this host the same day, so this is not a Mac-only claim.**
> `llvm-config --assertion-mode` here reports **OFF**, confirming the
> mechanism rather than assuming it; and the *control* — same commit, clean
> tree, unmodified tensor-only registration — passes
> `test_native_cpu_jit.py` **27/27**, i.e. the defect really is invisible
> here. With the patch applied, `tessera_jit` rebuilds clean and the
> native-CPU-JIT / signature-guard / totality / boundary-discovery /
> native-required packet is **73/73**. An alternating A/B of the two
> shared libraries at n=512, three reps each, is **indistinguishable**
> (60.0 / 67.6 / 74.2 pre-fix vs 75.7 / 86.9 / 55.3 GFLOP/s post-fix):
> run-to-run variance on this box swamps the effect, so **no performance
> change is claimed in either direction** — an earlier single-shot pair
> looked like a 24% regression and did not survive repetition. **(2) With the env var set,
every non-matmul module failed to compile** (the transform's empty
matmul match hard-failed stage 1b — measured 114 packet failures). The
lane now engages only for matmul-bearing modules, transforms a CLONE,
and swaps it in only on full success — any transform failure falls
back to the always-correct scalar pipeline. Pinned by
`test_vectorize_env_does_not_break_non_matmul_modules`. **(3) The
lane's `libmlir_c_runner_utils` dlopen was process-fatal beside HIP:**
that library links the dynamic `libLLVM.so`, and loading it alongside
libtessera_jit's static LLVM made a later `dlopen("libamdhip64.so")`
(comgr embeds a third LLVM) segfault — the ROCm state-machine tests
died the moment they loaded HIP after a vectorized compile (staged
probe: compile/invoke/heap all clean, dlopen fatal; runner-utils + HIP
without tessera_jit coexist fine). Fixed by implementing `memrefCopy`
in-process and registering it via ORC `registerSymbols`; the engine no
longer dlopens anything, which also deletes the hardcoded Homebrew
path. Full packet green with the lane ON and OFF (incl. the ROCm
state-machine suite + HIP loads). Mac note: the un-gating affects
Darwin too (also LLVM 23) — the vectorized path there is UNPROVEN
until the Darwin compat test is re-run on the M1 Max (recorded in the
apple plan). Follow-on opportunity: throughput falls with size (106 →
44 GFLOP/s at 1024) — no cache-level blocking above the 8x16x16
register tiles yet.


Cross-backend sync `JIT-MATH-AUDIT-2026-08-23` — **x86 host-JIT boundary
hardening + math-correctness pass: CLOSED (AVX-512 host).** Two defects
fixed, both measured on this host. **(1) `jb.invoke` boundary now fails
closed on signature mismatch.** The identity-layout ABI bakes static
extents into the generated indexing math, so a wrong-shape buffer was
guaranteed out-of-bounds (reproduced: a 2-element array against a
`tensor<7xf32>` module corrupted the heap and aborted in
`malloc_consolidate`). `tessera_jit` now records each function's
tensor-level signature at compile time (new `tessera_jit_signature` ABI),
`tessera_jit_invoke` validates arity + static extents from the
descriptors (memory-safety backstop for any caller), and Python
`invoke()` validates arity/rank/extents/dtype with actionable errors;
dynamic extents stay unconstrained.
`tests/unit/test_jit_invoke_signature_guard.py` (12 tests) pins both
layers, including the heap-corruption repro. **(2) The pipeline's
blanket `fastmath<fast>` stamp is narrowed to `reassoc|contract`.**
`fast` (nnan|ninf|nsz|arcp|afn) applied to every float add/sub/mul/div/neg
in the module: `arcp` measurably rewrote elementwise x/y into x*(1/y)
(1-ulp divergence vs correctly-rounded division), and nnan/ninf made
NaN/Inf inputs (e.g. -inf attention-mask biases) poison — latent on this
toolchain, legal to break. With reassoc|contract, division is bit-exact
vs numpy, NaN/Inf propagate bit-exactly through all four ops, and GEMM
throughput is retained (n=256 scalar lane: fast 3.34, narrowed 3.45,
no-fastmath control 2.7-2.8 GFLOP/s; n=512 was not comparable — WSL2
process-level timing variance swamped it). New regression tests pin
NaN/Inf propagation and correctly-rounded division in
`test_jit_tensor_elementwise_totality.py`. Full packet after both
changes: 219 passed (Darwin-only lanes skipped). Cross-target note: the
gfx1151 binary lane's signed-zero tie semantics deliberately differ —
see the rocm entry under this key.


Cross-backend sync `JIT-ELEMENTWISE-LINALG-2026-08-21` — **shared
`tessera_jit` pipeline change; x86 outcome: parity validated, widened
family included (AVX-512 host, 2026-08-23).** `tessera_jit` rebuilt on the
Strix Halo box against the module-scope `convert-elementwise-to-linalg` +
residual legality gate; the widened packet passed:
`test_jit_tensor_elementwise_totality.py` **22/22**, native CPU JIT +
production phase 1/3 lanes **182 passed** (all 160 skips are Darwin-only
Apple/MTL4 lanes, correctly unevaluable on this host), and the paired
state machines re-ran green (x86 3/3 native; the sibling gfx1151 rows
5/5 exact-device, recorded in the rocm plan).  The rerun surfaced one
defect — in the test, not the lane: the totality suite's signed-zero
min/max assertions used numpy as oracle, whose ±0 tie resolution is
host-dependent (SSE returns the second operand; NEON orders the zeros),
so the assertion encoded the test host's ISA and failed 4 cases here
while passing on M1 Max.  Probed on this host, the JIT is
contract-correct per the arith ODS: `maximumf`/`minimumf` order signed
zeros (ties → +0.0 / −0.0), and `maxnumf`/`minnumf` ties are "either of
them".  The test now keys signed-zero expectations off the arith
contract (num-variant tie signs unasserted, as the contract specifies
neither); NaN propagation and finite-value checks keep the numpy oracle,
which is host-stable for those.


Cross-backend sync `APPLE-MINMAX-1-2026-08-23` — **Apple closed the
contract on its own hardware; x86 outcome: no change required, one open
question inherited.** The M1 Max audit found MPSGraph's max/min
non-conforming on both NaN and the ±0 tie and fixed it with the *same*
bitwise AND/OR tie blend this backend's AVX-512 vector body uses, so the
two agree by construction rather than coincidence. Apple has no
Adafactor device kernel and no `max(stat, eps)` floor anywhere, so the
NaN-laundering half of `JIT-MATH-AUDIT-2026-08-23` had no Apple sibling.
Metal evidence transfers nothing here. **Inherited open item:** the Apple
audit measured `relu(NaN) = 0` on device against a `np.maximum(0.0, x)`
reference that returns NaN, and deliberately did not fix it — the x86
relu lane is equally unaudited for this. Decide the `relu` NaN contract
fleet-wide before fixing any single backend.

Cross-backend sync `IEEE-MINMAX-CONTRACT-2026-08-23` — **x86 outcome:
VALIDATED (AVX-512 host).** The fleet-wide IEEE-754-2019 ±0-tie
decision (rocm plan owns the key) lands here in three routes: the
AVX-512 binary shim (`avx512_binary_f32.cpp` — scalar tail was
`a > b ? a : b`, vector body `vmaxps`, both second-operand-on-tie; now
signbit-select / bitwise AND-OR tie blend), and the two
backend-neutral numpy routes the codex review caught as still
route-dependent: the eager op namespace (`__init__.py` maximum/minimum)
and the Apple-lane numpy fallback (`_apple_gpu_binary_numpy`) now share
one reference implementation (`tessera/_ieee_minmax.py`, Decision #31)
instead of delegating to np.maximum/np.minimum, whose tie sign is
host-ISA-dependent (SSE second operand, NEON IEEE). Pinned by
`test_x86_binary_max_min_signed_zero_ties_are_ieee_ordered` (n=19:
vector body + scalar tail) and the route-consistency tests in
`test_ieee_minmax_reference.py`.


Cross-backend sync `JIT-MATH-AUDIT-FIXES-2026-08-23` — **x86 outcome:
VALIDATED (AVX-512 host).** The Adafactor NaN-floor fix (rocm plan owns
the key): the AVX-512 optimizer shim floored second-moment statistics
with `std::fmax` (NaN-suppressing), laundering a NaN statistic into eps
— identical to the ROCm maxnumf defect. All 13 floors now go through a
NaN-propagating helper; exact-host test
(`test_adafactor_factored_nan_gradient_propagates_like_reference`)
proves a NaN gradient poisons the full row+col exactly as the optim.py
reference does. The softmax maxnumf change is ROCm-kernel-only — no
x86 surface compiles through it.


Cross-backend sync `W4-SM-ROCM-2026-08-21` — **W4-PRODUCT-1 x86 outcome:
VALIDATED (AVX-512 host, 2026-08-21).** The sibling row landed: the same
paired `bounded_state_machine_v1` functions (forward + recompute_all
backward, both entry paths of the two-entry irreducible SCC, plus the
per-element cmpf→select machine) compile through `tessera_jit`'s
MLIR→LLVM→ORC chain and execute natively — digest/residual-policy bound,
native `cf.assert` bound trap (stronger than ROCm's host-checked STATUS),
proof-of-execution counter (`tests/unit/test_x86_state_machine_exec.py`).
One shared fix rode along: `convert-elementwise-to-linalg` joined the
tessera_jit pipeline (tensor-typed arith from the paired cotangent
accumulation had no bufferization interface). Correctness-only rows;
clean Zen 5 timing remains open per W4.3. Original follow-up text: gfx1151 landed the
first irreducible-state-machine execution rows (see the rocm entry). The
W4-PRODUCT-1 acceptance names native x86 rows too: an AVX-512-host
consumer for the same paired `bounded_state_machine_v1` functions (the
per-thread scalarization model ports directly; the x86 lane can execute
the structurized SCF via the existing compiled-kernel path or an
LLVM-lowered CPU kernel) plus the same digest-binding and host-enforced
bound check. Same box as ROCm — schedulable in a follow-up session.


Cross-backend sync `AD-DATUM-POLYGAMMA-2026-08-21` — **autodiff reference
numerical policy, wave 3; x86 outcome: parity validated (AVX-512 host).**
Same contract change as the rocm entry (lgamma/digamma datum-derived over
the polygamma tower; rmsnorm γ operand). Validated on the AVX-512 primary
box: all six `test_x86_*_loss_compiled.py` backward lanes green against
the switched registry (part of the 95/96 device-lane run recorded in the
rocm entry).


Cross-backend sync `AD-RETIRE-2-2026-08-20` — **autodiff reference numerical
policy, wave 2; x86 outcome: parity validated (AVX-512 host).** Same
contract change as the rocm entry (structured trio jet-derived; six ops
datum-grown and switched). Validated on the AVX-512 primary box: 169 tests
across the six `test_x86_*_loss_compiled.py` backward lanes green against
the switched registry. AMX half not applicable (no fleet hardware).


Cross-backend sync `AD-RETIRE-1-POINTWISE-2026-08-20` — **autodiff reference
numerical policy; x86 outcome: parity validated (AVX-512 host).** PR #600
retires the ODE-family pointwise hand rules behind the `DerivativeContract`
datum (dtype-preserving reference rules; unified log/sqrt boundary guard —
see the rocm entry for the full contract statement). x86 impact: the
compiled-loss backward lanes compare against this reference; validated on the
AVX-512 primary box — 169 tests across
`test_x86_{loss,binary_loss,class_loss,metric_loss,ebm_loss,rl_loss}_compiled.py`
green against the retired registry. The AMX half remains not applicable (no
fleet hardware; capability-gated as before).


Cross-backend sync `APPLE-RUNTIME-SINGLE-IMAGE-2026-08-19` — **Apple runtime
loading; x86 outcome: not applicable.** The single-image slice fixes duplicate loading of the Apple GPU runtime.
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
x86 impact: none. The AVX-512 lane loads `libtessera_x86_elementwise.so`
through its own path, not through `_apple_gpu_dispatch`. No Zen 5 retest
required and no device evidence is produced or claimed.


Cross-backend sync `APPLE-STUB-BINARY-OPCODES-2026-08-19` — **shared runtime
contract; x86 outcome: not applicable today, but one follow-up recorded.**
The portable-stub opcode slice fixes a silent wrong-answer class in the Apple
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
x86 impact: none from this change. The AVX-512 elementwise binary lane is
separate code (`_execute_x86_compiled_binary` →
`tessera_x86_avx512_binary_f32` / `tessera_x86_reference_binary_f32` in
`src/compiler/codegen/tessera_x86_backend/src/kernels/avx512_binary_f32.cpp`)
and does not call the Apple symbol.

**Follow-up recorded (not fixed here):** that kernel carries the SAME
silent-fallthrough shape — `scalar_binary` ends `default: return a;` and the
vector loop ends `default: y = a; break;`, both returning operand A for an
unrecognised kind. It is **latent, not live**: the kernel implements kinds 0-7
(`kSub`/`kDiv`/`kMax`/`kMin`/`kAdd`/`kMul`/`kMod`/`kFloorDiv`) and
`_X86_BINARY_OPS` maps exactly onto that range, so no reachable call hits the
default today. It becomes live the moment an opcode is added on one side only —
which is precisely how the Apple stub defect arose. Fixing it belongs in an x86
change with AVX-512 evidence from the Zen 5 box; no Zen 5 retest is required for
the present change and no device evidence is produced or claimed.
Cross-backend sync `ZERO-FUNCTION-CANDIDATE-2026-08-19` — **shared frontend ABI
and diagnostics; x86 outcome: not applicable today; parity by construction.**
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
x86 impact: none today, for the same reason as ROCm — a non-apple_gpu target
re-raises at decoration instead of deferring to the tracer. No Zen 5 retest
required and no device evidence is produced or claimed.


Cross-backend sync `SCALAR-SIDE-ORDERING-2026-08-19` — **shared Graph IR
runtime contract; x86 outcome: not applicable today, fails closed by
construction.** The `scalar_side` slice (PR #589) makes the Graph IR lifted-scalar form carry
operand order. `graph_ir._OpExtractor._try_map_binop` lifts a literal out of
either side of a `BinOp` into the `scalar` attribute and records the side; until
now no code in `python/`, `src/`, or `tools/` read that record (Decision #29), so
`2.0 - x` and `x - 2.0` emitted indistinguishable IR and any consumer binding
`scalar` as the right operand computed `x - 2.0` for both — sign-flipped for
`sub`, reciprocal for `div`, with no diagnostic. Shared contract changed: a lone
`scalar` means the RIGHT operand, `scalar_side="left"` requests the mirrored
binding, and any other value is rejected rather than guessed (Decision #21).
x86 impact: none. `runtime._execute_x86_compiled_binary` binds both operands
positionally and raises `"binary math requires two operands (a, b)"` when fewer
are present, so the lifted-scalar form cannot reach the AVX-512 elementwise
kernel. No Zen 5 retest required and no device evidence is produced or claimed.
Follow-up: `_X86_BINARY_OPS` covers sub/div/mod/floor_div, so if that lane later
accepts a scalar kwarg form it must honor `scalar_side` first.


Cross-backend sync `AD-LAW-SERIES-2026-08-19` — **shared reference rules and
test infrastructure; X86 outcome: parity validated, no AVX-512 evidence changed.** The AD-LAW series (PR #588)
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
x86 impact: the AVX-512 backward packages (incl. the selective_ssm `bwd_hardware_proven` row) compare against these oracles. Same split as ROCm — forward-mode references moved, VJP-side values did not. No Zen 5 retest required; no device evidence produced or claimed. The previously recorded open spectral/quantize swallow findings are
therefore CLOSED; `_OPEN_FORWARD_KEY_SWALLOWS` (42 entries from the tape
positional-routing scan) remains the open set.


Cross-backend sync `AD-LAW-1-SHARED-ORACLE-2026-08-18` — **shared test
infrastructure; x86 outcome: parity validated, no AVX-512 evidence
changed.** AD-LAW-1 (PR #584) adds law oracles (adjoint `⟨Jv,u⟩ = ⟨v,Jᵀu⟩` +
canonical-forward chain check) over the shared numpy reference JVP/VJP
registries — the oracle lane the AVX-512 backward packages (incl. the
selective_ssm `bwd_hardware_proven` row) differentially compare against —
plus the byte-gated `autodiff_law_audit` dashboard. Reference-rule fixes in
the same PR: `jvp_rmsnorm` eps default (1e-6 → the forward's 1e-5) and
`jvp_clamp` swallowing canonical `min`/`max`. x86 impact: norm bindings pass
`eps` explicitly and VJP-side defaults did not move, so no Zen 5 retest is
required; no device evidence is produced or claimed by this gate. Open
shared follow-up: 20 pinned swallowed-kwarg findings
(`test_autodiff_laws.py`) await triage. *Triage update, same key (AD-LAW-1b):* reference JVP fixes landed for `clip` alias deafness, `add`/`mul` unary-`scalar`, and fft/ifft/rfft/irfft `norm` handling (√n-wrong under `norm="ortho"`); five entries benign-classified with recorded reasons; the open set is now the stft/istft/spectral_conv and quantize families only, riding their owning family reviews. *Spec-growth update, same key (AD-LAW-1c):* law coverage roughly doubled (109 adjoint / 87 chain rows green incl. attention, spectral-complex, structural, and loss families); two more silent reference JVP defects found and fixed — `lgamma` (derivative was a dead stub returning 0) and `digamma` (whole JVP was an identity placeholder) — plus `jvp_cast` crashing on canonical dtype strings. Forward-mode reference oracles for those three ops changed; reverse-mode VJPs are untouched, so no backend backward package is affected. Reflection formulas added to the shared polygamma helpers (`_digamma_positive`/`_trigamma_positive`): the upward recurrence advanced by 1 per step, so a valid input like -1e9+0.5 spun for ~10^9 iterations — a live defect in the **reverse** path too, since `vjp_lgamma`/`vjp_digamma` already call these. Now O(1) on the whole real line, exact against the canonical forward, poles -> nan..

Cross-backend sync `W4-DYNAMIC-EFFECT-NONLINEAR-CFG-2026-08-18` — **shared
contract parity; AVX-512 evidence unchanged.** Dynamic region state now uses
bounded per-slot data tapes plus recorded logical-shape tapes. Polynomial shape
guards require complete concrete witnesses and do not enter Presburger proofs.
Variadic branch state reaches the structured CFG carrier. Only compiler-owned
extent assertions are replay-safe; mutation/RNG/I/O/ordered collectives remain
fail closed. A clean Zen 5 irreducible/dynamic-state packet is still required.

Cross-backend sync `W2.4-E2E6-SYMBOLIC-2026-08-18` — **shared parity
validated; AVX-512 evidence unchanged.** Relational Tile legality now has one
staged production authority, while old CLI passes are wrappers. Pure static
annotations abstract-trace before AST compatibility capture. NVIDIA
Lion/DeltaNet dispatch moved into family plugins; x86 keeps its independent
AVX-512 state-lineage packages and inherits no CUDA evidence.

Cross-backend sync `W1-W3-AUTHORITY-CLOSEOUT-2026-08-18` — **shared and x86
plugin parity validated; no Zen 5 timing claim.** Bare Tile fragments and unknown
semantic selectors fail closed, whole-program memory activity is conservative,
and regression/BCE/class-loss VJPs use explicit x86 plugins. The x86 no-async
path is unchanged. Remaining public-module/frontend decomposition and clean
AVX-512 evidence are independent follow-ups.

Cross-backend sync `W4-PRODUCT-1-RESIDUAL-CONTRACT-2026-08-17` — **shared
carrier and bounded AVX-512 correctness consumer landed.** SAVE/HYBRID selection now
retains its exact checkpoint, CFG, and residual identity through Graph→SCF.
Shared paired AD now consumes dynamic branch-local residual extents through
zero-extent inactive sentinels, bounded SAVE and sparse HYBRID `while` state
tapes, and bounded-dynamic counted-loop tapes. Source-CFG SCC analysis now
distinguishes acyclic, reducible, and true multi-entry irreducible graphs.
Bounded pure native graphs lower to a typed program-counter state machine with
nested canonical structured bodies and mixed control/tensor state. Saved
dynamic slots now require total data/shape-tape envelopes; unbounded,
unsupported-region, and unrecorded effectful forms remain fail closed. The committed Zen 5
WSL packet binds paired-IR/CFG/residual digests and executes AVX-512 children
without Graph re-entry or predicate replay. It is correctness evidence only;
clean bare-metal timing remains required for performance promotion. An exact
Zen 5 irreducible-state-machine row remains required before physical x86
execution of that new form is claimed.

Cross-backend sync `E2E-REAL-6F-OPTIMIZER-VJP-2026-08-17` — **bounded
AVX-512 SGD and Momentum/Nesterov reverse authority complete.** The plugins
bind one-execution tracer proof and functional state lineage through typed
`schedule.optimizer_vjp` → `tile.training_kernel` packages before the existing
AVX-512 calls. Adam/AdamW remain fail closed on x86 because no physical reverse
consumer exists; clean Zen 5 timing remains a separate evidence gate.

Cross-backend sync `E2E-REAL-6E-STATEFUL-VJP-2026-08-17` — **bounded AVX-512
Adafactor and sequence-mixer reverse authority complete.** Explicit plugins
now own full/factored Adafactor and causal gated/Kimi/modified DeltaNet
backward; the compiler route binds tracer identity, one-execution proof,
state/workspace lineage, Schedule, and exact Tile identity without passing
Graph metadata to the runtime. Existing AVX-512 numerical tests remain valid;
clean Zen 5 selector-grade timing is a separate evidence gate and no AMX claim
is implied.

Cross-backend sync `E2E-REAL-6D-LION-VJP-2026-08-17` — **bounded AVX-512
Lion reverse authority closed.** The flat functional Lion Graph ABI now has
exactly two results, `(new_param, new_moment)`, and concrete tracing executes
that source once. A structural non-reexecuting certificate binds the retained
AST candidate to tracer Graph IR; the family plugin then builds the existing
`schedule.lion_vjp` → `tile.training_kernel` state-lineage package. Runtime
receives no Graph operation metadata and validates the frontend certificate,
state lineage, Schedule identity, and exact Tile digest before the AVX-512
call. Public numerical, one-execution, and tamper tests pass. Clean Zen 5
performance evidence remains independent.

Cross-backend sync `E2E-REAL-6C-ATTENTION-VJP-2026-08-17` — **bounded public
AVX-512 canonical rank-4 attention reverse authority closed.** `flash_attn`,
GQA, and MQA now enter one native-VJP family plugin from tracer-produced Graph
IR. The plugin builds the exact `schedule.attention_backward` →
`tile.attention_backward_kernel` package, preserves x86’s saved-LSE identity,
and binds parent Graph, Schedule, Tile, native-image, and aggregate digests
before runtime. `JitFn` no longer constructs the x86 attention package. Public
ragged GQA numerical proof and lineage/tamper negatives pass for the explicit
zero-dropout differential envelope; active dropout remains fail-closed until
keyed replay can be certified without duplicate effects. The rank-3
`multi_head_attention` wrapper remains a compatibility path until its
reshape/transpose product is explicit; clean Zen 5 performance evidence is
independent and remains open.

Cross-backend sync `E2E-REAL-6B-SPECTRAL-VJP-2026-08-17` — **bounded AVX-512
compound spectral reverse execution closed.** `spectral_filter` and
unbroadcast full `spectral_conv` now enter one native-VJP family plugin from
tracer-produced Graph IR. Source, Schedule, and Tile digests are validated
before the existing AVX-512 adjoint ABI executes; public numerical and
fail-closed tests cover filter, convolution, unsupported broadcasting, and
tampered lineage. Broader axes/dtypes and STFT/ISTFT backward remain separate
work, and clean Zen 5 performance evidence remains open.

Cross-backend sync `GFX1151-CALIB-BAREMETAL-2026-08-16` — **shared calibration
authority parity validated; no x86 evidence transfers.** `target_perf` now
rejects explicitly provisional and WSL-hosted corpora from its measured
selector registry while exposing a non-mutating pruning reader. AVX-512 code
and selectors are unchanged; clean Zen 5 perf/IBS evidence remains the x86
promotion authority.

Cross-backend sync `LAYOUT-SCHEDULE-OBJECT-2026-08-16` — **shared layout and
SO-1 parity validated; SO-2 no-async proof explicit.** The versioned C++
layout ABI is host-free and the GQA-fold consumer is common. Schedule Object
digests now bind R3 actions, edges, roles, residency, and resource vectors. x86
has no asynchronous-copy/mbarrier lane, so every typed family plugin declares
`async_role_policy = no_async_noop`; the SO-2 role carrier remains a verified
no-op for AVX-512 lowering. No x86 index template or raster output is
changed until L4, and no Zen 5/AMX evidence transfers.
Nested Schedule resource metadata is now frozen before hashing, and dynamic
rearrange/GQA-fold inference preserves ranked `?` dimensions. Neither change
selects a new x86 layout or raster policy.

Cross-backend sync `ATTN-BWD-ARCH-2026-08-16` — **deterministic parallel VJP
implemented; bare-metal selector packet remains open.** AVX-512 attention
backward partitions query rows into deterministic contiguous worker ranges,
uses bounded private dK/dV partials, and reduces workers in fixed order. MHA,
GQA/MQA, saved/recomputed LSE, bias, window, softcap, ragged shapes, and repeated
bit-determinism pass. An equal-flags Zen 5 WSL comparison measured 1.85x median
speedup at B2/Hq8/Hkv2/S128/D64 with 1.67e-6 maximum absolute delta. This is
regression evidence only; clean bare-metal timing remains required for a
selector-grade claim.
Allocation and thread-construction failures are contained inside the C ABI:
started workers are joined, outputs reset, and the deterministic serial path
replayed, so no exception or joinable-thread destructor can terminate a caller.

Cross-backend sync `PDE-EXACT-CONTRACT-2026-08-14` — **shared exact semantic
authority landed; AVX-512 physical follow-up required.** The compiler now owns
a typed constant-coefficient PDE carrier, exact-rational principal-symbol
classification, and a fail-closed centered-FTCS diagonal-diffusion certificate
with non-unit spacing. This adds no x86 Target record, vector package,
transport, or clean Zen 5 packet.

Cross-backend sync `DIST-SHARD-HVP-2026-08-14` — **shared SSA/product
foundation landed; native process transport remains open.** Planned bulk
reshards now become real Graph→Schedule→Tile SSA carrying digest, subgroup,
region, and deterministic all-to-all matching-round identity. Exact compiler
forward-over-reverse HVP is also live. This adds no MPI/OFI/SHMEM launcher,
native AVX-512 HVP package, or multi-rank/Zen 5 performance packet. Local-shard
result typing and process-transport evidence remain architecture-owned gates.

Cross-backend sync `E2E-REAL-6-NATIVE-VJP-2026-08-14` — **AVX-512
normalization ownership migrated.** RMSNorm and LayerNorm reverse execution
now enters one native-VJP family plugin with explicit Graph/Schedule/Tile/x86
Target consumers. `JitFn` binds inputs and records the result but no longer
constructs the normalization backward package. Existing Zen 5 correctness
evidence remains authoritative; this synchronization adds no performance or
AMX claim. Other x86 backward families remain compatibility paths.

Cross-backend sync `AMD-ISA-DTYPE-2026-08-14` — **parity assessed; no x86
physical change required.** The shared change is confined to AMD architecture,
datatype-role, and matrix-instruction selection. It changes no Graph dtype
spelling, public operation, Schedule contract, AVX-512/VNNI ABI, or AMX gate;
x86 evidence remains independent.

The shared `OP-DTYPE-FLOW-1` generator now audits every x86 operator/storage
pair from frontend through physical manifests. Accumulator-compatible dtypes
derived from target-wide AVX-512 legality remain `legal_only`; only explicit
x86 per-operation rows can claim a physical consumer or execution evidence.

## P0 root-caused — x86 dialect load was a build-flag leak, not an IR defect

`X86-DIALECT-LOAD-CRASH-2026-08-12` was **reopened by CI on 2026-08-15** (run
31893492411) after being closed on the Strix Halo host, and is now root-caused.
Both observations were correct; the host is what differed.

The dialect and its `TileType` registration were never at fault. The x86 kernel
project applied its detected AVX-512/AMX flags with **`add_compile_options`**,
which is *directory* scoped, and `add_subdirectory(lib/IR)` sat below that call
— so the hardware-free Target IR dialect was compiled with `-mavx512f
-mavx512bw … -mamx-tile …`. The compiler was then entitled to emit an AVX-512
encoding into dialect registration itself (confirmed by disassembly:
`vpbroadcastq %rgpr, %xmm`, a GPR-source broadcast that exists only under
AVX-512). On Zen 5 that instruction is legal, so the Strix Halo build ran
clean and the P0 read as closed. On the GitHub runner it is not, and
`tessera-opt` died the first time it touched the dialect.

The tell was in the exit status all along: **all 14 fixtures failed with signal
4 (SIGILL) at one identical address**, not SIGSEGV. An illegal instruction is a
build-configuration fact, not a memory-safety one — a crash *inside*
`Dialect::addType<TileType>()` was the first code from that translation unit to
execute, not the code that was wrong.

Fix: the flags are collected into `TESSERA_X86_ARCH_FLAGS` and applied with
`target_compile_options` to this directory's kernel targets only, enumerated via
`BUILDSYSTEM_TARGETS` so a later kernel target is covered automatically while
subdirectories are structurally excluded. `lib/IR/CMakeLists.txt` additionally
**fails configure** if any host-specific ISA flag is in scope, so the next
occurrence is a build error naming the cause rather than a runtime SIGILL that
reads as a corrupt MLIR install (Decision #21a, fail closed).

Standing lesson for this backend: a host that *has* the ISA cannot falsify a
host-portability claim. Decision #19's "lit-testable on any host" is only
evidenced by a host without AVX-512 — which, in the current fleet, means CI.

The named `x86_dialect_load.mlir` regression isolates dialect initialization
from operation lowering, so a future `TileType` registration failure cannot hide
behind the larger lit suite. Acceptance evidence for this fix is the green
`Validate / lit` lane on a non-AVX-512 runner, not a local rebuild.

## P0 fixed 2026-08-16 — `TileToX86Pass` declares its dialect dependency

Found while verifying the fix above, on an **assertions-enabled** LLVM/MLIR 23.
Distinct defect, same code path, and **CI structurally cannot see it**.

`src/transforms/lib/TileToX86Pass.cpp:1045` calls
`getContext().getOrLoadDialect("tessera_x86")` from inside `runOnOperation()`,
by name, to avoid linking the optional backend library. MLIR forbids loading a
dialect during pass execution:

```
LLVM ERROR: Loading a dialect (tessera_x86) while in a multi-threaded execution
context (maybe the PassManager): this can indicate a missing `dependentDialects`
in a pass for example.
```

That guard is `#ifndef NDEBUG`. The CI lit lane builds Release against apt
LLVM/MLIR 23 (assertions off), so the check is compiled out and the load becomes
silent undefined behaviour that happens to survive. On an assertions-enabled
build it is a hard error that fails **12 of the 15** x86 fixtures — i.e. the
whole x86 lit suite is currently unrunnable on a normal assert-enabled developer
toolchain, which is the second reason this backend's Decision #19 claim has been
easy to overstate.

The layering decision is now explicit. The hardware-free `TesseraX86IR` target
is created before `TesseraPasses` when x86 is enabled; the pass links that small
IR library, declares `TesseraX86Dialect` from `getDependentDialects()`, and is
compiled with `TESSERA_HAS_X86_TARGET_IR`. Native AVX-512/AMX kernel targets
remain later and do not enter the shared pass dependency. Builds without x86
retain the pass registration but fail before rewriting with an instruction to
enable `TESSERA_BUILD_X86_BACKEND`; there is no permissive unregistered marker
path. The forbidden `getOrLoadDialect("tessera_x86")` call is deleted.

The final composite-pipeline failure was a second namespace-authority defect:
`DistributionLoweringPass` defined a permissive private `schedule` dialect in
parallel with the ODS Schedule dialect. Composite x86 pipelines attempted to
register both classes and aborted. The canonical dialect is now isolated in
`TesseraScheduleIR`, both the programming-model and transform libraries link
that one target, and the private dialect is deleted.

Host LLVM/MLIR 23 validation rebuilt both x86-present and x86-absent drivers.
The positive Target fixture, negative verifier fixture, direct dialect-load
fixture, and executable family pipeline pass 4/4; the absent driver returns the
named fail-closed diagnostic before mutation. The installed WSL LLVM reports
assertions `OFF`, so an assertions-enabled toolchain rerun remains required as
the final external evidence packet; the source regression test independently
forbids reintroducing a runtime dialect load.

The production HIP build was also rebuilt with both
`TESSERA_BUILD_ROCM_BACKEND=ON` and `TESSERA_BUILD_X86_BACKEND=ON`. Its feature
ledger reports both `rocm-backend` and `x86-target-ir`; the complete shared lit
suite reports 329 enabled passes, 52 configuration-gated tests, and zero
failures across 381 discovered tests; the x86 verifier pair plus all twelve explicitly
gated ROCm fixtures execute 14/14 rather than becoming unsupported. This is
host-free compiler proof, not clean Zen 5 performance or AMX evidence.

---

Cross-backend sync `CI-LIT-BACKEND-DIALECTS-2026-08-12` — **x86 and combined
host proof closed; host-free lit coverage restored, no Zen 5 timing evidence
added.** The
`Validate / lit` lane was dead from 2026-08-11 to 2026-08-12 (pytest collection
aborted on a missing `ml_dtypes`, fixed in #554). The first green-collection run
discovered 367 fixtures, passed 318, and failed 27 — all from one cause: the
lane's `tessera-opt` was configured with `TESSERA_BUILD_APPLE_BACKEND=ON` only,
so `TesseraX86IR` never existed and `tessera_x86` went unregistered
(``error: Dialect `tessera_x86' not found for custom op
'tessera_x86.amx_tile_zero'``). x86 Target IR has therefore had **zero CI lit
coverage since the W0.10 dialect landed (2026-08-02)** — Decision #19's
"lit-testable hardware-free layer" was asserted but never exercised on CI.

This PR adds `-DTESSERA_BUILD_X86_BACKEND=ON` to that configure, recovering:
`phase2/x86_target_ir{,_invalid}.mlir` (including the negative fixture Decision
#19 requires), `phase2/tile_to_x86.mlir`, `phase2/tile_x86{,_base}_e2e.mlir`,
`phase2/x86_executable_pipeline.mlir`, `phase2/x86_kv_cache_lowering.mlir`,
`phase2/e2e_matmul_scheduled_x86_consumer.mlir`, and the x86 half of
`phase_f4/{es_low_rank_correction,spectral_backward}_x86_native.mlir`.

Scope limits, stated explicitly: this is **host-free IR/registration coverage
only**. It adds no AVX-512 physical consumer, no Zen 5 timing packet, and no
AMX evidence (no AMX hardware exists in the fleet; that lane stays
capability-gated). The flag does not gate on `CMAKE_SYSTEM_PROCESSOR`
(`src/CMakeLists.txt:44`), so it builds the dialect on the runner without
requiring AMX/AVX-512 at build time. Green-on-CI is the acceptance evidence;
if any recovered fixture fails, it is a real x86 defect that this lane was
previously hiding and it outranks the CI change.

**Sequencing note — x86 and ROCm share one host, so schedule them together.**
Strix Halo is Zen 5 + gfx1151 in a single box: it is the fleet's x86 host and
its ROCm host at once. Because `TESSERA_ENABLE_HIP=ON` avoids the `:95` lean
condition, one configure there carries core + ROCm + x86 in a single
`tessera-opt` — the only fleet configuration that runs both fixture families in
one `lit` invocation. Whoever picks up the ROCm side of
`CI-LIT-BACKEND-DIALECTS-2026-08-12` should add `-DTESSERA_BUILD_X86_BACKEND=ON`
and close the x86 host-side verification in the same session rather than
scheduling a second visit to the same machine; the procedure lives in
`docs/audit/backend/rocm/todo.md` under this sync key. Two caveats that do not
change: the box runs under WSL2, so host-wall timing there stays
**regression-only** and does not satisfy the clean AVX-512 timing packet still
open above; and Zen 5 has AVX-512 but **no AMX**, so nothing on this host can
retire the AMX lane.
Cross-backend sync `CI-LIT-DEPS-2026-08-12` — **parity validated; no physical
follow-up required.** PR 554 made the shared opt-in MLIR lit lane install the
workflow-owned Python dependency set before `lit`/FileCheck collection. This is
backend-neutral test infrastructure, changes no compiler/runtime contract, and
requires no x86 package or exact-device evidence.

Cross-backend sync `PDE-STENCIL-FOUNDATION-1-2026-08-12` — **CPU gradient
correctness landed; broader AVX-512 follow-up required.** The executable CPU
gradient ABI consumes explicit axis spacing and passes non-unit-grid numerical
tests. General stencil apply, boundary, and halo primitives remain
artifact-only; x86 still owns their typed consumers and a clean Zen 5 packet.
No gfx1151 evidence transfers.

Cross-backend sync `BLOCK-ATTNRES-ROCM-2026-08-12` — **follow-up required.**
The shared Block AttnRes numerical-policy, balanced-partition, VJP, and
softmax-merge oracle contracts apply to x86. The exact static all-f32 Graph
contract now lowers through the same content-addressed
`schedule.depth_attention` → `tile.depth_attention_kernel` boundary, with an
x86-owned workgroup policy included in its digest. This adds no AVX-512 Target
consumer or Zen 5 evidence. After the gfx1151 contract is proven, x86 must own
its vectorized stats-attention/merge package and independent correctness and
performance packet; ROCm schedules and evidence do not transfer.

Cross-backend sync `MODEL-FUSED-PHYS-1-2026-08-12` — **MiniMax MSA now owns a
digest-bound AVX-512 package; clean Zen 5 evidence remains open.** The package
binds Graph, Schedule, Tile, and Target identities and executes the declared
`x86_msa_compiled` lane without legacy Graph `ops` metadata or runtime
resynthesis. Structural and differential execution tests pass in WSL, but this
host cannot provide the required clean AVX-512 timing packet. DeepSeek MLA/DSA
remain separate family migrations.

Cross-backend sync `MODEL-WEIGHT-PHYS-1-2026-08-12` — **shared physical-byte
weight ABI landed; AVX-512/VNNI consumption remains follow-up required.** The
content-addressed carrier preserves INT4/FP8 checkpoint bytes and separate fp32
scales while prohibiting full-weight materialization. This slice adds no x86
packed-byte GEMM package and transfers no gfx1151 evidence. x86 still needs an
architecture-owned AVX-512/VNNI physical consumer and a clean Zen 5 packet.

Cross-backend sync `W4-PRESBURGER-SHARD-2026-08-12` — **shared typed shape and
placement analyses landed; AVX-512/MPI physical follow-up remains open.** The
C++ compiler consumes coefficient-vector integer-affine and exact
modular/divisibility constraints with MLIR Presburger analysis. Shared placement
propagation now distinguishes replicated, tiled, partial-reduction, and unknown
states; catalog pointwise/reduction/collective rules feed an explicit fail-closed
reshard planner. Lowered `control_scan` has shared JVP/VJP products under
`recompute_all`; saved checkpoint policies remain rejected. No MPI/OFI/SHMEM
reshard materialization, native region product, or clean Zen 5 packet was added.

Cross-backend sync `W4-CFG-RESIDUAL-W5.2G-2026-08-14` — **shared compiler
carriers and scalable scheduler landed; AVX-512 physical follow-up required.**
The tracer-owned structured CFG, block-wide Presburger identity, and executable
SAVE/HYBRID residual ABI change shared lineage only. The action-DAG model now
uses deterministic critical-path/list scheduling with safe lower-bound pruning
and a small-DAG exhaustive oracle. No native region-product package,
inferred-edge producer wiring, clean Zen 5 calibration, or selection claim was
added.

Cross-backend sync `E2E-AUTH-DAG-2026-08-12` — **shared reduction and
normalization product authority landed; exact AVX-512 rerun is blocked.**
Reduction now requires exact `schedule.reduce → tile.reduce_kernel → native
descriptor` children, and normalization owns a content-addressed composite
Schedule/Tile action program. Mandatory tracer/AST differential gates cover
pure native JVPs and VJPs. This WSL process cannot load the production AVX-512
shared image, so it cannot refresh the Zen 5 packet; no ROCm result transfers.
A clean bare-metal Zen 5 run with that image and calibrated timing remains the
architecture-owned gate.

Cross-backend sync `E2E-AUTH-DAG-2026-08-11` — **shared frontend authority and
automatic dependence-edge contracts landed; AVX-512 physical evidence is
unchanged.** Pure straight-line tensor signatures now cache tracer-owned Graph
IR and can be differentially certified against the retained AST candidate.
Native-JVP plugins declare Graph/Schedule/Tile/AVX-512 disposition and own parent
package construction; compatibility gaps are explicit. W2.1 facts now generate conservative Tile action-DAG
edges with reason and analysis digests. The existing JVP children remain the
physical implementation; this slice adds no selector promotion. Carrying the
generated edges through x86 family pipelines and a clean Zen 5 calibrated
packet remain follow-up; the no-async path stays valid.

Cross-backend sync `AD-SOLVER-ISTFT-PHYSICAL-2026-08-11` — **bounded general-
residual and exact ISTFT-window products execute on AVX-512.** A content-
addressed solver parent binds five native child packages and performs restarted
GMRES with a true-residual check. The compiler now derives all five packages
from a verified typed residual Graph. Pointwise, sum/mean, rank-2 matmul,
transpose, distinct parameter/solution spaces, bounded-dynamic dimensions,
mixed f16/bf16 storage with explicit f32 widening, and statically bounded
`control_for` are represented in the content-addressed child program. Pure
scalar predicates also lower data-dependent `if` and bounded `while` to
digest-bound compare/select SSA with deterministic recomputation in every
primal and product child; the nonlinear
`R=x*x+sin(x)-theta` proof executes without a dense Jacobian. The new
`tessera.istft_jvp` carrier reaches one AVX-512 package
that reuses packed inverse frames and differentiates both overlap-add numerator
and quadratic window-energy normalization. The committed 30-sample WSL packet
records cold/warm state, resources, device/toolchain identity, artifact
digests, and numerical error for nonlinear, reduction, reduced-storage matmul,
bounded-dynamic mixed-storage, data-dependent `if`, data-dependent `while`, and
ISTFT-window products (maximum error 2.33e-6). Timing remains
regression-only because this is not a clean bare-metal run and has no
independent device clock. Odd/low-precision ISTFT products and the clean Zen 5
selector packet remain open.

Cross-backend sync `E2E-REAL-6-JVP-SOLVER-2026-08-11` — **the first frontend/
family-plugin cohort is landing.** Native forward-product specialization now
binds tracer-produced canonical Graph IR and dispatches through explicit family
plugins outside `JitFn`. General solver contracts bind exact residual/JVP/VJP
identities and support exact matrix-free reference actions without numerical
re-entry. The six-case AVX-512 native forward-product cohort, including DCT-IV
with corrected logical-length normalization, passes on this Zen 5 WSL host.
General residual child-composite packages now execute and the typed residual
compiler derives all five children automatically for pointwise, reduction,
rank-2 matmul, bounded-dynamic/mixed-storage, and counted-region programs.
Architecture-owned packets for that expanded envelope and a clean bare-metal packet remain open. The AST lane stays available
for unmigrated families until their differential gates close.

Cross-backend sync `AD-FWD-DIST-3-2026-08-11` — **shared exact JVP and
structured-region products landed; CPU process transport remains open.** Public
JVP/jacfwd no longer substitutes finite differences, and compiler forward mode
carries primal/tangent state through bounded SCF. The typed
`collective_permute` contract executes in the portable runtime, but it does not
constitute MPI/OFI/SHMEM execution. x86 still needs subgroup-aware process
launch plus a clean multi-rank Zen 5 correctness/performance packet.

Cross-backend sync `W4-SOLVER-REGION-2026-08-11` — **shared bounded-region
adjoints and general matrix-free solver policy landed; AVX-512 now has a
bounded child-composite proof.** Portable tracing now emits bounded SCF, and the paired compiler
differentiates effect-safe single-block `if`, counted `for`, and canonical
bounded `while` with implicit captures. General residual execution uses
restarted GMRES/CG policy and exposes convergence work; counted-region evidence
executes SAVE/RECOMPUTE/HYBRID cohorts. The monolithic diagonal-sqrt pilot is
retained, while the general parent now consumes compiler-derived, digest-bound
nonlinear product children. Typed reduction/matmul and statically counted-region
lowering is now shared software; pure scalar predicate-bearing residuals and
the expanded-family WSL correctness packet are now closed. Selected-checkpoint
lowering and a clean Zen 5 selector packet remain open; AMX is
unaffected.

Cross-backend sync `COMP-GRAPH-DATAFLOW-W2.1-2026-08-11` — **shared
analysis substrate landed; AVX-512 remains the proven no-async path.** Graph IR
now has one fail-closed, invalidatable shape/alias/liveness/memory-dependence/
activity analysis with C++ and Python query surfaces. Reverse AD and await
sinking consume it. No physical x86 schedule changes, and clean Zen 5 overlap
or performance evidence remains architecture-owned.

Cross-backend sync `AD-FWD-FAMILY-2-2026-08-11` — **affine normalization,
compound spectral, and matrix-free solver products execute on AVX-512.** The
multi-active product ABI carries named child outputs. Affine LayerNorm binds
data/scale/bias tangents to AVX-512 norm and binary lanes; spectral-filter JVP
executes both bilinear terms through the typed TSOL artifact; and the solver
artifact carries residual-JVP/non-transposed-solve lineage distinct from VJP.
All three pass numerical tests on the WSL-visible Zen 5 CPU. The native
multi-rank collective product requires a live NCCL hardware adapter and world
size >= 2; clean bare-metal Zen 5 timing and native process-transport evidence
remain open.

Cross-backend sync `AD-FWD-NATIVE-1-2026-08-11` — **the first native AVX-512
JVP package is executable; clean-host promotion remains open.** The
content-addressed parent binds compiler-emitted paired JVP IR to ordered native
child packages without Graph redispatch. Sum, non-affine RMSNorm, and packed
RFFT primal/tangent products pass independent formulas on this Zen 5 WSL host
(**3/3**). The packet records architecture and artifact lineage, but it is not
a clean bare-metal timing packet. Broader products and process-collective
hardware evidence remain architecture-owned.

Cross-backend sync `COMP-EFFECTS-W2.2-2026-08-10` — **shared registered-effect
analysis closed; AVX-512 remains a no-async path.** Canonical Graph records now
carry effect, alias, mutation, and stochastic identity; Python and C++ consume
the same fail-closed facts and internal calls reach a fixed point. Await
sinking remains a proven x86 no-op. Native process transport and Zen 5 overlap
evidence remain x86-owned.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R4-2026-08-10` — **shared functional
MegaMoE plan consumption landed; native CPU transport remains open.** The
content-addressed plan binds chunk slices, per-expert capacity, two-live-frame
workspace limits, true-use dependencies, ordered collectives, and deterministic
combine order. R3 only prunes complete measured records; scalar clean-host
latency selects. Mock multi-rank execution is not MPI/OFI/SHMEM proof. x86
still needs native transport integration plus Zen 5 correctness/performance
evidence with stable affinity.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R3-2026-08-10` — **shared prune-only
Tile action-DAG model landed; no AVX-512 selection claim.** R3 validates
explicit dependencies and calibration identity, uses deterministic
critical-path/list scheduling, and composes compute/memory/communication lanes
with queue serialization. Exact small DAGs and proven lower-bound losers may be
pruned; every estimate is promotion-ineligible
and scalar measured latency remains authoritative. WSL timing and blocked PMU
access do not become calibration evidence; clean Zen 5 vectors remain required
before architecture-owned composition analysis or R4 transport overlap.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R2-2026-08-10` — **shared measured
resource-vector schema landed; Zen 5 evidence remains architecture-owned.**
Successful measured autotune rows may record compute time, dtype-correct bytes
moved, communication bytes, queue/resource identity, timing provenance, and
the measured-candidate digest. Analytical rows cannot claim the vector, and
scalar measured latency remains selector authority. The contract does not
upgrade WSL wall-clock or blocked PMU observations; clean-host AVX-512 timing,
affinity, and resource identity remain required before R3 composition use.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R1-2026-08-10` — **shared explicit
async lineage landed; x86 is a proven no-op path.** Python Schedule→Tile no
longer emits internal `tessera.queue.*` markers, and the registered collective
await pass sinks only across operations proven memory-effect-free. Mutation,
RNG, aliases/casts, regions, and ordered collectives remain barriers. A direct
no-async function test records zero moved awaits; no AVX-512 package, ABI, or
performance claim changes. Future MPI/OFI/SHMEM overlap remains separately
architecture-owned.

Cross-backend sync `AD-STOCHASTIC-RNG-1-2026-08-10` — **native x86 base RNG
transforms landed; clean Zen 5 packet remains required.** Explicit key/counter
Graph ops, estimator provenance, dropout replay, fixed-key EGGROLL JVP, and the
shared derivative proof matrix are compiler contracts. The x86 library now owns
uniform scaling, Box–Muller normal, and dropout masking instead of applying the
transforms in Python. Uniform words are bit exact and normal is one-f32-ULP
bounded. This local WSL host is not a clean Zen 5 evidence environment, so no
new AVX-512 performance or target-JVP promotion is claimed.

Cross-backend sync `AD-FWD-PRODUCT-2-2026-08-10` — **public JVP ABI landed;
AVX-512 execution remains follow-up required.** Forward/JVP requests now carry
mode-neutral provenance and stable `wrt_indices`, and the compiler emits only
requested tangent terms. Tanh/sigmoid add direct CPU-oracle proof. No native
JVP package or Zen 5 evidence is claimed; target promotion remains fail-closed.

Cross-backend sync `AD-FWD-CORE-1-2026-08-09` — **shared compiler JVP
foundation landed; x86 physical consumption remains architecture-owned.** The
Graph dialect now exposes compiler-owned tangent rules and a paired
`--tessera-autodiff-forward` function contract. Matmul/mul has independent CPU
IR numerical proof, while unsupported active operations and regions fail
closed. The generated ledger distinguishes compiler `ir_tangent` evidence from
Python JVP registration. This changes no AVX-512 package or Zen 5 evidence; x86
must lower and prove any native JVP package independently.

Cross-backend sync `X86-TYPED-FAMILY-PLUGIN-2026-08-09` — **the production
AVX-512 native packager now crosses one registered, closed-family executable
pipeline.** `tessera-x86-executable` validates the semantic family against
exactly one primary Tile carrier, binds `x86_64_avx512`, records the Tile producer,
registered `tessera_x86` Target consumer, and prebuilt native-image boundary,
then rejects surviving Tile operations or packages without both the typed
Target marker and native ABI call. Python package producers use
`X86ExecutablePipeline`; the former direct construction of
`--tessera-tile-to-x86=...` is gone from production packaging. The base x86
profile remains limited to its proven softmax/reduction subset, and AMX remains
access-gated. Spectral backward, solver IFT, scheduled matmul, attention, and
the broader AVX-512 ABI families all use the same boundary. Standalone legacy
pass fixtures remain only to test the underlying lowering pass, not as a
production configuration escape hatch.
Attention backward explicitly admits one forward-recompute companion carrier;
all other families remain single-carrier. This keeps LSE recompute identity in
the same content-addressed program without weakening family discrimination.

Cross-backend sync `EGGROLL-ES-LOWRANK-2026-08-09` — **the shared Graph,
Schedule, Tile, lineage, member-RNG-v1, and fp32 numeric-policy contract has
landed; rank-1 fp32 Zen 5 AVX-512 consumption is complete.** The typed x86
family lowers the exact architecture-bound Schedule→Tile artifact to
`tessera_x86_avx512_es_low_rank_correction_f32`; runtime execution preserves
the shared SplitMix64/Philox4x32/Box–Muller member identity, caches only
O(in+out) factors, and never materializes the perturbation matrix. The durable
packet `benchmarks/baselines/x86_zen5_es_low_rank_2026_08_09.json` records
correct aligned, ragged, and 16×32×1024→1024 cases (0.844 ms median for the
largest case) on Ryzen AI MAX+ 395. The separate integer lane remains
fail-closed: VNNI u8×s8→s32 cannot implement the Gaussian fp32 correction until
Graph/Schedule carry explicit quantization scales and saturating requantization.
W4 scalar-gather/member reconstruction passes mock-mesh proof; native x86
transport integration remains open.
Contract: `docs/audit/compiler/EGGROLL_SUPPORT_PLAN.md`.

Cross-backend sync `COLLECTIVE-RCCL-ADVANCED-LANES-2026-08-09` — **shared
fail-closed artifact vocabulary adopted; not applicable to AVX-512.** Zero-CU
Copy Engine, GIN/RMA, and gfx1250 DDA are distinct RCCL lanes and cannot be
selected by the x86 runtime. The eventual MPI/OFI/SHMEM transport keeps its
own communicator, one-sided-memory, and evidence contracts.
The registered window and put/signal/wait Target operations are RCCL-gated;
they do not create an x86 RMA implementation or replace the future OFI/SHMEM
lane.
The launcher-neutral RCCL GIN harness is not an x86 transport. Only its explicit
rank/rendezvous and dual-clock evidence discipline is reusable by the future
MPI/OFI/SHMEM lane.

Cross-backend sync `COLLECTIVE-NATIVE-FOUNDATION-2026-08-09` — **shared
artifact vocabulary adopted; no x86 accelerator transport claim.** Target
collective artifacts now bind initiation, registration, ordering, capture
policy, backend/source identity, and capability evidence. NCCL/RCCL host calls
and AMD LSA/Copy Engine/DDA paths do not apply to AVX-512. The portable
CPU/software adapter remains the x86 path until a separately owned
MPI/OFI/SHMEM transport and multi-rank evidence are introduced. Shared
runtime/artifact capability-digest rejection applies to that future transport.

Cross-backend sync `COLLECTIVE-ASYNC-UNIFY-2026-08-09` — **shared software
contract closed; process transport evidence open.** Active transform producers
now use the registered async `tessera_collective` dialect, explicit awaits, and
real SSA rewiring instead of unregistered marker strings. Runtime mesh-axis
validation fails closed when the adapter cannot implement a subgroup. x86
still needs a production multi-process transport and exact Zen 5 multi-rank
packet; AVX-512/AMX selectors and performance claims are unchanged.

Cross-backend sync `DIST-SHARD-ALIAS-1-2026-08-09` — **portable alias bridge
landed; native x86 transport unchanged.** Three public entries are compiler
placement/region contracts; `psum`/`pmean`/`pmax`/`pmin` and
`broadcast_to_axis` now resolve to the registered collective runtime instead
of remaining nine undifferentiated backend rows. `collective_permute` remains
an ordered point-to-point gap and fails closed. MPI/OFI/SHMEM binding and exact
multi-rank Zen 5 evidence remain open; AVX-512/AMX selectors are unchanged.

Cross-backend sync `AD-SOLVER-RESIDUAL-EVAL-2026-08-08` — **bounded AVX-512
physical pilot and packet landed.** The diagonal-sqrt residual lowers as one
content-addressed `schedule.solver_ift` → `tile.solver_ift_kernel` package and
executes residual, transposed diagonal matrix-free solve, and parameter adjoint
in the native AVX-512 image. The committed [Zen 5 3×257 f32 packet](../../../../benchmarks/baselines/x86_zen5_solver_ift_evidence.json)
reports 4.58e-7 maximum residual error, exact linear/adjoint outputs, 30 complete-
backward samples, and 3,084 retained bytes. General residuals and
iterative/Krylov solvers remain open and fail closed; AMX is unaffected.

Cross-backend sync `AD-CORE-EFFECT-CONTROL-COLLECTIVE-2026-08-08` — **shared
effect/control parity and native x86 consumption validated; multi-rank
transport open.** The native x86 spine passes 31 tests on the AVX-512 host;
three availability-gated cases skip. The loader now preserves
content-addressed base/AVX-512 image identity when the host Python lacks
`memfd_create` by using a unique unlinked temporary image. The four typed Tile
collectives now lower into a content-addressed portable Target queue and run
through the deterministic two-rank adapter. A production process transport
and exact multi-rank x86 proof remain open. No new performance or selector
promotion is made; AMX is unaffected.

Cross-backend sync `GRAPH-VERIFY-SIGNED-1-2026-08-08` — **shared legality
parity validated; AVX-512/AMX packages unchanged.** Graph and canonical-
attention verifiers now inspect signed `IntegerAttr` values before enforcing
positive and non-negative bounds, so MLIR 23 unsigned value accessors cannot
admit negative schedules, cache windows, seeds, or control bounds. Direct
negative IR cases cover both dialects. No native ABI, schedule, selector,
package, or Zen 5/AMX evidence changes.

Cross-backend sync `AD-TSOL-SPECTRAL-NATIVE-2026-08-09` — **bounded AVX-512
backward packages and exact-host correctness landed; performance open.** The
content-addressed multi-output carrier now lowers complex-f32 spectral-filter
and unbroadcast last-axis full-f32 spectral-convolution adjoints to native x86
ABI calls without returning to Graph IR. Both gradient outputs pass direct
numerical checks on the local Zen 5 AVX-512 host; convolution preserves the
artifact-selected `backward`, `forward`, and `ortho` scale. Native STFT/ISTFT
backward, broader axes/dtypes/broadcasting, and a clean timing/resource packet
remain open and fail closed. Existing forward AVX-512 TSOL packages are
unchanged; AMX is not implicated.

Cross-backend sync `AD-CORE-LINEAR-1-2026-08-08` — **shared Graph-IR parity
validated; AVX-512 package unchanged.** Both compiler autodiff passes now
consume compiler-owned transposes for structural views, broadcast, and
operand-wise matmul. Paired CPU execution proves the emitted inverse view chain
and all matmul transpose-flag combinations. This transfers no AVX-512/AMX
schedule, selector, performance, or exact-device claim.

Cross-backend sync `COMPILER-DASHBOARD-PROOF-TRUTH-2026-08-08` — **portable
CPU and native AVX-512 proof separated; no x86 physical change.** The x86 row
now uses the `x86` manifest grain instead of accidentally reporting only five
portable-CPU runtime rows. Implementation-present entries remain distinct
from exact Zen 5 proof; AMX access state and selectors are unchanged.

Cross-backend sync `X86-BUILD-ARTIFACT-DISCOVERY-2026-08-08` — **runtime and
packager discovery closed.** Both x86 entry points now honor the explicit
library override, then the common fail-closed `TESSERA_BUILD_DIR`, before
canonical defaults. The current clean LLVM/MLIR 23 tree loads the production
AVX-512 image and compiler together: the exact-host FFT suite passes 41/41.
The repaired comparison runner measures both normalized forward and inverse
C2C paths; a 64/256/1024, batch-4 smoke passes numerically. Its short WSL
host-wall samples are validation only, not a new performance promotion packet.

Cross-backend sync `STANDALONE-COVERAGE-TRUTH-2026-08-08` — **Zen 5 package
evidence is now represented without changing a selector.** The standalone
dashboard generates its registry and compiler-layer counts, exact-target
manifest summary, and open queues. The audit no longer hides the verified
Adafactor Schedule→Tile/native package behind a single-GPU terminal override,
and the benchmark inventory now names the physical TSOL and Adafactor
harnesses. This is an audit correction only; AMX and clean bare-metal timing
gates remain unchanged.

Cross-backend sync `TSOL-NATIVE-REAL-FFT-2026-08-08` — **Zen 5 packed-real
lane retained.** The v3 FFT artifact distinguishes logical and physical
lengths and hashes its Hermitian policy. Supported even RFFT/IRFFT shapes now
enter architecture-owned native C ABI functions using an N/2 complex FFT;
odd or unevidenced factorisations retain an explicit full-complex fallback.
Cached Hermitian twiddles were required: recomputing sin/cos made the sum of
the parts slower than the old lane. After caching, WSL synchronized-host-wall
measurements at batch=32 show 1.33x/2.23x RFFT/IRFFT at N=256 and
1.76x/2.53x at N=1024. All 66 Schedule/Tile and x86 numerical tests pass.
Bare-metal timing remains required for a performance-promotion packet.

Cross-backend sync `ROCM-BUILD-ARTIFACT-DISCOVERY-2026-08-07` — **parity
validated; no x86 physical change.** Shared compiler-test discovery now accepts
fail-closed `TESSERA_BUILD_DIR` selection while retaining explicit
`TESSERA_OPT` precedence. The migrated runtime-library and backend-tool users
are ROCm-owned; AVX-512/AMX packages, evidence, and selectors are unchanged.

Cross-backend sync `AUTODIFF-RELAXATION-1-2026-08-07` — **shared
Python-reference contract; x86 physical follow-up required.** `sparsemax`,
`entmax15`, `soft_top_k`, `gumbel_softmax`, and `perturbed_argmax` now have
storage-preserving reference semantics and autodiff rules, but no AVX-512
lowering or exact Zen 5 evidence. They remain explicitly reference-only until
an x86-owned physical package is selected and proven.

Cross-backend sync `MATH-PHYSICAL-2-2026-08-06` — **Zen 5 scan selector and
dtype boundary retained.** The f32 scan ABI now selects a 16-lane AVX-512
Hillis--Steele prefix for `cumsum` and `cumprod`; paired interleaved measurement
against the scalar reference records 1.48x and 1.47x speedups on Ryzen AI Max+
395. The same implementation regressed extrema, so `cummax` and `cummin`
deliberately retain the scalar recurrence. NaN propagation and signed-zero
behavior are exact-tested. The complete x86 physical math cohort passes 167
tests, and the benchmark packet covers unary, transcendental, binary,
reduction, and scan families. General x86 math remains an explicit f32 ABI;
target-wide bf16 capability does not imply bf16 support for these packages.
Binary packages now reject mixed input dtypes. Evidence:
`benchmarks/baselines/math_physical_zen5_2026_08_06.json`.

Cross-backend sync `TSOL-CONTRACT-GENERALIZE-2026-08-06` — **Zen 5 physical
policy expansion implemented and retained.** The v3 contract specializes
bounded dynamic shapes into exact content-addressed packages, packs arbitrary
axes inside the native AVX-512 package, admits fp32/fp16/bf16 real storage with f32
accumulation, and carries backward/forward/ortho scaling into the native
package. ABI v4 and 36 focused contract/package/evidence tests pass on AMD Ryzen AI
Max+ 395. The policy
packet `benchmarks/baselines/tsol_physical_policies_zen5_2026_08_06.json`
now contains 30 numerical-and-performance rows across all five compound
operations, including seven digest-changing bounded specializations and
combined dynamic-axis-reduced-storage-ortho cases. Warm medians span
0.058--0.177 ms; every row meets its recorded error limit. Reduced storage
conversion and arbitrary-axis pack/unpack are package-owned host-side work
around f32 native FFT accumulation, not claims of reduced-arithmetic FFT
instructions or device-side packing.

Cross-backend sync `X86-WELFORD-PARITY-2026-08-06` — **native implementation
and exact Zen 5 validation complete.** `var`/`std` now call a dedicated
`tessera_x86_avx512_welford_f32` ABI. It accumulates independent SIMD-lane
f64 Welford states and merges them deterministically after the existing
arbitrary-axis fold, replacing the cancellation-prone `mean(x²)-mean(x)²`
composition. On the AMD Ryzen AI Max+ 395 Zen 5 host, the native image rebuilt
with LLVM/MLIR 23 and all 17 focused tests passed, including
large-offset/low-variance data, ragged extents, multiple axes, and tuple axes.
Cross-backend sync `TPROF-MULTICLOCK-2026-08-06` — **shared schema and native
`TPROF-X86-TIME-1` timing provider landed; exact-host evidence open.** The ROCm
work establishes the shared
rule that every applicable clock is an independent provenance/validity record;
it does not require HIP-specific slots in x86 packets. The existing `cpu`
provider already exposes runtime callback spans and `steady_clock` timestamps,
but that is not yet an AVX-512 PMU or sampling profiler.

`TPROF-X86-TIME-1` now extends the same tprof artifact/CLI with four independently
validated x86 records:

1. `host_wall_ns`: raw `steady_clock` batch time with one terminal
   synchronization plus separate empty-call/dispatch calibration.
2. `monotonic_raw_ns`: Linux `CLOCK_MONOTONIC_RAW` for OS-adjustment and clock-
   agreement checks where available.
3. `tsc_cycles`: fenced `RDTSCP` entry/exit with invariant-TSC capability,
   calibrated frequency, start/end logical CPU, and migration invalidation.
4. `perf_task_clock_ns`: `perf_event_open` task-clock timing with explicit
   permission, multiplexing, enabled-time, and running-time fields.

The first portable PMU group is cycles, reference cycles, instructions,
branches, branch misses, cache references, and cache misses. Model-specific Zen
5 events, IBS samples, PEBS, raw event encodings, and derived cache-level claims
remain capability-gated by vendor/family/model, kernel support, and an exact-
machine event-map digest. Unsupported or permission-denied events are explicit
unavailable records, never zero counters.

The x86 provider runs the canonical typed AVX-512 package in a fresh subprocess
with recorded affinity, NUMA node, SMT state, governor/frequency context,
microcode, kernel, compiler, image/artifact digests, warm/cold state, and sample
distribution. Sampling uses Linux perf IP/callchain/branch-stack or AMD IBS only
when advertised; it must correlate samples to the package image and symbol
range. Instrumented and uninstrumented images stay paired so probe overhead is
visible.

WSL host timing may support same-host regression and retain/reject decisions.
PMU- or sampling-dependent promotion requires a non-virtualized exact Zen 5
host, non-multiplexed or correctly scaled counters, stable affinity, and an
independent clock-agreement packet. Intel AMX remains a separate named-host
evidence obligation. HIP `wall_clock64()`, `rtg_hsa_dispatch`, ROCprofiler, and
gfx1151 evidence do not transfer to x86.

Execution order:

1. **Complete:** generalize the TPROF clock-record validator and CLI away from
   GPU-only names.
2. **Complete:** add x86 host/raw/TSC timing with migration and invariant-TSC
   checks.
3. **Complete:** add the `perf_event_open` provider, permission diagnostics,
   scaling, portable PMU group, `tprof x86 timing-status`, and optional native
   proof snapshot workflow.
4. **Landing:** `tprof_x86_event_map.py` now digests the exact CPU identity,
   microcode, sysfs PMU encodings, and `perf` catalog. `tprof_x86_sample.py`
   embeds that map, pins the command with `taskset`, and records perf samples against an image
   build ID, DSO-aware symbolization, and the matching static ELF symbol range,
   avoiding invalid ASLR-relative comparisons. AMD IBS is admitted only for
   family 26 with AVX-512 and an advertised `ibs_op`/`ibs_fetch` event. Exact
   perf/IBS samples remain open because this WSL host has no `perf` executable
   and denies `perf_event_open`. The truthful host packet
   [`../../../../benchmarks/baselines/x86_zen5_event_map_2026_08_07.json`](../../../../benchmarks/baselines/x86_zen5_event_map_2026_08_07.json)
   identifies AMD family 26/model 112 and captures the visible sysfs event
   sources, but marks the catalog and promotion gates unavailable under WSL.
5. **Landing:** the exact-host runner now binds the existing aligned/ragged
   E2E-REAL-4 comparison to clock, sampling, source-commit, dirty-worktree, and
   artifact provenance. The current packet is
   [`../../../../benchmarks/baselines/x86_zen5_profiler_packet_2026_08_06.json`](../../../../benchmarks/baselines/x86_zen5_profiler_packet_2026_08_06.json):
   scheduled/production is 1.018x aligned and 1.008x ragged with exact route
   parity. Verdict `retain`; WSL clocks, denied PMU access, unavailable symbol
   samples, and the development worktree block promotion. A clean bare-metal
   rerun with real samples remains open; AMX stays access-gated.

The next exact-host action is one clean bare-metal Zen 5 run with a fixed CPU,
stable governor/NUMA placement, working `perf_event_open`, the model-specific
event-map digest, and symbol-correlated cycles plus advertised IBS samples.
That packet must be regenerated from the same commit as the aligned/ragged
benchmark; WSL timing and the event-source inventory cannot satisfy it.

Cross-backend sync `TSOL-ROCM-E2E-1-2026-08-06` — **shared typed carrier and
x86/Zen 5 physical consumer complete.** `tessera.scheduled_spectral.v5`
materializes one verified `schedule.spectral_program` →
`tile.spectral_program_kernel` edge and binds child FFT digests plus the full
compound policy. The native AVX-512 package now owns DCT mirroring,
padding/cropping, framing/windowing, half-spectrum expansion/compaction,
complex multiplication, deterministic overlap-add, and a bounded thread-local
digest-keyed workspace; runtime no longer reconstructs these programs with
host NumPy. Exact Zen 5 aligned, ragged, and prime/Bluestein cases agree with
NumPy. This is architecture-owned evidence and does not inherit gfx1151
performance or scheduling choices.

Cross-backend sync `TSOL-GFX1151-FUSED-BATCH-2026-08-08` — **shared artifact
truth assessed; no x86 physical change.** The FFT artifact can now identify
gfx1151's batched fused-LDS residency and v4 native family, but Zen 5 retains
its independently selected AVX-512 FFT/package policy. The HIP image build
dependency, LDS schedule, and WSL timings do not transfer to x86; no x86
correctness or performance claim changes.

Cross-backend sync `TSOL-SPECTRAL-POLICY-2026-08-08` — **DCT physical coverage
expanded on Zen 5; clean-host promotion remains open.** DCT-I/II/III/IV now
carry distinct public, autodiff, Graph, Schedule, and Tile identities.
`tessera.x86.spectral_composite.v6` phase-corrects the FFT-backed type-II path
and provides separately hashed direct cosine kernels for types I/III/IV; exact
Zen 5 correctness coverage passes. The causal chunked-STFT state now binds its
policy digest and overlap lineage; centred streaming fails closed pending
explicit lookahead. The new DCT direct paths have no performance promotion.
The same boundary audit preserves scalar convolution and one-sample
STFT/ISTFT through explicit scalar and odd full-complex paths; native AVX-512
regressions pass on the Zen 5 host.

Cross-backend sync `ROCM-MATH-EVIDENCE-2026-08-06` — **shared atan2 semantic
fix and x86 Welford parity apply; ROCm physical kernels are not applicable.** Shared quadrant logic
now preserves signed-zero origins, infinity diagonals, and NaN propagation.
The x86 atan kernel remains the magnitude consumer and requires its existing
Zen 5 differential gate. The x86 statistical path now has an independently
implemented and Zen 5-tested native Welford ABI under
`X86-WELFORD-PARITY-2026-08-06`; it does not inherit gfx1151 evidence.

Cross-backend sync `ROCM-FFT-PREBUILT-2026-08-05` — **not applicable; x86
parity assessed.** The opaque device-plan ABI, persistent HIP allocations, and
prebuilt ROCm shared image are architecture-owned. x86 retains its existing
content-addressed native package and thread-local cached Bluestein plan; no x86
schedule or numerical policy changed.

Cross-backend sync `FFT-PERF-2-2026-08-05` — **Bluestein cache and work-gated
mixed-radix Stockham promoted; Rader conditional; Bailey rejected.** Immutable
chirp/kernel FFT plans plus thread-local padded workspaces improve warm
Bluestein by 1.57x--1.76x at N=127--1009. The native mixed-radix ABI caches
stage matrices/twiddles and executes 16 contiguous butterflies per AVX-512
codelet. It beat Bluestein on 12/13 measured composite shapes; representative
speedups are 1.91x at N=68, 3.75x at N=289, and 2.82x--4.90x at
N=768--5120. A factor-work gate promotes those wins through the exact
Schedule→Tile artifact while rejecting N=255 to cached Bluestein.

Rader is retained as a named candidate only: it wins at N=257 because its
convolution and workspace are half Bluestein's, but loses at N=127/509/1009.
The native Bailey candidate fuses its middle transpose with twiddle
multiplication yet remains 1.66x--2.23x slower at N=64K--1M, so it is rejected
without coefficient tuning. The sweep also fixed a correctness defect where
the out-of-place runtime helper mutated contiguous caller inputs.

Cross-backend sync `FFT-PERF-FOUNDATION-2026-08-05` — **AVX-512 plan/twiddle
cache, generated permutation, and every planner-admitted mixed-radix codelet
are retained.** The production C2C ABI now keeps a
bounded thread-local immutable plan containing gather offsets and all
per-stage twiddles, removing allocation and transcendental work from warm
execution. Same-host ratios against `scipy.fft(workers=1)` improved from
8.63x/22.75x/17.20x to 3.46x/3.85x/3.10x at N=1024/8192/65536, with the
existing numerical gates passing. The former scalar random-swap bit reversal
is now a cached AVX-512 gather into reusable thread-local workspace; the gap
improved again to 2.92x/1.86x/2.73x with the C++ correctness corpus passing.
This is a retain, not library parity. The runtime-radix DFT loop is now a
compile-time specialization for radices 2/3/4/5/7/9/11/13/15/17, including
AVX-512 butterfly batches and unrolled scalar tails. Native tests cover every
radix plus a multistage N=68 round trip. The 2026-08-09 packet records correct
N=68/225/289/768/3072/5120 transforms: native is 2.79× and 2.65× faster than
NumPy at N=3072 and N=5120, while smaller rows remain slower. Therefore the
existing work gate is retained and no global selector promotion is made.
Remaining expansion is nontrivial stride, dtype, and larger-transform coverage.

The v2 content-addressed artifact records `cooley_tukey_dit`, host-inplace
residency, cached-f32 twiddles, workspace policy, radix sequence, and the
complex64/f32 numeric policy. Evidence is in
`benchmarks/baselines/fft_plan_cache_radix17_2026_08_05.json`.

Cross-backend sync `E2E-REAL-FFT-2026-08-05` — **typed artifact consumption
implemented; physical package parity validated on Zen 5.**
ROCm's public runtime now consumes its proven Stockham/Bluestein package rather
than a duplicate O(N²) DFT. x86 keeps its existing AVX-512 radix-2/Bluestein
package unchanged and remains numerically covered. It now consumes the exact
content-addressed `schedule.fft`→`tile.fft_kernel` identity without Graph
metadata or a second planner decision. The contract preserves x86's radix-2,
tiny-DFT, and Bluestein choices rather than transferring gfx1151's physical
stage sequence. Focused Schedule/Tile tamper tests and exact Zen 5 FFT tests
pass. The remaining shared packaging action is ROCm-specific runtime-`hipcc`
removal; x86 keeps its existing prebuilt native ABI.

Cross-backend sync `FFT-MIXED-RADIX-BLUESTEIN-2026-08-03` — **parity validated on host; the reference lane for the family.**
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

The CPU hook is the F4 reference every other lane is checked against, so its
correctness gates the others. Verified against a naive fp64 DFT across 63 sizes
(51 mixed-radix, 12 Bluestein), zero failures, round trips to ~3e-6.

Its generic radix-r stage precomputes the r-point DFT matrix once per stage and
reuses it across every butterfly -- the opposite of the GPU choice, and the
clearest evidence the shared/per-target split is drawn in the right place.

No AVX-512 specialisation: the stages are scalar C++. Vectorising the butterfly
is open work, and this change neither helps nor blocks it.


Cross-backend sync `SHAPE-RULE-REGISTRY-2026-08-03` — **parity validated at the capability level; device evidence missing.**
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

**This is the Python reference lane, not generated device code.** Complex FFT is SUPPORTED on `x86` - one of three targets (with `cpu` and
`apple_cpu`) declaring an `fft` capability entry, and complex maps onto the
interleaved fp32 pair AVX-512 already handles. No storage-contract change:
appending complex to any target's `supported_dtypes` was tried and correctly
broke `test_x86_dtype_contract`, because that tuple answers "what STORAGE has
this backend proven" and complex is not a storage format the ISA has. Complex is
declared on the transform ops that carry it, not on the target.

`x86_ready_storage_dtypes()` is unchanged. The reduced-precision compute contract
(promote -> compute at f32 -> store back) is reference-level; **the AVX-512 lane
has no exact-device proof that generated kernels honour it.**


Cross-backend sync `SUBBYTE-STORAGE-PATH-2026-08-03` — **follow-up required, emulated path.**
The x86 dtype contract records `fp8_e4m3` as `emulated`: packed-byte storage
with software conversion and fp32 compute, since Zen 5 has no native FP8
arithmetic. So x86 CAN carry real sub-byte STORAGE even though it cannot
compute in it — which makes it a useful place to prove the storage path
independently of native arithmetic. x86 owns deciding whether to materialize
packed fp8 storage or keep the f32 fake-quant reference.

Cross-backend sync `REDUCED-PRECISION-COMPUTE-2026-08-03` — **follow-up required, reference-level only.**
The reference lane now computes reduced-precision ops at f32 and stores back.
x86's AVX-512 kernels are the executable lane on this fleet and were not
re-verified against the corrected reference; the fp16/bf16 accumulate contract
is now stated explicitly, so a mismatch would be a real divergence rather than
an ambiguity. x86 owns an AVX-512 execute-and-compare — obtainable on the
Strix Halo box, unlike the AMX lane.

Cross-backend sync `TILE-MMA-DATA-OPERANDS-2026-08-03` — **not applicable, with a reason.**
x86 has no `tile.mma` consumer: `TileToX86Pass` lowers through the C-ABI shim
and the `tessera_x86` Target IR models the boundary with `abi_call`. The shared
`tessera::tile::dataOperands` helper is available to it but currently unused,
so there is nothing to migrate. Re-assess when the `x86vector.*` (AVX-512)
lowering lands, since that is where x86 would gain a matrix-op consumer.

Cross-backend sync `TARGET-IR-CONFORMANCE-2026-08-02` — **parity validated, new dialect landed.**
W0.10 closed x86's Decision #19 gap: it was the one backend with no Target IR
dialect at all (`TileToX86Pass` lowered to 21 `func::CallOp`s into a
hand-written C shim, and the Python emitter named a `tessera_x86.func` op no
dialect defined). No carve-out was granted. `tessera_x86` now exists with a real
`!tessera_x86.tile` type, is registered in `tessera-opt`, and separates
value-carrying AMX ops from directives; `abi_call` models the C-shim boundary
instead of hiding it. Positive and negative lit fixtures ship — the negative one
proves the verifier rejects a dot-product whose operands never came from a tile
load. The Python x86 emitter now parses, loads the dialect, and verifies.
**Scope limits:** AMX *lowering* is optional per project direction (expected
supersession by ACE), so the AMX ops are the IR-level contract only. The live
follow-up is `x86vector.*` (AVX-512) lowering instead of terminating in
`func.call` — that changes generated code and needs AVX-512
execute-and-compare, which IS obtainable on the Strix Halo box. No AMX
execution evidence is claimed: no machine in the fleet reports AMX.

## X86 attention and training closeout

Cross-backend sync `CORE-ATTENTION-TRAINING-X86-2026-07-30` — **closed for
Zen 5 AVX-512; no AMX claim.**

The pre-existing inventory is now explicit:

- AVX-512 rank-4 attention forward, Lion forward, SGD and
  Momentum/Nesterov VJPs, loss-to-SGD/AdamW fusion, and physical DeltaNet
  backward were already complete.
- `X86-ATTN-CANON-1` is complete. Canonical x86 packaging begins from
  `tessera.flash_attn`, runs the shared rank-4 batch/query-head/KV recurrence,
  fails closed unless the streaming `scf.for` structure is present, and only
  then selects the established typed AVX-512 attention ABI. The package source
  no longer presents a freshly synthesized `tile.attention_kernel` as its
  semantic authority. Existing f32 MHA/GQA, bias, causal/window, and softcap
  numerical behavior remains covered on the Ryzen AI MAX+ 395.
- `X86-ATTN-BWD-1` is complete. The x86 package structurally consumes the
  canonical tensor-valued dQ, split-dK/dV, and ascending fixed-order reduction
  loops. Its AVX-512 ABI executes MHA/GQA/MQA gradients with optional bias,
  causal/window, and softcap modifiers.
- `X86-LSE-1` is complete for this Zen 5 target. A 21-sample resident packet
  compares the established forward plus recomputed-LSE backward with the
  forward-with-LSE plus saved-LSE backward at sequence lengths 32/64/128.
  Saved LSE wins by 1.45x/1.23x/1.06x, so x86 selects `save_lse`. Evidence:
  [`../../../../benchmarks/baselines/x86_avx512_attention_lse_2026_07_30.json`](../../../../benchmarks/baselines/x86_avx512_attention_lse_2026_07_30.json).
- `X86-LION-BWD-1` is complete. One AVX-512 call implements the canonical
  stop-gradient-through-sign VJP for parameter, gradient, and carried moment.
- `X86-ADAFACTOR-1` is complete. AVX-512 factored row/column and lower-rank
  full-moment forward execution and analytic physical adjoints match the shared
  optimizer/VJP oracles.

The x86 work does not transfer physical schedules or evidence to sibling
backends. ROCm parity was already complete; Apple and NVIDIA retain their
architecture-owned canonical attention/backward and training materializer
items. Validation for this closeout is recorded by the owning PR.

## X86-SPINE-1: reconcile C synthesis with the MLIR/LLVM lane

Cross-backend sync `EXECUTION-SPINE-2026-07-29` — **AVX-512 lane landing; AMX
remains planned/access-gated.** Canonical target `x86` now has one meaning:
Graph/Tile IR lowered by `TileToX86Pass`, packaged with the C++ backend shared
image and typed launch descriptor. Vendor-family selection moved out of the
shared driver into `x86_native.native_package_kind()` / `package_native()`.
Apple CPU/GPU now use the same backend-owned admission shape while retaining an
explicit Value Target-IR compatibility/probe opt-out; this does not transfer any
AVX-512 ABI, schedule, or exact-device evidence.

APPLE-RASTER-1 subsequently consumed the shared mapping in emitted MSL and
retained row-major after mixed Apple7 timing. X86 remains not applicable because
CPU work partitioning is not GPU workgroup rasterization.

The former `emit/x86_llvm.py` implementation never emitted LLVM IR. It is now
`emit/x86_c.py`, registered under source target `x86_c`, and remains a measured
fused-region candidate under the canonical x86 arbiter. A compatibility import
preserves old module imports without reclaiming target `x86`. Its artifact
profile is explicit `x86-64-v4` rather than build-host-dependent
`-march=native`; the source carries that profile into the content-addressed
cache identity and the runner declines on hosts lacking the required AVX-512
feature set. Its execution tag is `x86_c_native`, distinct from the canonical
descriptor lane.

The native loader also keeps each memfd alive with its image. Previously Linux
could reuse an fd number and glibc could return the base-x86 handle for a later
AVX-512 `/proc/self/fd/N` load, making valid descriptor symbols appear absent.
A base-then-AVX-512 regression test guards distinct handles and symbols.

**Zen 5 proof.** The Ryzen AI Max+ 395 WSL host reports the complete
`x86-64-v4` AVX-512 feature set. The broader 2026-07-30 cleanup run passed 261
focused Python candidate/canonical/native/audit tests with one expected
capability skip. A fresh spine verification on the same host passes **63/63**
canonical-x86/native-descriptor plus explicit-`x86_c` source-candidate tests.
The current x86 dtype/ISA/capability and manifest gate is **26/26** (superseding
the earlier recorded count of 24), and all **18/18** rebuilt C++ backend
executables pass.

The seven x86-owned Tile-to-x86 MLIR fixtures pass **7/7**. The expanded
cross-target set now discovers 12 fixtures: **11 pass and 1 is expected
unsupported**. The unsupported `layout_target_materializers.mlir` fixture
requires the Apple backend; it is not an x86 failure. The native GEMM executable
reports `AMX not available; skipping` and runs the AVX-512 path. No AMX
readiness, numerical, or performance claim is inferred.

## X86-CALIB-1: split verdict on the hardware-free score calibration

Cross-backend sync `COSTMODEL-CALIB-2026-07-29` — **retired step-distance;
bank-conflict not applicable; T1 hierarchy rejected for latency ranking.** Owning host
Zen 5 (Ryzen AI Max+ 395 CPU complex, AVX-512, no AMX).

The original two static device-free scores were assessed against measured latency
([`../../compiler/AMD_KERNEL_COMPILER_SURVEY.md`](../../compiler/AMD_KERNEL_COMPILER_SURVEY.md)
§3.7–3.8; motivation in
[`TILESIGHT_ASSESSMENT.md`](../../compiler/TILESIGHT_ASSESSMENT.md) §2). They do
not get the same verdict here, and reporting one blended state would hide that.

**Bank-conflict analyzer — not applicable, architecture-specific reason.** It
counts N-way conflicts across a fixed number of software-managed scratchpad banks
under a wave's phase-grouped access. The x86 lane has no software-managed
scratchpad and no wave phases: AVX-512 loads go through a hardware-managed
L1/L2/L3 hierarchy where the analogous hazards are 4 KiB aliasing, cache-set
associativity conflicts, and store-forwarding stalls. Those are real, but they
are a different model with different inputs — not this analyzer with different
constants.

**Step-distance locality — rejected fleet-wide.** It failed on the ROCm
architecture from which it was derived, so x86 will not retune or revive it.

**T1 reuse/cache model — rejected for ranking, retained as a pruning
diagnostic.** The measured model now represents independent L1D/L2/L3 LRU
levels plus DRAM, using sysfs capacities, CPU-pinned resident-copy bandwidths,
and an architecture-derived single-core peak. Ten physically distinct blocked
AVX-512 GEMM candidates were measured on three workloads. The durable packet
`benchmarks/baselines/x86_zen5_t1_cache_model_2026_08_09.json` records median
Spearman rho -0.4062 and 0/3 measured-winner matches. This fails the
predeclared rho>0.25 promotion threshold. No coefficients were tuned, the
production selector is unchanged, and measured latency remains authoritative.

**Fleet outcome (2026-07-29).** ROCM-CALIB-1 reproduced 0/6 measured winners on
the AMD home architecture (median rho -0.1381, 0% positive), triggering the
agreed no-retuning stop rule. x86 no longer owes a calibration run for adoption
of this score. CPU cache-blocking or reuse-distance research remains valid as a
different model; it must not be presented as a resurrection of the rejected
step-distance latency ranker.

Cross-backend sync `RASTER-CONTRACT-2026-07-28` — **not applicable, with an
architecture-specific reason.** Schedule IR gained `raster_order` /
`raster_group` on `schedule.tile` / `schedule.knob` (arch-neutral definition in
`compiler/tile_rasterization.py`; rationale in
[`compiler/TILESIGHT_ASSESSMENT.md`](../../compiler/TILESIGHT_ASSESSMENT.md)
§3.2). The knob permutes *block ids across a 2-D launch grid* so that the set of
workgroups resident at one instant shares operand panels in a hardware-managed
L2. The x86 AMX/AVX-512 lane has no launch grid: `tessera_x86_backend` emits
loop nests over tiles executed by OpenMP threads, so there is no block id to
permute and no equivalent of a wave of co-resident workgroups contending for a
shared last-level cache in that pattern. The analogous x86 lever — loop order and
cache blocking in the C/LLVM emitter — already exists as a separate mechanism and
is not expressible as this permutation.

**This is a not-applicable for the *contract*, not for the underlying idea.**
Tile-granular reuse-distance analysis, the T1 item the same assessment proposes,
*does* port to AMX/AVX-512 cache blocking — that literature is a CPU literature
(Lam/Rothberg/Wolf 1991 on blocked algorithms). T1 v1 is now built; revisit x86
through `X86-CALIB-1` when the hierarchy inputs and Zen 5 corpus correlation are
ready, not by consuming `raster_order`. No exact-device evidence is owed for the
raster contract itself.

Cross-backend sync `APPLE-AOT-METALLIB-2026-07-28` — **not applicable**. Apple
added `apple_gpu_air`, a precompiled-artifact lane behind the shared
`register_compiler(target, compile_fn)` seam, measured against its compile-on-
launch lane (cold pipeline creation 29.7 ms -> 15.2 ms, ~1.95x; host-wall
timing on Apple M1 Max, not device-event evidence). x86 needs nothing here:
`_x86_compile_fn` already returns a real `.so` from clang, and the x86 lane has
no runtime-compilation path to compare it against. Recorded so the fast-path
shape is documented fleet-wide; the measurement method transfers when X86-1 is
proven on a Zen 5 host. No shared IR, ABI, dtype/op registration, or numerical
contract changed.

The fourth architecture queue, alongside
[`apple/todo.md`](../apple/todo.md), [`nvidia/todo.md`](../nvidia/todo.md), and
[`rocm/todo.md`](../rocm/todo.md). Opened 2026-07-28 because x86 work was being
discovered on an Apple Silicon host, where none of it can be proven.

**Owning-host rule.** x86 lanes are proven on an x86 host — the Zen 5 box for
AVX-512 (Core Ultra 7 265F has neither AVX-512 nor AMX; see the fleet notes).
Nothing in this queue may be marked complete from a Mac. An arm64 host can
author and structurally gate; it cannot produce device evidence.

**Scope split.** The exact-device target of this queue is `x86_avx512`, proven
on the Zen 5 host. AMX is **planned, access-gated**, and cannot inherit AVX-512
evidence: no AMX-capable owning host is currently named. X86-3 may reconcile
the compiler-lane architecture, but its AMX half remains open until a separate
AMX host, target identity, numerical packet, and performance evidence are
recorded. The access-gated correctness command now exists at
`scripts/run_x86_amx_release_gate.sh`; its existence is not device evidence.

## X86-1: close the portable-C plugin provenance and host gate

**Status: closed for the Zen 5 AVX-512 host.**

Historically `tests/unit/test_x86_plugin.py` asserted
`execution == "x86_native"` in 15 places. On the original Apple Silicon audit
host every one returned `"reference"`. The historical cause was not a defect:

* the former `emit/x86_llvm.py::_x86_compile_fn` compiled emitted C with
  `clang -O3 -march=native -fPIC -shared`;
* `platform.machine()` was `arm64`, so `-march=native` targeted ARM. The
  produced `.so` was not an x86 kernel, the runner declined, and it reported
  `reference` — the honest answer.

These failures were invisible until 2026-07-28 because `clang` was not on
`PATH`; the lane skipped for the wrong reason. Putting LLVM 23 on `PATH`
un-gated them. They fail identically on `main`.

Required work, on an x86 host:

1. **Complete.** `X86CEmitter` → `_x86_compile_fn` → `X86CRunner` returns
   `x86_c_native` with numerics matched against the F4 numpy oracle.
2. **Complete.** `native` baked in the build host, which made
   a cached artifact non-portable across the fleet and interacts badly with the
   content-addressed `kernel_cache` key (the key hashes source + dtype + target,
   *not* the host ISA — two hosts would collide on one entry). The selected
   profile is explicit `-march=x86-64-v4`, recorded in emitted source/cache
   identity and guarded before execution.
3. **Corrected.** AOCL-DLP registers only an unavailable hand-tuned candidate;
   it never registered a compiler. The C compiler now registers for `x86_c`,
   leaving canonical target `x86` unambiguous.

**Interim (landed 2026-07-28):** the assertions are host-gated so an arm64 host
skips instead of failing. The gate is `platform.machine()`, not a capability
probe — it says "this host cannot prove an x86 kernel", which is exactly the
claim. Removing the gate is not the fix; proving the lane on x86 is.

## X86-2: `_LANG = "c"` — the file name says LLVM, the emitter says C

**Status: closed.**

`emit/x86_c.py` sets `_LANG = "c"` and emits C for `clang`, matching its name.
`emit/x86_llvm.py` is a compatibility-only re-export.
That is the selected design: source-synthesis modules emit vendor source text
(CUDA C, HIP C++, MSL, C), while canonical `x86` reaches LLVM through the typed
C++ compiler spine. The compatibility shim preserves imports without restoring
the misleading compiler authority.

## X86-3: reconcile the two x86 lanes

**Status: AVX-512 half closed on Zen 5; AMX correctness lane landed but remains
planned/access-gated. A separately named AMX-capable host is still required.**

x86 reaches hardware two ways, and nothing arbitrates between them:

* **C++ MLIR** — `src/compiler/codegen/tessera_x86_backend/`, AMX BF16 +
  AVX-512 GEMM. Decision #1 records the existing end-to-end architecture;
  this plan may revalidate AVX-512 on Zen 5 but cannot refresh the AMX claim
  without an AMX-capable host.
* **Python C candidate** — `emit/x86_c.py` + optional `emit/x86_aocl_dlp.py`
  behind the arbiter; it no longer owns canonical target `x86`.

This is the same two-compiler split documented for Apple in
[`apple/todo.md`](../apple/todo.md); x86 has it too, and the resolution should
be consistent across the fleet rather than decided per backend. Blocked on the
spine decision in
[`../../compiler/COMPILER_THEORY_OF_OPERATION.md`](../../compiler/COMPILER_THEORY_OF_OPERATION.md).
The two required terminal outcomes are now explicit: AVX-512 is selected and
proven on Zen 5; AMX is planned/access-gated until a named capable host supplies
its own packet. Neither architecture promotes the other.

The AMX regression is now owned by
`tests/device/x86/test_amx_int8_gemm.py` and selected by
`scripts/run_x86_amx_release_gate.sh`. The gate fails closed on missing
AMX-TILE/AMX-INT8, runs native execution in a crash-isolated child, repeats the
K>64 numerical comparison twice without xdist, and retains identity, JUnit,
collection, and status artifacts. This closes the validation-ownership gap; it
does **not** close X86-3. A named Intel AMX host must still produce the packet,
and a separate measured-performance gate and baseline remain open.

Cross-backend sync `X86-AMX-DEVICE-2026-08-02` — **not applicable to Apple,
NVIDIA, and ROCm.** This change moves one x86-native regression and adds an
x86-owned local proof command; shared IR, runtime ABI, marker policy, and peer
backend device commands are unchanged.

## Cross-backend sync

`TESSERA-OPT-CAPABILITY-SKIP-2026-07-27` — **parity validated**. x86 tests that
drive `tessera-opt` route through the shared capability-aware helper, so a build
without the owning backend skips with the missing pass named. No x86 pass body,
ABI, or numerical contract changed; no exact-device evidence claimed.

## Cross-backend sync `TILE-FRAGMENT-TYPE-PARAM-2026-08-03` — `!tile.fragment` parameterized (W1.1 step 1)

Shared Tile IR type changed: `!tile.fragment` gained `(m, n, k, elem, acc, role, layout, family)` and a domain verifier. **No behaviour changes in this PR** — the bare `!tile.fragment` still parses AND still prints bare, so every existing producer and fixture is unaffected. All 7 C++ `FragmentType` uses are `isa<>` checks, so there were no construction sites to migrate.

**Outcome: not applicable — architecture-specific reason.** Zero files under `tessera_x86_backend/` reference `FragmentType` or `!tile.fragment` (measured 2026-08-03), and x86 has no cooperative-matrix fragment to model: it carries its own `!tessera_x86.tile` value type over AMX/AVX-512 ops (Decision #19, built typed from the start in W0.10).

That backend is in fact the reference shape for where W1.1 is heading — 0 `AnyType` / 0 `Variadic<AnyType>`, with a negative fixture proving the verifier rejects a dot-product whose operands never came from a tile load. Per project direction the AMX half stays an IR-level contract with no `amx.*` lowering, so no follow-up is created here.

## Cross-backend sync `TILE-FRAGMENT-KLOOP-ACCUM-2026-08-03` — typed `tile.mma` K-loop (W1.1 step 2)

Shared Tile IR contract changed: `MMAOp::verify()` (and the `fragment_pack` / `fragment_zero` producers) now read the operand contract from the fragment TYPE when it is parameterized, falling back to producer-chasing for the bare form. `#tile.mma_desc` is optional on the typed path and cross-checked when present. **The canonical K-loop now verifies.** No lowering changed in this PR, and no existing IR is affected — the bare form keeps its old path.

**Outcome: not applicable — architecture-specific reason.** Unchanged from `TILE-FRAGMENT-TYPE-PARAM-2026-08-03`: no cooperative-matrix fragment on this backend (it carries `!tessera_x86.tile`), so neither the typed `tile.mma` contract nor the accumulator-threading follow-up applies. AVX-512 K-loop accumulation is expressed in its own ops and is unaffected.

## Cross-backend sync `NVWGMMA-ACCUMULATOR-GUARD-2026-08-03` — WGMMA accumulator drop (W1.1 step 2b guard)

A `tile.mma` carrying an accumulator was lowered by `NVWGMMALoweringPass` to a **two-operand** WGMMA call: the accumulator was discarded, the shape hardcoded `m64n64k16`, and the dtype inferred through `dyn_cast<ShapedType>` (which a `!tile.fragment` is not, so it defaulted to bf16) — with **rc=0 and no diagnostic**. A K-loop recomputed A×B from nothing each step and returned the last partial product.

Measured on merged main, this was **not** specific to the typed fragment form: a legacy bare `tile.mma(A, B, C)` — what `LowerKReductionAddToTileMMA` emits for the canonical K-step — was dropped identically. **No fixture in the tree covered either case**, which is how it survived. The guard therefore keys on *has an accumulator*, not *is typed*.

**Outcome: not applicable — architecture-specific reason.** Probed: `--tessera-lower-to-x86` leaves `tile.mma` unlowered. x86 has no cooperative-matrix MMA path; AVX-512 K-loop accumulation is expressed in its own ops and never routes through this lowering.

## Cross-backend sync `ROCM-COMPILED-STRICT-DISPATCH-2026-08-04` — compiled-lane failures stop masquerading

Runtime dispatch contract changed. A compiled-ROCm **failure** (tessera-opt ran and serialized no kernel, or emitted a non-ELF blob) now routes through the existing `_note_dispatch_fallback` funnel, so `TESSERA_STRICT_DISPATCH=1` raises instead of degrading. **Envelope limits** (no libamdhip64, hipInit failed, tessera-opt not built, dtype/rank/arch out of range) are unchanged and still degrade silently — making those raise would break strict runs on every CPU-only host.

Measured before the fix: a deliberately broken pass pipeline returned `ok=True, compiler_path="rocm_compiled", execution_kind="native_gpu"` with correct numbers. Strict-mode suite results are identical before and after (18 fail both ways, all pre-existing), so this adds no new failures.

**Outcome: not applicable — architecture-specific reason.** x86 elementwise lanes raise `_RocmCompiledUnavailable` only for `lib is None` / missing-symbol conditions — envelope limits by construction, since there is no compile step whose output could be malformed. No x86 site was reclassified.

## Cross-backend sync `ROCM-PIPELINE-TILE-LOWERING-2026-08-04` — the compiled pipeline can lower `tile.mma`

Both ROCm compiled pipelines (plain and canonical) now run `lower-tile-to-rocm{arch=<chip>}` after `generate-wmma-gemm-kernel`. Verified byte-identical hsaco with and without the pass on the default path, so the production lane is unchanged.

**Outcome: not applicable — architecture-specific reason.** x86 carries `!tessera_x86.tile` and has no cooperative-matrix `tile.mma` path; its pipelines are untouched.

## Cross-backend sync `TILE-VIEW-BOUNDED-CONTRACT-2026-08-04` — bounded `tile.view` is a shared contract

`ViewOp::verify` now defines the pointer-backed operand contract: exactly 3 `(base, rowOrigin, colOrigin)` or 5 with `(rowBound, colBound)`. It previously accepted any count >= 3, so a 4-operand view was legal and meaningless and the bounded form's validity was decided by whichever backend looked.

**Outcome: not applicable — architecture-specific reason.** x86 carries `!tessera_x86.tile` and has no `tile.view`-backed fragment path.

## Cross-backend sync `TILE-VIEW-LINEAR-BASE-2026-08-05` — should `tile.view` carry a precomputed linear base?

ROCm W1.1 step 3 (`W1_1_TYPING_DESIGN.md` §4.7) established that isolated
fragment address derivation could not express the direct lane's shared row
offset. Measurement selected an optional precomputed `linear_base` operand on
`tile.view`; logical row/column origins remain present for bounds.

ROCm implemented explicit `tile.view` linear-base sharing. Its new same-run
final rebuilt measurement improves typed/direct from 0.685x to 0.711x, but does not close the
gap; load scheduling/wait overhead remains the ROCm-owned follow-up.

**Outcome for x86: NOT APPLICABLE.** This backend consumes neither `tile.view`
nor `tile.fragment_pack` (0 files). AMX/AVX-512 operands come from
`tessera_x86.amx_tile_load` over the `!tessera_x86.tile` type (Decision #19),
which addresses its own source directly; there is no `tile.view`-backed fragment
path whose base could be hoisted. If a future x86 path adopts Tile fragments,
re-open under this key.

## Cross-backend sync `TILE-DYNAMIC-LEADING-DIM-2026-08-04` — generic typed fragment addresses

Shared `tile.view` / `tile.store` can now carry an SSA leading dimension when
`#tile.memory_layout` states zero. **Outcome for x86: NOT APPLICABLE.** AVX-512
and access-gated AMX consume `!tessera_x86.tile`, not Tile fragments or
pointer-backed `tile.view`; no x86 lowering changed. Host Zen 5 validation:
x86 dtype + matmul-family suites, 21 passed.

## Cross-backend sync `E2E-REAL-LINEAGE-SCHEDULE-2026-08-05`

Shared compiler orchestration now records explicit artifact ancestry and
production `tessera-opt` registers the generated Schedule dialect. **x86
outcome: follow-up required under E2E-REAL-3.** Canonical x86 packaging still
accepts `GraphIRModule` and re-derives its launch Tile program, so the recorded
Graph→package-Tile edge exposes the fork and `lineage_complete` remains false.
No AVX-512 ABI, generated code, selector, or AMX gate changed. The consumer PR
must accept the canonical launch-Tile artifact and rerun Zen 5 exact execution;
this does not supply the separately access-gated Intel AMX packet.

## Cross-backend sync `E2E-REAL-SCHEDULED-MATMUL-2026-08-05`

The shared C++ spine now preserves a bounded static Graph matmul behind a
content-addressed `schedule.matmul` SSA edge and lowers it exactly once to the
portable A/B/D/M/N/K `tile.matmul_kernel` contract. The x86 instance is f32
storage/accumulation/output with m16n16k16 row/col layout and explicit
pipeline/raster fields. **x86 outcome: structural parity validated; physical
follow-up required under E2E-REAL-3.** No AVX-512 execution or performance is
claimed by this host-free conversion. Canonical x86 packaging must accept this
exact Tile artifact, run TileToX86, and repeat the Zen 5 numerical/performance
ratchet without reconstructing the launch contract from Graph IR. Intel AMX
evidence remains separately access-gated.

## Cross-backend sync `E2E-REAL-PHYSICAL-CONSUMERS-2026-08-05`

The bounded f32 matmul package now accepts `ScheduledMatmulArtifact` and
consumes its exact launch-level Tile text through TileToX86. The compile bundle
records adjacent Graph→Schedule→Tile→Target→backend digests rather than a
Graph-owned package fork. **x86 outcome: parity validated for E2E-REAL-3.**
Exact Zen 5 descriptor execution agrees numerically on the established
`1x1x1`, `5x17x9`, and `16x31x19` corpus, and the physical lit fixture proves
no Graph, Schedule, or launch-level matmul op survives. E2E-REAL-4 still owns
the AVX-512 performance ratchet and promotion decision. This is not Intel AMX
evidence; that named-host packet remains access-gated.

## Cross-backend sync `E2E-REAL-PERFORMANCE-2026-08-05`

The scheduled artifact now separates the physical 16x16 instruction tile from
an architecture-owned macro tile; x86 selects 16x16 for both. **x86 outcome:
promote.** On the exact Ryzen AI MAX+ 395 Zen 5 host, the established aligned
`64x128x96` and ragged `127x65x79` rows are bit-identical to the production
AVX-512 package. Scheduled/production median ratios are 1.031x and 0.988x,
inside the existing 10% ratchet. The report records compiler/toolchain and all
Graph/Schedule/Tile/Target/image digests, compile state, image size, CPU
features, and host-wall operation-total timing:
[`../../../../benchmarks/baselines/x86_avx512_e2e_real4_matmul_2026_08_05.json`](../../../../benchmarks/baselines/x86_avx512_e2e_real4_matmul_2026_08_05.json).
This is AVX-512 evidence only; Intel AMX remains access-gated.

## Cross-backend sync `E2E-REAL-SEMANTIC-KERNELS-2026-08-05`

The bounded canonical f32 softmax/reduction route now crosses real
Graph→Schedule→Tile boundaries. `schedule.softmax` and `schedule.reduce` bind
architecture, numeric policy, axis/kind, launch width, and durable SHA-256
identity; `ScheduledKernelArtifact` feeds the exact Tile text to TileToX86
without Graph re-entry. Static last-axis softmax and last-axis rank-reducing
sum/mean/max are lineage-complete, and tampered policy fails closed. Exact Zen
5 AVX-512 descriptor launches for scheduled softmax and reduction agree with
NumPy. **x86 outcome: parity validated for the bounded E2E-REAL-5 slice; no new
selector or performance promotion.** `keepdims=true` remains on the explicit
Graph-owned descriptor route because canonical `tessera.reduce` is presently
rank-reducing. This is AVX-512 evidence only; the named Intel AMX lane remains
access-gated and unchanged.

## Cross-backend sync `E2E-REAL-ATTENTION-2026-08-05`

`schedule.attention` now binds the shared static rank-4 online-softmax
recurrence, modifiers, launch contract, and architecture-owned backward-LSE
policy into one SHA-256 identity. The x86 package consumes the exact emitted
`tile.attention_kernel` through TileToX86 without returning to Graph IR and
preserves `save_lse/saved`. **x86 outcome: parity validated for E2E-REAL-5A.**
On the exact Ryzen AI MAX+ 395 Zen 5 host, the scheduled AVX-512 descriptor
launch agrees with the NumPy oracle for ragged `Sq=5/Sk=7` f32 attention.
This changes no selector and supplies no Intel AMX evidence. Canonical
attention backward was the next x86 family boundary.

## Cross-backend sync `E2E-REAL-ATTENTION-BACKWARD-2026-08-05`

`schedule.attention_backward` now carries the canonical tensor-valued dQ,
split-dK/dV, and ascending-reduction loops as one content-addressed three-result
program artifact. The exact Tile program lowers to
`tessera_x86_flash_attn_bwd_f32`; its descriptor requires the forward-owned
`row_lse` buffer, so `save_lse/saved` is explicit data identity rather than an
untracked policy string. **x86 outcome: parity validated for E2E-REAL-5B.**
Exact Zen 5 tests pass for MHA, GQA, and MQA; aligned and ragged shapes; and the
combined causal, symmetric-window, bias, and softcap envelope while preserving
the established AVX-512 modifier contract. No AMX evidence or selector
promotion is inferred.

## Cross-backend sync `E2E-REAL-5C-STATE-LINEAGE-2026-08-05`

The Zen 5 Lion VJP, factored/full Adafactor VJP, and sequence-mixer backward
launchers now enforce the shared content-addressed logical-buffer lineage and
consume exact typed Schedule→Tile artifacts before native launch. Runtime
consumers no longer retain or reconstruct Graph-op metadata. **x86 outcome:
parity validated for the bounded E2E-REAL-5C slice.** Exact Zen 5 Lion,
factored/full Adafactor, and gated/modified DeltaNet backward tests pass. No AMX
evidence is inferred.

## Cross-backend sync `ROCM-TYPED-EXECUTABLE-PIPELINE-2026-08-07`

The configuration schema makes the Tile producer, Target-IR consumer, and
backend image boundary explicit instead of accepting Python-composed pass
strings. **x86 outcome: parity validated.** The architecture-owned
`tessera-x86-executable` registry now applies the same closed-family rule while
preserving Schedule→Tile→`tessera_x86` identity and the stable AVX-512 shared
image. AMD async-copy/waitcnt semantics remain not applicable; Intel AMX stays
separately access-gated. No ROCm schedule or evidence transfers.

## Cross-backend sync `TSOL-PACKED-FUSION-2026-08-08`

`tessera.x86.spectral_composite.v5` consumes the new hashed compound-fusion
topology directly. Even-length convolution uses two packed N/2 RFFTs, a
Hermitian-bin multiply, and packed IRFFT; STFT frames into real storage before
the N/2 transform; ISTFT sends the half spectrum directly through packed IRFFT
before deterministic overlap-add. Odd windows retain a separately identified
full-complex fallback so the public shape envelope is unchanged.

**x86 verdict: retain.** All 16 native AVX-512 composite tests pass on the
Ryzen AI MAX+ 395 Zen 5 host. The 20-sample synchronized-host-wall packet is
[`../../../../benchmarks/baselines/tsol_packed_fusion_zen5_2026_08_08.json`](../../../../benchmarks/baselines/tsol_packed_fusion_zen5_2026_08_08.json);
baseline f32 maximum errors are `2.87e-6`, `9.71e-7`, and `8.77e-6` for
convolution/STFT/ISTFT. This changes no AMX claim. A same-run comparison with
the retired full-complex composite is still required before calling the v5
package a performance promotion rather than a correctness/workspace win.

That same-process comparison is now runnable through the comparison-only
full-complex symbols in the v6 native image; production artifacts cannot select
them. A pinned WSL Zen 5 20-sample run at batch 32 measured packed speedups of
1.470x, 1.525x, and 1.557x for `(input,kernel)` lengths `(256,65)`,
`(1024,257)`, and `(4096,513)`; maximum packed error was 3.05e-5. The runner
records cold and warm samples,
Schedule/Tile/child digests, CPU identity, affinity, governor visibility,
worktree state, and timing provenance. **Retain, not promote:** this checkout
was dirty, the prior model-specific event map remains promotion-ineligible,
and WSL exposes neither a frequency governor nor valid PMU provenance. A clean
pinned bare-metal Zen 5 `tprof` timing/symbol packet is
still required; the historical v5 selector packet is not relabeled as v6.

## Cross-backend sync `TILE-SYNC-RECONCILE-2026-08-10`

`tile.async_copy`/`tile.wait_async` now have one declared contract (ODS dual
form, `TileOps.td`); shared diagnostic `TILE_ASYNC_STAGE_NEGATIVE` added.
**x86 outcome: core parity validated; the async-copy model is not applicable
to physical x86 execution.** The x86 lane lowers tiled
matmul through `TileToX86Pass` `func.call`s and emits no
`tile.async_copy`/`tile.wait_async` — CPU lanes have no async-copy
double-buffering stage model to reconcile. PR #544's required build and unit
lanes validated the shared Tile dialect on an x86 CI host. This transfers no
AVX-512 schedule, Zen 5 performance result, or AMX evidence.

## TILE-SYNC-TYPED-2026-08-15 — shared Tile sync ABI assessment (PR #566)

**Not applicable.** x86 has no asynchronous-copy or mbarrier lane: the
x86/no-async path is a recorded no-op in the await-sinking contract (W5.2a),
and no x86 lowering consumes `tile.mbarrier.*`, `tile.tma.*`, or the new
registered sync ops. The shared Tile dialect still builds and verifies on
this configuration (combined-driver lit run green in the same 324/0 suite);
no AVX-512/AMX evidence transfers or is claimed.

## REF-TIER-OPS-2026-08-15 — reference-tier op registration assessment (PR #568)

PR #568 registered ten new public operations through the canonical op catalog
and the primitive coverage registry — `tridiagonal_solve` (Thomas recurrence,
PDE plan §III.1 / TSOL-A1) and the nine-op coalition-lattice family
(`game_subset_zeta`, `game_subset_mobius`, `game_superset_zeta`,
`game_superset_mobius`, `game_coalition_marginal`, `game_semivalue`,
`game_boltzmann_value`, `game_coalition_excess`, `game_mex`). Op registration
is a shared contract, so this queue records the outcome per AGENTS.md
"Cross-backend work coordination"; PR #568 itself landed without these records.

**2026-08-16 physical follow-up (`REF-TIER-PHYS-2026-08-16`): implemented for
the solver and four lattice transforms.** `schedule.tridiagonal_solve` selects
`batch_vector_thomas_v1`; the native kernel packs recurrence rows and executes
eight independent systems per AVX-512 fp64 vector before the final fp32 store.
`schedule.coalition_butterfly` carries `(half, sign, stage_order)` and one
AVX-512 Yates consumer implements all subset/superset zeta/Mobius variants in
fp64 working storage. Direct numerical tests cover scalar/tail/full SIMD
batches, non-power-of-two rejection, zero-pivot rejection, and all four
butterfly modes. This is a correctness retain, not a performance promotion;
a clean Zen 5 benchmark packet remains open. Coalition marginal, semivalue,
Boltzmann value, excess, and MEX remain reference/composition follow-ups rather
than five new one-off emitters. No AMX evidence is claimed.
Native solver and butterfly calls now assert their returned status in Target
IR, and the manifest records the shipped C symbols as `device_verified_abi`
rather than overclaiming generated-JIT proof. This is ABI/package evidence;
the existing direct fixtures remain the numerical authority.

## APPLE-SCHEDULED-REDUCE-NAN-2026-08-16 — shared reduce NaN semantics (PR #571)

**Shared contract changed; assess before relying on extrema reductions.** The
synthesizer's reduce vocabulary (`compiler/fusion_core.py::_PW_REDUCE_KINDS`)
emitted `max(acc, v)` / `min(acc, v)` for `amax`/`amin`. Metal's `max`/`min` are
IEEE maxNum/minNum-style and **suppress** a NaN operand, so the emitted kernel
disagreed with the table's own numpy reference (`a.max(-1)`, which propagates)
and with the `nan_mode = "propagate"` the reduce Schedule artifact declares.
With the `-INFINITY` seed an all-NaN row reduced to **`-inf`** — missing data
silently becoming a finite extreme. The accumulators now propagate explicitly.

**x86 outcome: not applicable — no consumer.** `_PW_REDUCE_KINDS` supplies MSL
accumulate expressions consumed only by `compiler/emit/apple_msl.py`; the Zen 5
AVX-512 scheduled reduction consumes its own `tile.reduce_kernel` lowering and is
unaffected. No x86 schedule, ABI, selector, or evidence changes. Same caution as
NVIDIA for any future `emit/x86_llvm.py` reduce emitter: `maxps`/`maxss` also
suppress NaN, so propagation must be explicit to honour `nan_mode = "propagate"`.

Also recorded for coordination: PR #571 admits Apple GPU into the shared
scheduled reduce contract (`scheduled_kernel.py`, last axis only) and closes
APPLE-DEVICE-EVENT-1 by giving the Apple MPSGraph BMM route an owned command
buffer. Both are Apple-guarded — the shared `scheduled_kernel` gate adds an
`apple_gpu` branch beside the existing x86/ROCm ones and changes neither — and
the runtime edit is in `apple_gpu_runtime.mm`, which no sibling links.

## APPLE-ATTN-BWD-PERF-1-2026-08-16 — backward row-prepass assessment (PR #572 merged)

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

**x86-specific note.** This backend already models a delegated boundary
explicitly (`tessera_x86.abi_call`, Decision #19), which makes it the natural
place to prototype how a LAPACK-backed matrix-function lane should be
represented in Target IR without hiding the delegation.

## Cross-backend sync `NVIDIA-BARRIER-AT-BIRTH-2026-08-21`

**Not applicable to x86 physical lowering.** The shared role-bearing Tile
contract remains compatible, while barrier binding and local TMA slots are
CUDA-owned. No CPU schedule, runtime ABI, or x86 evidence state changes.

## Cross-backend sync `TARGET-IR-CONFORMANCE-2026-08-21`

**NVIDIA-only conformance closure.** NVIDIA's Target-value restriction and
registered-dialect test leave the x86 ABI/call vocabulary and AVX-512 evidence
unchanged.


## Cross-backend sync `NVIDIA-SPECTRAL-PHILOX-JVP-2026-08-22`

**Outcome: parity already validated; no x86 physical change.** NVIDIA joined the
existing shared spectral and native-product contracts with CUDA-owned cuFFT and
Philox consumers. x86 retains its AVX-512 spectral/Philox packages and evidence;
no CUDA workspace or physical schedule is shared.


Cross-backend sync `NUMPOL-CARRIER-1-SCHEMA-AND-REDUCTION-2026-08-25` — **the
policy gets a schema and the reduction family carries its accumulator;
x86 outcome: shared contract landed; AVX-512 is where it was PROVEN.** Two measured defects closed, both shared,
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

This backend supplied the executed evidence: the native object was built and run on the Ryzen AI Max+ 395. That is a CPU correctness result on this host's toolchain and is not a timing claim.

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

**Outcome for this backend: `parity validated` — defect present here and fixed
here, with exact-device evidence.** `src/compiler/codegen/tessera_x86_backend/
src/kernels/avx512_deltanet_f32.cpp:291` held
`const float safe_norm = std::max(norm, 1.0f);`; the correction now divides by
`norm`. Verified on **Princess-Luna AVX-512** (Zen 5 — no AMX here and none
coming, so x86 proof means AVX-512): `tests/unit/test_x86_deltanet_compiled.py`
went from **2 failed / 16 passed** to **18 passed**.

**This is the two-transcription failure mode, not a copied bug.** ROCm and x86
reached the identical wrong divisor independently, in different languages
(MLIR builder vs C++ intrinsic kernel). A per-backend patch would have left the
other wrong, which is why the fix landed against the shared formula in one PR
and was re-measured on both lanes rather than inferred from one. Both lanes
happen to live on the same host; they were still rebuilt and run separately,
because sharing a box is not sharing evidence.

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

**Outcome for this backend: `not applicable` — no device clock exists to be
cross-checked.** The x86 lane times on the host, where the wall clock is not a
*witness* for a separate device counter but the measurement itself. There is no
event API, no asynchronous launch queue, and therefore neither of the two
failure modes this key is about: nothing can be queued behind a start marker,
and there is no second clock that could disagree.

**The one thing that does carry over is the shape of the mistake, not its
mechanism.** This key exists because a rule was ported between architectures by
analogy and its precondition was left behind — the band travelled, the drain
did not. x86 is the backend most exposed to that pattern, because it has the
fewest hardware-specific constraints of its own and so is the likeliest place
for a GPU-shaped rule to be adopted wholesale. Before importing any timing
discipline from the NVIDIA or ROCm plans, check which of its steps exist to
defend against a hazard this lane actually has.

**No AVX-512 evidence is claimed** — no x86 code changed.

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

**Outcome for this backend: `parity validated by construction` — x86 gets the
dispersion for free, and is the one lane where the fix required no new
measurement.** The end-to-end path already collected per-rep samples inside
`measure_latency` and threw them away to return a median;
`measure_latency_samples` simply stops discarding them. x86 times on the host
through that path exclusively — there is no device-event lane here — so every
x86 row acquires a real noise floor on its next regeneration with no extra
timing cost. The `device_repeats` multiplier does not apply.

**The one thing worth watching on this backend.** Host wall-clock timing is the
noisiest of the fleet's clocks: no device counter isolates the kernel, so
scheduler jitter, frequency scaling and page faults all land in the spread.
That is not a defect of the metric — it is what the caller actually pays, which
is why this lane times end-to-end by choice — but it does mean x86 rows will
separate *less often* than GPU rows at the same true speed difference. An
unseparated x86 row is therefore weaker evidence of "the lanes are equal" than
an unseparated device-timed row would be, and should not be read as one.

**No AVX-512 evidence is claimed** — no x86 code changed, and no x86 row has
been regenerated.

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

**Outcome for this backend: `not applicable` — there is no second clock to
witness.** x86 times on the host, so the wall clock is the measurement rather
than a check on something else; the two failure modes this key is about (a
self-reported device interval believed unchecked, and a witness scoped to the
wrong region) both need a device clock this lane does not have.

**What does apply here is the clock-selection evidence, and it is directly
reusable.** Measured on Apple Silicon while grounding the witness:
`std::chrono::steady_clock` **is** `high_resolution_clock` on libc++, and is
backed by the same constant-frequency counter as `CNTVCT_EL0` — mach timebase
125/3 ns per tick = 41.67 ns = **exactly 24.000 MHz**, confirmed against a
direct `mrs cntvct_el0` read over the same span. Constant rate is the property
that matters: it does not move with DVFS, so a sample taken under one power
state is comparable to one taken under another. The register read is cheaper
(0.3 ns vs 16.3 ns) but that is 0.00003% versus 0.0016% of a 1 ms span.

**Where that stops being negligible is this backend.** 16 ns of read overhead
against a ~1 ms GPU dispatch is nothing; against an AVX-512 microkernel it is
not, and a tick period is a *resolution floor* as well. An x86 timing lane
measuring individual kernels should size its region against that floor — or
batch reps until the span clears it — rather than assume host timing is free.
**Measure that floor on the ROCm box before designing around it**: 41.67 ns is
Apple Silicon's figure, and the Zen 5 host's differs.

**The Linux equivalents, so this is not re-derived from the Apple entry.**

| need | Apple Silicon | Linux (Zen 5, the x86 host) |
|---|---|---|
| monotonic wall | `steady_clock` → 24 MHz `CNTVCT_EL0` | `steady_clock` → **VDSO**, so the hardware clock is read with no real syscall |
| constant-rate raw counter | `mrs cntvct_el0` | `rdtsc` / `rdtscp` |
| CPU time excluding idle | `clock_gettime(CLOCK_THREAD_CPUTIME_ID)` | same POSIX call, natively TSC-backed and correspondingly cheaper |
| timeline probes | `os_signpost` → Instruments | `STAP_PROBE` (`sys/sdt.h`) → `perf`/eBPF, or Tracy |

Two consequences specific to this lane. First, `high_resolution_clock` is
usable on both hosts for the same reason — no syscall in the hot path — so a
shared timing helper needs no per-OS branching for the wall clock, only for the
raw-counter and probe layers.

Second, and more important: **`CLOCK_THREAD_CPUTIME_ID` is the more interesting
clock here than any wall clock.** An AVX-512 kernel timed on a shared,
frequency-scaling host has scheduler idle charged to it by a wall measurement,
and thread CPU time excludes exactly that — it is the closest thing this
backend has to the *isolation* a GPU device counter provides, which is the
property the whole three-clock discipline exists to obtain. It also gives this
lane a witness relationship it currently cannot form at all: for a
single-threaded region thread CPU time cannot exceed wall time, so the two
bound each other the way a device clock and a host clock do on the GPU
backends. That is the concrete route to an x86 acceptance rule, and it is
available today without new hardware.

`rdtsc` carries the same caveat as `cntvct_el0` — its rate is fixed and
unrelated to the core's current frequency, which is what makes it comparable
across power states and equally what makes it *not* a cycle count.

**No AVX-512 evidence is claimed** — no x86 code changed.

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

**Outcome for this backend: `not applicable` — no device clock to mis-select.**
x86 times on the host, so there is no pair of properties where one measures GPU
execution and the other does not.

**But the trap generalises to this lane's own clock pair, and it is the one
already recommended here.** `APPLE-TIMER-WITNESS` recorded that
`CLOCK_THREAD_CPUTIME_ID` is the more interesting clock for x86 because it
excludes scheduler idle a wall measurement charges to the kernel. That is
exactly a "two clocks, different regions" situation, and the Apple defect is
what it looks like when the wrong one is chosen and nothing checks: a number
that is plausible, stable, and not a measurement of the work. **Before that
pairing is built, wire the metamorphic tracking check with it** — vary the
problem size and require thread-CPU time and wall time to move together. It is
a handful of lines and it is the only check that would have caught this class.

**No AVX-512 evidence is claimed** — no x86 code changed.

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
`record_x86_base_packet.py:98` stamps `tested_commit` from `git rev-parse HEAD` with no dirtiness
check, exactly as Apple's did.

**Why it is not fixed in this PR.** The Apple guard works because that packet
declares which file it fingerprints, so the set to check is unambiguous. This
lane fingerprints measured resources rather than sources, so choosing the right
file set — plausibly the AVX-512 kernel sources — is a judgement about what this backend's
measurement actually depends on, and getting it wrong fails in the worse
direction: a too-narrow set is a guard that passes while the provenance is
false, which reads as protection and is not. That call belongs with someone
looking at this backend's build, on **Princess-Luna (AVX-512)**.

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

**Outcome for this backend: `follow-up required` — x86 has its own spectral
kernels and was not touched.** `avx512_fft_f32.cpp` and
`avx512_spectral_backward_f32.cpp` implement this family natively, and
`TileToX86Pass.cpp` lowers to them, so the Python-level check does not
necessarily gate the x86 route.

**What to check on Princess-Luna:** whether the AVX-512 spectral path validates
operand ranks at all, or whether it too computes something for a mismatch. The
host-side tests that exercise it (`test_autodiff_spectral_target_binding.py`)
all use equal ranks, so a permissive native path would be untested in exactly
the way the dispatch lanes were — and this defect's whole shape is that a
permissive path is invisible until something written against it collides with a
strict one.

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

**Outcome for this backend: `follow-up required` — same question, and x86 is
the lane where a per-call compile would be least visible.** Nothing x86-side
changed here.

x86 kernels are AOT-compiled into the extension rather than built per launch, so
the exact defect is unlikely. What is worth confirming is the *shape*: any
lane that attempts an expensive operation, fails deterministically on this host,
and does not remember the failure. The x86 timing lane is host wall-clock only
(see `APPLE-TIMER-WITNESS`), so a fixed per-call cost lands directly in every
measurement this backend reports rather than being isolated to a device counter.

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
question is cheap to ask of the host lane: no callback exists and none is needed — the value is available the moment the call returns.

**The test worth applying, whatever the API.** Does the timestamp read path
contain a wait that has no timeout? On Apple the answer was yes, twice — once
outright, once behind a flag that six call sites had to pass correctly. Neither
was visible as a failure until a reviewer traced what happens when the shared
event times out, because the hang only occurs on a GPU that is already broken.

**If this is ever checked on Princess-Luna**, the control that made the Apple answer
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
Princess-Luna, where the test PASSES rather than skips.**
`test_attention_package_rejects_stale_parent_and_tile_lineage` now checks
`rt._x86_elementwise_available()` alongside `find_tessera_opt()`, matching the
sibling test one function away that already did both.

**Not restructured to be host-free, deliberately.** The test builds a *real*
package through `build_native_attention_vjp_package` and mutates copies of it,
and `package_x86` needs the shared image before `validate()` is ever reached.
Mutating a genuinely-built package is stronger evidence than mutating a
synthetic one, so the right answer is an honest skip on hosts without AVX-512 —
not a weaker test that runs everywhere.

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

**Outcome for this backend: `not applicable` — x86 has no rows in this corpus
and no device timer.** Its lane times end-to-end on the host, where
`measure_latency_samples` already collects per-rep samples, so it gets a real
noise floor for free and `device_repeats` does not apply.

**The transferable warning is about the CONSTANT, not the clock.** This run's
central finding is that a sample count chosen for cost (3) produces a noise
estimate 2.5× the truth, and that the estimate is *published*. x86's timing
constants have the same property: `reps`/`warmup` are chosen for suite speed,
and whatever spread they yield becomes the floor every x86 separation verdict
is judged against. Before x86 rows enter this corpus, measure how its reported
sd moves with sample count — the NVIDIA curve (48% → 31% → 19%) took one probe
and changed what the recorder does.

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

**Outcome for this backend: `not applicable` — no x86 candidate declines by
shape.** Both registered lanes were audited: `x86_aocl_dlp` gates on
`_aocl_epilogue(region)`, which is purely structural, and `x86_generic_c`
serves any shape. Notably `x86_aocl_dlp.available()` already cites the same
PR #289 review this PR's ROCm half invokes — "which would let it win by tier
and demote a supported fused GEMM to the reference" — so this backend had
internalized the lesson on the availability axis before the shape axis existed.
The new seams are inherited and cost nothing here.

Follow-up if AOCL-DLP later gains an alignment-restricted microkernel: state it
in `applies_to_inputs`, not in `run`.

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

Apple's `promotion_rules` block turned out to be a declaration no code read;
fixed there (see the Apple plan). Assessed here because the pattern travels.

**Outcome for this backend: `not applicable` — x86 has no route ledger.** The
committed x86 baselines (`x86_avx512_e2e_*_comparison.json`, the
`core_compiler_*_avx512.json` family) are direct measurement comparisons, not
promotion ledgers: there is no incumbent-vs-candidate selector, so there is no
promotion to certify and no rules block to leave unread. If an x86 route
selector is ever added, the Apple loader's shape is the one to copy —
**re-derive the promotion from the retained evidence at load time**, rather
than trusting a status string and carrying the thresholds as documentation.
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

**Outcome for this backend: `not applicable` — no promotion artifact.** The
committed x86 baselines are direct measurement comparisons with no
incumbent-vs-candidate verdict, so there is no promotion to re-derive. If an
x86 selector is added, copy Apple's shape: put the thresholds **in the
artifact** and re-derive the verdict from retained evidence at load time, so
the gate does not live only in the recorder that produced the file.

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

**Outcome for this backend: `not applicable` — no matrix-core lane to gate.**
AVX-512 is the live x86 ISA here and its GEMM microkernel is not a
fixed-fragment matrix unit, so there is no 16x8x16-style tile whose edges need
predicating; the AMX ops that would have one are closed by direction (ACE
supersedes AMX, and no fleet box has it).

If an ACE matrix lane is ever added, the rule above is the one to apply from
the start: **zero-pad a contraction dimension, suppress the store on an output
dimension.** Getting that backwards is silent corruption rather than a fault.

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

**Outcome for this backend: `parity validated`.** `x86` is fully classified and
is the only classifier of the `elementwise` family. Its two false rows
(`min`/`amin`) and the false `depth_attn` compiled claim are removed. Its
`sum` route is the reason the literal check accepts both spellings:
`x86_native` names `tessera.sum` while coverage records the canonical
`tessera.reduce`.

## LAUNCH-OVERHEAD-BOUND-1 — cross-backend assessment (recorded 2026-09-02)

`tests/_support/launch_overhead.py` (PR #686) is shared test infrastructure: it
bounds `rt.launch` overhead against the `execution_kind` the launch reports.
A device dispatch gets a flat ceiling; every other lane keeps the
self-calibrating `max(2.0, direct_ms*4)` against the oracle arm. Review on #686
asked for a per-backend verdict, and the four are NOT the same.

**x86 — not applicable by construction, and deliberately so.** `native_cpu` is
reachable (`runtime.py` selects it for `target == "x86"`) but is excluded from
the flat-ceiling set. A native CPU lane executes on the same silicon as the
`direct` oracle arm, so the self-calibrating ratio stays meaningful and is far
tighter than any GPU-derived ceiling: at the first cut `native_cpu` was inside
the ceiling set, and an AVX-512 launch regressing from ~0.15 ms to 19 ms would
have passed a 20 ms bound. That hole is closed. If an x86 row ever needs a flat
floor it must be measured on an AVX-512 host and given its own constant --
never by widening the GPU one.

## MSW-4A-CODIFF-SIGN-1 — cross-backend assessment (recorded 2026-09-02)

`ga.calculus.codiff` changed from the unsigned `⋆d⋆` composition to the true
codifferential `δ = (-1)^(n(k+1)+1) ⋆d⋆` (PR #688), and its `clifford_codiff`
VJP changed with it. That is a shared numerical contract, so each backend gets
an explicit verdict.

**x86 — parity validated, no native change required.** Same shape as ROCm: the
AVX-512 Clifford lanes call `_clifford_ops.clifford_codiff`, which delegates to
the signed `ga.calculus.codiff`; no x86 symbol implements the codifferential
directly, so nothing carries the old convention. Covered by the same
Princess-Luna sweep (that host runs both the ROCm and AVX-512 lanes).

## Cross-backend sync `DISPATCH-BREAKER-RESIDENT-2026-09-03`

**Owning item:** `APPLE-DISPATCH-WEDGE-1` (Apple plan) · **synchronization
key:** `DISPATCH-BREAKER-RESIDENT-2026-09-03`

**Shared contract changed.** `runtime._apple_gpu_run_checked` gained an
optional `silent_failure_timeout_s`, and a new
`runtime._apple_gpu_device_call_checked` routes the eight device-resident
(`DeviceTensor`) Apple dispatch paths through the dispatch circuit breaker.
Shared test infrastructure changed with it: `test_apple_gpu_dispatch_breaker.py`
gained a per-dispatch AST drift gate. Nothing outside the `_apple_gpu_*`
namespace is touched, and no IR, ABI, dtype, diagnostic code or benchmark
schema changes.

**The premise that makes this Apple-shaped.** The breaker exists because Metal's
`waitUntilSignaledValue:timeoutMS:` **returns** when its deadline expires. The
caller then falls back to host, the next dispatch asks the device again, and
each one pays the full 30 s — an observed 70-minute sweep against a 4-minute
healthy run. The repeated cost, not the hang, is what a breaker cuts.

**x86 — not applicable, and structurally so.** There is no device and no
asynchronous dispatch: the AVX-512 lane is a synchronous host function call
that either returns a result or raises. There is no wait to bound, no deadline
to expire, and no streak to accumulate, so neither the breaker nor the
duration-based silent-timeout classifier has anything to apply to. This entry
exists to record that the assessment was made rather than skipped; no x86
change is expected under this key now or later.

**Extended 2026-09-03 for the runtime-side follow-ups (#710, #711).** The
earlier note above covers the Python dispatch helpers. Three further contract
changes landed in the Apple runtime itself, and each is assessed here:

1. **A bounded wait now publishes timeout kind 1.** `ts_enc_commit_wait` used
   to print an expiry to stderr and touch nothing, so the Python accounting had
   to infer a stall from wall time. It now reports on the shared error channel,
   as the other bounded waits already did.
2. **Each bounded wait owns its event.** Both Apple wait helpers reserved
   increasing values on one context-wide `MTLSharedEvent` under a lock released
   before commit, so a later dispatch could signal first and satisfy an earlier
   waiter while its own command buffer was still running.
3. **A timed-out dispatch quarantines its pooled buffers.** A guard whose
   acquire predates a timeout drops its buffer instead of returning it to the
   shared pool, since the stalled command may still read or write it.

**x86 — not applicable, structurally.** The CPU backend runs synchronous host
calls: there is no device to stop answering, so no wait to bound, no expiry to
publish, no event to share and no device buffer pool to quarantine. All three
changes are about what happens when a device stalls, which is not a state this
backend can enter.

**Validation performed:** none required; no x86 code changed. **Missing
exact-device evidence:** none.

**Extended again 2026-09-03 — the event-less fallbacks are bounded.** Four
Apple waits fell back to an untimed `waitUntilCompleted` when `newSharedEvent`
failed. That is the case where the device is already in trouble, so the one
path taken *because* it was unhealthy was the only one that could hang
forever. They now poll `status` against a deadline and, on expiry, report
timeout kind 1 and quarantine their pooled buffers.

**x86 — not applicable, structurally.** Synchronous host calls, no device, no
command buffer, so there is no wait to bound and no completion status to poll.

**Validation performed:** none required; no x86 code changed. **Missing
exact-device evidence:** none.


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

**x86 outcome:** Shared-contract parity validated by host-free tests only. The coordinate and contraction additions are Python reference/front-end algebra; no native x86 lowering or schedule is added. No AVX-512 result is claimed from these tests.

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

x86 outcome: host-free compiler contract validation only; native recipe selection and performance evidence remain follow-up required.

## Engineering sync `EVAL-TELEMETRY-IKF-2026-09-04`

Owners: MSW-9, APPLE-DISPATCH-WEDGE-1 and IKF-1. Program-pair evaluator
comparison keeps both native provenance gates; ANN composition and ReLU
identity checks are separately registered reference laws. IKF-1 is now bound
in the integrated plan: P0 timing first, P2 after validated clocks and
Schedule-Object region identity, P1 host schema/math independently.

x86: not applicable to Apple capture state and GPU wait mechanics. Native
ANN fusion remains follow-up required. IKF indexed-slot instrumentation stays
deferred to a measured CPU need under the existing TPROF-X86 owner.


## Cross-backend sync `EVIDENCE-POLICY-20260904`

Decision #26 coverage snapshots move to revision-bound CI artifacts with source
commit/tree fingerprints and output hashes. The canonical renderer and semantic
coverage checks remain shared; this changes evidence delivery, not backend
capability or execution status. Host-free validation applies to this contract.
No device measurements or schedule parity are inferred.

The optional median bound is Apple-route-only and is not applicable here:
this backend does not consume the Apple route ledger or inherit its timings.

IKF-1 admission guard: the shared D2 cache and persisted-record consumer refuse
L2/L3 intra-kernel timings (`evidence.instr_level`) and malformed levels as
dispatch evidence. L0/L1 and existing pre-instrumentation records retain their
semantics. This is a host-contract check for this backend, not a device-clock
or instrumentation implementation claim.

Sync `APPLE-POLICY-COMPARE-20260904`: not applicable to x86 selection. The Apple
experiment keeps observed and synthetic sensitivity data separate; no CPU timing
policy, runtime contract or execution claim changes.


## Compiler foundation sync `IR-NATIVE-FOUNDATION-1` — 2026-09-04

Migrate Graph-owned cohort/elementwise/breadth packages to verified IR inputs; use the existing MLIR-to-LLVM JIT for general bodies and preserve explicit tuned-library calls as distinct routes. Follow-up required for Zen 5 exact-host proof. A prebuilt shared-library call is native execution, not proof of a generated LLVM kernel body.

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

Sibling outcome: not applicable to execution parity. x86 scheduled packagers and LLVM/library execution paths are unchanged.
The driver change is confined to the NVIDIA branch; no device evidence transfers
to this backend. Its F2 migration obligations remain open.

## Foundation F2 unary slice — `IR-NATIVE-FOUNDATION-1` — 2026-09-05

Owning item: E2E-REAL-5 / foundation F2. The shared native scheduling passes now
admit SM120 f32 softmax and serial rank-reducing sum/mean/max. They emit a raw
LLVM launch wrapper with the established NVIDIA symbol/ABI, and retain the
Schedule hash in Tile IR. NVIDIA packaging consumes that artifact directly.
NVIDIA's existing `approx_exp2` policy is explicit and hashed; other architectures
retain `accurate`. The verifier rejects a policy inconsistent with its architecture.
The wrapper consumer refuses extra function work rather than erasing it.

Sibling outcome: not applicable to native execution changes. The x86 schedule
policy is unchanged; shared compiler fixtures cover its existing boundary.
No new x86 exact-device evidence is claimed. Its F2 families remain open.

## F2 direct unary clients — `IR-NATIVE-FOUNDATION-1` — 2026-09-05

The public NVIDIA unary package APIs now enter the same native Schedule/Tile
boundary as the driver for the migrated f32 envelope. Missing `tessera-opt` is
an explicit failure for that envelope, not a return to Python kernel emission.
Private Graph constructors remain for unmigrated dtype/policy cases and retained
differential baselines; they are not new production routes.

Sibling outcome: not applicable. The x86 package APIs, native passes,
runtime ABI and numerical policy are unchanged; no device proof transfers.

## F2-U1–U10 unary closure — `IR-NATIVE-FOUNDATION-1` — 2026-09-05

NVIDIA unary packaging now consumes native Schedule/Tile for f16/bf16/f32
softmax and sum/mean/max/min reductions with f32 accumulation/output, arbitrary
static axes, keepdims and serial/cooperative_128 policy. Production Graph
constructors were removed; differential baselines live only under test support.
The shared Graph reduce verifier admits narrow-to-f32 and retained dimensions;
reverse AD refuses those new envelopes explicitly. Generic Linalg lowering still
declines them. No new dtype, operation, runtime ABI or physical schedule is added.

Sibling outcome: follow-up required for any expansion to the new Graph reduce
envelopes. Existing x86 schedules and runtime ABI are unchanged and shared
compiler fixtures remain gated. No new x86 exact-device evidence is claimed;
NVIDIA's cooperative schedule does not transfer to this architecture.

## F2 norm/attention — `IR-NATIVE-FOUNDATION-1` — 2026-09-05

The shared compiler adds `schedule.norm` for SM120 unweighted f16/bf16/f32 row
normalization and admits SM120 forward `schedule.attention` with explicit
recompute policy. NVIDIA direct package/driver paths consume native Tile wrappers;
old constructors are test-only baselines. Runtime ABI and physical kernels are
unchanged. Norm rejects unsupported policy overrides and epsilon that cannot be
represented as positive finite f32.

Shared finding fixed: forward attention hashes now encode exact f32 policy bits
instead of six-decimal strings. Regenerate old forward Schedule artifacts;
stale serialized hashes fail closed. Backward hash encoding remains follow-up.

Sibling outcome: shared forward hash correctness is regression gated; no new
x86 exact-device proof is claimed. Physical policies and runtime ABI are
unchanged. Follow-up required for old Schedule regeneration, backward hash review
and remaining direct Graph-package migration. NVIDIA norm admission is not
x86 execution support.


### F2-A2/A3/P1/S1 package-contract synchronization — 2026-09-05

Owner: E2E-REAL-5; synchronization key `IR-NATIVE-FOUNDATION-1`.
Shared backward Schedule hashes now encode exact f32 policy bits. Regenerate
older backward Schedule artifacts; no runtime pointer ABI changes. Shared
ReplaySSM geometry and spans reject lossy integer inputs and overflowing native
workspace sizes. Native packed/stateful producers remain follow-up required.
x86 native Schedule replay and adjacent-f32 backward hash regression are covered
on Princess-Luna WSL. NVIDIA physical mask and paired package API changes are not
applicable to x86's split-reduced kernel. Persistent ReplaySSM execution is not
added; native constructor retirement and dedicated owning-host performance proof
remain follow-up required.


### Deleted functionality reassessment — 2026-09-05

Synchronization key: `IR-NATIVE-FOUNDATION-1`. The
[central reassessment](../../compiler/INTEGRATED_COMPILER_PLAN.md#deleted-functionality-reassessment--2026-09-05)
routes pipeline ownership to W2.4a/CAKE/SO-2, verifier coverage to W2.4,
native residual policy to W5.1, and sharding/halo composition to W5.4 and
COMP-SCHED-OVERLAP-1. StableHLO interoperability is deferred pending a named
consumer. This is planning only: no dialect restoration, capability promotion,
or new hardware evidence. Existing F2 implementation ordering is unchanged.

Assess CPU ownership/effects and native adjoint/sharding consumption; retain explicit no-async behavior where appropriate and require CPU-specific performance evidence.


### Scheduled unary replay review fix — 2026-09-05

Owner: E2E-REAL-5; synchronization key `IR-NATIVE-FOUNDATION-1`.
Not applicable to x86 physical lowering: this change is confined to the
NVIDIA scheduled unary package consumer. Shared IR and runtime ABIs are unchanged.
Existing x86 package validation and owning-host evidence remain separate;
no sibling replay completeness or performance claim follows from this fix.


### Native checkpoint, packed/state boundaries and ownership audit — 2026-09-05

Owner: E2E-REAL-5 / W2.4 / W2.4a; sync `IR-NATIVE-FOUNDATION-1`.
The integrated plan's five-action loop records native saved-LSE, signed-INT4 and
bounded paged-read package migrations, the recovered prefetch-space check, and
the nonblocking-poll reuse-verifier fix. Shared legality is assessed on every
backend; allocation-scoped release and control-flow lifetime proof remain open.
Not applicable to x86 physical package generation: SM120 pointer launchers and
packed views are not CPU schedules. Shared native verification applies where
used. Follow-up required for CPU-specific packed/state consumers and lifetime
proof; retain explicit no-async behavior and separate CPU performance evidence.

### Physical paged-read mnemonic isolation — PR #728 review

Owner E2E-REAL-5 / F2-S1; sync `IR-NATIVE-FOUNDATION-1`.
The bounded native tensor producer is `tessera.paged_kv_read`, distinct from the
public `tessera.kv_cache.read(cache, start, end) -> (K, V)` contract. The public
handle form remains unchanged and its Graph ODS declaration remains open.
This shared-contract correction prevents the NVIDIA physical form from claiming
Apple, ROCm or x86 public cache semantics; no sibling runtime or ABI changes.

### Allocation lifetime analysis and device regression comparison — 2026-09-05

Owner W2.4a / CAKE / SO-2; sync `IR-NATIVE-FOUNDATION-1`.
Shared analysis now retains all pending allocation accesses, scopes direct-token
and keyed completion, joins branch/CFG paths, checks loop backedges and rejects
premature frees and use-after-free. Thread rendezvous cannot retire DMA.
Unknown origins/regions and loop-token generation remain conservatively gated;
see the integrated plan's matching section for the exact admission boundary.
Follow-up required: CPU ownership/alias contracts and the memref reuse-group consumer; no GPU timing is CPU evidence.


### Dynamic completion generations and memref arena proof — 2026-09-05

Owner W2.4a / CAKE / SO-2; sync `IR-NATIVE-FOUNDATION-1`.
The integrated plan's matching section supersedes the direct-token-only and
memref-planner limitations above. SSA edge renaming distinguishes dynamic
completion generations; structured/CFG allocation identity is shared by operation
and lifetime verifiers. Memref cast/view lifetimes now feed both reuse assignment
and arena preflight; forged groups fail before physical materialization, and
supported alias descriptors retain the arena address space. No runtime ABI,
selector or architecture schedule changes.

Shared static alias and SSA lifetime rules apply. GPU workgroup arenas and asynchronous ring schedules are not applicable to the CPU route; CPU ownership/escape summaries and CPU-specific reuse consumers remain follow-up required.


### Structured-path reuse, private borrowing and device ring experiment — 2026-09-05

Owner W2.4a / CAKE / SO-2; sync `IR-NATIVE-FOUNDATION-1`.
The integrated plan's matching section supersedes the blanket structured-region
and direct-call exclusions above. Coalescing requires all-path completion,
derived uniform branch exclusivity and release before loop backedges. Private
callee bodies establish borrowing; the arena preserves the existing helper ABI.
External/recursive ownership and general CFGs remain conservatively excluded.

Body-derived private-call borrowing is shared analysis; workgroup address-space
arenas and GPU ring timing are not applicable to CPU execution. Follow-up
required: CPU ownership/ABI consumers, general CFG coalescing and CPU-specific
reuse measurements. GPU occupancy and barrier results are not CPU evidence.

**Async experiment assessment (2026-09-05), W2.4a / `IR-NATIVE-FOUNDATION-1`:**
The [native asynchronous GEMM benchmark](../../../../benchmarks/nvidia/ASYNC_PRODUCER_CONSUMER.md)
is a CUDA-specific matched wait ablation. Physical schedule parity is not
applicable here: this backend does not consume the SM120 `cp.async`/MMA kernel.
No runtime, selector or shared ownership contract changes in this experiment;
existing backend-specific lifetime follow-ups remain open.
