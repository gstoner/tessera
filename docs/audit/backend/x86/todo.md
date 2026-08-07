---
last_updated: 2026-08-06
audit_role: plan
plan_state: open
owner: x86 backend
target: x86_avx512
scope: x86 AVX-512 implementation/proof and AMX access planning
---

# x86 backend TODO

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

Cross-backend sync `TSOL-ROCM-E2E-1-2026-08-06` — **shared typed carrier and
x86/Zen 5 physical consumer complete.** `tessera.scheduled_spectral.v3`
materializes one verified `schedule.spectral_program` →
`tile.spectral_program_kernel` edge and binds child FFT digests plus the full
compound policy. The native AVX-512 package now owns DCT mirroring,
padding/cropping, framing/windowing, half-spectrum expansion/compaction,
complex multiplication, deterministic overlap-add, and a bounded thread-local
digest-keyed workspace; runtime no longer reconstructs these programs with
host NumPy. Exact Zen 5 aligned, ragged, and prime/Bluestein cases agree with
NumPy. This is architecture-owned evidence and does not inherit gfx1151
performance or scheduling choices.

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
cache and generated permutation retained; mixed-radix SIMD remains open.** The production C2C ABI now keeps a
bounded thread-local immutable plan containing gather offsets and all
per-stage twiddles, removing allocation and transcendental work from warm
execution. Same-host ratios against `scipy.fft(workers=1)` improved from
8.63x/22.75x/17.20x to 3.46x/3.85x/3.10x at N=1024/8192/65536, with the
existing numerical gates passing. The former scalar random-swap bit reversal
is now a cached AVX-512 gather into reusable thread-local workspace; the gap
improved again to 2.92x/1.86x/2.73x with the C++ correctness corpus passing.
This is a retain, not parity; radix-2-only execution and the absence of
mixed-radix SIMD codelets remain the next x86 FFT work.

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
bank-conflict not applicable; T1 cache-model follow-up required.** Owning host
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

**T1 reuse/cache model — follow-up required.** The new shared model is
structurally applicable to blocked AVX-512/AMX GEMM, but the current target
profile does not yet provide trustworthy Zen 5 compute peaks and the generic
single-cache abstraction must be mapped to the L1/L2/L3 hierarchy. x86 executes
natively and has committed benchmarks
(`benchmarks/x86/benchmark_x86_e2e*.py`), so it can supply the non-GPU
retain/reject check once those inputs are evidence-backed.

**Missing exact-device evidence.** Evidence-backed Zen 5 compute/bandwidth/cache
inputs and rank correlation between T1 and recorded AVX-512 latencies over the
e2e benchmark rows. No evidence is owed for the bank-conflict metric.

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
