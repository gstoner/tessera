---
last_updated: 2026-08-03
audit_role: plan
plan_state: open
owner: x86 backend
target: x86_avx512
scope: x86 AVX-512 implementation/proof and AMX access planning
---

# x86 backend TODO

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
