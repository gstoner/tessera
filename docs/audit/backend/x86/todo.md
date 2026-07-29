---
last_updated: 2026-07-29
audit_role: plan
plan_state: open
owner: x86 backend
target: x86_avx512
scope: x86 AVX-512 implementation/proof and AMX access planning
---

# x86 backend TODO

## X86-CALIB-1: split verdict on the hardware-free score calibration

Cross-backend sync `COSTMODEL-CALIB-2026-07-29` — **split: bank-conflict metric
not applicable; locality metric follow-up required.** Owning host Zen 5 (Ryzen AI
Max+ 395 CPU complex, AVX-512, no AMX).

Two static device-free scores are being calibrated against measured latency
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

**Locality histogram — follow-up required.** The step-distance histogram over a
materialized access order is genuinely target-independent: it scores an access
*order*, not a memory technology. This is also the metric with the strongest
prior for CPUs, since blocked-algorithm cache analysis is a CPU literature
(Lam/Rothberg/Wolf 1991, cited in the same assessment). x86 executes natively and
has committed benchmarks (`benchmarks/x86/benchmark_x86_e2e*.py`), so it can supply a
non-GPU architecture to the calibration — valuable precisely because a score that
holds across CPU *and* GPU is far less likely to be fitting an accelerator
artifact.

**Missing exact-device evidence.** Rank correlation between the locality score
and recorded Zen 5 AVX-512 latencies over the e2e benchmark rows. No evidence is
owed for the conflict metric.
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
(Lam/Rothberg/Wolf 1991 on blocked algorithms). Revisit x86 when T1 is built, not
when an emitter consumes `raster_order`. No exact-device evidence is owed here;
nothing in the x86 lane changed and no x86 test was affected.

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
AMX host, target identity, numerical packet, and performance gate are recorded.

## X86-1: the plugin lane cannot report `x86_native` off x86 — 15 red tests

**Status: open. Host: needs a Zen 5 (AVX-512) runner.**

`tests/unit/test_x86_plugin.py` asserts `execution == "x86_native"` in 15
places. On this Apple Silicon host every one returns `"reference"`, and the
cause is not a defect:

* `python/tessera/compiler/emit/x86_llvm.py::_x86_compile_fn` compiles the
  emitted C with `clang -O3 -march=native -fPIC -shared`.
* `platform.machine()` is `arm64`, so `-march=native` targets ARM. The produced
  `.so` is not an x86 kernel, the runner declines, and it reports `reference` —
  which is the honest answer.

These failures were invisible until 2026-07-28 because `clang` was not on
`PATH`; the lane skipped for the wrong reason. Putting LLVM 23 on `PATH`
un-gated them. They fail identically on `main`.

Required work, on an x86 host:

1. Prove the lane end to end: `X86CEmitter` → `_x86_compile_fn` → `X86CRunner`
   returning `x86_native` with numerics matched against the F4 numpy oracle.
2. Decide the `-march` contract. `native` bakes in the build host, which makes
   a cached artifact non-portable across the fleet and interacts badly with the
   content-addressed `kernel_cache` key (the key hashes source + dtype + target,
   *not* the host ISA — two hosts would collide on one entry). An explicit
   `-march=x86-64-v4` or a target-profile-driven flag is the likely answer, and
   it must be reflected in the cache key.
3. Confirm whether `x86_aocl_dlp.py` (AOCL-DLP, Zen-tuned) and `x86_llvm.py`
   should both register for target `"x86"` — today they do, and the second
   `register_compiler("x86", ...)` silently replaces the first.

**Interim (landed 2026-07-28):** the assertions are host-gated so an arm64 host
skips instead of failing. The gate is `platform.machine()`, not a capability
probe — it says "this host cannot prove an x86 kernel", which is exactly the
claim. Removing the gate is not the fix; proving the lane on x86 is.

## X86-2: `_LANG = "c"` — the file name says LLVM, the emitter says C

**Status: open. Host-free.**

`emit/x86_llvm.py` sets `_LANG = "c"` and emits C for `clang`, not LLVM IR.
That is a legitimate design (every backend in `compiler/emit/` emits vendor
source text — CUDA C, HIP C++, MSL, C), but the module name states otherwise and
misleads anyone reasoning about the MLIR/LLVM spine.

Either rename to `x86_c.py`, or make it genuinely emit LLVM IR. The second is
only worth it if the x86 lane is meant to join the C++ MLIR pipeline, which
already reaches AMX/AVX-512 independently — so this is a naming decision first
and an architecture decision second. Record the choice here.

## X86-3: reconcile the two x86 lanes

**Status: open. Hosts: Zen 5 for the AVX-512 measured half; a separately named
AMX-capable host is required for the AMX half.**

x86 reaches hardware two ways, and nothing arbitrates between them:

* **C++ MLIR** — `src/compiler/codegen/tessera_x86_backend/`, AMX BF16 +
  AVX-512 GEMM. Decision #1 records the existing end-to-end architecture;
  this plan may revalidate AVX-512 on Zen 5 but cannot refresh the AMX claim
  without an AMX-capable host.
* **Python synthesizer** — `emit/x86_llvm.py` + `emit/x86_aocl_dlp.py` behind
  the `KernelEmitter`/`compile_fn`/`KernelRunner` seams.

This is the same two-compiler split documented for Apple in
[`apple/todo.md`](../apple/todo.md); x86 has it too, and the resolution should
be consistent across the fleet rather than decided per backend. Blocked on the
spine decision in
[`../../compiler/COMPILER_THEORY_OF_OPERATION.md`](../../compiler/COMPILER_THEORY_OF_OPERATION.md).
Closure requires separate terminal outcomes: an exact-device Zen 5 AVX-512
selection, and either exact-device AMX evidence from a named capable host or an
explicit planned/access-gated AMX state. Neither architecture may promote the
other.

## Cross-backend sync

`TESSERA-OPT-CAPABILITY-SKIP-2026-07-27` — **parity validated**. x86 tests that
drive `tessera-opt` route through the shared capability-aware helper, so a build
without the owning backend skips with the missing pass named. No x86 pass body,
ABI, or numerical contract changed; no exact-device evidence claimed.
