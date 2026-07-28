---
last_updated: 2026-07-28
audit_role: plan
plan_state: open
owner: x86 backend
target: x86_avx512
scope: x86 AMX/AVX-512 backend implementation and exact-device proof
---

# x86 backend TODO

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

**Status: open. Host: Zen 5 for the measured half.**

x86 reaches hardware two ways, and nothing arbitrates between them:

* **C++ MLIR** — `src/compiler/codegen/tessera_x86_backend/`, AMX BF16 +
  AVX-512 GEMM, works end to end (Decision #1).
* **Python synthesizer** — `emit/x86_llvm.py` + `emit/x86_aocl_dlp.py` behind
  the `KernelEmitter`/`compile_fn`/`KernelRunner` seams.

This is the same two-compiler split documented for Apple in
[`apple/todo.md`](../apple/todo.md); x86 has it too, and the resolution should
be consistent across the fleet rather than decided per backend. Blocked on the
spine decision in
[`../../compiler/COMPILER_THEORY_OF_OPERATION.md`](../../compiler/COMPILER_THEORY_OF_OPERATION.md).

## Cross-backend sync

`TESSERA-OPT-CAPABILITY-SKIP-2026-07-27` — **parity validated**. x86 tests that
drive `tessera-opt` route through the shared capability-aware helper, so a build
without the owning backend skips with the missing pass named. No x86 pass body,
ABI, or numerical contract changed; no exact-device evidence claimed.
