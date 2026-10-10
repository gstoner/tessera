# Native compensated scale-VJP regimes, gfx1201

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization key: SCALED-TRANSPOSE-COMPENSATED-2026-10-07.

## Correctness and repair

The original serial and wave schedules each failed six long shared-RHS
SB-gradient cases. prior-long-failures.json preserves those failures.
The native MLIR repair carries FP32 sum and rounding residual across the
proved additive chain, while preserving each K-group dot product. The wave
candidate partitions the innermost additive contribution loop, rather than
the small outer batch axis. Schedule identity binds compensated_fp32;
Tile/Target body seals bind actual arithmetic. Python performs no production
gradient arithmetic.

All 72 repaired cases pass the independent float64 oracle before timing and
after changed-cotangent replay, at unchanged rtol=4e-5 / atol=2e-4.
Three batching policies, KN/NK storage and four ordered gradient requests
cover tiny [1,1,1,1,128], ragged [2,3,7,19,129] and long [2,3,17,129,1536].
Maximum absolute error is 3.3721e-5.

## Device and measurement boundary

Tajasarus reports AMD Radeon RX 9070 XT, gfx1201,
GPU-28d9e7efbf2ef716. Images were compiled by the matching Super-Bear
LLVM/MLIR compiler and replayed by Tajasarus's existing native HIP owner.
This is cross-compiled exact-device replay, not a new Tajasarus compiler build.

Six alternating windows per arm, ten native invocations per window measure
one/two-member HIP event launch windows, excluding Python enqueue loops and
host transfer/readback. Separate allocations are used per arm.
Median serial/wave ratios are tiny 1.140, ragged 19.349, long 14.669.
Tiny minimum is 0.974: no universal wave promotion is supported.
These ratios compare repaired serial against repaired wave; they do not
establish public-call or isolated ISA speedups. Serial remains default.

## Gates and remaining work

52 native export/lowering tests passed after rebuild; the additional
compensation/seal regression lane passes 23 lowering tests. Shared pass
metadata, diagnostics and audit lifecycle gates pass 307 tests.
Public candidate timing, broader/dynamic/composed/storage AD, sibling physical
reverse proof, fresh full-unit green and aggregate publication remain open.

## Public JIT candidate integration

TESSERA_ROCM_SCALE_VJP_SCHEDULE explicitly selects serial_per_scale_element
(default) or wave_per_scale_element. Invalid values raise before device/package
work. Immutable cache identity includes the schedule; receipts record it.
No Python numerical lowering or automatic selector promotion is introduced.

The delivered compiler matches the image-emission SHA256
8bbfcddddd613636a3985d7ffc10a766992b4dbce6ddd16a7d05650179344036.
Its layout library SHA256 is
4523ff130f1ac60f7c55b9d8eb3c997f1eb541ba9785f6d0a6096bc9c57008c8.
The gfx1201 native runtime SHA256 is
203a6528e397605739621ac469ffbe1cb63c7151139459a9fd07425a0ac05a8c.

Both schedules pass 72 public scalar/one-map/two-map device tests (144 total).
Each run retains the host's pre-existing unknown pytest timeout-option warning.
48 alternating public A/B rows pass independent numerics, changed-cotangent
replay, exact-device receipts, different schedule artifact hashes and compiler
subprocess refusal during warm windows. Six windows and three calls per
window include frontend, ABI, upload, native execution and readback.
Serial/wave ratio min/median/max: 1.148 / 3.934 / 9.572.
Wave wall-time medians: 0.740-1.575 ms; serial: 1.105-11.170 ms.
These small static mapped shapes do not prove public long-shape performance
or general AD closure. Native tiny/ragged/long proof above remains distinct.

362 focused native/package/metadata/diagnostic/audit checks pass after public
integration. Ruff passes for the plugin and recorder.
