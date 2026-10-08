# Native gfx1201 scale-VJP baseline

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization: SCALED-TRANSPOSE-PROGRAM-2026-10-07.

## Route and proof

Typed Graph native paired AD exports actual generated adjoint SSA regions.
Schedule seals the region; Tile lowers its tensor reads to flattened buffers
and portable E4M3 decode; ROCm Target transfers that real GPU body into
ROCDL/LLVM HSACO generation. Symbol-table traits and typed kernel properties
preserve verified serialization. The existing native HIP owner performs
member launches, private allocation, lifetime and generation checks.

24 owning gfx1201 rows pass independent float64 scale-adjoint comparison
before and after timing, changed-cotangent replay and stale-generation refusal.
They cover three static rank-four batching policies, KN/NK RHS storage,
lhs-only, rhs-only, paired and reordered requested gradients. Shape is
[2, 3, 7, 19, 256], with K/N scale blocks of 128.
Compiler subprocesses are forbidden during replay.
Maximum absolute error is 1.107e-5.

15 native ancestry/image/corruption tests, 55 export/program regressions,
296 diagnostic/pass registry tests and 11 audit-document tests pass in host WSL.
Original failed symbol, property and legacy kernel lanes are retained.

## Timing and provenance

Native HIP launch-window event medians range from 0.173 to 9.789 ms.
Prepared host update/invoke/readback medians range from 0.517 to 10.530 ms.
These describe different gradient roles and one/two-member programs; they
are not isolated ISA timings, speedups or a selector-promotion result.

identity.json binds the actual Super-Bear compiler/source/package manifests.
The device receipt and runtime hash bind Tajasaurus execution. Images were
cross-compiled on Super-Bear; the existing native owner ABI required no runtime
change. This does not claim a freshly rebuilt Tajasaurus compiler.

Reproduce: emit packages with benchmark_native_scaled_vjp.py --emit-packages DIR
on the matching compiler host, then use --packages DIR --output FILE on
verified gfx1201 with its checked native movement runtime.

## Open work

Public Python JIT reverse capture remains unconnected. This is an initial
serial-per-scale-element baseline; optimize only after numerical and timing
attribution. Generic dynamic/nonleading/deeper maps, broader storage AD,
sibling physical execution and full-unit closure remain open.
