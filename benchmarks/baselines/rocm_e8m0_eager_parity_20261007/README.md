# Typed E8M0 eager/native parity

Owner: FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Synchronization: E8M0-EAGER-PARITY-2026-10-07.

The typed eager frontend reference now accepts explicit E8M0 block [1,32]
scale bytes. It decodes powers of two in float64 and preserves code 0 as
2**-127 and code 255 as NaN before final float32 output. It rejects conflicting
block layout, signed scale bytes or floating scale storage. Existing fp32
reference behavior is retained. This reference is diagnostic/eager arithmetic;
production lowering and sequencing remain native Graph/Schedule/Tile/Target
and HIP ownership. Encoded scale bytes remain discrete, with no implicit STE.

shared-tests.log records 474 passing capture, independent ml_dtypes decoder,
layout/storage rejection, dtype/operator and diagnostic/pass registry checks.
package-regression.log records 54 passing typed primal packaging, native SSA
projection and transpose-contract checks. The reference tests cover both A/B
transpose orientations and smallest/NaN
codes. Eager transpose support does not imply native transpose-left admission.

device-tests.log records 25 public gfx1201 primal/JVP cases on Tajasaurus
RX 9070 XT. Four MXFP8 KN/NK short/ragged cases compare eager and native JIT
with an independent float64 block oracle. An additional native JIT case proves
code 0 and code 255 against ml_dtypes decoding, preserving NaN locations and
the smallest positive scale contribution. Native execution receipts are
asserted; other cases retain compiler-free changed-input replay and cache
ownership checks. source-tools.json binds source/test/compiler/runtime bytes.

No physical image, schedule, ABI, selector or native arithmetic changes were
made for this reference repair. The prior paired public timing packet remains
the performance evidence for its recorded native implementation; no eager
reference speedup is claimed. No sibling GPU physical proof transfers.

Generic batching/linear transpose, active encoded-scale derivatives,
composed/dynamic AD, broader route/performance programs, a green full suite
and aggregate PR delivery remain open.
