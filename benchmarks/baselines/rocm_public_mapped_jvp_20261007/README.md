# Public mapped native FP32 scale-JVP integration

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Cross-backend synchronization key: ROCM-PUBLIC-MAPPED-JVP-2026-10-07.

Public vmap preserves forward differentiation intent for direct typed E4M3 scaled products on exact rocm_gfx1201. FP32 scale roles sa, sb or both are admitted under static leading shared-RHS, independent-RHS and shared-LHS policies with KN/NK RHS. The mapped owner remains independent of its scalar owner. Native AD, Schedule/Tile/Target and checked multi-step package execution own primal/tangent arithmetic.

Before native execution, a signature-cached frontend differential certificate compares the projected Graph and vectorized typed reference against an eager scalar-map oracle. Its Python iteration is diagnostic-only: it launches no backend and is never a production batch execution route. Warm certificates/packages avoid oracle replay and compiler subprocesses.

Validation: 72 owning gfx1201 numerical/warm-call tests pass (36 primal, 36 composed scale-JVP), including independent float64 analytic derivatives and central finite differences for sa, sb and both. Small and ragged M200 cases are covered. 503 shared dtype/frontend/op/diagnostic/pass gates pass. Encoded E8M0 scale derivatives, operand-storage derivatives and reverse requests remain rejected; no generic AD closure is claimed.

Twelve benchmark rows use the actual public cached native program, record its images and geometry, and separate public-call wall, prepared update/invoke/read wall, and native HIP sequence events. Native event sequences include enqueue gaps and are not isolated kernel cost. Small public medians range 1.41-1.61 ms; ragged M200 medians range 1.68-3.54 ms across policies/orientations. No comparative speedup is claimed.

The device/benchmark source-tools receipt predates the separate explicit gated-byte legacy-specialization repair, which affects MXFP8 frontend certificates and is covered by the shared tests. It changes no FP32 native arithmetic, image or ABI.

Remaining: dynamic/nested/nonleading maps, general AD/linear transpose, sibling physical consumers, full-suite validation and PR delivery.

## Public constraint gate

native_jvp now enforces the same source call-time shape constraints as ordinary
JIT execution before tracing or compiling. Three mapped policy counterexamples
with Range(M,1,6) and M=7 are rejected before frontend capture. 44 focused
frontend tests and 72 shared native-JVP/attention regression tests pass;
16 gated cases skip. The owning gfx1201 scalar/mapped JVP run passes 59 tests.
Twelve timings-constraints.json rows retain identical native images to the
original packet; source-tools-constraints.json binds the current owning code.
Original timings and source receipts remain intact.
