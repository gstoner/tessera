# Public ragged and long scale-VJP regimes, gfx1201

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization key: SCALED-REVERSE-RAGGED-2026-10-07.

The first ragged recorder failed before GPU execution because its test builder
allocated floor(K/scale_k) groups. Ceiling extents and final-group primal
oracle iteration repair this diagnostic defect. The original failure log is
retained. Six subsequent frontend regressions exposed an actual production
admission gap: reverse projection inherited the primal WMMA alignment gate.

The repair separates static FP32 scale-adjoint logical extents from aligned
WMMA primal scheduling. Public reverse maps validate exact ceiling-sized
scale storage and invoke native paired AD, sealed Schedule/Tile reduction,
ROCm Target/LLVM images and checked HIP program ownership. Python performs
no production gradient arithmetic. Missing final groups still reject before
capture. Primal/JVP admission is unchanged; no generic closure is promoted.

73 frontend/contract tests pass, including ragged certificates and missing-tail
counterexamples. 361 native/registry/lifecycle checks pass. Eleven existing
RTX 5070 NVFP4 map cases pass on GPU-cba12639-821a-7a10-4cd3-f918f9c0a545.

RX 9070 XT gfx1201 (GPU-28d9e7efbf2ef716) passes 48 ragged and 48 long public
serial/wave A/B rows. One/two leading maps, three batching policies, KN/NK
and four ordered scale-gradient requests are checked against a float64 oracle
before/after timing and changed-cotangent replay. Warm compiler subprocesses
are forbidden; schedule-bound artifact hashes differ. Bounds remain
rtol=4e-5 / atol=2e-4. Each row records its output shape.

Ragged B0/B1/M/N/K=[2,3,7,19,129]: median paired ratio 4.043;
wave public medians 0.713-1.680 ms, serial 1.020-13.521 ms.
Long [2,3,17,129,1536]: median paired ratio 10.321;
wave public medians 1.044-12.045 ms, serial 2.703-253.801 ms.
These are six alternating windows of three compiler-free public calls,
including frontend, ABI, transfer, native execution and readback. They are
not isolated kernel speedups. Serial remains default.

The compiler SHA256 is
8bbfcddddd613636a3985d7ffc10a766992b4dbce6ddd16a7d05650179344036,
cross-built on Super-Bear and delivered to Tajasarus with its matching layout
library. The HIP owner is the existing gfx1201 runtime; identities are recorded
in the preceding compensated packet. Public source fingerprints bind this
later frontend repair. This is not a Tajasarus compiler rebuild.

Dynamic/nonleading/deeper/mixed/composed/storage AD, sibling reverse execution,
generic batching/transpose closure, fresh full-unit green and PR delivery
remain open. The original five-slice objective is not complete.
