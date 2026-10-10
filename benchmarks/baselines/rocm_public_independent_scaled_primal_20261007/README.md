# Public independent-prefix primal and scale JVP

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / ROCM-FP8-BLOCKSCALE-1.
Synchronization: INDEPENDENT-SCALE-BATCH-2026-10-07.

Public leading vmap now admits independent mapped/shared matrix and scale operands
for the aligned native gfx1201 FP8/MXFP8 profile. Python records typed Graph
semantics and binds the ABI; native MLIR owns Schedule/Tile addressing, Target
lowering and LLVM images. Independent primal descriptors bind the actual
outlined native member image and geometry, rather than compiling a second image.

Host WSL: 487 focused frontend, certificate, original-Graph immutability and
native image-binding checks pass. Exact RX 9070 XT: 90 public device cases pass:
60 primal (all 15 nonempty axis masks, both RHS orientations, FP32/E8M0 scales)
and 30 FP32 scale JVP. Changed-input warm replay disables compiler subprocesses.
The compiler is LLVM/MLIR 23.1.1; packet hashes match the authoritative source
and compiler snapshot.

All 90 benchmark rows pass independent float64 numerics before and after timing,
changed inputs, compiler-free reuse and stale native generation refusal.
Public wall time includes frontend/binding/upload/execution/readback. Prepared
wall time excludes owner construction. Native HIP launch windows are already
averaged by the runtime over 20 repeats and are recorded without another division.
The three measurement domains are not speedup comparisons.

Maximum absolute error: 2.7016246173516834e-07.

Median ranges in milliseconds:

{
  "primal": {
    "public_warm_median_ms": [
      0.8708529639989138,
      1.1055790353566408
    ],
    "native_launch_window_median_ms": [
      0.00950194988399744,
      0.01988385058939457
    ],
    "prepared_update_invoke_read_median_ms": [
      0.38196337409317493,
      0.5803541745990515
    ]
  },
  "paired_jvp": {
    "public_warm_median_ms": [
      1.4172305818647146,
      1.6069747973233461
    ],
    "native_launch_window_median_ms": [
      0.033733852207660675,
      0.06247770041227341
    ],
    "prepared_update_invoke_read_median_ms": [
      0.5455879960209131,
      0.8154913783073425
    ]
  }
}

This exact-device envelope is static M3 N5 K64, prefix 2x3, aligned K32
groups, nontransposed A, and KN/NK B. Host projection also covers one, two and
three leading map levels. Transposed A, partial scale groups, dynamic/nonleading
maps, arbitrary composition, storage derivatives and sibling execution remain
open. No physical selector or generic coverage state is promoted. Full-suite
batching/transpose closure and publication remain unfinished.
