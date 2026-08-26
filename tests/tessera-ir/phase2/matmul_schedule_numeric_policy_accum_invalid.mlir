// RUN: not tessera-opt %s --tessera-graph-to-schedule --split-input-file 2>&1 \
// RUN:   | FileCheck %s

// NUMPOL-CARRIER-1 — the refusals. Accept-set:
// matmul_schedule_numeric_policy_accum.mlir.
//
// `accum` is a semantic key: it decides what the program COMPUTES. Honouring
// the inference over the declaration would report success for a different
// computation than the one written down, which is exactly the silent default
// Decision #21a forbids — and the declaration having no consumer at all was
// Decision #29's case.

// ── a narrower accumulator than the target provides ──
module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @accum_narrower(%a: tensor<17x19xf16>, %b: tensor<19x23xf16>)
      -> tensor<17x23xf32> {
    %0 = tessera.matmul %a, %b
        {numeric_policy = {storage = "fp16", accum = "fp16"}}
        : (tensor<17x19xf16>, tensor<19x23xf16>) -> tensor<17x23xf32>
    return %0 : tensor<17x23xf32>
  }
}
// CHECK: MATMUL_SCHEDULE_ACCUM_UNSUPPORTED
// CHECK-SAME: accum="fp16"
// CHECK-SAME: accumulates in "f32"

// -----

// ── and a WIDER one is refused too, which is the less obvious half ──
// Silently widening looks harmless and is not: the program would be more
// accurate than it asked for, at a cost it did not budget, and a later
// change that stopped widening would read as a regression nobody could
// attribute. Refusing keeps the declared contract the thing that decides.
module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @accum_wider(%a: tensor<17x19xf16>, %b: tensor<19x23xf16>)
      -> tensor<17x23xf32> {
    %0 = tessera.matmul %a, %b
        {numeric_policy = {storage = "fp16", accum = "fp64"}}
        : (tensor<17x19xf16>, tensor<19x23xf16>) -> tensor<17x23xf32>
    return %0 : tensor<17x23xf32>
  }
}
// CHECK: MATMUL_SCHEDULE_ACCUM_UNSUPPORTED
// CHECK-SAME: accum="fp64"
