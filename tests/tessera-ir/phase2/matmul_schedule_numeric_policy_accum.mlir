// RUN: tessera-opt %s --tessera-graph-to-schedule --split-input-file | FileCheck %s
//
// The accept-set. Refusals are in
// matmul_schedule_numeric_policy_accum_invalid.mlir, so neither file can pass
// by accident.

// NUMPOL-CARRIER-1 (queue row 3b) — the declared accumulator gets a consumer
// on the schedule path.
//
// `getMatmulSchedule` infers `accum` from operand/result element types per
// target. That inference is coherent and it is not the declared contract:
// `numeric_policy.accum` reached the op, was never read, and the inference
// won — the same shape as the ROCm WMMA generator, one level up. The typed
// sm_120 route then built `!tile.fragment` with the literal "f32" written
// three times, two lines from an unused `selected->accum`: one fact, four
// copies, and only the copies were reachable from the policy.
//
// The check wraps EVERY exit of the inference rather than being pasted at
// each `return schedule`, so no target path can be added that bypasses it.

// ── declared accumulator matches the inference ──
// Note the two spellings: the policy uses Decision #15a's vocabulary ("fp32")
// and the schedule uses MLIR's ("f32"). Same fact, and #32's point that a
// level may rename an attribute without losing it — so they are normalized
// rather than compared as strings.
module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @accum_matches(%a: tensor<17x19xf16>, %b: tensor<19x23xf16>)
      -> tensor<17x23xf32> {
    %0 = tessera.matmul %a, %b
        {numeric_policy = {storage = "fp16", accum = "fp32"}}
        : (tensor<17x19xf16>, tensor<19x23xf16>) -> tensor<17x23xf32>
    return %0 : tensor<17x23xf32>
  }
}
// CHECK-LABEL: @accum_matches
// CHECK: schedule.

// -----

// ── no policy: the inference stands, unchanged ──
module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @no_policy(%a: tensor<17x19xf16>, %b: tensor<19x23xf16>)
      -> tensor<17x23xf32> {
    %0 = tessera.matmul %a, %b
        : (tensor<17x19xf16>, tensor<19x23xf16>) -> tensor<17x23xf32>
    return %0 : tensor<17x23xf32>
  }
}
// CHECK-LABEL: @no_policy
// CHECK: schedule.
