// RUN: not tessera-opt %s --tessera-to-linalg 2>&1 | FileCheck %s

// NUMPOL-CARRIER-1, PR #631 review — the accumulator that is more precise but
// not wider, refused by the LOWERING rather than the contract checker (the
// diagnostic comes from --tessera-to-linalg, which is why it does not live
// beside the schema refusals).

// ── an accumulator that is more precise but not wider ──
// PR #631 review, and this file previously carried a FALSE claim about it: a
// comment in the pass said bf16 storage with an fp16 accumulator "is refused
// upstream ... and never reaches here". It is not. The legality rule compares
// SIGNIFICAND bits — fp16 has 11 against bf16's 8 — so the policy is accepted,
// and the width-based check here then returned null and the chain quietly
// computed in bf16. A declared accumulator accepted upstream and ignored
// downstream is exactly the silent default this slice removes.
//
// Both are 16 bits, so arith.extf cannot express the cast. Computing in fp32
// instead would deliver 24 significand bits where the program asked for 11 —
// the same unrequested substitution refused on the schedule and ROCm paths,
// differing only in being generous. So it fails closed.
module {
  func.func @accum_more_precise_same_width(%x: tensor<8x128xbf16>)
      -> tensor<8x128xbf16> {
    %s = "tessera.softmax"(%x) {
      axis = 1 : i64, numeric_policy = {storage = "bf16", accum = "fp16"}
    } : (tensor<8x128xbf16>) -> tensor<8x128xbf16>
    return %s : tensor<8x128xbf16>
  }
}
// CHECK: NUMERIC_POLICY_ACCUM_UNREALIZABLE
// CHECK-SAME: accum="fp16"
