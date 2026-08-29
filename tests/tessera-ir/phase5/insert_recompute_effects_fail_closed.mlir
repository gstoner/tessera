// RUN: tessera-opt --tessera-insert-recompute="memory-budget-mb=4096" --allow-unregistered-dialect %s | FileCheck %s
//
// The negative fixture Decision #10a asks for: an eligibility-marking pass must
// ship a case whose correct output is NO annotation.
//
// `isPureOp` used to describe itself as conservative while doing the opposite —
// an op with no `tessera.effect` attribute was assumed pure unless its NAME
// contained "alloc", "store", or "dealloc". An RNG draw and an opaque call both
// pass those substring tests, so both were tagged recomputable; a backward pass
// that honours the hint re-runs the draw, gets different randomness than the
// forward saw, and produces a wrong gradient with nothing to indicate it.
//
// Purity is now derived: without an explicit attribute the op must be provably
// memory-effect-free, and anything MLIR cannot see through stays unmarked.

module {
  func.func @mixed_effects(%x: tensor<8x8xbf16>, %w: tensor<8x8xbf16>)
      -> tensor<8x8xbf16> {

    // Declared pure — still marked recomputable.
    // CHECK: tessera.matmul
    // CHECK-SAME: tessera_sr.recompute_hint
    %a = "tessera.matmul"(%x, %w) {tessera.effect = "pure"}
        : (tensor<8x8xbf16>, tensor<8x8xbf16>) -> tensor<8x8xbf16>

    // An opaque op carrying no effect attribute is NOT provably pure. Its name
    // contains none of the old substrings, which is exactly how an RNG draw
    // slipped through and got recomputed with fresh randomness.
    // CHECK: test.rng_uniform
    // CHECK-NOT: tessera_sr.recompute_hint
    %b = "test.rng_uniform"(%a) : (tensor<8x8xbf16>) -> tensor<8x8xbf16>

    // Same for an opaque call: its callee's effects are not visible here.
    // CHECK: call @dropout_mask
    // CHECK-NOT: tessera_sr.recompute_hint
    %c = func.call @dropout_mask(%b) : (tensor<8x8xbf16>) -> tensor<8x8xbf16>
    return %c : tensor<8x8xbf16>
  }

  func.func private @dropout_mask(tensor<8x8xbf16>) -> tensor<8x8xbf16>
}
