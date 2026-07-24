// RUN: tessera-opt --tessera-activation-rematerialization --verify-diagnostics %s
//
// Phase F2 (IR form) — a `tessera.recompute`-tagged op that is NOT provably
// side-effect-free must be rejected before it is cloned. Re-executing an
// effectful op (RNG like dropout, a collective, a store/copy) on the backward
// path would change program semantics, not just trade memory for compute
// (Decision #10 — only pure ops qualify). An op that does not model its effects
// is conservatively treated as effectful (we never recompute what we cannot
// prove pure), which is what this unregistered stand-in exercises.

module {
  func.func @remat_bad_effect(%x: tensor<4xf32>) -> tensor<4xf32> {
    // expected-error @+1 {{REMAT_EFFECTFUL:}}
    %v = "test.stateful_dropout"(%x) {tessera.recompute} : (tensor<4xf32>) -> tensor<4xf32>
    %w = "test.use"(%v) : (tensor<4xf32>) -> tensor<4xf32>
    return %w : tensor<4xf32>
  }
}
