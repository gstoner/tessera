// RUN: tessera-opt --tessera-activation-rematerialization %s | FileCheck %s
// RUN: tessera-opt --tessera-activation-rematerialization %s | FileCheck %s --check-prefix=NOMARK
//
// Phase F2 (IR form) — ActivationRematerializationPass sinks a
// `tessera.recompute`-tagged pure activation to its backward consumer. The
// original op (near the forward) is erased and the recomputation is cloned
// immediately before the consumer, shrinking the activation's live range. The
// clone must NOT retain the tessera.recompute marker (it is the materialized
// value, not another checkpoint boundary), and the function records the count.
//
// Uses `arith` ops — they are `[Pure]`, so they clear the side-effect-free gate
// (an unregistered/effect-unmodeled op would be rejected as not-provably-pure).

module {
  // The function records how many activations were rematerialized (1).
  // CHECK: func.func @remat_single_use
  // CHECK-SAME: tessera.rematerialized = 1
  func.func @remat_single_use(%x: tensor<4xf32>) -> tensor<4xf32> {
    %a = arith.mulf %x, %x {tessera.recompute} : tensor<4xf32>
    %far = arith.negf %x : tensor<4xf32>
    // The recomputed mulf is cloned right before its only consumer.
    // CHECK: %[[CLONE:.*]] = arith.mulf %{{.*}}, %{{.*}} : tensor<4xf32>
    // CHECK-NEXT: arith.addf %[[CLONE]],
    %b = arith.addf %a, %far : tensor<4xf32>
    return %b : tensor<4xf32>
  }

  // The recompute marker is fully consumed — no occurrence survives anywhere.
  // NOMARK-NOT: tessera.recompute
}
