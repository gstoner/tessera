// RUN: tessera-opt --tessera-activation-rematerialization %s | FileCheck %s
// RUN: tessera-opt --tessera-activation-rematerialization %s | FileCheck %s --check-prefix=NOMARK
//
// Phase F2 (IR form) — a tagged PRODUCER CHAIN rematerializes together at the
// final consumer. %a feeds %b (both tagged) which feeds a later backward user.
// The pass walks recompute ops in reverse program order, so %b sinks to the
// user first and %a then lands next to %b's clone — the whole chain recomputes
// at the consumer, and nothing from the chain stays live across from the
// forward region. (Forward order would strand %a's clone next to %b in the
// forward block — the bug this test pins.)

module {
  // CHECK: func.func @remat_chain
  // CHECK-SAME: tessera.rematerialized = 2
  func.func @remat_chain(%x: tensor<4xf32>) -> tensor<4xf32> {
    %a = arith.mulf %x, %x {tessera.recompute} : tensor<4xf32>
    %b = arith.addf %a, %x {tessera.recompute} : tensor<4xf32>
    %far = arith.negf %x : tensor<4xf32>
    // The forward filler stays put; the rematerialized mulf→addf chain lands
    // immediately before its consumer (proving mulf did NOT strand in front of
    // the negf, which is what forward-order processing would produce).
    // CHECK: arith.negf
    // CHECK: %[[A:.*]] = arith.mulf %{{.*}}, %{{.*}} : tensor<4xf32>
    // CHECK-NEXT: %[[B:.*]] = arith.addf %[[A]], %{{.*}} : tensor<4xf32>
    // CHECK-NEXT: arith.subf %[[B]],
    %user = arith.subf %b, %far : tensor<4xf32>
    return %user : tensor<4xf32>
  }
  // NOMARK-NOT: tessera.recompute
}
