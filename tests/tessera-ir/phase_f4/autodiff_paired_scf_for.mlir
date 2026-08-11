// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// W4.3 recompute-all baseline for a counted region. The reverse loop walks
// iterations in reverse logical order, replays the primal prefix to recover
// the carried value, and accumulates the implicit %w capture cotangent.

module {
  func.func @loop(%x: tensor<4xf32>, %w: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %out = scf.for %i = %c0 to %c3 step %c1
        iter_args(%carry = %x) -> tensor<4xf32> {
      %next = "tessera.mul"(%carry, %w) :
          (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      scf.yield %next : tensor<4xf32>
    }
    return %out : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @loop__bwd
  // CHECK: scf.for
  // Reverse-order index construction is explicit and independent of a static
  // trip count, so dynamic affine bounds use the same path.
  // CHECK: arith.ceildivsi
  // CHECK: %[[REVERSE:.+]] = scf.for
  // Prefix replay recovers the exact primal state for the selected iteration.
  // CHECK: scf.for
  // CHECK: tessera.mul
  // CHECK: arith.addf
  // CHECK: return
}
