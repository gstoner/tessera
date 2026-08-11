// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// W4.3 canonical bounded while: the executed trip count is the first loop
// result. Reverse mode treats the discrete condition path as fixed, replays
// the pure after-region to each predecessor state, and transposes that step.

module {
  func.func @bounded_while(%x: tensor<4xf32>, %w: tensor<4xf32>)
      -> tensor<4xf32> attributes {tessera.autodiff = "reverse"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %count, %out = scf.while (%i = %c0, %carry = %x)
        : (index, tensor<4xf32>) -> (index, tensor<4xf32>) {
      %continue = arith.cmpi slt, %i, %c3 : index
      scf.condition(%continue) %i, %carry : index, tensor<4xf32>
    } do {
    ^bb0(%i: index, %carry: tensor<4xf32>):
      %next = "tessera.mul"(%carry, %w) :
          (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %next_i = arith.addi %i, %c1 : index
      scf.yield %next_i, %next : index, tensor<4xf32>
    }
    return %out : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @bounded_while__bwd
  // CHECK: scf.while
  // The actual count controls a reverse counted loop.
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: tessera.mul
  // CHECK: arith.addf
  // CHECK: return
}
