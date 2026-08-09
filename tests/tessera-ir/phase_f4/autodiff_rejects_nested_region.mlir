// RUN: not tessera-opt --tessera-autodiff --allow-unregistered-dialect %s 2>&1 | FileCheck %s
//
// Code-review P2: reverse-mode autodiff must reject an op that carries a nested
// region (e.g. scf.for / scf.if) with a stable diagnostic, rather than
// reverse-walking a flattened op list that interleaves parent/child adjoints
// out of structured order.
//
// CHECK: [AUTODIFF_NESTED_REGION]

module {
  func.func @loop_grad(%A: tensor<4x8xf32>, %B: tensor<8x16xf32>) -> tensor<4x16xf32>
      attributes {tessera.autodiff = "reverse"} {
    %C = "tessera.matmul"(%A, %B) :
        (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    // An op with a nested region on the active return path.
    %D = "test.region_op"(%C) ({
      "test.terminator"() : () -> ()
    }) : (tensor<4x16xf32>) -> tensor<4x16xf32>
    func.return %D : tensor<4x16xf32>
  }
}
