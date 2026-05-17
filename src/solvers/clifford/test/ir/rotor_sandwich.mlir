// RUN: ts-clifford-opt --tessera-clifford-pipeline %s 2>&1 | FileCheck %s
//
// End-to-end pipeline alias: annotate → expand-product-table → grade-fusion
// → rotor-sandwich-fold.  In GA7 the lowering passes are stubs that emit
// per-op remarks; the annotation pass is fully implemented.  This fixture
// verifies the pipeline dispatches all four passes on a rotor-sandwich
// chain that GA8 will eventually fold into a single fused kernel.

module {
  func.func @rotor_sandwich_chain(
      %R : tensor<8xf32>, %v : tensor<8xf32>) -> tensor<8xf32> {
    %y = "tessera_clifford.rotor_sandwich"(%R, %v)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %y : tensor<8xf32>
  }
}

// Annotation succeeds.
// CHECK: tessera.clifford.canonical
// CHECK: tessera.clifford.dim = 8

// GA8 stub remarks fire (lowering bodies not implemented yet).
// CHECK: expand-product-table stub
// CHECK: grade-fusion stub
// CHECK: rotor-sandwich-fold stub
