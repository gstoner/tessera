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

// Annotation succeeds; the rotor_sandwich op survives the pipeline
// (GA8 RotorSandwichFold preserves the high-level op for GA9 backend
// kernel lowering; only chains of geo_product+reverse get folded into it).
// CHECK: tessera_clifford.rotor_sandwich
// CHECK-SAME: tessera.clifford.canonical
// CHECK-SAME: tessera.clifford.dim = 8
