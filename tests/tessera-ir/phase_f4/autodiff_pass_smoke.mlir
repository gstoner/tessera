// RUN: tessera-opt --tessera-autodiff %s | FileCheck %s
//
// Phase F4 — `AutodiffPass` (reverse-mode via `Tessera_AdjointInterface`).
// Verifies the pass walks the IR in reverse, dispatches `buildAdjoint` on
// the matmul op, and rewrites the function to expose argument cotangents
// as additional outputs.

module {
  // CHECK-LABEL: func.func @train_step
  // CHECK-SAME:    -> (tensor<4x16xf32>, tensor<4x8xf32>, tensor<8x16xf32>)
  // CHECK-SAME:    tessera.autodiff.arg_cotangents
  func.func @train_step(%A: tensor<4x8xf32>, %B: tensor<8x16xf32>) -> tensor<4x16xf32>
      attributes {tessera.autodiff = "reverse"} {
    %C = "tessera.matmul"(%A, %B) :
        (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>

    // Cotangent seed (constant tensor of 1.0 matching the output shape),
    // followed by two transposed matmuls — dA = seed @ B^T and dB = A^T @ seed.
    //
    // CHECK: arith.constant
    // CHECK-SAME: dense<1.000000e+00>
    // CHECK: tessera.matmul {{.*}}transposeB = true
    // CHECK: tessera.matmul {{.*}}transposeA = true
    func.return %C : tensor<4x16xf32>
  }
}
