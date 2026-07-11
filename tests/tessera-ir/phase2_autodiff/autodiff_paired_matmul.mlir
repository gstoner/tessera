// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// Phase 2 — paired forward/backward autodiff. The in-place pass fuses the
// backward into the forward's return; this pass emits a SEPARATE backward
// function satisfying the residual ABI:
//   @f__bwd(inputs..., out_cotangents...) -> input_cotangents...
// under the recompute-all residual policy.

module {
  // Forward stays primals-only, gains a link to its paired backward.
  // CHECK-LABEL: func.func @loss
  // CHECK-SAME:    tessera.autodiff.paired = @loss__bwd
  // CHECK-SAME:    tessera.autodiff.residual_policy = "recompute_all"
  func.func @loss(%A: tensor<4x8xf32>, %B: tensor<8x16xf32>) -> tensor<4x16xf32>
      attributes {tessera.autodiff = "reverse"} {
    %C = "tessera.matmul"(%A, %B) :
        (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    return %C : tensor<4x16xf32>
  }

  // Backward: (A, B, dC) -> (dA, dB); dA = dC @ B^T, dB = A^T @ dC.
  // CHECK-LABEL: func.func @loss__bwd
  // CHECK-SAME:    (%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<4x16xf32>)
  // CHECK-SAME:    -> (tensor<4x8xf32>, tensor<8x16xf32>)
  // CHECK-SAME:    tessera.autodiff.forward = @loss
  // CHECK-SAME:    tessera.autodiff.role = "backward"
  // CHECK-DAG: tessera.matmul{{.*}}transposeB = true
  // CHECK-DAG: tessera.matmul{{.*}}transposeA = true
  // CHECK: return {{.*}} : tensor<4x8xf32>, tensor<8x16xf32>
}
