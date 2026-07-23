// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s

module {
  func.func @bce(%logits: tensor<?x7xf32>,
                 %target: tensor<?x7xf32>) -> tensor<f32>
      attributes {tessera.autodiff = "reverse"} {
    %loss = "tessera.loss.binary_cross_entropy"(%logits, %target)
        {reduction = "mean"} :
        (tensor<?x7xf32>, tensor<?x7xf32>) -> tensor<f32>
    return %loss : tensor<f32>
  }

  // CHECK-LABEL: func.func @bce__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: %[[DZ:.+]], %[[DT:.+]] = tessera.loss.binary_cross_entropy_backward
  // CHECK: return %[[DZ]], %[[DT]]
}
