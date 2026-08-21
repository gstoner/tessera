// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// Regression (PR #601 review): an OFF-gradient input that is a dynamically
// shaped *encoded* ranked tensor. Its zero cotangent is synthesized through
// the tensor.empty + linalg.fill path (DenseElementsAttr cannot splat a
// dynamic extent). That path must carry the input's encoding onto the empty
// tensor — otherwise the fill result is unencoded and func.return fails to
// verify against the backward function's declared (encoded) result type.
//
// `%unused` is off the gradient path, so its cotangent is a synthesized zero;
// `%logits` carries the real gradient.

module {
  func.func @encoded_dyn(%logits: tensor<?x4x?xf32>, %target: tensor<?x?xi64>,
                         %unused: tensor<?xf32, "sparse">) -> tensor<f32>
      attributes {tessera.autodiff = "reverse"} {
    %loss = "tessera.loss.cross_entropy"(%logits, %target)
        {axis = 1 : i64, reduction = "mean"} :
        (tensor<?x4x?xf32>, tensor<?x?xi64>) -> tensor<f32>
    return %loss : tensor<f32>
  }

  // The synthesized zero for the encoded input keeps its encoding end to end,
  // so the encoded result type is legal.
  // CHECK-LABEL: func.func @encoded_dyn__bwd
  // CHECK-SAME: -> (tensor<?x4x?xf32>, tensor<?x?xi64>, tensor<?xf32, "sparse">)
  // CHECK: %[[E:.+]] = tensor.empty(%{{.*}}) : tensor<?xf32, "sparse">
  // CHECK: %[[F:.+]] = linalg.fill ins(%{{.*}} : f32) outs(%[[E]] : tensor<?xf32, "sparse">) -> tensor<?xf32, "sparse">
  // CHECK: return %{{.*}}, %{{.*}}, %[[F]] : tensor<?x4x?xf32>, tensor<?x?xi64>, tensor<?xf32, "sparse">
}
