// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s --check-prefix=REV
// RUN: tessera-opt --tessera-autodiff-forward %s | FileCheck %s --check-prefix=FWD

module {
  func.func @depth(
      %query: tensor<4xf32>, %sources: tensor<3x2x4xf32>)
      -> tensor<2x4xf32> attributes {tessera.autodiff = "reverse"} {
    %result = "tessera.depth_attn"(%query, %sources) {
      eps = 1.000000e-06 : f64,
      numeric_policy = {accum = "fp32", softmax = "fp32", storage = "fp32"}
    } : (tensor<4xf32>, tensor<3x2x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }

  func.func @depth_forward(
      %query: tensor<4xf32>, %sources: tensor<3x2x4xf32>)
      -> tensor<2x4xf32> attributes {tessera.autodiff = "forward"} {
    %result = "tessera.depth_attn"(%query, %sources) {
      eps = 1.000000e-06 : f64,
      numeric_policy = {accum = "fp32", softmax = "fp32", storage = "fp32"}
    } : (tensor<4xf32>, tensor<3x2x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// REV-LABEL: func.func @depth__bwd
// REV-NOT: tessera.custom_adjoint_call
// REV: tessera.depth_attn_vjp
// REV-SAME: eps =
// REV-SAME: numeric_policy = {accum = "fp32", softmax = "fp32", storage = "fp32"}

// FWD-LABEL: func.func private @depth_forward__jvp
// FWD: tessera.depth_attn
// FWD: tessera.depth_attn_jvp
// FWD-SAME: eps =
// FWD-SAME: numeric_policy = {accum = "fp32", softmax = "fp32", storage = "fp32"}
