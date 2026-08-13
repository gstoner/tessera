// RUN: tessera-opt %s -split-input-file -verify-diagnostics

func.func @bad_width(%query: tensor<5xf32>, %sources: tensor<3x2x4xf32>)
    -> tensor<2x4xf32> {
  // expected-error @+1 {{query width must match the final source dimension}}
  %0 = "tessera.depth_attn"(%query, %sources) {
    eps = 1.0e-6 : f64,
    numeric_policy = {accum = "fp32", softmax = "fp32", storage = "fp32"}
  } : (tensor<5xf32>, tensor<3x2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @bad_policy(%query: tensor<4xf32>, %sources: tensor<3x2x4xf32>)
    -> tensor<2x4xf32> {
  // expected-error @+1 {{numeric_policy requires softmax=fp32 and accum=fp32}}
  %0 = "tessera.depth_attn"(%query, %sources) {
    eps = 1.0e-6 : f64,
    numeric_policy = {accum = "fp16", softmax = "fp32", storage = "fp32"}
  } : (tensor<4xf32>, tensor<3x2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @bad_merge_trait(
    %o: tensor<2x4xf32>, %m: tensor<2xf32>, %l: tensor<2xf32>)
    -> tensor<2x4xf32> {
  // expected-error @+1 {{reassociable must be true for the canonical merge}}
  %mo, %mm, %ml = "tessera.softmax_merge"(%o, %m, %l, %o, %m, %l) {
    reassociable = false,
    numeric_policy = {accum = "fp32", softmax = "fp32", storage = "fp32"}
  } : (tensor<2x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x4xf32>, tensor<2xf32>, tensor<2xf32>)
      -> (tensor<2x4xf32>, tensor<2xf32>, tensor<2xf32>)
  return %mo : tensor<2x4xf32>
}
