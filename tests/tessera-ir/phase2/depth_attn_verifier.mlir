// RUN: tessera-opt %s | FileCheck %s

module {
  func.func @contracts(
      %query: tensor<4xf32>, %sources: tensor<3x2x4xf32>,
      %oa: tensor<2x4xf32>, %ma: tensor<2xf32>, %la: tensor<2xf32>,
      %ob: tensor<2x4xf32>, %mb: tensor<2xf32>, %lb: tensor<2xf32>)
      -> tensor<2x4xf32> {
    %o, %m, %l = "tessera.attn_with_stats"(%query, %sources) {
      eps = 1.000000e-06 : f64,
      numeric_policy = {accum = "fp32", softmax = "fp32", storage = "fp32"}
    } : (tensor<4xf32>, tensor<3x2x4xf32>)
        -> (tensor<2x4xf32>, tensor<2xf32>, tensor<2xf32>)
    %mo, %mm, %ml = "tessera.softmax_merge"(%oa, %ma, %la, %ob, %mb, %lb) {
      reassociable = true,
      numeric_policy = {accum = "fp32", softmax = "fp32", storage = "fp32"}
    } : (tensor<2x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x4xf32>, tensor<2xf32>, tensor<2xf32>)
        -> (tensor<2x4xf32>, tensor<2xf32>, tensor<2xf32>)
    %result = "tessera.softmax_finalize"(%mo, %ml) {
      numeric_policy = {accum = "fp32", softmax = "fp32", storage = "fp32"}
    } : (tensor<2x4xf32>, tensor<2xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// CHECK: tessera.attn_with_stats
// CHECK: reassociable = true
// CHECK: tessera.softmax_finalize
