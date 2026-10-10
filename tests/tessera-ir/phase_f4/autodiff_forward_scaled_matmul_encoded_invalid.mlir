// RUN: not tessera-opt %s --tessera-autodiff-forward 2>&1 | FileCheck %s
// Encoded E8M0 scales are discrete storage, not floating derivative seeds.
module {
  func.func @encoded(%a: tensor<16x32xf8E4M3FN>,
                    %b: tensor<32x16xf8E4M3FN>,
                    %sa: tensor<16x1xi8>, %sb: tensor<1x16xi8>)
      -> tensor<16x16xf32>
      attributes {tessera.autodiff = "forward",
                  tessera.autodiff.wrt_indices = [2]} {
    %y = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"}
    } : (tensor<16x32xf8E4M3FN>, tensor<32x16xf8E4M3FN>,
         tensor<16x1xi8>, tensor<1x16xi8>) -> tensor<16x16xf32>
    return %y : tensor<16x16xf32>
  }
}
// CHECK: error: tessera-autodiff-forward: wrt_indices may select only floating or complex tensor arguments
