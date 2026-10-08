module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @mxfp8(%a: tensor<16x128xf8E4M3FN>, %b: tensor<16x128xf8E4M3FN>,
                    %sa: tensor<16x4xi8>, %sb: tensor<4x16xi8>) -> tensor<16x16xf32> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"}, transposeB = true
    } : (tensor<16x128xf8E4M3FN>, tensor<16x128xf8E4M3FN>,
          tensor<16x4xi8>, tensor<4x16xi8>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
}
