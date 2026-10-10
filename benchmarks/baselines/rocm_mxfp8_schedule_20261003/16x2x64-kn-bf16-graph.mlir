module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @mxfp8(%a: tensor<16x64xf8E4M3FN>, %b: tensor<64x2xf8E4M3FN>,
                    %sa: tensor<16x2xi8>, %sb: tensor<2x2xi8>) -> tensor<16x2xbf16> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"}
    } : (tensor<16x64xf8E4M3FN>, tensor<64x2xf8E4M3FN>,
          tensor<16x2xi8>, tensor<2x2xi8>) -> tensor<16x2xbf16>
    return %0 : tensor<16x2xbf16>
  }
}
