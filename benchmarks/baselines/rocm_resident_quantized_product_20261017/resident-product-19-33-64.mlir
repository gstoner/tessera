module attributes {tessera.target = "rocm_gfx1201", tessera.arch = "gfx1201",
        tessera.native.quantized_product} {
 func.func @block_quant(%x: tensor<19x64xbf16>, %b: tensor<33x64xf8E4M3FN>, %sb: tensor<4x33xf32>) -> tensor<19x33xbf16> {
  %blocks = tessera.reshape %x {shape = [76, 16]} : (tensor<19x64xbf16>) -> tensor<76x16xbf16>
  %wide = tessera.cast %blocks {dtype = "fp32"} : (tensor<76x16xbf16>) -> tensor<76x16xf32>
  %absolute = tessera.absolute %wide : (tensor<76x16xf32>) -> tensor<76x16xf32>
  %max = tessera.reduce %absolute {kind = "max", axis = 1 : i64, keepdims = true} : (tensor<76x16xf32>) -> tensor<76x1xf32>
  %divisor = arith.constant dense<448.0> : tensor<76x1xf32>
  %raw = tessera.div %max, %divisor : (tensor<76x1xf32>, tensor<76x1xf32>) -> tensor<76x1xf32>
  %floor = arith.constant dense<1.1754943508222875E-38> : tensor<76x1xf32>
  %scale = tessera.maximum %raw, %floor : (tensor<76x1xf32>, tensor<76x1xf32>) -> tensor<76x1xf32>
  %expanded = tessera.expand %scale {shape = [76, 16]} : (tensor<76x1xf32>) -> tensor<76x16xf32>
  %normalized = tessera.div %wide, %expanded : (tensor<76x16xf32>, tensor<76x16xf32>) -> tensor<76x16xf32>
  %encoded = tessera.cast %normalized {dtype = "fp8_e4m3"} : (tensor<76x16xf32>) -> tensor<76x16xf8E4M3FN>
  %codes = tessera.reshape %encoded {shape = [19, 64]} : (tensor<76x16xf8E4M3FN>) -> tensor<19x64xf8E4M3FN>
  %scales = tessera.reshape %scale {shape = [19, 4]} : (tensor<76x1xf32>) -> tensor<19x4xf32>
  %result = tessera.scaled_matmul %codes, %b scales(%scales, %sb) {
      transposeB = true,
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 16], format = "fp32"}
    } : (tensor<19x64xf8E4M3FN>, tensor<33x64xf8E4M3FN>, tensor<19x4xf32>, tensor<4x33xf32>) -> tensor<19x33xbf16>
    return %result : tensor<19x33xbf16>
 }
}