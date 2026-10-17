module attributes {tessera.target = "rocm_gfx1201", tessera.arch = "gfx1201",
        tessera.native.quantized_product} {
 func.func @block_quant(%x: tensor<256x12288xbf16>, %b: tensor<128x12288xf8E4M3FN>, %sb: tensor<768x128xf32>) -> tensor<256x128xf32> {
  %blocks = tessera.reshape %x {shape = [196608, 16]} : (tensor<256x12288xbf16>) -> tensor<196608x16xbf16>
  %wide = tessera.cast %blocks {dtype = "fp32"} : (tensor<196608x16xbf16>) -> tensor<196608x16xf32>
  %absolute = tessera.absolute %wide : (tensor<196608x16xf32>) -> tensor<196608x16xf32>
  %max = tessera.reduce %absolute {kind = "max", axis = 1 : i64, keepdims = true} : (tensor<196608x16xf32>) -> tensor<196608x1xf32>
  %divisor = arith.constant dense<448.0> : tensor<196608x1xf32>
  %raw = tessera.div %max, %divisor : (tensor<196608x1xf32>, tensor<196608x1xf32>) -> tensor<196608x1xf32>
  %floor = arith.constant dense<1.1754943508222875E-38> : tensor<196608x1xf32>
  %scale = tessera.maximum %raw, %floor : (tensor<196608x1xf32>, tensor<196608x1xf32>) -> tensor<196608x1xf32>
  %expanded = tessera.expand %scale {shape = [196608, 16]} : (tensor<196608x1xf32>) -> tensor<196608x16xf32>
  %normalized = tessera.div %wide, %expanded : (tensor<196608x16xf32>, tensor<196608x16xf32>) -> tensor<196608x16xf32>
  %encoded = tessera.cast %normalized {dtype = "fp8_e4m3"} : (tensor<196608x16xf32>) -> tensor<196608x16xf8E4M3FN>
  %codes = tessera.reshape %encoded {shape = [256, 12288]} : (tensor<196608x16xf8E4M3FN>) -> tensor<256x12288xf8E4M3FN>
  %scales = tessera.reshape %scale {shape = [256, 768]} : (tensor<196608x1xf32>) -> tensor<256x768xf32>
  %result = tessera.scaled_matmul %codes, %b scales(%scales, %sb) {
      transposeB = true,
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 16], format = "fp32"}
    } : (tensor<256x12288xf8E4M3FN>, tensor<128x12288xf8E4M3FN>, tensor<256x768xf32>, tensor<768x128xf32>) -> tensor<256x128xf32>
    return %result : tensor<256x128xf32>
 }
}