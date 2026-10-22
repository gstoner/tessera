module attributes {tessera.arch = "sm_120", tessera.frontend.authority = "tracer", tessera.ir.version = "1.0", tessera.native.tensor_program_shape_bounds_request = {K = 64 : i64, M = 65 : i64, N = 47 : i64}, tessera.target = "nvidia_sm120"} {
  func.func @two_matmuls(%arg0: tensor<1x32xbf16>, %arg1: tensor<32x1xbf16>, %arg2: tensor<1x1xbf16>) -> tensor<1x1xf32> attributes {tessera.frontend.authority = "tracer", tessera.structured_cfg.blocks = 2 : i64, tessera.structured_cfg.digest = "5a43c97574525e7ae71f930a6fca284d0dab42f710517c17154581b1597a7eaf", tessera.structured_cfg.schema = "tessera.structured_cfg.v1"} {
    %0 = tessera.matmul %arg0, %arg1 {output_dtype = "fp32", tessera.effect_kind = "pure"} : (tensor<1x32xbf16>, tensor<32x1xbf16>) -> tensor<1x1xf32>
    %1 = tessera.relu %0 {tessera.effect_kind = "pure"} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = tessera.matmul %1, %arg2 {output_dtype = "fp32", tessera.effect_kind = "pure"} : (tensor<1x1xf32>, tensor<1x1xbf16>) -> tensor<1x1xf32>
    return %2 : tensor<1x1xf32>
  }
}

