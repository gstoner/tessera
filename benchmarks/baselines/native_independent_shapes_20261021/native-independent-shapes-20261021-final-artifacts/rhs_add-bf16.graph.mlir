module attributes {tessera.arch = "sm_120", tessera.frontend.authority = "tracer", tessera.ir.version = "1.0", tessera.native.tensor_program_shape_bounds_request = {K = 64 : i64, M = 65 : i64, N = 47 : i64}, tessera.target = "nvidia_sm120"} {
  func.func @rhs_add(%arg0: tensor<1x1xbf16>, %arg1: tensor<1x1xbf16>, %arg2: tensor<1x1xbf16>) -> tensor<1x1xf32> attributes {tessera.frontend.authority = "tracer", tessera.structured_cfg.blocks = 2 : i64, tessera.structured_cfg.digest = "4d5bdb2da288e2ca2bb616997d480ee8c6fc320fdc60ac5f8e1af46d758ab482", tessera.structured_cfg.schema = "tessera.structured_cfg.v1"} {
    %0 = tessera.relu %arg0 {tessera.effect_kind = "pure"} : (tensor<1x1xbf16>) -> tensor<1x1xbf16>
    %1 = tessera.add %arg1, %arg2 {tessera.effect_kind = "pure"} : (tensor<1x1xbf16>, tensor<1x1xbf16>) -> tensor<1x1xbf16>
    %2 = tessera.matmul %0, %1 {output_dtype = "fp32", tessera.effect_kind = "pure"} : (tensor<1x1xbf16>, tensor<1x1xbf16>) -> tensor<1x1xf32>
    return %2 : tensor<1x1xf32>
  }
}

