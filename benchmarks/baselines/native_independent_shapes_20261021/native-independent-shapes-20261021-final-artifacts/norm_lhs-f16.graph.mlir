module attributes {tessera.arch = "sm_120", tessera.frontend.authority = "tracer", tessera.ir.version = "1.0", tessera.native.tensor_program_shape_bounds_request = {K = 64 : i64, M = 65 : i64, N = 47 : i64}, tessera.target = "nvidia_sm120"} {
  func.func @norm_lhs(%arg0: tensor<1x1xf16>, %arg1: tensor<1x1xf16>) -> tensor<1x1xf32> attributes {tessera.frontend.authority = "tracer", tessera.structured_cfg.blocks = 2 : i64, tessera.structured_cfg.digest = "95c9e304d0a75dc18cf890b24c90c8a8e9799dd275a50c29e378c8c24c575401", tessera.structured_cfg.schema = "tessera.structured_cfg.v1"} {
    %0 = tessera.rmsnorm %arg0 {tessera.effect_kind = "pure"} : (tensor<1x1xf16>) -> tensor<1x1xf16>
    %1 = tessera.matmul %0, %arg1 {output_dtype = "fp32", tessera.effect_kind = "pure"} : (tensor<1x1xf16>, tensor<1x1xf16>) -> tensor<1x1xf32>
    return %1 : tensor<1x1xf32>
  }
}

