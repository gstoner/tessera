module attributes {tessera.arch = "sm_120", tessera.frontend.authority = "tracer", tessera.ir.version = "1.0", tessera.native.tensor_program_shape_bounds_request = {K = 64 : i64, M = 65 : i64, N = 47 : i64}, tessera.target = "nvidia_sm120"} {
  func.func @chain(%arg0: tensor<1x1xf16>, %arg1: tensor<1x1xf16>, %arg2: tensor<1x1xf16>) -> tensor<1x1xf32> attributes {tessera.frontend.authority = "tracer", tessera.structured_cfg.blocks = 2 : i64, tessera.structured_cfg.digest = "308156bd393c72dac3f6eff73d59adaf9e01b10f81bc22c40a95a813c840df0c", tessera.structured_cfg.schema = "tessera.structured_cfg.v1"} {
    %0 = tessera.add %arg0, %arg2 {tessera.effect_kind = "pure"} : (tensor<1x1xf16>, tensor<1x1xf16>) -> tensor<1x1xf16>
    %1 = tessera.relu %0 {tessera.effect_kind = "pure"} : (tensor<1x1xf16>) -> tensor<1x1xf16>
    %2 = tessera.matmul %1, %arg1 {output_dtype = "fp32", tessera.effect_kind = "pure"} : (tensor<1x1xf16>, tensor<1x1xf16>) -> tensor<1x1xf32>
    return %2 : tensor<1x1xf32>
  }
}

