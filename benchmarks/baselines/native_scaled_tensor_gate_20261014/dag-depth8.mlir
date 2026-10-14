module attributes {tessera.ir.version = "1.0", tessera.frontend.authority = "tracer", tessera.autodiff = "reverse", tessera.autodiff.wrt = ["gate"], tessera.autodiff.wrt_indices = [4], tessera.metadata_snapshot = {gated = {dim_names = ["[G, C]", 1, "[K, N]", 1, "[M, G]", 1, "[M, K]", 1, "[M, N]", 1]}}, tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @gated(%a0: tensor<2x9xf32>, %a1: tensor<9x5xf32>, %a2: tensor<2x3xf32>, %a3: tensor<3x2xf32>, %a4: tensor<2x5xf32>) -> (tensor<2x5xf32>) attributes {tessera.frontend.authority = "tracer", tessera.structured_cfg.schema = "tessera.structured_cfg.v1", tessera.structured_cfg.digest = "202d41b8a63b6785826be621457c4900489d2b65f034f72e8eaa0c820220f6ac", tessera.structured_cfg.blocks = 2, tessera.autodiff = "reverse", tessera.autodiff.wrt = ["gate"], tessera.autodiff.wrt_indices = [4], tessera.lowering.dropped = {dim_names = "not_yet_carried:FRONTEND-IR-MEDIUM-1"}} {
    %v0 = tessera.scaled_matmul %a0, %a1 scales (%a2, %a3) {numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"}, scale_layout = {block = [4, 4], format = "fp32", granularity = "block"}, tessera.effect_kind = "pure"} : (tensor<2x9xf32>, tensor<9x5xf32>, tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x5xf32>
    %v1 = tessera.mul %v0, %v0 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v2 = tessera.mul %v1, %v1 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v3 = tessera.mul %v2, %v2 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v4 = tessera.mul %v3, %v3 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v5 = tessera.mul %v4, %v4 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v6 = tessera.mul %v5, %v5 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v7 = tessera.mul %v6, %v6 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v8 = tessera.mul %v7, %v7 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v9 = tessera.mul %v8, %a4 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    return %v9 : tensor<2x5xf32>
  }
}