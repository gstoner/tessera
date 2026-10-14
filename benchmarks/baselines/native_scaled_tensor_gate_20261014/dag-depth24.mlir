module attributes {tessera.ir.version = "1.0", tessera.frontend.authority = "tracer", tessera.autodiff = "reverse", tessera.autodiff.wrt = ["gate"], tessera.autodiff.wrt_indices = [4], tessera.metadata_snapshot = {gated = {dim_names = ["[G, C]", 1, "[K, N]", 1, "[M, G]", 1, "[M, K]", 1, "[M, N]", 1]}}, tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @gated(%a0: tensor<2x9xf32>, %a1: tensor<9x5xf32>, %a2: tensor<2x3xf32>, %a3: tensor<3x2xf32>, %a4: tensor<2x5xf32>) -> (tensor<2x5xf32>) attributes {tessera.frontend.authority = "tracer", tessera.structured_cfg.schema = "tessera.structured_cfg.v1", tessera.structured_cfg.digest = "4f5ce3873a5358574aef87968addce0ea01da414521a7b58399091c15eea1dc6", tessera.structured_cfg.blocks = 2, tessera.autodiff = "reverse", tessera.autodiff.wrt = ["gate"], tessera.autodiff.wrt_indices = [4], tessera.lowering.dropped = {dim_names = "not_yet_carried:FRONTEND-IR-MEDIUM-1"}} {
    %v0 = tessera.scaled_matmul %a0, %a1 scales (%a2, %a3) {numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"}, scale_layout = {block = [4, 4], format = "fp32", granularity = "block"}, tessera.effect_kind = "pure"} : (tensor<2x9xf32>, tensor<9x5xf32>, tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x5xf32>
    %v1 = tessera.mul %v0, %v0 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v2 = tessera.mul %v1, %v1 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v3 = tessera.mul %v2, %v2 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v4 = tessera.mul %v3, %v3 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v5 = tessera.mul %v4, %v4 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v6 = tessera.mul %v5, %v5 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v7 = tessera.mul %v6, %v6 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v8 = tessera.mul %v7, %v7 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v9 = tessera.mul %v8, %v8 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v10 = tessera.mul %v9, %v9 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v11 = tessera.mul %v10, %v10 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v12 = tessera.mul %v11, %v11 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v13 = tessera.mul %v12, %v12 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v14 = tessera.mul %v13, %v13 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v15 = tessera.mul %v14, %v14 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v16 = tessera.mul %v15, %v15 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v17 = tessera.mul %v16, %v16 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v18 = tessera.mul %v17, %v17 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v19 = tessera.mul %v18, %v18 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v20 = tessera.mul %v19, %v19 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v21 = tessera.mul %v20, %v20 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v22 = tessera.mul %v21, %v21 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v23 = tessera.mul %v22, %v22 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v24 = tessera.mul %v23, %v23 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    %v25 = tessera.mul %v24, %a4 {tessera.effect_kind = "pure"} : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    return %v25 : tensor<2x5xf32>
  }
}