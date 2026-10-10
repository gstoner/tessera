module attributes {tessera.arch = "gfx1151", tessera.frontend.authority = "tracer", tessera.ir.version = "1.0", tessera.target = "rocm"} {
  func.func @normalized(%arg0: tensor<1024x8x128xf32>) -> tensor<1024x8x128xf32> attributes {tessera.frontend.authority = "tracer", tessera.structured_cfg.blocks = 2 : i64, tessera.structured_cfg.digest = "4f06b08357b7ef2800d4287f2daacefa4ec19fb2ca2ecf11cbeae13330757ed6", tessera.structured_cfg.schema = "tessera.structured_cfg.v1"} {
    %0 = tessera.softmax %arg0 {axis = -1 : i64, schedule.artifact_hash = "a6eaf1766ae294586eee998f8fc7d74480c07a329168f09beeb0745f3dcf3871", tessera.effect_kind = "pure"} : (tensor<1024x8x128xf32>) -> tensor<1024x8x128xf32>
    %1 = schedule.softmax %0 {accum = "f32", arch = "gfx1151", artifact_hash = "a6eaf1766ae294586eee998f8fc7d74480c07a329168f09beeb0745f3dcf3871", axis = -1 : i64, exp_mode = "accurate", ftz = false, storage = "f32", workgroup_size = 256 : i64} : tensor<1024x8x128xf32> -> tensor<1024x8x128xf32>
    schedule.artifact {arch = "gfx1151", hash = "a6eaf1766ae294586eee998f8fc7d74480c07a329168f09beeb0745f3dcf3871", numeric_policy = "f32->f32", shape_key = "family=softmax;storage=f32;axis=-1", tile = {workgroup_size = 256 : i64}}
    return %1 : tensor<1024x8x128xf32>
  }
}

