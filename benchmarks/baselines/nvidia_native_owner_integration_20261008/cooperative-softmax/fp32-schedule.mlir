module attributes {tessera.arch = "sm_120", tessera.ir.version = "1.0", tessera.target = "nvidia_sm120"} {
  func.func @nvidia_sm120_softmax(%arg0: tensor<2x3x4096xf32>) -> tensor<2x3x4096xf32> {
    %0 = tessera.softmax %arg0 {axis = -1 : i64, schedule = "cooperative_128", schedule.artifact_hash = "9d3b4bf250bf3f9199314ab3950b38ef15c37cb05f576417bdcf98b0f374c29d", tessera.effect_kind = "pure"} : (tensor<2x3x4096xf32>) -> tensor<2x3x4096xf32>
    %1 = schedule.softmax %0 {accum = "f32", arch = "sm_120", artifact_hash = "9d3b4bf250bf3f9199314ab3950b38ef15c37cb05f576417bdcf98b0f374c29d", axis = -1 : i64, exp_mode = "approx_exp2", ftz = false, schedule = "cooperative_128", storage = "f32", workgroup_size = 128 : i64} : tensor<2x3x4096xf32> -> tensor<2x3x4096xf32>
    schedule.artifact {arch = "sm_120", hash = "9d3b4bf250bf3f9199314ab3950b38ef15c37cb05f576417bdcf98b0f374c29d", numeric_policy = "f32->f32", shape_key = "family=softmax;storage=f32;axis=-1", tile = {workgroup_size = 128 : i64}}
    return %1 : tensor<2x3x4096xf32>
  }
}

