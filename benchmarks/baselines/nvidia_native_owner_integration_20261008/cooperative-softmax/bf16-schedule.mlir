module attributes {tessera.arch = "sm_120", tessera.ir.version = "1.0", tessera.target = "nvidia_sm120"} {
  func.func @nvidia_sm120_softmax(%arg0: tensor<2x3x4096xbf16>) -> tensor<2x3x4096xbf16> {
    %0 = tessera.softmax %arg0 {axis = -1 : i64, schedule = "cooperative_128", schedule.artifact_hash = "c49459512612a437ab62f0cc312e82ee1a8b661a939895610aa9e2a6e5b2d0ea", tessera.effect_kind = "pure"} : (tensor<2x3x4096xbf16>) -> tensor<2x3x4096xbf16>
    %1 = schedule.softmax %0 {accum = "f32", arch = "sm_120", artifact_hash = "c49459512612a437ab62f0cc312e82ee1a8b661a939895610aa9e2a6e5b2d0ea", axis = -1 : i64, exp_mode = "approx_exp2", ftz = false, schedule = "cooperative_128", storage = "bf16", workgroup_size = 128 : i64} : tensor<2x3x4096xbf16> -> tensor<2x3x4096xbf16>
    schedule.artifact {arch = "sm_120", hash = "c49459512612a437ab62f0cc312e82ee1a8b661a939895610aa9e2a6e5b2d0ea", numeric_policy = "bf16->f32", shape_key = "family=softmax;storage=bf16;axis=-1", tile = {workgroup_size = 128 : i64}}
    return %1 : tensor<2x3x4096xbf16>
  }
}

