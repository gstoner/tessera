module attributes {tessera.arch = "sm_120", tessera.ir.version = "1.0", tessera.target = "nvidia_sm120"} {
  func.func @nvidia_sm120_softmax(%arg0: tensor<2x3x4096xf16>) -> tensor<2x3x4096xf16> {
    %0 = tessera.softmax %arg0 {axis = -1 : i64, schedule = "cooperative_128", schedule.artifact_hash = "d1b6b72f8e4afd0047197bb9d7adabf028c281afac00aef1f8b162dbaa2cc5ca", tessera.effect_kind = "pure"} : (tensor<2x3x4096xf16>) -> tensor<2x3x4096xf16>
    %1 = schedule.softmax %0 {accum = "f32", arch = "sm_120", artifact_hash = "d1b6b72f8e4afd0047197bb9d7adabf028c281afac00aef1f8b162dbaa2cc5ca", axis = -1 : i64, exp_mode = "approx_exp2", ftz = false, schedule = "cooperative_128", storage = "f16", workgroup_size = 128 : i64} : tensor<2x3x4096xf16> -> tensor<2x3x4096xf16>
    schedule.artifact {arch = "sm_120", hash = "d1b6b72f8e4afd0047197bb9d7adabf028c281afac00aef1f8b162dbaa2cc5ca", numeric_policy = "f16->f32", shape_key = "family=softmax;storage=f16;axis=-1", tile = {workgroup_size = 128 : i64}}
    return %1 : tensor<2x3x4096xf16>
  }
}

