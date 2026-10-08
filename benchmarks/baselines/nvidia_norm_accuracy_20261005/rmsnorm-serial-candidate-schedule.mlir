module attributes {tessera.arch = "sm_120", tessera.ir.version = "1.0", tessera.target = "nvidia_sm120"} {
  func.func @cooperative_norm(%arg0: tensor<128x4096xbf16>) -> tensor<128x4096xbf16> {
    %0 = tessera.rmsnorm %arg0 {eps = 9.9999997473787516E-6 : f64, schedule = "serial", schedule.artifact_hash = "0124369db05e93c7bb8d304cdfbdae177c94d87c4192a8b5d4aa1edbd3400694", tessera.effect_kind = "pure"} : (tensor<128x4096xbf16>) -> tensor<128x4096xbf16>
    %1 = schedule.norm %0 {accum = "f32", arch = "sm_120", artifact_hash = "0124369db05e93c7bb8d304cdfbdae177c94d87c4192a8b5d4aa1edbd3400694", axis = -1 : i64, epsilon = 9.99999974E-6 : f32, kind = "rmsnorm", storage = "bf16", workgroup_size = 128 : i64} : tensor<128x4096xbf16> -> tensor<128x4096xbf16>
    schedule.artifact {arch = "sm_120", hash = "0124369db05e93c7bb8d304cdfbdae177c94d87c4192a8b5d4aa1edbd3400694", numeric_policy = "bf16->f32", shape_key = "family=norm;storage=bf16;axis=-1", tile = {workgroup_size = 128 : i64}}
    return %1 : tensor<128x4096xbf16>
  }
}

