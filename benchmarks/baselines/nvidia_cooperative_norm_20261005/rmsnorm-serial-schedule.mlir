module attributes {tessera.arch = "sm_120", tessera.ir.version = "1.0", tessera.target = "nvidia_sm120"} {
  func.func @cooperative_norm(%arg0: tensor<128x1024xf16>) -> tensor<128x1024xf16> {
    %0 = tessera.rmsnorm %arg0 {eps = 9.9999997473787516E-6 : f64, schedule.artifact_hash = "c2e66acb3795505b1e56af4acaab18937d1a123b20fa8fa8d6a6be3fb5318ef9", tessera.effect_kind = "pure"} : (tensor<128x1024xf16>) -> tensor<128x1024xf16>
    %1 = schedule.norm %0 {accum = "f32", arch = "sm_120", artifact_hash = "c2e66acb3795505b1e56af4acaab18937d1a123b20fa8fa8d6a6be3fb5318ef9", axis = -1 : i64, epsilon = 9.99999974E-6 : f32, kind = "rmsnorm", storage = "f16", workgroup_size = 128 : i64} : tensor<128x1024xf16> -> tensor<128x1024xf16>
    schedule.artifact {arch = "sm_120", hash = "c2e66acb3795505b1e56af4acaab18937d1a123b20fa8fa8d6a6be3fb5318ef9", numeric_policy = "f16->f32", shape_key = "family=norm;storage=f16;axis=-1", tile = {workgroup_size = 128 : i64}}
    return %1 : tensor<128x1024xf16>
  }
}

