module attributes {tessera.arch = "sm_120", tessera.ir.version = "1.0", tessera.target = "nvidia_sm120"} {
  func.func @cooperative_norm(%arg0: tensor<128x1024xf16>) -> tensor<128x1024xf16> {
    %0 = tessera.rmsnorm %arg0 {eps = 9.9999997473787516E-6 : f64, schedule.artifact_hash = "142c3b737fcae29e397a699165bc4081bcac328620c674449e10bc4b78cd3430", tessera.effect_kind = "pure"} : (tensor<128x1024xf16>) -> tensor<128x1024xf16>
    %1 = schedule.norm %0 {accum = "f32", arch = "sm_120", artifact_hash = "142c3b737fcae29e397a699165bc4081bcac328620c674449e10bc4b78cd3430", axis = -1 : i64, epsilon = 9.99999974E-6 : f32, kind = "rmsnorm", schedule = "cooperative_128", storage = "f16", workgroup_size = 128 : i64} : tensor<128x1024xf16> -> tensor<128x1024xf16>
    schedule.artifact {arch = "sm_120", hash = "142c3b737fcae29e397a699165bc4081bcac328620c674449e10bc4b78cd3430", numeric_policy = "f16->f32", shape_key = "family=norm;storage=f16;axis=-1", tile = {workgroup_size = 128 : i64}}
    return %1 : tensor<128x1024xf16>
  }
}

