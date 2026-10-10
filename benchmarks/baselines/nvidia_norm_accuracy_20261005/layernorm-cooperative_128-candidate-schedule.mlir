module attributes {tessera.arch = "sm_120", tessera.ir.version = "1.0", tessera.target = "nvidia_sm120"} {
  func.func @cooperative_norm(%arg0: tensor<128x4096xbf16>) -> tensor<128x4096xbf16> {
    %0 = tessera.layer_norm %arg0 {eps = 9.9999997473787516E-6 : f64, schedule = "cooperative_128", schedule.artifact_hash = "960acb2a23e1329a8ca2b8e985ed95a8402228ac86689f463081d9f5f8a68d74", tessera.effect_kind = "pure"} : (tensor<128x4096xbf16>) -> tensor<128x4096xbf16>
    %1 = schedule.norm %0 {accum = "f32", arch = "sm_120", artifact_hash = "960acb2a23e1329a8ca2b8e985ed95a8402228ac86689f463081d9f5f8a68d74", axis = -1 : i64, epsilon = 9.99999974E-6 : f32, kind = "layernorm", schedule = "cooperative_128", storage = "bf16", workgroup_size = 128 : i64} : tensor<128x4096xbf16> -> tensor<128x4096xbf16>
    schedule.artifact {arch = "sm_120", hash = "960acb2a23e1329a8ca2b8e985ed95a8402228ac86689f463081d9f5f8a68d74", numeric_policy = "bf16->f32", shape_key = "family=norm;storage=bf16;axis=-1", tile = {workgroup_size = 128 : i64}}
    return %1 : tensor<128x4096xbf16>
  }
}

