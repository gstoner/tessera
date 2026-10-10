module attributes {tessera.ir.version = "1.0", tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @cooperative_norm(%a0: tensor<128x1024xf16>) -> (tensor<128x1024xf16>) {
    %v0 = tessera.layer_norm %a0 {eps = 9.999999747378752e-06, schedule = "serial", tessera.effect_kind = "pure"} : (tensor<128x1024xf16>) -> tensor<128x1024xf16> loc("tests/device/nvidia/test_lhs_tensor_jit.py":18:26)
    return %v0 : tensor<128x1024xf16>
  }
}