module attributes {tessera.ir.version = "1.0", tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @nvidia_sm120_softmax(%x: tensor<2x3x4096xf32>) -> (tensor<2x3x4096xf32>) {
    %o = tessera.softmax %x {axis = -1, schedule = "cooperative_128", tessera.effect_kind = "pure"} : (tensor<2x3x4096xf32>) -> tensor<2x3x4096xf32>
    return %o : tensor<2x3x4096xf32>
  }
}