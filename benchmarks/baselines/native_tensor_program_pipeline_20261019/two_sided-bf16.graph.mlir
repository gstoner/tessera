module attributes {tessera.ir.version = "1.0", tessera.frontend.authority = "tracer", tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @two_sided(%a0: tensor<17x32xbf16>, %a1: tensor<32x11xbf16>) -> (tensor<17x11xf32>) attributes {tessera.frontend.authority = "tracer", tessera.structured_cfg.schema = "tessera.structured_cfg.v1", tessera.structured_cfg.digest = "a47321b348d47a3c29b13e5b2e20bb5beceb73825169b4291ff54f51abfa5dd2", tessera.structured_cfg.blocks = 2} {
    %v0 = tessera.relu %a0 {tessera.effect_kind = "pure"} : (tensor<17x32xbf16>) -> tensor<17x32xbf16> loc("/home/angstorms/scratch/w11-program-pipeline-20261019/device-probe.py":20:26)
    %v1 = tessera.softmax %a1 {axis = -1, tessera.effect_kind = "pure"} : (tensor<32x11xbf16>) -> tensor<32x11xbf16> loc("/home/angstorms/scratch/w11-program-pipeline-20261019/device-probe.py":20:41)
    %v2 = tessera.matmul %v0, %v1 {output_dtype = "fp32", tessera.effect_kind = "pure"} : (tensor<17x32xbf16>, tensor<32x11xbf16>) -> tensor<17x11xf32> loc("/home/angstorms/scratch/w11-program-pipeline-20261019/device-probe.py":20:12)
    return %v2 : tensor<17x11xf32>
  }
}