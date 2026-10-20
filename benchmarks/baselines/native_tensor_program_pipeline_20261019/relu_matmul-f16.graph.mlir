module attributes {tessera.ir.version = "1.0", tessera.frontend.authority = "tracer", tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @relu_matmul(%a0: tensor<17x32xf16>, %a1: tensor<32x11xf16>) -> (tensor<17x11xf32>) attributes {tessera.frontend.authority = "tracer", tessera.structured_cfg.schema = "tessera.structured_cfg.v1", tessera.structured_cfg.digest = "4e57e8782eae35b87f773dc3e829612ba6b24d538cc5d82be6b72fe3f2368023", tessera.structured_cfg.blocks = 2} {
    %v0 = tessera.relu %a0 {tessera.effect_kind = "pure"} : (tensor<17x32xf16>) -> tensor<17x32xf16> loc("/home/angstorms/scratch/w11-program-pipeline-20261019/device-probe.py":16:26)
    %v1 = tessera.matmul %v0, %a1 {output_dtype = "fp32", tessera.effect_kind = "pure"} : (tensor<17x32xf16>, tensor<32x11xf16>) -> tensor<17x11xf32> loc("/home/angstorms/scratch/w11-program-pipeline-20261019/device-probe.py":16:12)
    return %v1 : tensor<17x11xf32>
  }
}