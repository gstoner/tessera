module attributes {tessera.ir.version = "1.0", tessera.frontend.authority = "tracer", tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @norm_matmul(%a0: tensor<17x32xf16>, %a1: tensor<32x11xf16>) -> (tensor<17x11xf32>) attributes {tessera.frontend.authority = "tracer", tessera.structured_cfg.schema = "tessera.structured_cfg.v1", tessera.structured_cfg.digest = "95c9e304d0a75dc18cf890b24c90c8a8e9799dd275a50c29e378c8c24c575401", tessera.structured_cfg.blocks = 2} {
    %v0 = tessera.rmsnorm %a0 {eps = 1.0e-05, numeric_policy = {accum = "fp32"}, tessera.effect_kind = "pure"} : (tensor<17x32xf16>) -> tensor<17x32xf16> loc("/home/angstorms/scratch/w11-program-pipeline-20261019/device-probe.py":18:26)
    %v1 = tessera.matmul %v0, %a1 {output_dtype = "fp32", tessera.effect_kind = "pure"} : (tensor<17x32xf16>, tensor<32x11xf16>) -> tensor<17x11xf32> loc("/home/angstorms/scratch/w11-program-pipeline-20261019/device-probe.py":18:12)
    return %v1 : tensor<17x11xf32>
  }
}