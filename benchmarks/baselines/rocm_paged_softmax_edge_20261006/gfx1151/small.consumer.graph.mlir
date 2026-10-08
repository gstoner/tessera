module attributes {tessera.ir.version = "1.0", tessera.frontend.authority = "tracer", tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @normalized(%a0: tensor<5x3x8xf32>) -> (tensor<5x3x8xf32>) attributes {tessera.frontend.authority = "tracer", tessera.structured_cfg.schema = "tessera.structured_cfg.v1", tessera.structured_cfg.digest = "4f06b08357b7ef2800d4287f2daacefa4ec19fb2ca2ecf11cbeae13330757ed6", tessera.structured_cfg.blocks = 2} {
    %v0 = tessera.softmax %a0 {axis = -1, tessera.effect_kind = "pure"} : (tensor<5x3x8xf32>) -> tensor<5x3x8xf32> loc("benchmarks/rocm/benchmark_paged_softmax_edge.py":19:12)
    return %v0 : tensor<5x3x8xf32>
  }
}