module attributes {tessera.ir.version = "1.0", tessera.frontend.authority = "tracer", tessera.target = "rocm_gfx1201", tessera.arch = "gfx1201"} {
  func.func @paged_full(%a0: tensor<32x16x8x128xf32>, %a1: tensor<64xi32>) -> (tensor<1024x8x128xf32>) attributes {tessera.frontend.authority = "tracer", tessera.structured_cfg.schema = "tessera.structured_cfg.v1", tessera.structured_cfg.digest = "c754085d55b2dfdc01758b11f1761205a49ed509abeaa8d28a787fb162a90553", tessera.structured_cfg.blocks = 2, tessera.bindings = ["a0", "a1", "v0"]} {
    %v0 = tessera.paged_kv_read %a0, %a1 {start = 0, end = 1024} : (tensor<32x16x8x128xf32>, tensor<64xi32>) -> tensor<1024x8x128xf32> loc("benchmarks/rocm/benchmark_captured_movement.py":20:12)
    return %v0 : tensor<1024x8x128xf32>
  }
}