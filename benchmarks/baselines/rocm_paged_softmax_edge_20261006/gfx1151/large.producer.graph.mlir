module attributes {tessera.ir.version = "1.0", tessera.frontend.authority = "tracer", tessera.target = "rocm_gfx1151", tessera.arch = "gfx1151"} {
  func.func @paged(%a0: tensor<8x8x4x32xf32>, %a1: tensor<4xi32>) -> (tensor<5x4x32xf32>) attributes {tessera.frontend.authority = "tracer", tessera.structured_cfg.schema = "tessera.structured_cfg.v1", tessera.structured_cfg.digest = "c754085d55b2dfdc01758b11f1761205a49ed509abeaa8d28a787fb162a90553", tessera.structured_cfg.blocks = 2, tessera.bindings = ["a0", "a1", "v0"]} {
    %v0 = tessera.paged_kv_read %a0, %a1 {start = 1, end = 6} : (tensor<8x8x4x32xf32>, tensor<4xi32>) -> tensor<5x4x32xf32> loc("tests/unit/test_public_movement_frontend.py":12:12)
    return %v0 : tensor<5x4x32xf32>
  }
}