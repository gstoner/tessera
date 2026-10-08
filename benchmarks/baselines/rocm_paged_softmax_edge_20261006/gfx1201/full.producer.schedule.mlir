module attributes {tessera.arch = "gfx1201", tessera.frontend.authority = "tracer", tessera.ir.version = "1.0", tessera.target = "rocm_gfx1201"} {
  func.func @paged_full(%arg0: tensor<32x16x8x128xf32>, %arg1: tensor<64xi32>) -> tensor<1024x8x128xf32> attributes {tessera.bindings = ["a0", "a1", "v0"], tessera.frontend.authority = "tracer", tessera.structured_cfg.blocks = 2 : i64, tessera.structured_cfg.digest = "c754085d55b2dfdc01758b11f1761205a49ed509abeaa8d28a787fb162a90553", tessera.structured_cfg.schema = "tessera.structured_cfg.v1"} {
    %0 = tessera.paged_kv_read %arg0, %arg1 {end = 1024 : i64, schedule.artifact_hash = "9bc7d0a2e77931e017c04d2c01a10cfcb101fd55ae446db1b42c26ceef3a0d5f", start = 0 : i64} : (tensor<32x16x8x128xf32>, tensor<64xi32>) -> tensor<1024x8x128xf32>
    %1 = schedule.paged_kv_read %0 {artifact_hash = "9bc7d0a2e77931e017c04d2c01a10cfcb101fd55ae446db1b42c26ceef3a0d5f", contract = {arch = "gfx1201", bindings = ["a0", "a1", "v0"], layout = "row_major", page_ownership = "read_only_borrow", shape = array<i64: 32, 64, 16, 8, 128, 0, 1024>, table_bounds = "runtime_checked_physical_page_indices", target = "rocm_gfx1201"}} : tensor<1024x8x128xf32> -> tensor<1024x8x128xf32>
    return %1 : tensor<1024x8x128xf32>
  }
}

