module attributes {tessera.arch = "gfx1201", tessera.frontend.authority = "tracer", tessera.ir.version = "1.0", tessera.target = "rocm_gfx1201"} {
  func.func @paged(%arg0: tensor<8x8x4x32xf32>, %arg1: tensor<4xi32>) -> tensor<5x4x32xf32> attributes {tessera.bindings = ["a0", "a1", "v0"], tessera.frontend.authority = "tracer", tessera.structured_cfg.blocks = 2 : i64, tessera.structured_cfg.digest = "c754085d55b2dfdc01758b11f1761205a49ed509abeaa8d28a787fb162a90553", tessera.structured_cfg.schema = "tessera.structured_cfg.v1"} {
    %0 = tessera.paged_kv_read %arg0, %arg1 {end = 6 : i64, schedule.artifact_hash = "feba3c6f330da79bace2573c838a606ec3d9f8203f00c6d07500dde66580fe01", start = 1 : i64} : (tensor<8x8x4x32xf32>, tensor<4xi32>) -> tensor<5x4x32xf32>
    %1 = schedule.paged_kv_read %0 {artifact_hash = "feba3c6f330da79bace2573c838a606ec3d9f8203f00c6d07500dde66580fe01", contract = {arch = "gfx1201", bindings = ["a0", "a1", "v0"], layout = "row_major", page_ownership = "read_only_borrow", shape = array<i64: 8, 4, 8, 4, 32, 1, 5>, table_bounds = "runtime_checked_physical_page_indices", target = "rocm_gfx1201"}} : tensor<5x4x32xf32> -> tensor<5x4x32xf32>
    return %1 : tensor<5x4x32xf32>
  }
}

