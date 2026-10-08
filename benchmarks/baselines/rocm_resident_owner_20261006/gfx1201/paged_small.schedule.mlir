module attributes {tessera.arch = "gfx1201", tessera.frontend.authority = "tracer", tessera.ir.version = "1.0", tessera.target = "rocm_gfx1201"} {
  func.func @paged(%arg0: tensor<4x4x3x8xf32>, %arg1: tensor<4xi32>) -> tensor<5x3x8xf32> attributes {tessera.bindings = ["a0", "a1", "v0"], tessera.frontend.authority = "tracer", tessera.structured_cfg.blocks = 2 : i64, tessera.structured_cfg.digest = "c754085d55b2dfdc01758b11f1761205a49ed509abeaa8d28a787fb162a90553", tessera.structured_cfg.schema = "tessera.structured_cfg.v1"} {
    %0 = tessera.paged_kv_read %arg0, %arg1 {end = 6 : i64, schedule.artifact_hash = "c5fb6219e7e53afc0c69391546762fec5fa9e1087f81c20fe7251bd08e09b38f", start = 1 : i64} : (tensor<4x4x3x8xf32>, tensor<4xi32>) -> tensor<5x3x8xf32>
    %1 = schedule.paged_kv_read %0 {artifact_hash = "c5fb6219e7e53afc0c69391546762fec5fa9e1087f81c20fe7251bd08e09b38f", contract = {arch = "gfx1201", bindings = ["a0", "a1", "v0"], layout = "row_major", page_ownership = "read_only_borrow", shape = array<i64: 4, 4, 4, 3, 8, 1, 5>, table_bounds = "runtime_checked_physical_page_indices", target = "rocm_gfx1201"}} : tensor<5x3x8xf32> -> tensor<5x3x8xf32>
    return %1 : tensor<5x3x8xf32>
  }
}

