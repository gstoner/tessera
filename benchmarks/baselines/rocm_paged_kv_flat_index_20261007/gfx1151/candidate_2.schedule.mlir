module attributes {tessera.arch = "gfx1151", tessera.ir.version = "1.0", tessera.target = "rocm_gfx1151"} {
  func.func @gfx1151_paged_kv_benchmark(%arg0: tensor<5x9x1x31xf32>, %arg1: tensor<8xi32>) -> tensor<51x1x31xf32> attributes {tessera.bindings = ["pages", "page_table", "slice"]} {
    %0 = tessera.paged_kv_read %arg0, %arg1 {end = 59 : i64, schedule.artifact_hash = "7f6614a3c8a6a3fa373cb1a94752888df959c33e0bc518fd63f66c246595bfc9", start = 8 : i64} : (tensor<5x9x1x31xf32>, tensor<8xi32>) -> tensor<51x1x31xf32>
    %1 = schedule.paged_kv_read %0 {artifact_hash = "7f6614a3c8a6a3fa373cb1a94752888df959c33e0bc518fd63f66c246595bfc9", contract = {arch = "gfx1151", bindings = ["pages", "page_table", "slice"], layout = "row_major", page_ownership = "read_only_borrow", shape = array<i64: 5, 8, 9, 1, 31, 8, 51>, table_bounds = "runtime_checked_physical_page_indices", target = "rocm_gfx1151"}} : tensor<51x1x31xf32> -> tensor<51x1x31xf32>
    return %1 : tensor<51x1x31xf32>
  }
}

