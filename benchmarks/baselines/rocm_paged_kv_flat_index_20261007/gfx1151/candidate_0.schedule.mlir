module attributes {tessera.arch = "gfx1151", tessera.ir.version = "1.0", tessera.target = "rocm_gfx1151"} {
  func.func @gfx1151_paged_kv_benchmark(%arg0: tensor<4x16x3x8xf32>, %arg1: tensor<4xi32>) -> tensor<41x3x8xf32> attributes {tessera.bindings = ["pages", "page_table", "slice"]} {
    %0 = tessera.paged_kv_read %arg0, %arg1 {end = 48 : i64, schedule.artifact_hash = "ccbe3f74f023d46ccc10282bca02d7bc2a8b5e60292d37023fc4f984e60024b8", start = 7 : i64} : (tensor<4x16x3x8xf32>, tensor<4xi32>) -> tensor<41x3x8xf32>
    %1 = schedule.paged_kv_read %0 {artifact_hash = "ccbe3f74f023d46ccc10282bca02d7bc2a8b5e60292d37023fc4f984e60024b8", contract = {arch = "gfx1151", bindings = ["pages", "page_table", "slice"], layout = "row_major", page_ownership = "read_only_borrow", shape = array<i64: 4, 4, 16, 3, 8, 7, 41>, table_bounds = "runtime_checked_physical_page_indices", target = "rocm_gfx1151"}} : tensor<41x3x8xf32> -> tensor<41x3x8xf32>
    return %1 : tensor<41x3x8xf32>
  }
}

