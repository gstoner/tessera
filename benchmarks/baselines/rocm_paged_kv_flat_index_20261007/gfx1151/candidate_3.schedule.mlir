module attributes {tessera.arch = "gfx1151", tessera.ir.version = "1.0", tessera.target = "rocm_gfx1151"} {
  func.func @gfx1151_paged_kv_benchmark(%arg0: tensor<4x1x2x257xf32>, %arg1: tensor<3xi32>) -> tensor<3x2x257xf32> attributes {tessera.bindings = ["pages", "page_table", "slice"]} {
    %0 = tessera.paged_kv_read %arg0, %arg1 {end = 3 : i64, schedule.artifact_hash = "31dd211cc242bf963c47248dcda96210b03e8912be1e2cd0d9274ce4333b0d41", start = 0 : i64} : (tensor<4x1x2x257xf32>, tensor<3xi32>) -> tensor<3x2x257xf32>
    %1 = schedule.paged_kv_read %0 {artifact_hash = "31dd211cc242bf963c47248dcda96210b03e8912be1e2cd0d9274ce4333b0d41", contract = {arch = "gfx1151", bindings = ["pages", "page_table", "slice"], layout = "row_major", page_ownership = "read_only_borrow", shape = array<i64: 4, 3, 1, 2, 257, 0, 3>, table_bounds = "runtime_checked_physical_page_indices", target = "rocm_gfx1151"}} : tensor<3x2x257xf32> -> tensor<3x2x257xf32>
    return %1 : tensor<3x2x257xf32>
  }
}

