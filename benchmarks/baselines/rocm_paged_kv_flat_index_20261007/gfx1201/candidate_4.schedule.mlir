module attributes {tessera.arch = "gfx1201", tessera.ir.version = "1.0", tessera.target = "rocm_gfx1201"} {
  func.func @gfx1151_paged_kv_benchmark(%arg0: tensor<2x3x1x1xf32>, %arg1: tensor<3xi32>) -> tensor<5x1x1xf32> attributes {tessera.bindings = ["pages", "page_table", "slice"]} {
    %0 = tessera.paged_kv_read %arg0, %arg1 {end = 7 : i64, schedule.artifact_hash = "05df87d7c237ecd4d1251f47a29bfb333a73a1e4f3a8fee85bea95300a9bf7f6", start = 2 : i64} : (tensor<2x3x1x1xf32>, tensor<3xi32>) -> tensor<5x1x1xf32>
    %1 = schedule.paged_kv_read %0 {artifact_hash = "05df87d7c237ecd4d1251f47a29bfb333a73a1e4f3a8fee85bea95300a9bf7f6", contract = {arch = "gfx1201", bindings = ["pages", "page_table", "slice"], layout = "row_major", page_ownership = "read_only_borrow", shape = array<i64: 2, 3, 3, 1, 1, 2, 5>, table_bounds = "runtime_checked_physical_page_indices", target = "rocm_gfx1201"}} : tensor<5x1x1xf32> -> tensor<5x1x1xf32>
    return %1 : tensor<5x1x1xf32>
  }
}

