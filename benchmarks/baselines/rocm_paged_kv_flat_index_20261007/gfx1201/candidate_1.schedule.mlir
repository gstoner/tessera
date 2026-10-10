module attributes {tessera.arch = "gfx1201", tessera.ir.version = "1.0", tessera.target = "rocm_gfx1201"} {
  func.func @gfx1151_paged_kv_benchmark(%arg0: tensor<3x7x3x17xf32>, %arg1: tensor<5xi32>) -> tensor<25x3x17xf32> attributes {tessera.bindings = ["pages", "page_table", "slice"]} {
    %0 = tessera.paged_kv_read %arg0, %arg1 {end = 30 : i64, schedule.artifact_hash = "d25f69c4c3a507d9d1d518113b990216a0b30e33f69624a2c7410a12a825ce89", start = 5 : i64} : (tensor<3x7x3x17xf32>, tensor<5xi32>) -> tensor<25x3x17xf32>
    %1 = schedule.paged_kv_read %0 {artifact_hash = "d25f69c4c3a507d9d1d518113b990216a0b30e33f69624a2c7410a12a825ce89", contract = {arch = "gfx1201", bindings = ["pages", "page_table", "slice"], layout = "row_major", page_ownership = "read_only_borrow", shape = array<i64: 3, 5, 7, 3, 17, 5, 25>, table_bounds = "runtime_checked_physical_page_indices", target = "rocm_gfx1201"}} : tensor<25x3x17xf32> -> tensor<25x3x17xf32>
    return %1 : tensor<25x3x17xf32>
  }
}

