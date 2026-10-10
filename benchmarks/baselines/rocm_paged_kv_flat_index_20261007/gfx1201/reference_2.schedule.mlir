module attributes {tessera.arch = "gfx1201", tessera.ir.version = "1.0", tessera.target = "rocm_gfx1201"} {
  func.func @gfx1151_paged_kv_benchmark(%arg0: tensor<5x9x1x31xf32>, %arg1: tensor<8xi32>) -> tensor<51x1x31xf32> attributes {tessera.bindings = ["pages", "page_table", "slice"]} {
    %0 = tessera.paged_kv_read %arg0, %arg1 {end = 59 : i64, schedule.artifact_hash = "a66988e34f52fcbb7b1c693a10d078ab75ee4537eecd83c99eeddbda4075eb2f", start = 8 : i64} : (tensor<5x9x1x31xf32>, tensor<8xi32>) -> tensor<51x1x31xf32>
    %1 = schedule.paged_kv_read %0 {artifact_hash = "a66988e34f52fcbb7b1c693a10d078ab75ee4537eecd83c99eeddbda4075eb2f", contract = {arch = "gfx1201", bindings = ["pages", "page_table", "slice"], layout = "row_major", page_ownership = "read_only_borrow", shape = array<i64: 5, 8, 9, 1, 31, 8, 51>, table_bounds = "runtime_checked_physical_page_indices", target = "rocm_gfx1201"}} : tensor<51x1x31xf32> -> tensor<51x1x31xf32>
    return %1 : tensor<51x1x31xf32>
  }
}

