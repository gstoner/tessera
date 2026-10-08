module attributes {tessera.arch = "gfx1201", tessera.ir.version = "1.0", tessera.target = "rocm_gfx1201"} {
  func.func @gfx1151_paged_kv_benchmark(%arg0: tensor<32x16x8x128xf32>, %arg1: tensor<64xi32>) -> tensor<1024x8x128xf32> attributes {tessera.bindings = ["pages", "page_table", "slice"]} {
    %0 = tessera.paged_kv_read %arg0, %arg1 {end = 1024 : i64, schedule.artifact_hash = "f49ca767119f392f4e96778d1cab08784c3d906b40e9c80cd6c727147a638518", start = 0 : i64} : (tensor<32x16x8x128xf32>, tensor<64xi32>) -> tensor<1024x8x128xf32>
    %1 = schedule.paged_kv_read %0 {artifact_hash = "f49ca767119f392f4e96778d1cab08784c3d906b40e9c80cd6c727147a638518", contract = {arch = "gfx1201", bindings = ["pages", "page_table", "slice"], layout = "row_major", page_ownership = "read_only_borrow", shape = array<i64: 32, 64, 16, 8, 128, 0, 1024>, table_bounds = "runtime_checked_physical_page_indices", target = "rocm_gfx1201"}} : tensor<1024x8x128xf32> -> tensor<1024x8x128xf32>
    return %1 : tensor<1024x8x128xf32>
  }
}

