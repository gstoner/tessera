module attributes {tessera.ir.version = "1.0", tessera.target = "rocm_gfx1201", tessera.arch = "gfx1201"} {
  func.func @gfx1151_paged_kv_benchmark(%pages: tensor<4x16x3x8xf32>, %page_table: tensor<4xi32>) -> (tensor<41x3x8xf32>) attributes {tessera.bindings = ["pages", "page_table", "slice"]} {
    %slice = tessera.paged_kv_read %pages, %page_table {start = 7, end = 48} : (tensor<4x16x3x8xf32>, tensor<4xi32>) -> tensor<41x3x8xf32>
    return %slice : tensor<41x3x8xf32>
  }
}