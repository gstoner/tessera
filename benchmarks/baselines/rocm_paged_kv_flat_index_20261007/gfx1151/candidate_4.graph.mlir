module attributes {tessera.ir.version = "1.0", tessera.target = "rocm_gfx1151", tessera.arch = "gfx1151"} {
  func.func @gfx1151_paged_kv_benchmark(%pages: tensor<2x3x1x1xf32>, %page_table: tensor<3xi32>) -> (tensor<5x1x1xf32>) attributes {tessera.bindings = ["pages", "page_table", "slice"]} {
    %slice = tessera.paged_kv_read %pages, %page_table {start = 2, end = 7} : (tensor<2x3x1x1xf32>, tensor<3xi32>) -> tensor<5x1x1xf32>
    return %slice : tensor<5x1x1xf32>
  }
}