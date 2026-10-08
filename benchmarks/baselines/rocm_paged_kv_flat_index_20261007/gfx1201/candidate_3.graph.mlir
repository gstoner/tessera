module attributes {tessera.ir.version = "1.0", tessera.target = "rocm_gfx1201", tessera.arch = "gfx1201"} {
  func.func @gfx1151_paged_kv_benchmark(%pages: tensor<4x1x2x257xf32>, %page_table: tensor<3xi32>) -> (tensor<3x2x257xf32>) attributes {tessera.bindings = ["pages", "page_table", "slice"]} {
    %slice = tessera.paged_kv_read %pages, %page_table {start = 0, end = 3} : (tensor<4x1x2x257xf32>, tensor<3xi32>) -> tensor<3x2x257xf32>
    return %slice : tensor<3x2x257xf32>
  }
}