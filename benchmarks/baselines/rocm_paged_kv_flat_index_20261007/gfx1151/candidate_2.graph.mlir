module attributes {tessera.ir.version = "1.0", tessera.target = "rocm_gfx1151", tessera.arch = "gfx1151"} {
  func.func @gfx1151_paged_kv_benchmark(%pages: tensor<5x9x1x31xf32>, %page_table: tensor<8xi32>) -> (tensor<51x1x31xf32>) attributes {tessera.bindings = ["pages", "page_table", "slice"]} {
    %slice = tessera.paged_kv_read %pages, %page_table {start = 8, end = 59} : (tensor<5x9x1x31xf32>, tensor<8xi32>) -> tensor<51x1x31xf32>
    return %slice : tensor<51x1x31xf32>
  }
}