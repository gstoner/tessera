module attributes {tessera.ir.version = "1.0", tessera.target = "rocm_gfx1151", tessera.arch = "gfx1151"} {
  func.func @gfx1151_paged_kv_benchmark(%pages: tensor<32x16x8x128xf32>, %page_table: tensor<64xi32>) -> (tensor<1024x8x128xf32>) attributes {tessera.bindings = ["pages", "page_table", "slice"]} {
    %slice = tessera.paged_kv_read %pages, %page_table {start = 0, end = 1024} : (tensor<32x16x8x128xf32>, tensor<64xi32>) -> tensor<1024x8x128xf32>
    return %slice : tensor<1024x8x128xf32>
  }
}