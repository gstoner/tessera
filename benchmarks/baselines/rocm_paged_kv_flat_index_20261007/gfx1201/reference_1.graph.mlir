module attributes {tessera.ir.version = "1.0", tessera.target = "rocm_gfx1201", tessera.arch = "gfx1201"} {
  func.func @gfx1151_paged_kv_benchmark(%pages: tensor<3x7x3x17xf32>, %page_table: tensor<5xi32>) -> (tensor<25x3x17xf32>) attributes {tessera.bindings = ["pages", "page_table", "slice"]} {
    %slice = tessera.paged_kv_read %pages, %page_table {start = 5, end = 30} : (tensor<3x7x3x17xf32>, tensor<5xi32>) -> tensor<25x3x17xf32>
    return %slice : tensor<25x3x17xf32>
  }
}