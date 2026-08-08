// RUN: not tessera-opt %s 2>&1 | FileCheck %s

module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @invalid_macro_tile(%a: tensor<32x32xf16>,
      %b: tensor<32x32xf16>) -> tensor<32x32xf32> {
    %graph = tessera.matmul %a, %b
        : (tensor<32x32xf16>, tensor<32x32xf16>) -> tensor<32x32xf32>
    // CHECK: 'schedule.matmul' op macro tile dimensions must be positive multiples of tile_m/tile_n
    %scheduled = schedule.matmul %graph {
      a_layout = "row_major", accum = "f32", arch = "gfx1151",
      artifact_hash = "0000000000000000000000000000000000000000000000000000000000000000",
      b_layout = "col_major", macro_tile_m = 24 : i64,
      macro_tile_n = 64 : i64, pipeline_depth = 1 : i64,
      raster_group = 1 : i64, raster_order = "row_major", storage = "f16",
      tile_k = 16 : i64, tile_m = 16 : i64, tile_n = 16 : i64,
      warps = 1 : i64
    } : tensor<32x32xf32> -> tensor<32x32xf32>
    return %scheduled : tensor<32x32xf32>
  }
}
