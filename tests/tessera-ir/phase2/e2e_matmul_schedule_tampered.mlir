// RUN: not tessera-opt --tessera-schedule-to-tile %s 2>&1 | FileCheck %s

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512"} {
  func.func @tampered(%a: tensor<64x32xf32>, %b: tensor<32x48xf32>)
      -> tensor<64x48xf32> {
    %graph = tessera.matmul %a, %b {
      schedule.artifact_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    } : (tensor<64x32xf32>, tensor<32x48xf32>) -> tensor<64x48xf32>
    // CHECK: scheduled decision does not match the retained Graph matmul contract
    %scheduled = schedule.matmul %graph {
      a_layout = "row_major", accum = "f32", arch = "zen5-avx512",
      artifact_hash = "0000000000000000000000000000000000000000000000000000000000000000",
      b_layout = "col_major", pipeline_depth = 1 : i64,
      macro_tile_m = 16 : i64, macro_tile_n = 16 : i64,
      raster_group = 1 : i64, raster_order = "row_major", storage = "f32",
      bias = false, activation = "none", residual = false, output = "f32",
      tile_k = 16 : i64, tile_m = 16 : i64, tile_n = 16 : i64,
      warps = 1 : i64
    } : tensor<64x48xf32> -> tensor<64x48xf32>
    schedule.artifact {
      arch = "zen5-avx512",
      hash = "0000000000000000000000000000000000000000000000000000000000000000",
      shape_key = "M=64;N=48;K=32;dtype=f32"
    }
    return %scheduled : tensor<64x48xf32>
  }
}
