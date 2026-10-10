module attributes {tessera.arch = "gfx1201", tessera.target = "rocm"} {
  func.func @mxfp8(%arg0: tensor<17x64xf8E4M3FN>, %arg1: tensor<64x19xf8E4M3FN>, %arg2: tensor<17x2xi8>, %arg3: tensor<2x19xi8>) -> tensor<17x19xf32> {
    %0 = tessera.scaled_matmul %arg0, %arg1 scales(%arg2, %arg3) {numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"}, scale_layout = {block = [1, 32], format = "e8m0", granularity = "block"}, schedule.artifact_hash = "62b69f4209770e72be556db17c775076e7dd599e2b3e63dca72f29e85aaa2873"} : (tensor<17x64xf8E4M3FN>, tensor<64x19xf8E4M3FN>, tensor<17x2xi8>, tensor<2x19xi8>) -> tensor<17x19xf32>
    %1 = schedule.matmul %0 {a_layout = "row_major", accum = "f32", activation = "none", arch = "gfx1201", artifact_hash = "62b69f4209770e72be556db17c775076e7dd599e2b3e63dca72f29e85aaa2873", b_layout = "col_major", bias = false, block_k = 32 : i64, macro_tile_m = 16 : i64, macro_tile_n = 16 : i64, output = "f32", physical_contract = "rocm_mxfp8_e4m3_e8m0_k32_v1", pipeline_depth = 1 : i64, raster_group = 1 : i64, raster_order = "row_major", residual = false, scale_format = "e8m0", scale_k = 32 : i64, scale_n = 1 : i64, storage = "e4m3", storage_b = "e4m3", tile_k = 16 : i64, tile_m = 16 : i64, tile_n = 16 : i64, warps = 1 : i64} : tensor<17x19xf32> -> tensor<17x19xf32>
    schedule.artifact {arch = "gfx1201", hash = "62b69f4209770e72be556db17c775076e7dd599e2b3e63dca72f29e85aaa2873", numeric_policy = "e4m3->f32", shape_key = "M=17;N=19;K=64;dtype=e4m3", tile = {k = 16 : i64, m = 16 : i64, macro_m = 16 : i64, macro_n = 16 : i64, n = 16 : i64, pipeline_depth = 1 : i64, warps = 1 : i64}}
    return %1 : tensor<17x19xf32>
  }
}

