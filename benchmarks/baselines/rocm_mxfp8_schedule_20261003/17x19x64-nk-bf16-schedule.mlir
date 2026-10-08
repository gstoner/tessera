module attributes {tessera.arch = "gfx1201", tessera.target = "rocm"} {
  func.func @mxfp8(%arg0: tensor<17x64xf8E4M3FN>, %arg1: tensor<19x64xf8E4M3FN>, %arg2: tensor<17x2xi8>, %arg3: tensor<2x19xi8>) -> tensor<17x19xbf16> {
    %0 = tessera.scaled_matmul %arg0, %arg1 scales(%arg2, %arg3) {numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"}, scale_layout = {block = [1, 32], format = "e8m0", granularity = "block"}, schedule.artifact_hash = "6ed6b559dcd47e2b2857563138b19d420a8a4b26817e4fac8a58c3bd6e9b501d", transposeB = true} : (tensor<17x64xf8E4M3FN>, tensor<19x64xf8E4M3FN>, tensor<17x2xi8>, tensor<2x19xi8>) -> tensor<17x19xbf16>
    %1 = schedule.matmul %0 {a_layout = "row_major", accum = "f32", activation = "none", arch = "gfx1201", artifact_hash = "6ed6b559dcd47e2b2857563138b19d420a8a4b26817e4fac8a58c3bd6e9b501d", b_layout = "col_major", bias = false, block_k = 32 : i64, macro_tile_m = 16 : i64, macro_tile_n = 16 : i64, output = "bf16", physical_contract = "rocm_mxfp8_e4m3_e8m0_k32_nk_v1", pipeline_depth = 1 : i64, raster_group = 1 : i64, raster_order = "row_major", residual = false, scale_format = "e8m0", scale_k = 32 : i64, scale_n = 1 : i64, storage = "e4m3", storage_b = "e4m3", tile_k = 16 : i64, tile_m = 16 : i64, tile_n = 16 : i64, warps = 1 : i64} : tensor<17x19xbf16> -> tensor<17x19xbf16>
    schedule.artifact {arch = "gfx1201", hash = "6ed6b559dcd47e2b2857563138b19d420a8a4b26817e4fac8a58c3bd6e9b501d", numeric_policy = "e4m3->f32", shape_key = "M=17;N=19;K=64;dtype=e4m3", tile = {k = 16 : i64, m = 16 : i64, macro_m = 16 : i64, macro_n = 16 : i64, n = 16 : i64, pipeline_depth = 1 : i64, warps = 1 : i64}}
    return %1 : tensor<17x19xbf16>
  }
}

