module attributes {tessera.arch = "gfx1201", tessera.frontend.authority = "tracer", tessera.ir.version = "1.0", tessera.target = "rocm_gfx1201"} {
  func.func @composed(%arg0: tensor<24576x2048xui8>, %arg1: tensor<24576x256xf8E4M3FN>, %arg2: tensor<2xf64>, %arg3: tensor<256x4096xf8E4M3FN>, %arg4: tensor<256x256xf32>) -> tensor<256x12288xbf16> attributes {tessera.frontend.authority = "tracer", tessera.structured_cfg.blocks = 2 : i64, tessera.structured_cfg.digest = "4c70c17f86ed6da7040cb7945dc86a302ab176d0c7d355d36f676e4b042238be", tessera.structured_cfg.schema = "tessera.structured_cfg.v1"} {
    %values, %scale_plane = tessera.nvfp4_expand %arg0, %arg1, %arg2 {numeric_policy = {destination_code_selection = "exact_e2m1_signed_zero", destination_format = "fp8_e4m3_k16_f32_scales", destination_scale_order = "k_group_n", execution_mode = "source_preserving_expansion", scale_rounding = "f64_product_to_f32_rne", source_format = "nvfp4_e2m1_e4m3_k16", source_scale_application = "projection_global_times_e4m3"}, row_offsets = [0, 12288, 24576], tessera.effect_kind = "pure"} : (tensor<24576x2048xui8>, tensor<24576x256xf8E4M3FN>, tensor<2xf64>) -> (tensor<24576x4096xf8E4M3FN>, tensor<256x24576xf32>)
    %0 = tessera.scaled_matmul %arg3, %values scales(%arg4, %scale_plane) {numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"}, scale_layout = {block = [1, 16], format = "fp32", granularity = "block"}, tessera.effect_kind = "pure", transposeB = true} : (tensor<256x4096xf8E4M3FN>, tensor<24576x4096xf8E4M3FN>, tensor<256x256xf32>, tensor<256x24576xf32>) -> tensor<256x24576xf32>
    %1 = tessera.cast %0 {dtype = "bf16", tessera.effect_kind = "pure"} : (tensor<256x24576xf32>) -> tensor<256x24576xbf16>
    %2 = tessera.slice %1 {slice_sizes = [256, 12288], start_indices = [0, 0], tessera.effect_kind = "pure"} : (tensor<256x24576xbf16>) -> tensor<256x12288xbf16>
    %3 = tessera.slice %1 {slice_sizes = [256, 12288], start_indices = [0, 12288], tessera.effect_kind = "pure"} : (tensor<256x24576xbf16>) -> tensor<256x12288xbf16>
    %4 = tessera.cast %2 {dtype = "fp32", tessera.effect_kind = "pure"} : (tensor<256x12288xbf16>) -> tensor<256x12288xf32>
    %5 = tessera.cast %3 {dtype = "fp32", tessera.effect_kind = "pure"} : (tensor<256x12288xbf16>) -> tensor<256x12288xf32>
    %6 = tessera.silu_mul %4, %5 {tessera.effect_kind = "pure"} : (tensor<256x12288xf32>, tensor<256x12288xf32>) -> tensor<256x12288xf32>
    %7 = tessera.cast %6 {dtype = "bf16", tessera.effect_kind = "pure"} : (tensor<256x12288xf32>) -> tensor<256x12288xbf16>
    return %7 : tensor<256x12288xbf16>
  }
}

