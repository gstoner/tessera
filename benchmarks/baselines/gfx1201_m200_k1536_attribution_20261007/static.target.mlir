module attributes {tessera.arch = "gfx1201", tessera.pipeline.arch = "gfx1201", tessera.pipeline.backend_codegen = "rocdl_hsaco", tessera.pipeline.family = "matmul", tessera.pipeline.output = "target", tessera.pipeline.schema = "tessera.executable_pipeline.v1", tessera.pipeline.target_ir_consumer = "tessera_rocm", tessera.pipeline.tile_producer = "content_addressed_tile", tessera.target = "rocm"} {
  func.func @w8a8_blockscale(%arg0: tensor<200x1536xf8E4M3FN>, %arg1: tensor<1024x1536xf8E4M3FN>, %arg2: tensor<200x12xf32>, %arg3: tensor<12x8xf32>) -> tensor<200x1024xbf16> {
    %0 = bufferization.to_buffer %arg0 : tensor<200x1536xf8E4M3FN> to memref<200x1536xf8E4M3FN>
    %intptr = memref.extract_aligned_pointer_as_index %0 : memref<200x1536xf8E4M3FN> -> index
    %1 = arith.index_cast %intptr : index to i64
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %3 = bufferization.to_buffer %arg1 : tensor<1024x1536xf8E4M3FN> to memref<1024x1536xf8E4M3FN>
    %intptr_0 = memref.extract_aligned_pointer_as_index %3 : memref<1024x1536xf8E4M3FN> -> index
    %4 = arith.index_cast %intptr_0 : index to i64
    %5 = llvm.inttoptr %4 : i64 to !llvm.ptr
    %6 = bufferization.to_buffer %arg2 : tensor<200x12xf32> to memref<200x12xf32>
    %intptr_1 = memref.extract_aligned_pointer_as_index %6 : memref<200x12xf32> -> index
    %7 = arith.index_cast %intptr_1 : index to i64
    %8 = llvm.inttoptr %7 : i64 to !llvm.ptr
    %9 = bufferization.to_buffer %arg3 : tensor<12x8xf32> to memref<12x8xf32>
    %intptr_2 = memref.extract_aligned_pointer_as_index %9 : memref<12x8xf32> -> index
    %10 = arith.index_cast %intptr_2 : index to i64
    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
    %alloc = memref.alloc() : memref<200x1024xbf16>
    %intptr_3 = memref.extract_aligned_pointer_as_index %alloc : memref<200x1024xbf16> -> index
    %12 = arith.index_cast %intptr_3 : index to i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    %c200_i64 = arith.constant 200 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c1536_i64 = arith.constant 1536 : i64
    tessera_rocm.scaled_wmma_gemm {abi = "a_bnk_lhs_scale_rhs_scale_d_m_n_k", block_m = 16 : i64, block_n = 32 : i64, instruction_k = 16 : i64, k = 1536 : i64, k_step_schedule = "isolated_scale_group", m = 200 : i64, macro_k = 128 : i64, n = 1024 : i64, name = "w8a8_blockscale", numeric_policy = {accum = "f32", execution_mode = "exact_per_block", storage = "e4m3"}, output = "bf16", package_abi = "tessera.rocm.fp8_w8a8_blockscale.a_bnk_sa_sb_o_m_n_k.e4m3_e4m3_f32_bf16.wmma_exact.v1", partial_combine = "scale_outer_product_then_add", physical_contract = "rocm_fp8_w8a8_blockscale_nk_v1", pipeline_depth = 1 : i64, scale_format = "fp32", scale_k = 128 : i64, scale_n = 128 : i64, schedule_raster_group = 1 : i64, schedule_raster_order = "row_major", staging = "global", tessera.schedule_hash = "2e8a7c75676d711841b23cb7af6b9855aeb9739622b5fd199a3ebd5d5c27c4ec", warps = 1 : i64}
    %14 = bufferization.to_tensor %alloc : memref<200x1024xbf16> to tensor<200x1024xbf16>
    return %14 : tensor<200x1024xbf16>
  }
}

