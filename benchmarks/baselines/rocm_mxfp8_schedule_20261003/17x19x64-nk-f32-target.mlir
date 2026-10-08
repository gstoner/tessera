module attributes {tessera.arch = "gfx1201", tessera.pipeline.arch = "gfx1201", tessera.pipeline.backend_codegen = "rocdl_hsaco", tessera.pipeline.family = "matmul", tessera.pipeline.output = "target", tessera.pipeline.schema = "tessera.executable_pipeline.v1", tessera.pipeline.target_ir_consumer = "tessera_rocm", tessera.pipeline.tile_producer = "content_addressed_tile", tessera.target = "rocm"} {
  func.func @mxfp8(%arg0: tensor<17x64xf8E4M3FN>, %arg1: tensor<19x64xf8E4M3FN>, %arg2: tensor<17x2xi8>, %arg3: tensor<2x19xi8>) -> tensor<17x19xf32> {
    %0 = bufferization.to_buffer %arg0 : tensor<17x64xf8E4M3FN> to memref<17x64xf8E4M3FN>
    %intptr = memref.extract_aligned_pointer_as_index %0 : memref<17x64xf8E4M3FN> -> index
    %1 = arith.index_cast %intptr : index to i64
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %3 = bufferization.to_buffer %arg1 : tensor<19x64xf8E4M3FN> to memref<19x64xf8E4M3FN>
    %intptr_0 = memref.extract_aligned_pointer_as_index %3 : memref<19x64xf8E4M3FN> -> index
    %4 = arith.index_cast %intptr_0 : index to i64
    %5 = llvm.inttoptr %4 : i64 to !llvm.ptr
    %6 = bufferization.to_buffer %arg2 : tensor<17x2xi8> to memref<17x2xi8>
    %intptr_1 = memref.extract_aligned_pointer_as_index %6 : memref<17x2xi8> -> index
    %7 = arith.index_cast %intptr_1 : index to i64
    %8 = llvm.inttoptr %7 : i64 to !llvm.ptr
    %9 = bufferization.to_buffer %arg3 : tensor<2x19xi8> to memref<2x19xi8>
    %intptr_2 = memref.extract_aligned_pointer_as_index %9 : memref<2x19xi8> -> index
    %10 = arith.index_cast %intptr_2 : index to i64
    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
    %alloc = memref.alloc() : memref<17x19xf32>
    %intptr_3 = memref.extract_aligned_pointer_as_index %alloc : memref<17x19xf32> -> index
    %12 = arith.index_cast %intptr_3 : index to i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    %c17_i64 = arith.constant 17 : i64
    %c19_i64 = arith.constant 19 : i64
    %c64_i64 = arith.constant 64 : i64
    tessera_rocm.scaled_wmma_gemm {abi = "a_bnk_lhs_scale_rhs_scale_d_m_n_k", block_m = 16 : i64, block_n = 16 : i64, instruction_k = 16 : i64, k = 64 : i64, k_step_schedule = "isolated_scale_group", m = 17 : i64, macro_k = 32 : i64, n = 19 : i64, name = "mxfp8", numeric_policy = {accum = "f32", execution_mode = "exact_per_block", storage = "e4m3"}, output = "f32", package_abi = "tessera.rocm.mxfp8_e4m3_e8m0_k32.a_bnk_sa_sb_o_m_n_k.f32.wide_scale.v1", partial_combine = "scale_outer_product_then_add", physical_contract = "rocm_mxfp8_e4m3_e8m0_k32_nk_v1", pipeline_depth = 1 : i64, scale_format = "e8m0", scale_k = 32 : i64, scale_n = 1 : i64, schedule_raster_group = 1 : i64, schedule_raster_order = "row_major", staging = "global", tessera.schedule_hash = "ff1204c98a751e5a0cfa675c4e8ffe6efa0e39e6b644366cecc0e5be844e7932", warps = 1 : i64}
    %14 = bufferization.to_tensor %alloc : memref<17x19xf32> to tensor<17x19xf32>
    return %14 : tensor<17x19xf32>
  }
}

