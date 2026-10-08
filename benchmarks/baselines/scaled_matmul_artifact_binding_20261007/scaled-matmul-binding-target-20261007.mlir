module attributes {tessera.arch = "gfx1201", tessera.pipeline.arch = "gfx1201", tessera.pipeline.backend_codegen = "rocdl_hsaco", tessera.pipeline.family = "matmul", tessera.pipeline.output = "target", tessera.pipeline.schema = "tessera.executable_pipeline.v1", tessera.pipeline.target_ir_consumer = "tessera_rocm", tessera.pipeline.tile_producer = "content_addressed_tile", tessera.target = "rocm"} {
  func.func @scales(%arg0: tensor<17x256xf8E4M3FN>, %arg1: tensor<256x19xf8E4M3FN>, %arg2: tensor<17x2xf32>, %arg3: tensor<2x1xf32>) -> tensor<17x19xf32> attributes {tessera.autodiff = "forward", tessera.autodiff.jvp = @scales__jvp, tessera.autodiff.wrt_indices = [2, 3]} {
    %0 = bufferization.to_buffer %arg0 : tensor<17x256xf8E4M3FN> to memref<17x256xf8E4M3FN>
    %intptr = memref.extract_aligned_pointer_as_index %0 : memref<17x256xf8E4M3FN> -> index
    %1 = arith.index_cast %intptr : index to i64
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %3 = bufferization.to_buffer %arg1 : tensor<256x19xf8E4M3FN> to memref<256x19xf8E4M3FN>
    %intptr_0 = memref.extract_aligned_pointer_as_index %3 : memref<256x19xf8E4M3FN> -> index
    %4 = arith.index_cast %intptr_0 : index to i64
    %5 = llvm.inttoptr %4 : i64 to !llvm.ptr
    %6 = bufferization.to_buffer %arg2 : tensor<17x2xf32> to memref<17x2xf32>
    %intptr_1 = memref.extract_aligned_pointer_as_index %6 : memref<17x2xf32> -> index
    %7 = arith.index_cast %intptr_1 : index to i64
    %8 = llvm.inttoptr %7 : i64 to !llvm.ptr
    %9 = bufferization.to_buffer %arg3 : tensor<2x1xf32> to memref<2x1xf32>
    %intptr_2 = memref.extract_aligned_pointer_as_index %9 : memref<2x1xf32> -> index
    %10 = arith.index_cast %intptr_2 : index to i64
    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
    %alloc = memref.alloc() : memref<17x19xf32>
    %intptr_3 = memref.extract_aligned_pointer_as_index %alloc : memref<17x19xf32> -> index
    %12 = arith.index_cast %intptr_3 : index to i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    %c17_i64 = arith.constant 17 : i64
    %c19_i64 = arith.constant 19 : i64
    %c256_i64 = arith.constant 256 : i64
    tessera_rocm.scaled_wmma_gemm {abi = "a_b_lhs_scale_rhs_scale_d_m_n_k", block_m = 16 : i64, block_n = 16 : i64, instruction_k = 16 : i64, k = 256 : i64, k_step_schedule = "isolated_scale_group", m = 17 : i64, macro_k = 128 : i64, n = 19 : i64, name = "scales", numeric_policy = {accum = "f32", execution_mode = "exact_per_block", storage = "e4m3"}, output = "f32", package_abi = "tessera.rocm.fp8_w8a8_blockscale.a_b_sa_sb_o_m_n_k.e4m3_e4m3_f32_f32.wmma_exact.v1", partial_combine = "scale_outer_product_then_add", physical_contract = "rocm_fp8_w8a8_blockscale_v1", pipeline_depth = 1 : i64, scale_format = "fp32", scale_k = 128 : i64, scale_n = 128 : i64, schedule_raster_group = 1 : i64, schedule_raster_order = "row_major", staging = "global", tessera.schedule_hash = "248ef92f25a529cc407da2d4bd06c1842a9c6c412d288336a10231fdc7288b10", warps = 1 : i64}
    %14 = bufferization.to_tensor %alloc : memref<17x19xf32> to tensor<17x19xf32>
    return %14 : tensor<17x19xf32>
  }
  func.func private @scales__jvp(%arg0: tensor<17x256xf8E4M3FN>, %arg1: tensor<256x19xf8E4M3FN>, %arg2: tensor<17x2xf32>, %arg3: tensor<2x1xf32>, %arg4: tensor<17x2xf32>, %arg5: tensor<2x1xf32>) -> (tensor<17x19xf32>, tensor<17x19xf32>) attributes {tessera.autodiff.forward = @scales, tessera.autodiff.role = "jvp"} {
    %0 = bufferization.to_buffer %arg0 : tensor<17x256xf8E4M3FN> to memref<17x256xf8E4M3FN>
    %intptr = memref.extract_aligned_pointer_as_index %0 : memref<17x256xf8E4M3FN> -> index
    %1 = arith.index_cast %intptr : index to i64
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %3 = bufferization.to_buffer %arg1 : tensor<256x19xf8E4M3FN> to memref<256x19xf8E4M3FN>
    %intptr_0 = memref.extract_aligned_pointer_as_index %3 : memref<256x19xf8E4M3FN> -> index
    %4 = arith.index_cast %intptr_0 : index to i64
    %5 = llvm.inttoptr %4 : i64 to !llvm.ptr
    %6 = bufferization.to_buffer %arg2 : tensor<17x2xf32> to memref<17x2xf32>
    %intptr_1 = memref.extract_aligned_pointer_as_index %6 : memref<17x2xf32> -> index
    %7 = arith.index_cast %intptr_1 : index to i64
    %8 = llvm.inttoptr %7 : i64 to !llvm.ptr
    %9 = bufferization.to_buffer %arg3 : tensor<2x1xf32> to memref<2x1xf32>
    %intptr_2 = memref.extract_aligned_pointer_as_index %9 : memref<2x1xf32> -> index
    %10 = arith.index_cast %intptr_2 : index to i64
    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
    %alloc = memref.alloc() : memref<17x19xf32>
    %intptr_3 = memref.extract_aligned_pointer_as_index %alloc : memref<17x19xf32> -> index
    %12 = arith.index_cast %intptr_3 : index to i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    %c17_i64 = arith.constant 17 : i64
    %c19_i64 = arith.constant 19 : i64
    %c256_i64 = arith.constant 256 : i64
    tessera_rocm.scaled_wmma_gemm {abi = "a_b_lhs_scale_rhs_scale_d_m_n_k", block_m = 16 : i64, block_n = 16 : i64, instruction_k = 16 : i64, k = 256 : i64, k_step_schedule = "isolated_scale_group", m = 17 : i64, macro_k = 128 : i64, n = 19 : i64, name = "scales__jvp", numeric_policy = {accum = "f32", execution_mode = "exact_per_block", storage = "e4m3"}, output = "f32", package_abi = "tessera.rocm.fp8_w8a8_blockscale.a_b_sa_sb_o_m_n_k.e4m3_e4m3_f32_f32.wmma_exact.v1", partial_combine = "scale_outer_product_then_add", physical_contract = "rocm_fp8_w8a8_blockscale_v1", pipeline_depth = 1 : i64, scale_format = "fp32", scale_k = 128 : i64, scale_n = 128 : i64, schedule_raster_group = 1 : i64, schedule_raster_order = "row_major", staging = "global", tessera.schedule_hash = "248ef92f25a529cc407da2d4bd06c1842a9c6c412d288336a10231fdc7288b10", warps = 1 : i64}
    %14 = bufferization.to_tensor %alloc : memref<17x19xf32> to tensor<17x19xf32>
    %15 = bufferization.to_buffer %arg0 : tensor<17x256xf8E4M3FN> to memref<17x256xf8E4M3FN>
    %intptr_4 = memref.extract_aligned_pointer_as_index %15 : memref<17x256xf8E4M3FN> -> index
    %16 = arith.index_cast %intptr_4 : index to i64
    %17 = llvm.inttoptr %16 : i64 to !llvm.ptr
    %18 = bufferization.to_buffer %arg1 : tensor<256x19xf8E4M3FN> to memref<256x19xf8E4M3FN>
    %intptr_5 = memref.extract_aligned_pointer_as_index %18 : memref<256x19xf8E4M3FN> -> index
    %19 = arith.index_cast %intptr_5 : index to i64
    %20 = llvm.inttoptr %19 : i64 to !llvm.ptr
    %21 = bufferization.to_buffer %arg4 : tensor<17x2xf32> to memref<17x2xf32>
    %intptr_6 = memref.extract_aligned_pointer_as_index %21 : memref<17x2xf32> -> index
    %22 = arith.index_cast %intptr_6 : index to i64
    %23 = llvm.inttoptr %22 : i64 to !llvm.ptr
    %24 = bufferization.to_buffer %arg3 : tensor<2x1xf32> to memref<2x1xf32>
    %intptr_7 = memref.extract_aligned_pointer_as_index %24 : memref<2x1xf32> -> index
    %25 = arith.index_cast %intptr_7 : index to i64
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr
    %alloc_8 = memref.alloc() : memref<17x19xf32>
    %intptr_9 = memref.extract_aligned_pointer_as_index %alloc_8 : memref<17x19xf32> -> index
    %27 = arith.index_cast %intptr_9 : index to i64
    %28 = llvm.inttoptr %27 : i64 to !llvm.ptr
    %c17_i64_10 = arith.constant 17 : i64
    %c19_i64_11 = arith.constant 19 : i64
    %c256_i64_12 = arith.constant 256 : i64
    tessera_rocm.scaled_wmma_gemm {abi = "a_b_lhs_scale_rhs_scale_d_m_n_k", block_m = 16 : i64, block_n = 16 : i64, instruction_k = 16 : i64, k = 256 : i64, k_step_schedule = "isolated_scale_group", m = 17 : i64, macro_k = 128 : i64, n = 19 : i64, name = "scales__jvp", numeric_policy = {accum = "f32", execution_mode = "exact_per_block", storage = "e4m3"}, output = "f32", package_abi = "tessera.rocm.fp8_w8a8_blockscale.a_b_sa_sb_o_m_n_k.e4m3_e4m3_f32_f32.wmma_exact.v1", partial_combine = "scale_outer_product_then_add", physical_contract = "rocm_fp8_w8a8_blockscale_v1", pipeline_depth = 1 : i64, scale_format = "fp32", scale_k = 128 : i64, scale_n = 128 : i64, schedule_raster_group = 1 : i64, schedule_raster_order = "row_major", staging = "global", tessera.schedule_hash = "248ef92f25a529cc407da2d4bd06c1842a9c6c412d288336a10231fdc7288b10", warps = 1 : i64}
    %29 = bufferization.to_tensor %alloc_8 : memref<17x19xf32> to tensor<17x19xf32>
    %30 = bufferization.to_buffer %arg0 : tensor<17x256xf8E4M3FN> to memref<17x256xf8E4M3FN>
    %intptr_13 = memref.extract_aligned_pointer_as_index %30 : memref<17x256xf8E4M3FN> -> index
    %31 = arith.index_cast %intptr_13 : index to i64
    %32 = llvm.inttoptr %31 : i64 to !llvm.ptr
    %33 = bufferization.to_buffer %arg1 : tensor<256x19xf8E4M3FN> to memref<256x19xf8E4M3FN>
    %intptr_14 = memref.extract_aligned_pointer_as_index %33 : memref<256x19xf8E4M3FN> -> index
    %34 = arith.index_cast %intptr_14 : index to i64
    %35 = llvm.inttoptr %34 : i64 to !llvm.ptr
    %36 = bufferization.to_buffer %arg2 : tensor<17x2xf32> to memref<17x2xf32>
    %intptr_15 = memref.extract_aligned_pointer_as_index %36 : memref<17x2xf32> -> index
    %37 = arith.index_cast %intptr_15 : index to i64
    %38 = llvm.inttoptr %37 : i64 to !llvm.ptr
    %39 = bufferization.to_buffer %arg5 : tensor<2x1xf32> to memref<2x1xf32>
    %intptr_16 = memref.extract_aligned_pointer_as_index %39 : memref<2x1xf32> -> index
    %40 = arith.index_cast %intptr_16 : index to i64
    %41 = llvm.inttoptr %40 : i64 to !llvm.ptr
    %alloc_17 = memref.alloc() : memref<17x19xf32>
    %intptr_18 = memref.extract_aligned_pointer_as_index %alloc_17 : memref<17x19xf32> -> index
    %42 = arith.index_cast %intptr_18 : index to i64
    %43 = llvm.inttoptr %42 : i64 to !llvm.ptr
    %c17_i64_19 = arith.constant 17 : i64
    %c19_i64_20 = arith.constant 19 : i64
    %c256_i64_21 = arith.constant 256 : i64
    tessera_rocm.scaled_wmma_gemm {abi = "a_b_lhs_scale_rhs_scale_d_m_n_k", block_m = 16 : i64, block_n = 16 : i64, instruction_k = 16 : i64, k = 256 : i64, k_step_schedule = "isolated_scale_group", m = 17 : i64, macro_k = 128 : i64, n = 19 : i64, name = "scales__jvp", numeric_policy = {accum = "f32", execution_mode = "exact_per_block", storage = "e4m3"}, output = "f32", package_abi = "tessera.rocm.fp8_w8a8_blockscale.a_b_sa_sb_o_m_n_k.e4m3_e4m3_f32_f32.wmma_exact.v1", partial_combine = "scale_outer_product_then_add", physical_contract = "rocm_fp8_w8a8_blockscale_v1", pipeline_depth = 1 : i64, scale_format = "fp32", scale_k = 128 : i64, scale_n = 128 : i64, schedule_raster_group = 1 : i64, schedule_raster_order = "row_major", staging = "global", tessera.schedule_hash = "248ef92f25a529cc407da2d4bd06c1842a9c6c412d288336a10231fdc7288b10", warps = 1 : i64}
    %44 = bufferization.to_tensor %alloc_17 : memref<17x19xf32> to tensor<17x19xf32>
    %45 = tessera.add %29, %44 : (tensor<17x19xf32>, tensor<17x19xf32>) -> tensor<17x19xf32>
    return %14, %45 : tensor<17x19xf32>, tensor<17x19xf32>
  }
}

