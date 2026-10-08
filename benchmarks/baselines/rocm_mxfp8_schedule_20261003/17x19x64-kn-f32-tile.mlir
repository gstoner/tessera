module attributes {tessera.arch = "gfx1201", tessera.target = "rocm"} {
  func.func @mxfp8(%arg0: tensor<17x64xf8E4M3FN>, %arg1: tensor<64x19xf8E4M3FN>, %arg2: tensor<17x2xi8>, %arg3: tensor<2x19xi8>) -> tensor<17x19xf32> {
    %0 = bufferization.to_buffer %arg0 : tensor<17x64xf8E4M3FN> to memref<17x64xf8E4M3FN>
    %intptr = memref.extract_aligned_pointer_as_index %0 : memref<17x64xf8E4M3FN> -> index
    %1 = arith.index_cast %intptr : index to i64
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %3 = bufferization.to_buffer %arg1 : tensor<64x19xf8E4M3FN> to memref<64x19xf8E4M3FN>
    %intptr_0 = memref.extract_aligned_pointer_as_index %3 : memref<64x19xf8E4M3FN> -> index
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
    tile.scaled_matmul_kernel %2, %5, %8, %11, %13, %c17_i64, %c19_i64, %c64_i64 {epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">, mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16, a = "e4m3", b = "e4m3", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 2, scale_k = 32, scale_fmt = "e8m0">, numeric_policy = {accum = "f32", execution_mode = "exact_per_block", storage = "e4m3"}, partial_accumulator = {combine = "scale_outer_product_then_add", cross_step_motion = "forbid", init = "zero", instruction_steps = 2 : i64, schedule_scope = "scale_group", scope = "scale_group"}, physical_contract = "rocm_mxfp8_e4m3_e8m0_k32_v1", staging = "global", tessera.canonical_k_loop = true, tessera.macro_tile_m = 16 : i64, tessera.macro_tile_n = 16 : i64, tessera.pipeline_depth = 1 : i64, tessera.problem_k = 64 : i64, tessera.problem_m = 17 : i64, tessera.problem_n = 19 : i64, tessera.raster_group = 1 : i64, tessera.raster_order = "row_major", tessera.scale_block_n = 1 : i64, tessera.schedule_hash = "62b69f4209770e72be556db17c775076e7dd599e2b3e63dca72f29e85aaa2873", tessera.tile_k = 16 : i64, tessera.tile_m = 16 : i64, tessera.tile_n = 16 : i64, warps = 1 : i64} : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    %14 = bufferization.to_tensor %alloc : memref<17x19xf32> to tensor<17x19xf32>
    return %14 : tensor<17x19xf32>
  }
}

