// RUN: tessera-opt %s -split-input-file -verify-diagnostics
//
// ROCM-SPLIT-K-1: the split-K contract is a semantic PAIR (Decision #21a) --
// the slice count and the order the partials are summed in. Neither level may
// accept one without the other, a split without a macro K block to align to,
// or any reduction but the deterministic `ordered` one.

func.func @schedule_split_without_order(%x: tensor<16x256xf32>) -> tensor<16x256xf32> {
  // expected-error @+1 {{SCHEDULE_SPLIT_K_BAD_CONTRACT: split_k > 1 requires split_k_reduction = "ordered"}}
  %0 = schedule.matmul %x {a_layout = "row_major", accum = "f32", activation = "none", arch = "gfx1201", artifact_hash = "0000000000000000000000000000000000000000000000000000000000000000", b_layout = "col_major", bias = false, block_k = 32 : i64, macro_tile_m = 16 : i64, macro_tile_n = 16 : i64, output = "f32", pipeline_depth = 1 : i64, raster_group = 1 : i64, raster_order = "row_major", residual = false, split_k = 2 : i64, storage = "f16", tile_k = 16 : i64, tile_m = 16 : i64, tile_n = 16 : i64, warps = 1 : i64} : tensor<16x256xf32> -> tensor<16x256xf32>
  return %0 : tensor<16x256xf32>
}

// -----

func.func @schedule_atomic_reduction(%x: tensor<16x256xf32>) -> tensor<16x256xf32> {
  // expected-error @+1 {{SCHEDULE_SPLIT_K_BAD_CONTRACT: split_k > 1 requires split_k_reduction = "ordered"}}
  %0 = schedule.matmul %x {a_layout = "row_major", accum = "f32", activation = "none", arch = "gfx1201", artifact_hash = "0000000000000000000000000000000000000000000000000000000000000000", b_layout = "col_major", bias = false, block_k = 32 : i64, macro_tile_m = 16 : i64, macro_tile_n = 16 : i64, output = "f32", pipeline_depth = 1 : i64, raster_group = 1 : i64, raster_order = "row_major", residual = false, split_k = 2 : i64, split_k_reduction = "atomic", storage = "f16", tile_k = 16 : i64, tile_m = 16 : i64, tile_n = 16 : i64, warps = 1 : i64} : tensor<16x256xf32> -> tensor<16x256xf32>
  return %0 : tensor<16x256xf32>
}

// -----

func.func @schedule_order_without_split(%x: tensor<16x256xf32>) -> tensor<16x256xf32> {
  // expected-error @+1 {{SCHEDULE_SPLIT_K_BAD_CONTRACT: split_k_reduction requires split_k > 1}}
  %0 = schedule.matmul %x {a_layout = "row_major", accum = "f32", activation = "none", arch = "gfx1201", artifact_hash = "0000000000000000000000000000000000000000000000000000000000000000", b_layout = "col_major", bias = false, block_k = 32 : i64, macro_tile_m = 16 : i64, macro_tile_n = 16 : i64, output = "f32", pipeline_depth = 1 : i64, raster_group = 1 : i64, raster_order = "row_major", residual = false, split_k_reduction = "ordered", storage = "f16", tile_k = 16 : i64, tile_m = 16 : i64, tile_n = 16 : i64, warps = 1 : i64} : tensor<16x256xf32> -> tensor<16x256xf32>
  return %0 : tensor<16x256xf32>
}

// -----

func.func @schedule_split_without_k_block(%x: tensor<16x256xf32>) -> tensor<16x256xf32> {
  // expected-error @+1 {{SCHEDULE_SPLIT_K_BAD_CONTRACT: split_k > 1 requires a macro K block}}
  %0 = schedule.matmul %x {a_layout = "row_major", accum = "f32", activation = "none", arch = "gfx1201", artifact_hash = "0000000000000000000000000000000000000000000000000000000000000000", b_layout = "col_major", bias = false, block_k = 0 : i64, macro_tile_m = 16 : i64, macro_tile_n = 16 : i64, output = "f32", pipeline_depth = 1 : i64, raster_group = 1 : i64, raster_order = "row_major", residual = false, split_k = 2 : i64, split_k_reduction = "ordered", storage = "f16", tile_k = 16 : i64, tile_m = 16 : i64, tile_n = 16 : i64, warps = 1 : i64} : tensor<16x256xf32> -> tensor<16x256xf32>
  return %0 : tensor<16x256xf32>
}

// -----

func.func @tile_split_without_order(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr) {
  %m = arith.constant 16 : i64
  %n = arith.constant 256 : i64
  %k = arith.constant 2048 : i64
  // expected-error @+1 {{TILE_SPLIT_K_BAD_CONTRACT: tessera.split_k and tessera.split_k_reduction must appear together}}
  tile.matmul_kernel %a, %b, %d, %m, %n, %k {
    mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 2>,
    epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
    tessera.canonical_k_loop = true, tessera.tile_m = 16 : i64, tessera.tile_n = 16 : i64, tessera.tile_k = 16 : i64,
    tessera.split_k = 2 : i64
  } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
  return
}

// -----

func.func @tile_split_unaligned_k(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr) {
  %m = arith.constant 16 : i64
  %n = arith.constant 256 : i64
  %k = arith.constant 2080 : i64
  // expected-error @+1 {{TILE_SPLIT_K_BAD_CONTRACT: K=2080 does not split into 2 slices of whole macro K blocks (32)}}
  tile.matmul_kernel %a, %b, %d, %m, %n, %k {
    mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 2>,
    epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
    tessera.canonical_k_loop = true, tessera.tile_m = 16 : i64, tessera.tile_n = 16 : i64, tessera.tile_k = 16 : i64,
    tessera.split_k = 2 : i64, tessera.split_k_reduction = "ordered"
  } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
  return
}

// -----

func.func @tile_split_integer_accumulator(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr) {
  %m = arith.constant 16 : i64
  %n = arith.constant 256 : i64
  %k = arith.constant 2048 : i64
  // expected-error @+1 {{TILE_SPLIT_K_BAD_CONTRACT: split-K partials are an fp32 workspace}}
  tile.matmul_kernel %a, %b, %d, %m, %n, %k {
    mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16, a = "int8", b = "int8", acc = "i32", a_layout = "row_major", b_layout = "col_major", k_blocks = 2>,
    epilogue = #tile.epilogue<bias = false, activation = "none", output = "i32">,
    tessera.canonical_k_loop = true, tessera.tile_m = 16 : i64, tessera.tile_n = 16 : i64, tessera.tile_k = 16 : i64,
    tessera.split_k = 2 : i64, tessera.split_k_reduction = "ordered"
  } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
  return
}
