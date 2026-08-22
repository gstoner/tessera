// RUN: %tnv --lower-tile-to-nvidia=sm=120 --lower-tessera-nvidia-to-nvvm %s | FileCheck %s

!fa = !tile.fragment<m = 16, n = 8, k = 64, elem = "nvfp4", acc = "f32", role = "a", layout = "row_major", family = "mma_sync">
!fb = !tile.fragment<m = 16, n = 8, k = 64, elem = "nvfp4", acc = "f32", role = "b", layout = "col_major", family = "mma_sync">
!fc = !tile.fragment<m = 16, n = 8, k = 64, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "mma_sync">
!fsa = !tile.fragment<m = 16, n = 8, k = 64, elem = "nvfp4", acc = "f32", role = "scale_a", layout = "row_major", family = "mma_sync">
!fsb = !tile.fragment<m = 16, n = 8, k = 64, elem = "nvfp4", acc = "f32", role = "scale_b", layout = "col_major", family = "mma_sync">

// The data tile stays at the base while the logical scale views select a later
// A row tile and B column tile.  This isolates scale-view origin addressing.
module {
  llvm.func @nvfp4_fragment_origins(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr,
                                    %scale_a_ptr: !llvm.ptr,
                                    %scale_b_ptr: !llvm.ptr,
                                    %d_ptr: !llvm.ptr, %zero: i64,
                                    %scale_row: i64, %scale_col: i64)
      attributes {nvvm.kernel} {
    %a_tile = tile.view %a_ptr, %zero, %zero {
      tile.layout = #tile.layout<shard = [16, 64] : [64, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 32>
    } : (!llvm.ptr, i64, i64) -> !tile.tile
    %b_tile = tile.view %b_ptr, %zero, %zero {
      tile.layout = #tile.layout<shard = [64, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 32>
    } : (!llvm.ptr, i64, i64) -> !tile.tile
    %scale_a_tile = tile.view %scale_a_ptr, %scale_row, %zero {
      tile.layout = #tile.layout<shard = [16, 4] : [4, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 4>
    } : (!llvm.ptr, i64, i64) -> !tile.tile
    %scale_b_tile = tile.view %scale_b_ptr, %zero, %scale_col {
      tile.layout = #tile.layout<shard = [4, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 4>
    } : (!llvm.ptr, i64, i64) -> !tile.tile
    %a = tile.fragment_pack %a_tile {role = "a", mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 64, a = "nvfp4", b = "nvfp4", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>} : (!tile.tile) -> !fa
    %b = tile.fragment_pack %b_tile {role = "b", mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 64, a = "nvfp4", b = "nvfp4", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>} : (!tile.tile) -> !fb
    %scale_a = tile.fragment_pack %scale_a_tile {role = "scale_a", mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 64, a = "nvfp4", b = "nvfp4", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>} : (!tile.tile) -> !fsa
    %scale_b = tile.fragment_pack %scale_b_tile {role = "scale_b", mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 64, a = "nvfp4", b = "nvfp4", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>} : (!tile.tile) -> !fsb
    %c = tile.fragment_zero {role = "acc", mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 64, a = "nvfp4", b = "nvfp4", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>} : !fc
    %d = tile.mma %a, %b, %c, %scale_a, %scale_b {mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 64, a = "nvfp4", b = "nvfp4", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>} : (!fa, !fb, !fc, !fsa, !fsb) -> !fc
    %out = tile.fragment_unpack %d {
      tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 64, a = "nvfp4", b = "nvfp4", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!fc) -> !tile.tile
    "tile.store"(%out, %d_ptr, %zero, %zero) {
      tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 8>
    } : (!tile.tile, !llvm.ptr, i64, i64) -> ()
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @nvfp4_fragment_origins
// CHECK: arith.addi %arg6
// CHECK: llvm.getelementptr %arg2
// CHECK: arith.addi %arg7
// CHECK: llvm.getelementptr %arg3
// CHECK: llvm.inline_asm
// CHECK-SAME: mxf4nvf4.block_scale
// CHECK-NOT: tile.fragment_pack
