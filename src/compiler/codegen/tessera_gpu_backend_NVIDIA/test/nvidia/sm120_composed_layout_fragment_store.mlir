// RUN: %tnv --lower-tile-to-nvidia='sm=120' --lower-tessera-nvidia-to-nvvm %s | FileCheck %s
// Static affine composed-layout addresses feeding the real SM120 fragment path.
// A is row-major at (a_row, a_col); B is column-major at (b_row, b_col).
// The physical fragment layout remains the existing m16n8k16 MMA contract.

!fa = !tile.fragment<m = 16, n = 8, k = 16, elem = "f16", acc = "f32", role = "a", layout = "row_major", family = "mma_sync">
!fb = !tile.fragment<m = 16, n = 8, k = 16, elem = "f16", acc = "f32", role = "b", layout = "col_major", family = "mma_sync">
!fc = !tile.fragment<m = 16, n = 8, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "mma_sync">

module {
  llvm.func @composed_layout_fragment_store(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr,
                                            %d_ptr: !llvm.ptr, %a_row: i64,
                                            %a_col: i64, %b_row: i64,
                                            %b_col: i64) attributes {nvvm.kernel} {
    %zero = arith.constant 0 : i64
    %a_linear = "tile.materialize_composed_layout"(%a_row, %a_col) {
      layout = #tile.composed_layout<[32, 16], [16, 1], [[[32], [1]], [[16], [1]]], [0, 0]>
    } : (i64, i64) -> i64
    %b_linear = "tile.materialize_composed_layout"(%b_row, %b_col) {
      layout = #tile.composed_layout<[16, 16], [1, 16], [[[16], [1]], [[16], [1]]], [0, 0]>
    } : (i64, i64) -> i64
    %a_tile = tile.view %a_ptr, %a_linear, %a_row, %a_col {tile.linear_base,
      tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 16>
    } : (!llvm.ptr, i64, i64, i64) -> !tile.tile
    %b_tile = tile.view %b_ptr, %b_linear, %b_row, %b_col {tile.linear_base,
      tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 16>
    } : (!llvm.ptr, i64, i64, i64) -> !tile.tile
    %a = tile.fragment_pack %a_tile {role = "a", mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>} : (!tile.tile) -> !fa
    %b = tile.fragment_pack %b_tile {role = "b", mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>} : (!tile.tile) -> !fb
    %c = tile.fragment_zero {role = "acc", mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>} : !fc
    %d = tile.mma %a, %b, %c {mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>} : (!fa, !fb, !fc) -> !fc
    %out = tile.fragment_unpack %d {tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>, mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>} : (!fc) -> !tile.tile
    "tile.store"(%out, %d_ptr, %zero, %zero) {tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>, tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 8>} : (!tile.tile, !llvm.ptr, i64, i64) -> ()
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @composed_layout_fragment_store
// CHECK: nvvm.mma.sync
// CHECK: llvm.store
// CHECK-NOT: tile.materialize_composed_layout
// CHECK-NOT: tile.fragment_pack
