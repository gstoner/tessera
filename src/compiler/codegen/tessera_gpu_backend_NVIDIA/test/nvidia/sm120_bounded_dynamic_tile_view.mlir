// RUN: %tnv --lower-tile-to-nvidia='sm=120' --lower-tessera-nvidia-to-nvvm %s | FileCheck %s

!fa = !tile.fragment<m = 16, n = 8, k = 16, elem = "f16", acc = "f16", role = "a", layout = "row_major", family = "auto">
!fb = !tile.fragment<m = 16, n = 8, k = 16, elem = "f16", acc = "f16", role = "b", layout = "col_major", family = "auto">
!fc = !tile.fragment<m = 16, n = 8, k = 16, elem = "f16", acc = "f16", role = "acc", layout = "row_major", family = "auto">

// Dynamic leading dimensions and runtime bounds are materialized with masked
// scalar loads, preserving zero fill for lanes outside the logical tile.
module {
  func.func @bounded_dynamic_fragment_pack(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr,
                                           %row: i64, %col: i64,
                                           %rows: i64, %cols: i64, %ld: i64) {
    %a_tile = tile.view %a_ptr, %row, %col, %rows, %cols, %ld {
      tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 0>
    } : (!llvm.ptr, i64, i64, i64, i64, i64) -> !tile.tile
    %b_tile = tile.view %b_ptr, %row, %col, %rows, %cols, %ld {
      tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 0>
    } : (!llvm.ptr, i64, i64, i64, i64, i64) -> !tile.tile
    %a = tile.fragment_pack %a_tile {
      role = "a",
      mma = #tile.mma_desc<family = "auto", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f16", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.tile) -> !fa
    %b = tile.fragment_pack %b_tile {
      role = "b",
      mma = #tile.mma_desc<family = "auto", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f16", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.tile) -> !fb
    %c = tile.fragment_zero {
      role = "acc",
      mma = #tile.mma_desc<family = "auto", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f16", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : !fc
    %d = tile.mma %a, %b, %c {
      mma = #tile.mma_desc<family = "auto", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f16", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!fa, !fb, !fc) -> !fc
    return
  }
}

// CHECK-LABEL: func.func @bounded_dynamic_fragment_pack
// CHECK: arith.cmpi
// CHECK: llvm.load
// CHECK: arith.select
// CHECK: nvvm.mma.sync A[
// CHECK-NOT: tile.fragment
