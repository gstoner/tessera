// RUN: %tnv --lower-tile-to-nvidia='sm=120' --lower-tessera-nvidia-to-nvvm %s | FileCheck %s
//
// Complete compiler path for one m16n8k16 f16 tile with f32 accumulation:
// pointer-backed A/B loads, real MMA, accumulator unpack, row-major D stores.

module {
  llvm.func @pointer_fragment_store(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr,
                                    %d_ptr: !llvm.ptr, %zero: i64)
      attributes {nvvm.kernel} {
    %a_tile = tile.view %a_ptr, %zero, %zero {
      tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 16>
    } : (!llvm.ptr, i64, i64) -> !tile.tile
    %b_tile = tile.view %b_ptr, %zero, %zero {
      tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 16>
    } : (!llvm.ptr, i64, i64) -> !tile.tile
    %a = tile.fragment_pack %a_tile {
      role = "a",
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.tile) -> !tile.fragment
    %b = tile.fragment_pack %b_tile {
      role = "b",
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.tile) -> !tile.fragment
    %c = tile.fragment_zero {
      role = "acc",
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : !tile.fragment
    %d = tile.mma %a, %b, %c {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.fragment, !tile.fragment, !tile.fragment) -> !tile.fragment
    %out = tile.fragment_unpack %d {
      tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.fragment) -> !tile.tile
    "tile.store"(%out, %d_ptr, %zero, %zero) {
      tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 8>
    } : (!tile.tile, !llvm.ptr, i64, i64) -> ()
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @pointer_fragment_store
// CHECK: nvvm.read.ptx.sreg.tid.x
// CHECK: nvvm.mma.sync A[
// CHECK-SAME: C[{{.*}}] {layoutA
// CHECK: llvm.extractvalue
// CHECK-NEXT: llvm.store
// CHECK: llvm.extractvalue
// CHECK-NEXT: llvm.store
// CHECK: llvm.extractvalue
// CHECK-NEXT: llvm.store
// CHECK: llvm.extractvalue
// CHECK-NEXT: llvm.store
// CHECK-NOT: tile.fragment
// CHECK-NOT: tile.store
