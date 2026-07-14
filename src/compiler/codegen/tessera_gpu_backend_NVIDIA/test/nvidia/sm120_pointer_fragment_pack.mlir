// RUN: %tnv --lower-tile-to-nvidia='sm=120' --lower-tessera-nvidia-to-nvvm %s | FileCheck %s
//
// Pointer-backed logical A/B views materialize the proven m16n8k16 f16 lane
// map: four row-major A vector loads and two column-major B vector loads.

module {
  func.func @pointer_fragment_pack(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr,
                                  %m0: i64, %k0: i64, %n0: i64) {
    %a_tile = tile.view %a_ptr, %m0, %k0 {
      tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 16>
    } : (!llvm.ptr, i64, i64) -> !tile.tile
    %b_tile = tile.view %b_ptr, %k0, %n0 {
      tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 16>
    } : (!llvm.ptr, i64, i64) -> !tile.tile
    %a = tile.fragment_pack %a_tile {
      role = "a",
      mma = #tile.mma_desc<family = "auto", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f16", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.tile) -> !tile.fragment
    %b = tile.fragment_pack %b_tile {
      role = "b",
      mma = #tile.mma_desc<family = "auto", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f16", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.tile) -> !tile.fragment
    %c = tile.fragment_zero {
      role = "acc",
      mma = #tile.mma_desc<family = "auto", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f16", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : !tile.fragment
    %d = tile.mma %a, %b, %c {
      mma = #tile.mma_desc<family = "auto", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f16", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.fragment, !tile.fragment, !tile.fragment) -> !tile.fragment
    return
  }
}

// CHECK-LABEL: func.func @pointer_fragment_pack
// CHECK: nvvm.read.ptx.sreg.tid.x
// CHECK: llvm.getelementptr
// CHECK-NEXT: llvm.load
// CHECK: llvm.getelementptr
// CHECK-NEXT: llvm.load
// CHECK: llvm.getelementptr
// CHECK-NEXT: llvm.load
// CHECK: llvm.getelementptr
// CHECK-NEXT: llvm.load
// CHECK: llvm.getelementptr
// CHECK-NEXT: llvm.load
// CHECK: llvm.getelementptr
// CHECK-NEXT: llvm.load
// CHECK: nvvm.mma.sync A[
// CHECK-SAME: shape = #nvvm.shape<m = 16, n = 8, k = 16>
// CHECK-NOT: tile.fragment
