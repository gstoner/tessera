// RUN: %tnv --lower-tile-to-nvidia='sm=120' --lower-tessera-nvidia-to-nvvm %s | FileCheck %s
//
// Canonical bf16 pointer-backed fragment path. NVVM's bf16 MMA ABI uses packed
!fa = !tile.fragment<m = 16, n = 8, k = 16, elem = "bf16", acc = "f32", role = "a", layout = "row_major", family = "mma_sync">
!fb = !tile.fragment<m = 16, n = 8, k = 16, elem = "bf16", acc = "f32", role = "b", layout = "col_major", family = "mma_sync">
!fc = !tile.fragment<m = 16, n = 8, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "mma_sync">

// i32 A/B registers (unlike vector<2xf16>), so the pack boundary must bitcast
// the two bf16 lanes before materializing nvvm.mma.sync.

module {
  llvm.func @pointer_fragment_store_bf16(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr,
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
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "bf16", b = "bf16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.tile) -> !fa
    %b = tile.fragment_pack %b_tile {
      role = "b",
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "bf16", b = "bf16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.tile) -> !fb
    %c = tile.fragment_zero {
      role = "acc",
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "bf16", b = "bf16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : !fc
    %d = tile.mma %a, %b, %c {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "bf16", b = "bf16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!fa, !fb, !fc) -> !fc
    %out = tile.fragment_unpack %d {
      tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "bf16", b = "bf16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!fc) -> !tile.tile
    "tile.store"(%out, %d_ptr, %zero, %zero) {
      tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 8>
    } : (!tile.tile, !llvm.ptr, i64, i64) -> ()
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @pointer_fragment_store_bf16
// CHECK: llvm.bitcast {{.*}} : vector<2xbf16> to i32
// CHECK: nvvm.mma.sync A[{{.*}}] B[{{.*}}] C[{{.*}}]
// CHECK-SAME: {layoutA
// CHECK: llvm.store
// CHECK-NOT: tile.fragment
// CHECK-NOT: tile.store
