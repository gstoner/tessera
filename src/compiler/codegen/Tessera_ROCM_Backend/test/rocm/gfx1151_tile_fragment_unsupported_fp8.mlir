// RUN: not %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151})' %s 2>&1 | FileCheck %s

!frag_a = !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "a", layout = "row_major", family = "auto">
!frag_b = !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "b", layout = "col_major", family = "auto">
!frag_acc = !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">

module {
  gpu.module @fragment_mod {
    gpu.func @unsupported_fp8(%a_mem: memref<256xf8E4M3FN>,
                              %b_mem: memref<256xf8E4M3FN>) kernel {
      %zero = arith.constant 0 : index
      %a_tile = tile.view %a_mem, %zero, %zero {
        tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
        tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 16>
      } : (memref<256xf8E4M3FN>, index, index) -> !tile.tile
      %b_tile = tile.view %b_mem, %zero, %zero {
        tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
        tile.memory = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 16>
      } : (memref<256xf8E4M3FN>, index, index) -> !tile.tile
      %a = tile.fragment_pack %a_tile {
        role = "a",
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "e4m3", b = "e4m3", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!tile.tile) -> !frag_a
      %b = tile.fragment_pack %b_tile {
        role = "b",
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "e4m3", b = "e4m3", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!tile.tile) -> !frag_b
      %c = tile.fragment_zero {
        role = "acc",
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "e4m3", b = "e4m3", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : !frag_acc
      %d = tile.mma %a, %b, %c {
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "e4m3", b = "e4m3", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!frag_a, !frag_b, !frag_acc) -> !frag_acc
      gpu.return
    }
  }
}

// CHECK: ROCM_TILE_UNSUPPORTED_DTYPE: gfx1151 RDNA 3.5 WMMA has no FP8/BF8 matrix form
