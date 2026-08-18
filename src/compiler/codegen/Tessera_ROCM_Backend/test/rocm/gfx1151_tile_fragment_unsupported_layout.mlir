// RUN: not %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151})' %s 2>&1 | FileCheck %s

!frag_a = !tile.fragment<m = 16, n = 16, k = 16, elem = "f16", acc = "f32", role = "a", layout = "row_major", family = "auto">
!frag_b = !tile.fragment<m = 16, n = 16, k = 16, elem = "f16", acc = "f32", role = "b", layout = "col_major", family = "auto">
!frag_acc = !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">

module {
  gpu.module @bad_layout_mod {
    gpu.func @bad_layout(%a_mem: memref<256xf16>,
                         %b_mem: memref<256xf16>,
                         %d_mem: memref<256xf32>) kernel {
      %zero = arith.constant 0 : index
      %a_tile = tile.view %a_mem, %zero, %zero {
        tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
        tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 16>
      } : (memref<256xf16>, index, index) -> !tile.tile
      // B lives in LDS, which this materializer cannot address: it emits
      // global loads against a rank-1 gmem buffer.
      //
      // This fixture used to assert something else -- that a ROW-MAJOR B
      // contradicted the descriptor's `b_layout = "col_major"` and was
      // therefore rejected. That premise is gone as of the strided-K change,
      // and deliberately: `b_layout` on `#tile.mma_desc` is the orientation the
      // MMA OPERAND is read in, while `tile.memory`'s order is how the BUFFER
      // is stored. Conflating them made storage order a correctness gate when
      // it is only an addressing choice -- row-major B is now a legal
      // stride-`ld` gather (`rocm_fragment_strided_k.mlir`). The non-gmem space
      // keeps this file's actual job: role `b` still has source layouts the
      // materializer must refuse by name rather than mis-lower.
      %b_tile = tile.view %b_mem, %zero, %zero {
        tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
        tile.memory = #tile.memory_layout<space = "lds", order = "col_major", leading_dim = 16>
      } : (memref<256xf16>, index, index) -> !tile.tile
      %a = tile.fragment_pack %a_tile {
        role = "a",
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!tile.tile) -> !frag_a
      %b = tile.fragment_pack %b_tile {
        role = "b",
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!tile.tile) -> !frag_b
      %c = tile.fragment_zero {
        role = "acc",
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : !frag_acc
      %d = tile.mma %a, %b, %c {
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!frag_a, !frag_b, !frag_acc) -> !frag_acc
      %out = tile.fragment_unpack %d {
        tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
        mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
      } : (!frag_acc) -> !tile.tile
      "tile.store"(%out, %d_mem, %zero, %zero) {
        tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
        tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 16>
      } : (!tile.tile, memref<256xf32>, index, index) -> ()
      gpu.return
    }
  }
}

// CHECK: error: ROCM_FRAGMENT_UNSUPPORTED_SOURCE_LAYOUT: unsupported rdna3_wmma fragment source layout for role b
