// This file is fixture DATA for tests/unit/test_rocm_wmma_gemm_generated.py,
// which drives it itself. It is not a lit test: without this marker lit
// discovers it, reports Unresolved ("Test has no 'RUN:' line"), and fails
// `check-tessera-rocm` for the whole repository — the exact twin of the x86
// case fixed in PR #626, and equally invisible to CI, which does not run this
// suite at all (lane removed 2026-08-19).
// UNSUPPORTED: true
// The shared producer receives per-lane coordinates.  `tile.linear_base` is a
// scalar SSA value per GPU lane, so it must include the lane-resolved row/col
// before ROCm's fragment pack issues its contiguous-K vector load.
!frag_a = !tile.fragment<m = 16, n = 16, k = 16, elem = "f16", acc = "f32", role = "a", layout = "row_major", family = "auto">
!frag_b = !tile.fragment<m = 16, n = 16, k = 16, elem = "f16", acc = "f32", role = "b", layout = "col_major", family = "auto">
!frag_acc = !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">

module {
  gpu.module @composed_layout_fragment_mod {
    gpu.func @composed_layout_fragment_store(%a_mem: memref<512xf16>, %b_mem: memref<512xf16>, %d_mem: memref<256xf32>) kernel {
      %zero = arith.constant 0 : index
      %a_row = arith.constant 7 : index
      %b_col = arith.constant 5 : index
      %thread = gpu.thread_id x
      %sixteen = arith.constant 16 : index
      %lane = arith.remui %thread, %sixteen : index
      %a_lane = arith.addi %a_row, %lane : index
      %b_lane = arith.addi %b_col, %lane : index
      %a_lane64 = arith.index_cast %a_lane : index to i64
      %b_lane64 = arith.index_cast %b_lane : index to i64
      %zero64 = arith.constant 0 : i64
      %a_linear = "tile.materialize_composed_layout"(%a_lane64, %zero64) {layout = #tile.composed_layout<[16, 16], [16, 1], [[[16], [1]], [[16], [1]]], [0, 0]>} : (i64, i64) -> i64
      %b_linear = "tile.materialize_composed_layout"(%zero64, %b_lane64) {layout = #tile.composed_layout<[16, 16], [1, 16], [[[16], [1]], [[16], [1]]], [0, 0]>} : (i64, i64) -> i64
      %a_tile = tile.view %a_mem, %a_linear, %a_row, %zero {tile.linear_base, tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>, tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 16>} : (memref<512xf16>, i64, index, index) -> !tile.tile
      %b_tile = tile.view %b_mem, %b_linear, %zero, %b_col {tile.linear_base, tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>, tile.memory = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = 16>} : (memref<512xf16>, i64, index, index) -> !tile.tile
      %a = tile.fragment_pack %a_tile {role = "a", mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>} : (!tile.tile) -> !frag_a
      %b = tile.fragment_pack %b_tile {role = "b", mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>} : (!tile.tile) -> !frag_b
      %c = tile.fragment_zero {role = "acc", mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>} : !frag_acc
      %d = tile.mma %a, %b, %c {mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>} : (!frag_a, !frag_b, !frag_acc) -> !frag_acc
      %out = tile.fragment_unpack %d {tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>, mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>} : (!frag_acc) -> !tile.tile
      "tile.store"(%out, %d_mem, %zero, %zero) {tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>, tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 16>} : (!tile.tile, memref<256xf32>, index, index) -> ()
      gpu.return
    }
  }
}
