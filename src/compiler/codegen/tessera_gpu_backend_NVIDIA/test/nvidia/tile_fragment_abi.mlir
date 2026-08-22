// RUN: %tnv %s | FileCheck %s
//
// The portable Tile fragment ABI owns logical layout and descriptor agreement;
// it intentionally does not expose NVIDIA vector fragment or AMD VGPR shapes.

!fa = !tile.fragment<m = 16, n = 8, k = 16, elem = "f16", acc = "f16", role = "a", layout = "row_major", family = "auto">
!fb = !tile.fragment<m = 16, n = 8, k = 16, elem = "f16", acc = "f16", role = "b", layout = "col_major", family = "auto">
!fc = !tile.fragment<m = 16, n = 8, k = 16, elem = "f16", acc = "f16", role = "acc", layout = "row_major", family = "auto">

module {
  func.func @portable_fragment_abi(%source: tensor<16x16xf16>) {
    %a_tile = tile.view %source {
      tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>
    } : (tensor<16x16xf16>) -> !tile.tile
    %b_tile = tile.view %source {
      tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>
    } : (tensor<16x16xf16>) -> !tile.tile
    %c_tile = tile.view %source {
      tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>
    } : (tensor<16x16xf16>) -> !tile.tile
    %a = tile.fragment_pack %a_tile {
      role = "a",
      mma = #tile.mma_desc<family = "auto", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f16", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.tile) -> !fa
    %b = tile.fragment_pack %b_tile {
      role = "b",
      mma = #tile.mma_desc<family = "auto", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f16", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.tile) -> !fb
    %c = tile.fragment_pack %c_tile {
      role = "acc",
      mma = #tile.mma_desc<family = "auto", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f16", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.tile) -> !fc
    %d = tile.mma %a, %b, %c {
      mma = #tile.mma_desc<family = "auto", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f16", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!fa, !fb, !fc) -> !fc
    %out = tile.fragment_unpack %d {
      tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      mma = #tile.mma_desc<family = "auto", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f16", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!fc) -> !tile.tile
    return
  }
}

// CHECK-LABEL: func.func @portable_fragment_abi
// CHECK: !tile.tile
// CHECK: !tile.fragment
// CHECK: #tile.mma_desc<family = "auto", m = 16, n = 8, k = 16
