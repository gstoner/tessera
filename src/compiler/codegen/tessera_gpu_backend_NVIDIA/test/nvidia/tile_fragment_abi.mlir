// RUN: %tnv %s | FileCheck %s
//
// The portable Tile fragment ABI owns logical layout and descriptor agreement;
// it intentionally does not expose NVIDIA vector fragment or AMD VGPR shapes.

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
    } : (!tile.tile) -> !tile.fragment
    %b = tile.fragment_pack %b_tile {
      role = "b",
      mma = #tile.mma_desc<family = "auto", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f16", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.tile) -> !tile.fragment
    %c = tile.fragment_pack %c_tile {
      role = "acc",
      mma = #tile.mma_desc<family = "auto", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f16", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.tile) -> !tile.fragment
    %d = tile.mma %a, %b, %c {
      mma = #tile.mma_desc<family = "auto", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f16", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.fragment, !tile.fragment, !tile.fragment) -> !tile.fragment
    %out = tile.fragment_unpack %d {
      tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      mma = #tile.mma_desc<family = "auto", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f16", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    } : (!tile.fragment) -> !tile.tile
    return
  }
}

// CHECK-LABEL: func.func @portable_fragment_abi
// CHECK: !tile.tile
// CHECK: !tile.fragment
// CHECK: #tile.mma_desc<family = "auto", m = 16, n = 8, k = 16
