// RUN: tessera-opt -split-input-file -verify-diagnostics %s | FileCheck %s
//
// Structured C4 physical-view contract. Format semantics are independent of
// per-buffer packing axes and block-scale addressing.

// CHECK-LABEL: llvm.func @int4_roundtrip
llvm.func @int4_roundtrip(%src: !llvm.ptr, %dst: !llvm.ptr,
                          %row: i64, %col: i64, %rows: i64, %cols: i64) {
  // CHECK: #tile.packed_view<format = <logical = "int4", container = "int8", logical_bits = 4, elements_per_container = 2
  // CHECK-SAME: scale = <dtype = "none", block_size = 0, axis = -1, layout = "none", stride = 0, alignment = 1, offset = 0>>
  %tile = tile.packed_load %src, %row, %col, %rows, %cols {
    tile.layout = #tile.layout<shard = [8, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
    tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 4>,
    tile.packed_view = #tile.packed_view<
      format = #tile.packed_format<logical = "int4", container = "int8",
        logical_bits = 4, elements_per_container = 2,
        signedness = "signed_twos_complement", encoding = "twos_complement",
        lane_order = "low_to_high">,
      packing_axis = 1, strides = [4, 1], alignment = 1, offset = 0,
      scale = #tile.scale_layout<dtype = "none", block_size = 0, axis = -1,
        layout = "none", stride = 0, alignment = 1, offset = 0>>
  } : (!llvm.ptr, i64, i64, i64, i64) -> !tile.tile
  tile.packed_store %tile, %dst, %row, %col, %rows, %cols {
    tile.layout = #tile.layout<shard = [8, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
    tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 4>,
    tile.packed_view = #tile.packed_view<
      format = #tile.packed_format<logical = "int4", container = "int8",
        logical_bits = 4, elements_per_container = 2,
        signedness = "signed_twos_complement", encoding = "twos_complement",
        lane_order = "low_to_high">,
      packing_axis = 1, strides = [4, 1], alignment = 1, offset = 0,
      scale = #tile.scale_layout<dtype = "none", block_size = 0, axis = -1,
        layout = "none", stride = 0, alignment = 1, offset = 0>>
  } : !tile.tile, !llvm.ptr, i64, i64, i64, i64
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @scaled_fp6
llvm.func @scaled_fp6(%src: !llvm.ptr, %scale: !llvm.ptr,
                      %row: i64, %col: i64, %rows: i64, %cols: i64) {
  // FP6 remains logical_bits=6 even though one logical value occupies each i8.
  // CHECK: logical_bits = 6, elements_per_container = 1
  %tile = tile.packed_load %src, %scale, %row, %col, %rows, %cols {
    tile.layout = #tile.layout<shard = [8, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
    tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 8>,
    tile.packed_view = #tile.packed_view<
      format = #tile.packed_format<logical = "fp6_e2m3", container = "int8",
        logical_bits = 6, elements_per_container = 1,
        signedness = "format_defined", encoding = "e2m3",
        lane_order = "scalar_lsb">,
      packing_axis = 1, strides = [8, 1], alignment = 1, offset = 0,
      scale = #tile.scale_layout<dtype = "ue8m0", block_size = 32, axis = 1,
        layout = "row_major", stride = 1, alignment = 1, offset = 0>>
  } : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> !tile.tile
  llvm.return
}

// -----

llvm.func @fp6_factor_is_not_byte_width() {
  // expected-error @+1 {{TILE_PACKED_FORMAT_INVALID: logical_bits and elements_per_container do not fit the physical container}}
  "test.bad"() {p = #tile.packed_format<logical = "fp6_e2m3", container = "int8",
    logical_bits = 6, elements_per_container = 2, signedness = "format_defined",
    encoding = "e2m3", lane_order = "low_to_high">} : () -> ()
  llvm.return
}

// -----

llvm.func @scale_axis_mismatch(%src: !llvm.ptr, %scale: !llvm.ptr,
                               %row: i64, %col: i64, %rows: i64, %cols: i64) {
  %tile = tile.packed_load %src, %scale, %row, %col, %rows, %cols {
    tile.layout = #tile.layout<shard = [8, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
    tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 8>,
    // expected-error @+1 {{TILE_PACKED_VIEW_INVALID: block scales must index the packed logical axis}}
    tile.packed_view = #tile.packed_view<
      format = #tile.packed_format<logical = "nvfp4", container = "int8",
        logical_bits = 4, elements_per_container = 2,
        signedness = "format_defined", encoding = "nv_e2m1",
        lane_order = "low_to_high">,
      packing_axis = 1, strides = [4, 1], alignment = 1, offset = 0,
      scale = #tile.scale_layout<dtype = "ue4m3", block_size = 16, axis = 0,
        layout = "row_major", stride = 1, alignment = 1, offset = 0>>
  } : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> !tile.tile
  llvm.return
}
