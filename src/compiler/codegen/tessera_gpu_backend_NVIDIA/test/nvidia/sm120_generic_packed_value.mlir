// RUN: %tnv --lower-tile-to-nvidia=sm=120 %s | FileCheck %s

module {
  llvm.func @decode_nvfp4(
      %src: !llvm.ptr, %scale: !llvm.ptr, %dst: !llvm.ptr,
      %row: i64, %col: i64, %rows: i64, %cols: i64)
      attributes {nvvm.kernel} {
    %tile = tile.packed_load %src, %scale, %row, %col, %rows, %cols {
      tile.layout = #tile.layout<shard = [8, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 9>,
      tile.packed_view = #tile.packed_view<
        format = #tile.packed_format<logical = "nvfp4", container = "int8",
          logical_bits = 4, elements_per_container = 2,
          signedness = "format_defined", encoding = "nv_e2m1",
          lane_order = "low_to_high">,
        packing_axis = 1, strides = [9, 1], alignment = 1, offset = 1,
        scale = #tile.scale_layout<dtype = "ue4m3", block_size = 16, axis = 1,
          layout = "row_major", stride = 5, alignment = 1, offset = 2>>
    } : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> !tile.tile
    tile.store %tile, %dst, %row, %col {
      tile.layout = #tile.layout<shard = [8, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 17>
    } : !tile.tile, !llvm.ptr, i64, i64
    llvm.return
  }

  llvm.func @roundtrip_int4(
      %src: !llvm.ptr, %dst: !llvm.ptr,
      %row: i64, %col: i64, %rows: i64, %cols: i64)
      attributes {nvvm.kernel} {
    %tile = tile.packed_load %src, %row, %col, %rows, %cols {
      tile.layout = #tile.layout<shard = [8, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 11>,
      tile.packed_view = #tile.packed_view<
        format = #tile.packed_format<logical = "int4", container = "int8",
          logical_bits = 4, elements_per_container = 2,
          signedness = "signed_twos_complement",
          encoding = "twos_complement", lane_order = "low_to_high">,
        packing_axis = 1, strides = [11, 1], alignment = 1, offset = 1,
        scale = #tile.scale_layout<dtype = "none", block_size = 0, axis = -1,
          layout = "none", stride = 0, alignment = 1, offset = 0>>
    } : (!llvm.ptr, i64, i64, i64, i64) -> !tile.tile
    tile.packed_store %tile, %dst, %row, %col, %rows, %cols {
      tile.layout = #tile.layout<shard = [8, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 11>,
      tile.packed_view = #tile.packed_view<
        format = #tile.packed_format<logical = "int4", container = "int8",
          logical_bits = 4, elements_per_container = 2,
          signedness = "signed_twos_complement",
          encoding = "twos_complement", lane_order = "low_to_high">,
        packing_axis = 1, strides = [11, 1], alignment = 1, offset = 1,
        scale = #tile.scale_layout<dtype = "none", block_size = 0, axis = -1,
          layout = "none", stride = 0, alignment = 1, offset = 0>>
    } : !tile.tile, !llvm.ptr, i64, i64, i64, i64
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @decode_nvfp4
// CHECK: nvvm.read.ptx.sreg.tid.x
// CHECK: llvm.load
// CHECK: math.exp2
// CHECK: llvm.store
// CHECK-NOT: tile.packed_load

// CHECK-LABEL: llvm.func @roundtrip_int4
// CHECK: nvvm.read.ptx.sreg.tid.x
// CHECK: llvm.load
// CHECK: llvm.store
// CHECK-NOT: tile.packed
