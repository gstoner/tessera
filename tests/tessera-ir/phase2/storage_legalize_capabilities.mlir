// RUN: tessera-opt --allow-unregistered-dialect --tessera-storage-legalize='target=nvidia_sm120' --tessera-storage-pack-consume %s | FileCheck %s
//
// Terminal packed-storage legalization is a target + operation + physical-view
// capability decision.  A low-precision numeric policy alone is not evidence
// that an executable packed consumer exists.

module {
  llvm.func @decode_nvfp4(
      %src: !llvm.ptr, %scale: !llvm.ptr, %dst: !llvm.ptr,
      %row: i64, %col: i64, %rows: i64, %cols: i64) {
    %tile = tile.packed_load %src, %scale, %row, %col, %rows, %cols {
      numeric_policy = {storage = "nvfp4", accum = "fp32"},
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
      numeric_policy = {storage = "nvfp4", accum = "fp32"},
      tile.layout = #tile.layout<shard = [8, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 17>
    } : !tile.tile, !llvm.ptr, i64, i64
    llvm.return
  }

  llvm.func @roundtrip_int4(
      %src: !llvm.ptr, %dst: !llvm.ptr,
      %row: i64, %col: i64, %rows: i64, %cols: i64) {
    %tile = tile.packed_load %src, %row, %col, %rows, %cols {
      numeric_policy = {storage = "int4", accum = "int32"},
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
      numeric_policy = {storage = "int4", accum = "int32"},
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

  llvm.func @packed_matmul(
      %a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
      %m: i64, %n: i64, %k: i64) {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      numeric_policy = {storage = "int4", accum = "int32"},
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 32, a = "int4", b = "int4", acc = "int32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "i32">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @matmul_descriptor_disagrees(
      %a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
      %m: i64, %n: i64, %k: i64) {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      numeric_policy = {storage = "fp4_e2m1", accum = "fp32"},
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @orphan_fp6(
      %src: !llvm.ptr, %scale: !llvm.ptr,
      %row: i64, %col: i64, %rows: i64, %cols: i64) {
    %unused = tile.packed_load %src, %scale, %row, %col, %rows, %cols {
      numeric_policy = {storage = "fp6_e2m3", accum = "fp32"},
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

  func.func @logical_only() {
    "test.dtype_request"() {
      numeric_policy = {storage = "fp4_e2m1", accum = "fp32"}
    } : () -> ()
    return
  }
}

// CHECK-LABEL: llvm.func @decode_nvfp4
// CHECK: tile.packed_load
// CHECK-SAME: tessera.storage_container = "int8"
// CHECK-SAME: tessera.storage_pack = #tile.packed_format<logical = "nvfp4"
// CHECK-SAME: tessera.storage_packed = true
// CHECK: tile.store
// CHECK-SAME: tessera.storage_container = "int8"
// CHECK-SAME: tessera.storage_packed = true

// CHECK-LABEL: llvm.func @roundtrip_int4
// CHECK: tile.packed_load
// CHECK-SAME: tessera.storage_container = "int8"
// CHECK-SAME: tessera.storage_packed = true
// CHECK: tile.packed_store
// CHECK-SAME: tessera.storage_container = "int8"
// CHECK-SAME: tessera.storage_packed = true

// CHECK-LABEL: llvm.func @packed_matmul
// CHECK: tile.matmul_kernel
// CHECK-SAME: tessera.storage_container = "int8"
// CHECK-SAME: tessera.storage_pack = #tile.packed_format<logical = "int4"
// CHECK-SAME: tessera.storage_packed = true

// CHECK-LABEL: llvm.func @matmul_descriptor_disagrees
// CHECK: tile.matmul_kernel
// CHECK-NOT: tessera.storage_packed

// CHECK-LABEL: llvm.func @orphan_fp6
// CHECK: tile.packed_load
// CHECK-NOT: tessera.storage_packed

// CHECK-LABEL: func.func @logical_only
// CHECK: "test.dtype_request"
// CHECK-NOT: tessera.storage_packed
