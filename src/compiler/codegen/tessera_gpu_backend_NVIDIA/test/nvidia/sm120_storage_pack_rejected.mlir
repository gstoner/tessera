// RUN: not %tnv --lower-tile-to-nvidia=sm=120 %s 2>&1 | FileCheck %s

module {
  llvm.func @nvfp4_rejects_drifted_pack_descriptor(
      %a: !llvm.ptr, %b: !llvm.ptr, %scale_a: !llvm.ptr,
      %scale_b: !llvm.ptr, %d: !llvm.ptr,
      %m: i64, %n: i64, %k: i64) attributes {nvvm.kernel} {
    // CHECK: TILE_PACKED_FORMAT_INVALID
    tile.matmul_kernel %a, %b, %scale_a, %scale_b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 64, a = "nvfp4", b = "nvfp4", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global",
      tessera.storage_packed = true,
      tessera.storage_container = "int8",
      tessera.storage_pack = #tile.packed_format<logical = "nvfp4", container = "int8", logical_bits = 4, elements_per_container = 1, signedness = "format_defined", encoding = "nv_e2m1", lane_order = "scalar_lsb">
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}
