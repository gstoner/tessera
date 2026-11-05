// RUN: not %tnv --lower-tile-to-nvidia=sm=120 %s 2>&1 | FileCheck %s

// E2E-REAL-6 SM120 MX (2026-11-04): the UE8M0 K32 scale contract is a
// semantic key (Decision #21a). A launch op that omits it, or states a
// different block, is refused rather than lowered as if it said K32.
module {
// CHECK: error: sm_120 FP6/MXFP4 matmul_kernel requires physical_contract = "nvidia_sm120_mx_blockscale_v1"
  llvm.func @mx_without_scale_contract(
      %a: !llvm.ptr, %b: !llvm.ptr, %scale_a: !llvm.ptr,
      %scale_b: !llvm.ptr, %d: !llvm.ptr,
      %m: i64, %n: i64, %k: i64) attributes {nvvm.kernel} {
    tile.matmul_kernel %a, %b, %scale_a, %scale_b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 32, a = "e2m3", b = "e2m3", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global",
      tessera.storage_packed = true,
      tessera.storage_container = "int8",
      tessera.storage_pack = #tile.packed_format<logical = "fp6_e2m3", container = "int8", logical_bits = 6, elements_per_container = 1, signedness = "format_defined", encoding = "e2m3", lane_order = "scalar_lsb">
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}
