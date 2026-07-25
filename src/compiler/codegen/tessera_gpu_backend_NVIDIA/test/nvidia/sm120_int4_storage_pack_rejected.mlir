// RUN: not %tnv --tessera-lower-to-nvidia-sm120 %s 2>&1 | FileCheck %s

module {
  llvm.func @bad_int4_signedness(
      %a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
      %m: i64, %n: i64, %k: i64) attributes {nvvm.kernel} {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 32, a = "int4", b = "int4", acc = "int32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "i32">,
      warps = 1 : i64, staging = "global",
      tessera.storage_packed = true,
      tessera.storage_container = "int8",
      tessera.storage_pack = {logical = "int4", container = "int8", factor = 2 : i64, signedness = "unsigned"}
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}

// CHECK: error: NVIDIA packed matmul storage descriptor disagrees with the selected physical fragment ABI
