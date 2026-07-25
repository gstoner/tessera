// RUN: %tnv --tessera-lower-to-nvidia-sm120 %s | FileCheck %s

module {
  llvm.func @tessera_tile_matmul_int4(
      %a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
      %m: i64, %n: i64, %k: i64) attributes {nvvm.kernel} {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 32, a = "int4", b = "int4", acc = "int32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "i32">,
      warps = 1 : i64, staging = "global",
      tessera.storage_packed = true,
      tessera.storage_container = "int8",
      tessera.storage_pack = #tile.packed_format<logical = "int4", container = "int8", logical_bits = 4, elements_per_container = 2, signedness = "signed_twos_complement", encoding = "twos_complement", lane_order = "low_to_high">
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @tessera_tile_matmul_int4
// CHECK-DAG: nvvm.read.ptx.sreg.ctaid.x
// CHECK-DAG: nvvm.read.ptx.sreg.tid.x
// CHECK: llvm.load
// CHECK: arith.shrui
// CHECK: arith.select
// CHECK: arith.muli
// CHECK: llvm.store
// CHECK-NOT: tile.matmul_kernel
