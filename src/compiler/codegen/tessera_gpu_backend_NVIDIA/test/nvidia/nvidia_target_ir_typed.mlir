// RUN: %tnv %s | FileCheck %s
//
// The tessera_nvidia Target IR ops are REGISTERED (Decision #19): this fixture
// parses and round-trips them WITHOUT --allow-unregistered-dialect. If the
// dialect were still `isExtensible` with no typed ops, the parse would fail.
// Covers the operand/result shapes the lowering emits: a matmul contract
// (operands + result), a TMEM alloc (no operands, no results), an mbarrier
// (operand, no result), and a TMA async-copy (operands + token result).

module {
  func.func @typed_contracts(%a: f32, %b: f32,
                             %dst: !llvm.ptr, %src: !llvm.ptr, %bytes: i64,
                             %tok: !llvm.ptr) {
    // CHECK: tessera_nvidia.mma_sync
    // CHECK-SAME: arch = "sm_120"
    // CHECK-SAME: shape = "m16n8k16"
    %0 = tessera_nvidia.mma_sync %a, %b
        {arch = "sm_120", shape = "m16n8k16", dtype_ab = "bf16",
         dtype_c = "f32", block_scaled = false} : (f32, f32) -> f32

    // CHECK: tessera_nvidia.tmem_alloc
    // CHECK-SAME: columns = 128
    tessera_nvidia.tmem_alloc {arch = "sm_100a", columns = 128 : i64} : () -> ()

    // CHECK: tessera_nvidia.tma_async_copy
    // CHECK-SAME: dst_space = "shared"
    // CHECK-SAME: src_space = "global"
    %1 = tessera_nvidia.tma_async_copy %dst, %src, %bytes
        {arch = "sm_90a", src_space = "global", dst_space = "shared",
         bytes = 16 : i64} : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr

    // CHECK: tessera_nvidia.mbarrier
    tessera_nvidia.mbarrier %tok {arch = "sm_90a"} : (!llvm.ptr) -> ()
    return
  }
}
