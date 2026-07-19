// RUN: %tnv --tessera-lower-to-nvidia-sm120 %s | FileCheck %s

module {
  llvm.func @tessera_tile_softmax_f32(
      %x: !llvm.ptr, %o: !llvm.ptr, %rows: i64, %columns: i64)
      attributes {nvvm.kernel} {
    tile.softmax_kernel %x, %o, %rows, %columns {
      storage = "f32", accum = "f32", axis = -1 : i64,
      exp_mode = "approx_exp2", ftz = false
    } : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }

  llvm.func @tessera_tile_softmax_f16(
      %x: !llvm.ptr, %o: !llvm.ptr, %rows: i64, %columns: i64)
      attributes {nvvm.kernel} {
    tile.softmax_kernel %x, %o, %rows, %columns {
      storage = "f16", accum = "f32", axis = -1 : i64,
      exp_mode = "approx_exp2", ftz = false
    } : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }
}

// CHECK: llvm.func @tessera_tile_softmax_f32
// CHECK: nvvm.read.ptx.sreg.ctaid.x
// CHECK: nvvm.read.ptx.sreg.tid.x
// CHECK: nvvm.ex2
// CHECK: llvm.func @tessera_tile_softmax_f16
// CHECK: llvm.fpext
// CHECK: nvvm.ex2
// CHECK: llvm.fptrunc
// CHECK-NOT: tile.softmax_kernel
