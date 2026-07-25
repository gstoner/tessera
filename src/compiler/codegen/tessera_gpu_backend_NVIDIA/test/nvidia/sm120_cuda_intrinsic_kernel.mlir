// RUN: %tnv --tessera-lower-to-nvidia-sm120 %s | FileCheck %s

module {
  llvm.func @tessera_cuda_math(
      %a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr, %o: !llvm.ptr, %n: i64)
      attributes {nvvm.kernel} {
    tile.cuda_intrinsic_kernel %a, %b, %c, %o, %n {
      kind = "dp4a_s32", input_storage = "i32", output_storage = "i32",
      rounding = "none", saturation = false
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }

  llvm.func @tessera_cuda_cast(
      %a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr, %o: !llvm.ptr, %n: i64)
      attributes {nvvm.kernel} {
    tile.cuda_intrinsic_kernel %a, %b, %c, %o, %n {
      kind = "cvt_f32_i32_rn", input_storage = "f32", output_storage = "i32",
      rounding = "rn", saturation = false
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }

  llvm.func @tessera_cuda_simd(
      %a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr, %o: !llvm.ptr, %n: i64)
      attributes {nvvm.kernel} {
    tile.cuda_intrinsic_kernel %a, %b, %c, %o, %n {
      kind = "vadd2_u16x2", input_storage = "i32", output_storage = "i32",
      rounding = "none", saturation = false
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @tessera_cuda_math
// CHECK: llvm.inline_asm "dp4a.s32.s32 $0, $1, $2, $3;"
// CHECK-LABEL: llvm.func @tessera_cuda_cast
// CHECK: llvm.inline_asm "cvt.rni.s32.f32 $0, $1;"
// CHECK-LABEL: llvm.func @tessera_cuda_simd
// CHECK: llvm.inline_asm "add.u16x2 $0, $1, $2;"
// CHECK-NOT: tile.cuda_intrinsic_kernel
