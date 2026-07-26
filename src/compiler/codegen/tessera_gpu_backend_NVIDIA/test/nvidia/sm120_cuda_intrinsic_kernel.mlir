// RUN: %tnv --tessera-lower-to-nvidia-sm120 %s | FileCheck %s
// RUN: %tnv --lower-tile-to-nvidia=sm=120 %s | FileCheck %s --check-prefix=TARGET

module {
  llvm.func @tessera_cuda_math(
      %a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr, %o: !llvm.ptr, %n: i64)
      attributes {nvvm.kernel} {
    tile.cuda_intrinsic_kernel %a, %b, %c, %o, %n {
      kind = "dp4a_s32", input_storage = "i32", output_storage = "i32",
      rounding = "none", saturation = false, lane_width = 0 : i64,
      signedness = "scalar", predicate_form = "none"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }

  llvm.func @tessera_cuda_cast(
      %a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr, %o: !llvm.ptr, %n: i64)
      attributes {nvvm.kernel} {
    tile.cuda_intrinsic_kernel %a, %b, %c, %o, %n {
      kind = "cvt_f32_i32_rn", input_storage = "f32", output_storage = "i32",
      rounding = "rn", saturation = false, lane_width = 0 : i64,
      signedness = "scalar", predicate_form = "none"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }

  llvm.func @tessera_cuda_simd(
      %a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr, %o: !llvm.ptr, %n: i64)
      attributes {nvvm.kernel} {
    tile.cuda_intrinsic_kernel %a, %b, %c, %o, %n {
      kind = "vadd2_u16x2", input_storage = "i32", output_storage = "i32",
      rounding = "none", saturation = false, lane_width = 16 : i64,
      signedness = "unsigned", predicate_form = "none"
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

// TARGET-LABEL: llvm.func @tessera_cuda_math
// TARGET: tessera_nvidia.cuda_math_kernel
// TARGET-SAME: kind = "dp4a_s32"
// TARGET-SAME: lane_width = 0
// TARGET-SAME: predicate_form = "none"
// TARGET-NOT: tile.cuda_intrinsic_kernel
