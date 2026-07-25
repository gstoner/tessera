// RUN: not %tnv --tessera-lower-to-nvidia-sm120 %s 2>&1 | FileCheck %s

module {
  llvm.func @bad_rounding(
      %a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr, %o: !llvm.ptr, %n: i64)
      attributes {nvvm.kernel} {
    tile.cuda_intrinsic_kernel %a, %b, %c, %o, %n {
      kind = "cvt_f32_i32_rn", input_storage = "f32", output_storage = "i32",
      rounding = "rz", saturation = false, lane_width = 0 : i64,
      signedness = "scalar", predicate_form = "none"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }
}

// CHECK: error: 'tile.cuda_intrinsic_kernel' op numeric casts require f32->i32 and matching RN/RD/RU/RZ
