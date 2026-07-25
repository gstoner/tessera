// RUN: not %tnv --lower-tessera-nvidia-to-nvvm %s 2>&1 | FileCheck %s

module {
  llvm.func @bad_packed_contract(
      %a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr, %o: !llvm.ptr, %n: i64)
      attributes {nvvm.kernel} {
    tessera_nvidia.cuda_math_kernel %a, %b, %c, %o, %n {
      kind = "vaddss4_s8x4", input_storage = "i32",
      output_storage = "i32", rounding = "none", saturation = false,
      lane_width = 8 : i64, signedness = "unsigned",
      predicate_form = "none"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }
}

// CHECK: error: 'tessera_nvidia.cuda_math_kernel' op packed CUDA Math signedness disagrees with kind
