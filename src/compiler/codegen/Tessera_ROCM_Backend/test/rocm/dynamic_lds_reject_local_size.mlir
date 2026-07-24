// RUN: not %trop %s --pass-pipeline='builtin.module(gpu.module(rocm-materialize-dynamic-lds))' 2>&1 | FileCheck %s

module {
  gpu.module @kernels {
    llvm.func @local_size(%n: i64)
        attributes {gpu.kernel, rocdl.kernel} {
      %one = llvm.mlir.constant(1 : i64) : i64
      %local_size = llvm.add %n, %one : i64
      %arena = llvm.alloca %local_size x i8 : (i64) -> !llvm.ptr<3>
      llvm.return
    }
  }
}

// CHECK: ROCM_DYNAMIC_LDS_SIZE_NOT_KERNEL_ARGUMENT:
