// RUN: not %trop %s --pass-pipeline='builtin.module(gpu.module(rocm-materialize-dynamic-lds))' 2>&1 | FileCheck %s

module {
  gpu.module @kernels {
    llvm.func @sequential_dynamic_lds(%lhs: i64, %rhs: i64)
        attributes {gpu.kernel, rocdl.kernel} {
      llvm.br ^later
    ^later:
      %a = llvm.alloca %lhs x i8 : (i64) -> !llvm.ptr<3>
      llvm.br ^final
    ^final:
      %b = llvm.alloca %rhs x i8 : (i64) -> !llvm.ptr<3>
      llvm.return
    }
  }
}

// CHECK: ROCM_DYNAMIC_LDS_MULTIPLE_ARENAS:
// CHECK-SAME: direct mutually-exclusive successors
