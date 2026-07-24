// RUN: %trop %s --pass-pipeline='builtin.module(gpu.module(rocm-materialize-dynamic-lds))' | FileCheck %s

module {
  gpu.module @kernels {
    // ptrtoint makes the pointer's future alias lifetime unknowable. The
    // allocator conservatively keeps it live for the kernel and does not reuse
    // its slot for the later arena.
    // CHECK-LABEL: llvm.func @escaping_dynamic_lds
    // CHECK-SAME: tessera.rocm.dynamic_lds_slots
    // CHECK-SAME: slot = 0
    // CHECK-SAME: slot = 1
    llvm.func @escaping_dynamic_lds(%lhs: i64, %rhs: i64)
        attributes {gpu.kernel, rocdl.kernel} {
      %a = llvm.alloca %lhs x i8 : (i64) -> !llvm.ptr<3>
      %escaped_a = llvm.ptrtoint %a : !llvm.ptr<3> to i64
      %b = llvm.alloca %rhs x i8 : (i64) -> !llvm.ptr<3>
      %escaped_b = llvm.ptrtoint %b : !llvm.ptr<3> to i64
      llvm.return
    }
  }
}
