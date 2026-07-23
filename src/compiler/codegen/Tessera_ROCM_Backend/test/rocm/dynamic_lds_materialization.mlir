// RUN: %trop %s --pass-pipeline='builtin.module(gpu.module(rocm-materialize-dynamic-lds))' | FileCheck %s

module {
  gpu.module @kernels {
    // CHECK: llvm.mlir.global external @__tessera_dynamic_lds() {addr_space = 3
    // CHECK-LABEL: llvm.func @dynamic_lds
    // CHECK-SAME: tessera.rocm.dynamic_lds_launch_bytes
    // CHECK: llvm.mlir.addressof @__tessera_dynamic_lds : !llvm.ptr<3>
    // CHECK-NOT: llvm.alloca
    llvm.func @dynamic_lds(%n: i64) attributes {gpu.kernel, rocdl.kernel} {
      %arena = llvm.alloca %n x i8 : (i64) -> !llvm.ptr<3>
      llvm.return
    }
  }
}
