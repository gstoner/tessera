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

    // CHECK-LABEL: llvm.func @packed_dynamic_lds
    // CHECK-SAME: tessera.rocm.dynamic_lds_launch_bytes
    // CHECK-SAME: tessera.rocm.dynamic_lds_packed_arenas
    // CHECK-DAG: llvm.mlir.addressof @__tessera_dynamic_lds
    // CHECK-DAG: llvm.mlir.addressof @__tessera_dynamic_lds
    // CHECK-DAG: llvm.getelementptr
    // CHECK-DAG: llvm.getelementptr
    // CHECK-DAG: llvm.and
    // CHECK-DAG: llvm.and
    // CHECK-NOT: llvm.alloca
    llvm.func @packed_dynamic_lds(%lhs_bytes: i64, %rhs_bytes: i64)
        attributes {gpu.kernel, rocdl.kernel} {
      %lhs = llvm.alloca %lhs_bytes x i8 : (i64) -> !llvm.ptr<3>
      %rhs = llvm.alloca %rhs_bytes x i8 : (i64) -> !llvm.ptr<3>
      llvm.return
    }

    // Mutually exclusive pointer lifetimes reuse one interference slot.
    // CHECK-LABEL: llvm.func @path_max_dynamic_lds
    // CHECK-SAME: tessera.rocm.dynamic_lds_launch_reduction = "aligned_sum_of_slot_maxima"
    // CHECK-SAME: tessera.rocm.dynamic_lds_slots
    // CHECK-COUNT-2: llvm.getelementptr
    // CHECK-NOT: llvm.alloca
    llvm.func @path_max_dynamic_lds(
        %cond: i1, %then_bytes: i64, %else_bytes: i64)
        attributes {gpu.kernel, rocdl.kernel} {
      llvm.cond_br %cond, ^then, ^else
    ^then:
      %then_arena = llvm.alloca %then_bytes x i8 : (i64) -> !llvm.ptr<3>
      llvm.br ^exit
    ^else:
      %else_arena = llvm.alloca %else_bytes x i8 : (i64) -> !llvm.ptr<3>
      llvm.br ^exit
    ^exit:
      llvm.return
    }
  }
}
