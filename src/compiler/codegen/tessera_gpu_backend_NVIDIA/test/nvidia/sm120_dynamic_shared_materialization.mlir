// RUN: %tnv --pass-pipeline='builtin.module(gpu.module(nvidia-materialize-dynamic-shared))' %s | FileCheck %s

module {
  gpu.module @kernels {
    // Mutually exclusive branch arenas share slot zero.  The launch evaluates
    // max(align16(then_bytes), align16(else_bytes)), not their sum.
    //
    // CHECK: llvm.mlir.global external @__tessera_dynamic_smem() {addr_space = 3
    // CHECK-LABEL: llvm.func @path_max_dynamic_shared
    // CHECK-SAME: tessera.nvidia.dynamic_smem_launch_bytes
    // CHECK-SAME: tessera.nvidia.dynamic_smem_launch_reduction = "aligned_sum_of_slot_maxima"
    // CHECK-SAME: tessera.nvidia.dynamic_smem_slots
    // CHECK: llvm.mlir.addressof @__tessera_dynamic_smem
    // CHECK: llvm.getelementptr
    // CHECK: llvm.mlir.addressof @__tessera_dynamic_smem
    // CHECK: llvm.getelementptr
    // CHECK-NOT: llvm.alloca
    llvm.func @path_max_dynamic_shared(
        %cond: i1, %then_bytes: i64, %else_bytes: i64)
        attributes {gpu.kernel, nvvm.kernel} {
      llvm.cond_br %cond, ^then, ^else
    ^then:
      %then_arena = llvm.alloca %then_bytes x i8 : (i64) -> !llvm.ptr<3>
      llvm.store %cond, %then_arena : i1, !llvm.ptr<3>
      llvm.br ^exit
    ^else:
      %else_arena = llvm.alloca %else_bytes x i8 : (i64) -> !llvm.ptr<3>
      llvm.store %cond, %else_arena : i1, !llvm.ptr<3>
      llvm.br ^exit
    ^exit:
      llvm.return
    }

    // Simultaneously live arenas occupy distinct aligned slots.
    //
    // CHECK-LABEL: llvm.func @packed_dynamic_shared
    // CHECK-SAME: tessera.nvidia.dynamic_smem_launch_reduction = "aligned_sum_of_slot_maxima"
    // CHECK: llvm.and
    // CHECK: llvm.add
    // CHECK: llvm.getelementptr
    // CHECK-NOT: llvm.alloca
    llvm.func @packed_dynamic_shared(
        %lhs_bytes: i64, %rhs_bytes: i64, %value: i8)
        attributes {gpu.kernel, nvvm.kernel} {
      %lhs = llvm.alloca %lhs_bytes x i8 : (i64) -> !llvm.ptr<3>
      %rhs = llvm.alloca %rhs_bytes x i8 : (i64) -> !llvm.ptr<3>
      llvm.store %value, %lhs : i8, !llvm.ptr<3>
      llvm.store %value, %rhs : i8, !llvm.ptr<3>
      %left = llvm.load %lhs : !llvm.ptr<3> -> i8
      %right = llvm.load %rhs : !llvm.ptr<3> -> i8
      %sum = llvm.add %left, %right : i8
      llvm.return
    }

    // CFG block arguments recursively retain their kernel-argument leaves.
    //
    // CHECK-LABEL: llvm.func @cfg_forwarded_dynamic_shared
    // CHECK-SAME: kind = "path_max"
    // CHECK-SAME: kind = "argument"
    // CHECK-NOT: llvm.alloca
    llvm.func @cfg_forwarded_dynamic_shared(
        %cond: i1, %bytes: i64, %fallback: i64)
        attributes {gpu.kernel, nvvm.kernel} {
      llvm.cond_br %cond, ^selected(%bytes : i64), ^selected(%fallback : i64)
    ^selected(%size: i64):
      %arena = llvm.alloca %size x i8 : (i64) -> !llvm.ptr<3>
      llvm.store %cond, %arena : i1, !llvm.ptr<3>
      llvm.return
    }

    // Locally computed non-negative add/multiply expressions are serialized
    // and reconstructed from host-visible kernel arguments.
    //
    // CHECK-LABEL: llvm.func @local_expression_dynamic_shared
    // CHECK-SAME: kind = "add"
    // CHECK-SAME: kind = "multiply"
    // CHECK-SAME: kind = "constant"
    // CHECK-NOT: llvm.alloca
    llvm.func @local_expression_dynamic_shared(
        %elements: i64, %element_bytes: i64, %value: i8)
        attributes {gpu.kernel, nvvm.kernel} {
      %bias = llvm.mlir.constant(17 : i64) : i64
      %scaled = llvm.mul %elements, %element_bytes : i64
      %size = llvm.add %scaled, %bias : i64
      %arena = llvm.alloca %size x i8 : (i64) -> !llvm.ptr<3>
      llvm.store %value, %arena : i8, !llvm.ptr<3>
      llvm.return
    }
  }
}
