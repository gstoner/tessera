// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151})' %s | FileCheck %s --check-prefix=TARGET
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151},generate-rocm-tridiagonal-kernel)' %s | FileCheck %s --check-prefix=GENERATED
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(tessera-rocm-executable{family=tridiagonal_solve input=tile output=binary arch=gfx1151})' %s | FileCheck %s --check-prefix=BINARY

module {
  llvm.func @tridiagonal_pcr(%dl: !llvm.ptr, %d: !llvm.ptr, %du: !llvm.ptr,
      %rhs: !llvm.ptr, %output: !llvm.ptr) {
    %batch = llvm.mlir.constant(5 : i64) : i64
    %n = llvm.mlir.constant(32 : i64) : i64
    tile.tridiagonal_solve_kernel %dl, %d, %du, %rhs, %output, %batch, %n {
      accum = "f64", algorithm = "cooperative_lds_pcr_v1", arch = "gfx1151",
      storage = "f32",
      tessera.schedule_hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      workgroup_size = 32 : i64
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }
}

// TARGET-NOT: tile.tridiagonal_solve_kernel
// TARGET: tessera_rocm.tridiagonal_solve
// TARGET-SAME: accum = "f64"
// TARGET-SAME: algorithm = "cooperative_lds_pcr_v1"
// TARGET-SAME: arch = "gfx1151"
// TARGET-SAME: status_abi = "per_equation_i32_v1"
// TARGET-SAME: system_size = 32 : i64

// GENERATED-NOT: tile.tridiagonal_solve_kernel
// GENERATED-NOT: tessera_rocm.tridiagonal_solve
// GENERATED: gpu.module @tridiagonal_pcr_mod
// GENERATED: gpu.func @tridiagonal_pcr
// GENERATED-SAME: memref<?xf32>
// GENERATED-SAME: memref<?xi32>
// GENERATED: memref<32xf64, #gpu.address_space<workgroup>>
// GENERATED: gpu.barrier
// GENERATED: arith.cmpf
// GENERATED: arith.divf
// GENERATED: memref.store {{.*}} : memref<?xi32>

// BINARY: tessera.pipeline.family = "tridiagonal_solve"
// BINARY: gpu.binary @tridiagonal_pcr_mod
// BINARY-SAME: group_segment_fixed_size = 2048 : i64
