// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151})' %s | FileCheck %s --check-prefix=TARGET
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151},generate-rocm-coalition-butterfly-kernel)' %s | FileCheck %s --check-prefix=GENERATED
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(tessera-rocm-executable{family=coalition_butterfly input=tile output=binary arch=gfx1151})' %s | FileCheck %s --check-prefix=BINARY

module {
  llvm.func @subset_mobius(%input: !llvm.ptr, %output: !llvm.ptr) {
    %batch = llvm.mlir.constant(3 : i64) : i64
    %size = llvm.mlir.constant(32 : i64) : i64
    tile.coalition_butterfly_kernel %input, %output, %batch, %size {
      accum = "f64", arch = "gfx1151", half = 1 : i64, players = 5 : i64,
      sign = -1 : i64, stage_order = "ascending_bit_yates_v1", storage = "f32",
      tessera.schedule_hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      workgroup_size = 32 : i64
    } : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }
}

// TARGET-NOT: tile.coalition_butterfly_kernel
// TARGET: tessera_rocm.coalition_butterfly
// TARGET-SAME: accum = "f64"
// TARGET-SAME: arch = "gfx1151"
// TARGET-SAME: half = 1 : i64
// TARGET-SAME: sign = -1 : i64
// TARGET-SAME: stage_order = "ascending_bit_yates_v1"

// GENERATED-NOT: tile.coalition_butterfly_kernel
// GENERATED-NOT: tessera_rocm.coalition_butterfly
// GENERATED: gpu.module @subset_mobius_mod
// GENERATED: gpu.func @subset_mobius
// GENERATED: memref<32xf64, #gpu.address_space<workgroup>>
// GENERATED: scf.for
// GENERATED: arith.subf
// GENERATED: gpu.barrier
// GENERATED: arith.truncf

// BINARY: tessera.pipeline.family = "coalition_butterfly"
// BINARY: gpu.binary @subset_mobius_mod
// Thirty-two fp64 lattice values occupy 256 LDS bytes.  This is independent
// of the 32-thread launch size.
// BINARY-SAME: group_segment_fixed_size = 256 : i64
