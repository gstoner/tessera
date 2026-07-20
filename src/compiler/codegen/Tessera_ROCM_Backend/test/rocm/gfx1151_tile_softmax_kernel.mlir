// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(rocm-wave-lds-pipeline,rocm-wave-lds-legality,lower-tile-to-rocm{arch=gfx1151})' %s | FileCheck %s --check-prefix=TARGET
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(rocm-wave-lds-pipeline,rocm-wave-lds-legality,lower-tile-to-rocm{arch=gfx1151},generate-rocm-softmax-kernel)' %s | FileCheck %s --check-prefix=GENERATED

module {
  llvm.func @tessera_tile_softmax_f32(
      %x: !llvm.ptr, %o: !llvm.ptr, %rows: i64, %columns: i64) {
    tile.softmax_kernel %x, %o, %rows, %columns {
      storage = "f32", accum = "f32", axis = -1 : i64,
      exp_mode = "accurate", ftz = false
    } : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }
}

// TARGET-NOT: tile.softmax_kernel
// TARGET: tessera_rocm.softmax
// TARGET-SAME: accum = "f32"
// TARGET-SAME: arch = "gfx1151"
// TARGET-SAME: exp_mode = "accurate"
// TARGET-SAME: ftz = false
// TARGET-SAME: name = "tessera_tile_softmax_f32"
// TARGET-SAME: source = "tile.softmax_kernel"

// GENERATED-NOT: tile.softmax_kernel
// GENERATED-NOT: tessera_rocm.softmax
// GENERATED: gpu.module @tessera_tile_softmax_f32_mod
// GENERATED: gpu.func @tessera_tile_softmax_f32
// GENERATED-SAME: memref<?xf32>
// GENERATED: gpu.block_id x
// GENERATED: arith.maximumf
// GENERATED: math.exp
// GENERATED: arith.divf
