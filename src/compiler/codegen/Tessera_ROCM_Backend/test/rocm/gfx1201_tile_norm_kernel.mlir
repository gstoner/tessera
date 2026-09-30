// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1201})' %s | FileCheck %s --check-prefix=TARGET
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1201},generate-rocm-norm-kernel)' %s | FileCheck %s --check-prefix=GENERATED
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(tessera-rocm-executable{family=normalization input=tile output=target arch=gfx1201})' %s | FileCheck %s --check-prefix=PIPELINE

module {
  llvm.func @tessera_tile_norm_rmsnorm_f16_0123456789(
      %x: !llvm.ptr, %o: !llvm.ptr, %rows: i64, %columns: i64, %epsilon: f32) {
    tile.norm_kernel %x, %o, %rows, %columns, %epsilon {
      kind = "rmsnorm", storage = "f16", accum = "f32", axis = -1 : i64,
      affine = false, arch = "gfx1201",
      tessera.schedule_hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      tessera.workgroup_size = 256 : i64
    } : !llvm.ptr, !llvm.ptr, i64, i64, f32
    llvm.return
  }
}

// TARGET-NOT: tile.norm_kernel
// TARGET: tessera_rocm.norm
// TARGET-SAME: arch = "gfx1201"
// TARGET-SAME: dtype = "f16"
// TARGET-SAME: name = "tessera_tile_norm_rmsnorm_f16_0123456789"
// TARGET-SAME: scheduled_unary = true
// TARGET-SAME: source = "tile.norm_kernel"

// GENERATED-NOT: tile.norm_kernel
// GENERATED: gpu.module @tessera_tile_norm_rmsnorm_f16_0123456789_mod
// GENERATED: gpu.func @tessera_tile_norm_rmsnorm_f16_0123456789
// GENERATED-SAME: memref<?xf16>
// GENERATED: gpu.block_id x
// GENERATED: math.sqrt
// GENERATED: arith.divf

// PIPELINE: tessera.pipeline.output = "target"
// PIPELINE: tessera_rocm.norm
// PIPELINE-SAME: scheduled_unary = true
// PIPELINE-NOT: {{^ *tile\.norm_kernel}}
// PIPELINE-NOT: gpu.module
