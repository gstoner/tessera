// RUN: not %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151})' %s 2>&1 | FileCheck %s

module {
  func.func @packed_marker_without_consumer(%a: f16, %b: f16) -> f16 {
    %m = "tile.mma"(%a, %b) {
      numeric_policy = {storage = "int4", accum = "int32"},
      tessera.storage_packed = true,
      tessera.storage_container = "int8"
    } : (f16, f16) -> f16
    return %m : f16
  }
}

// CHECK: ROCM_LOWERING_UNCONSUMED_STORAGE_PACK
