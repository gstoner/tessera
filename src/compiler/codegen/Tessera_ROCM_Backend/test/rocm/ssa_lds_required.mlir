// RUN: not %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151})' %s 2>&1 | FileCheck %s

module {
  func.func @reject_structured_lds_without_ssa_ownership(
      %dst: memref<64xf16>, %src: memref<64xf16>, %bytes: index) {
    %copy = "tile.async_copy"(%dst, %src, %bytes) {
      tile.layout = #tile.layout<
        shard = [64] : [1] on ["lds"],
        replica = [] : [] on [], offset = 0>
    } : (memref<64xf16>, memref<64xf16>, index) -> memref<64xf16>
    return
  }
}

// CHECK: structured ROCm LDS copies require !tile.buffer allocation identity,
// CHECK-SAME: !tile.async_token result
// CHECK-SAME: !tile.pipeline_state operand
