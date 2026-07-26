// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(rocm-wave-lds-pipeline)' %s | FileCheck %s

module {
  func.func @branch_local_pipeline(
      %condition: i1,
      %dst0: memref<64xf16>,
      %dst1: memref<64xf16>,
      %src: memref<64xf16>,
      %bytes: index) {
    scf.if %condition {
      %copy0 = "tile.async_copy"(%dst0, %src, %bytes)
          : (memref<64xf16>, memref<64xf16>, index) -> i32
      "tile.wait_async"() : () -> ()
    } else {
      %copy1 = "tile.async_copy"(%dst1, %src, %bytes)
          : (memref<64xf16>, memref<64xf16>, index) -> i32
      "tile.wait_async"() : () -> ()
    }
    return
  }

  func.func @preserve_shared_state(
      %state: !tile.pipeline_state,
      %dst: memref<64xf16>,
      %src: memref<64xf16>,
      %bytes: index) {
    %copy = "tile.async_copy"(%dst, %src, %bytes, %state)
        : (memref<64xf16>, memref<64xf16>, index, !tile.pipeline_state) -> i32
    "tile.wait_async"(%state) : (!tile.pipeline_state) -> ()
    return
  }
}

// Both branch-local chains are rooted in entry-block states. A chain from one
// branch must never be used in its sibling.
// CHECK-COUNT-2: tile.pipeline_init
// CHECK-COUNT-2: tile.alloc
// CHECK: scf.if
// CHECK: tile.async_copy
// CHECK: tile.pipeline_advance
// CHECK: tile.wait_async
// CHECK: tile.pipeline_advance
// CHECK: } else {
// CHECK: tile.async_copy
// CHECK: tile.pipeline_advance
// CHECK: tile.wait_async
// CHECK: tile.pipeline_advance
// CHECK-COUNT-2: tile.dealloc
// CHECK-LABEL: func.func @preserve_shared_state
// CHECK-NOT: tile.pipeline_init
// CHECK: tile.async_copy
// CHECK: tile.wait_async
// CHECK-NOT: tile.pipeline_init
// CHECK: return
