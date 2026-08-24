// REQUIRES: tessera-nvidia-backend
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile %s | FileCheck %s

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @nvidia_sm120_scheduled_dynamic(
      %a: tensor<?x?xf16>, %b: tensor<?x?xf16>) -> tensor<?x?xf32> {
    %d = tessera.matmul %a, %b {shape_bounds = [32, 24, 32]}
      : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf32>
    return %d : tensor<?x?xf32>
  }
}

// CHECK-LABEL: llvm.func @nvidia_sm120_scheduled_dynamic_kernel(
// CHECK-SAME: %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr,
// CHECK-SAME: %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64)
// CHECK: tessera_nvidia.block_coordinate
// CHECK: scf.for
// CHECK: tile.materialize_composed_layout
// CHECK-SAME: {{<\[\[-1\], \[-1\]\], \[\[-1\], \[1\]\]}}
// CHECK: tile.view
// CHECK-SAME: leading_dim = 0
// CHECK: tile.mma
// CHECK: tile.store
// CHECK-SAME: leading_dim = 0
