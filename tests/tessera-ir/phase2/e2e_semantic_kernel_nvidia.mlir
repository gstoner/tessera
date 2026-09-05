// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --split-input-file %s | FileCheck %s
// NVIDIA F2: native launch wrappers preserve the established scalar ABI and policy.
module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @user_softmax(%x: tensor<3x17xf32>) -> tensor<3x17xf32> {
    %0 = "tessera.softmax"(%x) {axis = -1 : i64} : (tensor<3x17xf32>) -> tensor<3x17xf32>
    return %0 : tensor<3x17xf32>
  }
}
// CHECK-NOT: func.func
// CHECK-LABEL: llvm.func @tessera_tile_softmax_f32
// CHECK-SAME: !llvm.ptr
// CHECK-SAME: !llvm.ptr
// CHECK-SAME: i64
// CHECK-SAME: i64
// CHECK-SAME: nvvm.kernel
// CHECK: tile.softmax_kernel
// CHECK-SAME: exp_mode = "approx_exp2"
// CHECK-SAME: ftz = false
// CHECK-SAME: tessera.schedule_hash
// CHECK-SAME: tessera.workgroup_size = 128
// CHECK: llvm.return
// -----
module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @user_mean(%x: tensor<2x3x5xf32>) -> tensor<2x5xf32> {
    %0 = "tessera.reduce"(%x) {axis = 1 : i64, kind = "mean"} : (tensor<2x3x5xf32>) -> tensor<2x5xf32>
    return %0 : tensor<2x5xf32>
  }
}
// CHECK-NOT: func.func
// CHECK-LABEL: llvm.func @tessera_tile_reduce_mean_f32_serial
// CHECK-SAME: nvvm.kernel
// CHECK: tile.reduce_kernel
// CHECK-SAME: axis = 1
// CHECK-SAME: inner_is_one = false
// CHECK-SAME: kind = "mean"
// CHECK-SAME: nan_mode = "propagate"
// CHECK-SAME: schedule = "serial"
// CHECK-SAME: tessera.schedule_hash
// CHECK-SAME: tessera.workgroup_size = 128
// CHECK: llvm.return
