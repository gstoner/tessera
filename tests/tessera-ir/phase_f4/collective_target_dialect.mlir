// RUN: %tessera_strict_opt %s | FileCheck %s

module {
  func.func @target_queue(%x: tensor<8xf32>) -> tensor<8xf32> {
    %future = "tessera_collective.all_reduce"(%x) {
      mesh_axis = "dp", tensor_axis = 0 : i64, reduction = "sum",
      world_size = 2 : i64
    } : (tensor<8xf32>) -> !tessera_collective.future<tensor<8xf32>>
    %result = "tessera_collective.await"(%future)
      : (!tessera_collective.future<tensor<8xf32>>) -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}

// CHECK: tessera_collective.all_reduce
// CHECK-SAME: mesh_axis = "dp"
// CHECK-SAME: reduction = "sum"
// CHECK-SAME: world_size = 2
// CHECK: tessera_collective.await
