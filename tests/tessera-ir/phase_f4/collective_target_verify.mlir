// RUN: not %tessera_strict_opt %s 2>&1 | FileCheck %s

module {
  func.func @bad_target_shape(%x: tensor<8xf32>) -> tensor<5xf32> {
    %future = "tessera_collective.reduce_scatter"(%x) {
      mesh_axis = "dp", tensor_axis = 0 : i64, reduction = "sum",
      world_size = 2 : i64
    } : (tensor<8xf32>) -> !tessera_collective.future<tensor<5xf32>>
    %result = "tessera_collective.await"(%future)
      : (!tessera_collective.future<tensor<5xf32>>) -> tensor<5xf32>
    return %result : tensor<5xf32>
  }
}

// CHECK: 'tessera_collective.reduce_scatter' op awaited output shape disagrees with world_size
