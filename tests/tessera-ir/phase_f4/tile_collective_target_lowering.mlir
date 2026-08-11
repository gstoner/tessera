// RUN: tessera-opt --tessera-lower-tile-collectives %s | FileCheck %s

module {
  func.func @collectives(%x8: tensor<8xf32>, %x4: tensor<4xf32>)
      -> (tensor<8xf32>, tensor<4xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) {
    %r = "tile.all_reduce"(%x8) {
      mesh_axis = "dp", tensor_axis = 0 : i64, reduction = "sum",
      world_size = 2 : i64
    } : (tensor<8xf32>) -> tensor<8xf32>
    %s = "tile.reduce_scatter"(%x8) {
      mesh_axis = "dp", tensor_axis = 0 : i64, reduction = "sum",
      world_size = 2 : i64
    } : (tensor<8xf32>) -> tensor<4xf32>
    %g = "tile.all_gather"(%x4) {
      mesh_axis = "dp", tensor_axis = 0 : i64, reduction = "none",
      world_size = 2 : i64
    } : (tensor<4xf32>) -> tensor<8xf32>
    %a = "tile.all_to_all"(%x8) {
      mesh_axis = "dp", tensor_axis = 0 : i64, reduction = "none",
      world_size = 2 : i64
    } : (tensor<8xf32>) -> tensor<8xf32>
    %p = "tile.collective_permute"(%x8) {
      mesh_axis = "dp", source_peers = array<i64: 0, 1>,
      target_peers = array<i64: 1, 0>, world_size = 2 : i64
    } : (tensor<8xf32>) -> tensor<8xf32>
    return %r, %s, %g, %a, %p : tensor<8xf32>, tensor<4xf32>,
                                     tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
  }
}

// CHECK: module attributes {
// CHECK-DAG: tessera.collective.target_abi = "tessera.collective.v1"
// CHECK-DAG: tessera.collective.transport = "runtime_adapter"
// CHECK: %[[R_FUTURE:.*]] = tessera_collective.all_reduce
// CHECK-SAME: mesh_axis = "dp"
// CHECK-SAME: reduction = "sum"
// CHECK-SAME: tensor_axis = 0
// CHECK-SAME: world_size = 2
// CHECK: %[[R:.*]] = tessera_collective.await %[[R_FUTURE]]
// CHECK: %[[S_FUTURE:.*]] = tessera_collective.reduce_scatter
// CHECK: %[[S:.*]] = tessera_collective.await %[[S_FUTURE]]
// CHECK: %[[G_FUTURE:.*]] = tessera_collective.all_gather
// CHECK: %[[G:.*]] = tessera_collective.await %[[G_FUTURE]]
// CHECK: %[[A_FUTURE:.*]] = tessera_collective.all_to_all
// CHECK: %[[A:.*]] = tessera_collective.await %[[A_FUTURE]]
// CHECK: %[[P_FUTURE:.*]] = tessera_collective.collective_permute
// CHECK-SAME: source_peers = array<i64: 0, 1>
// CHECK-SAME: target_peers = array<i64: 1, 0>
// CHECK: %[[P:.*]] = tessera_collective.await %[[P_FUTURE]]
// CHECK-NOT: tile.all_
// CHECK-NOT: tile.reduce_scatter
// CHECK: return %[[R]], %[[S]], %[[G]], %[[A]], %[[P]]
