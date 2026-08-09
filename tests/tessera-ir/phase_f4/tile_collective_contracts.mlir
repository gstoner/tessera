// RUN: tessera-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// The Tile layer owns separate mesh/tensor axes and can prove static shape
// scaling whenever the package binds world_size.

func.func @collectives(%x8: tensor<8xf32>, %x4: tensor<4xf32>) {
  // CHECK: tile.all_reduce
  %r = "tile.all_reduce"(%x8) {
    mesh_axis = "dp", reduction = "sum", tensor_axis = 0 : i64,
    world_size = 2 : i64
  } : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK: tile.reduce_scatter
  %s = "tile.reduce_scatter"(%x8) {
    mesh_axis = "dp", reduction = "sum", tensor_axis = 0 : i64,
    world_size = 2 : i64
  } : (tensor<8xf32>) -> tensor<4xf32>
  // CHECK: tile.all_gather
  %g = "tile.all_gather"(%x4) {
    mesh_axis = "dp", tensor_axis = 0 : i64, world_size = 2 : i64
  } : (tensor<4xf32>) -> tensor<8xf32>
  // CHECK: tile.all_to_all
  %a = "tile.all_to_all"(%x8) {
    mesh_axis = "dp", tensor_axis = 0 : i64, world_size = 2 : i64
  } : (tensor<8xf32>) -> tensor<8xf32>
  return
}

// -----

func.func @bad_gather_shape(%x: tensor<4xf32>) {
  // expected-error @+1 {{collective-axis extent disagrees with world_size}}
  %g = "tile.all_gather"(%x) {
    mesh_axis = "dp", tensor_axis = 0 : i64, world_size = 2 : i64
  } : (tensor<4xf32>) -> tensor<7xf32>
  return
}

// -----

func.func @bad_reduce(%x: tensor<8xf32>) {
  // expected-error @+1 {{reduction must be one of sum, mean, max, min, prod}}
  %r = "tile.all_reduce"(%x) {
    mesh_axis = "dp", reduction = "xor", tensor_axis = 0 : i64
  } : (tensor<8xf32>) -> tensor<8xf32>
  return
}
