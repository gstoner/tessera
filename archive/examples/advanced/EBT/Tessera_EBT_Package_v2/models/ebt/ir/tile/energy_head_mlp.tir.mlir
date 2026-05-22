// RUN: mlir-opt %s | FileCheck %s
module {
  func.func @ebt_energy_mlp(%h: tensor<?x?x?xf32>, %y: tensor<?x?x?xf32>,
                            %W1: tensor<?x?xf32>, %b1: tensor<?xf32>,
                            %W2: tensor<?x?xf32>, %b2: tensor<?xf32>) -> tensor<?x1xf32> attributes { tessera.ebt.energy } {
    %z = tessera.tile.concat %y, %h {axis = -1} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %u = tessera.tile.linear %z, %W1, %b1 : (tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?x?xf32>
    %a = tessera.tile.gelu %u : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %o = tessera.tile.linear %a, %W2, %b2 : (tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?x1xf32>
    %E = tessera.tile.reduce_sum %o {dims=[1]} : (tensor<?x?x1xf32>) -> tensor<?x1xf32>
    return %E : tensor<?x1xf32>
  }
}
// CHECK-LABEL: func @ebt_energy_mlp
// CHECK: tessera.tile.concat
// CHECK: tessera.tile.linear
// CHECK: tessera.tile.gelu
// CHECK: tessera.tile.reduce_sum
