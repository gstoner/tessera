// RUN: mlir-opt %s | FileCheck %s
module {
  func.func @energy_bilinear(%h: tensor<?x?x?xf32>, %y: tensor<?x?x?xf32>, %W: tensor<?x?xf32>) -> tensor<?x1xf32> attributes { tessera.ebt.energy } {
    %t0 = tessera.tile.matmul %y, %W : (tensor<?x?x?xf32>, tensor<?x?xf32>) -> tensor<?x?x?xf32>
    %E_tok = tessera.tile.batched_dot %t0, %h : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x1xf32>
    %E = tessera.tile.reduce_sum %E_tok {dims=[1]} : (tensor<?x?x1xf32>) -> tensor<?x1xf32>
    return %E : tensor<?x1xf32>
  }

  func.func @energy_bilinear_jvp(%h: tensor<?x?x?xf32>, %y: tensor<?x?x?xf32>, %v: tensor<?x?x?xf32>, %W: tensor<?x?xf32>) -> tensor<?x1xf32> attributes { tessera.ebt.energy_jvp } {
    %t0 = tessera.tile.matmul %v, %W : (tensor<?x?x?xf32>, tensor<?x?xf32>) -> tensor<?x?x?xf32>
    %dE_tok = tessera.tile.batched_dot %t0, %h : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x1xf32>
    %dE = tessera.tile.reduce_sum %dE_tok {dims=[1]} : (tensor<?x?x1xf32>) -> tensor<?x1xf32>
    return %dE : tensor<?x1xf32>
  }
}
// CHECK-LABEL: func @energy_bilinear_jvp
// CHECK: tessera.tile.matmul
// CHECK: tessera.tile.batched_dot
// CHECK: tessera.tile.reduce_sum
