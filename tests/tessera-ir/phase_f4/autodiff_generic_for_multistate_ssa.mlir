// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s

// W4.3 generic counted-loop SAVE: independently typed carried states receive
// independent tapes and the paired backward consumes both without replay.

module {
  func.func @multistate(%x: tensor<2xf32>, %y: tensor<3xf32>,
                        %wx: tensor<2xf32>, %wy: tensor<3xf32>)
      -> (tensor<2xf32>, tensor<3xf32>) attributes {
        tessera.autodiff = "reverse"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %rx, %ry = scf.for %i = %c0 to %c3 step %c1
        iter_args(%sx = %x, %sy = %y) -> (tensor<2xf32>, tensor<3xf32>) {
      %nx = "tessera.mul"(%sx, %wx) :
          (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
      %ny = "tessera.add"(%sy, %wy) :
          (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
      scf.yield %nx, %ny : tensor<2xf32>, tensor<3xf32>
    } {tessera.autodiff.checkpoint_policy = "save",
       tessera.autodiff.checkpoint_indices = array<i64: 1, 2>}
    return %rx, %ry : tensor<2xf32>, tensor<3xf32>
  }
}

// CHECK-LABEL: func.func @multistate(
// CHECK-SAME: -> (tensor<2xf32>, tensor<3xf32>, tensor<2x2xf32>, tensor<2x3xf32>)
// CHECK-SAME: tessera.autodiff.residual_sources = ["generic_for:state_tape:0", "generic_for:state_tape:1"]
// CHECK: %[[LOOP:.+]]:4 = scf.for
// CHECK: } {tessera.autodiff.activity = "active"
// CHECK-SAME: tessera.autodiff.residual_owner = "generic_for"
// CHECK-SAME: tessera.autodiff.residual_result_indices = array<i64: 2, 3>
// CHECK: return %[[LOOP]]#0, %[[LOOP]]#1, %[[LOOP]]#2, %[[LOOP]]#3

// CHECK-LABEL: func.func @multistate__bwd(
// CHECK-SAME: tensor<2x2xf32>, %{{[^:]+}}: tensor<2x3xf32>
// CHECK-COUNT-2: tensor.extract_slice
// CHECK-NOT: scf.for
// CHECK: return
