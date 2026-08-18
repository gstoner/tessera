// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// W4-PRODUCT-1: a bounded SAVE while carries predecessor states in an explicit
// tape.  Backward indexes that tape by reverse ordinal and performs no prefix
// replay loop.

module {
  func.func @saved_while(%x: tensor<4xf32>, %w: tensor<4xf32>)
      -> tensor<4xf32> attributes {tessera.autodiff = "reverse"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %count, %out = "scf.while"(%c0, %x) ({
    ^bb0(%i: index, %carry: tensor<4xf32>):
      %continue = arith.cmpi slt, %i, %c3 : index
      scf.condition(%continue) %i, %carry : index, tensor<4xf32>
    }, {
    ^bb0(%i: index, %carry: tensor<4xf32>):
      %next = "tessera.mul"(%carry, %w) :
          (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %next_i = arith.addi %i, %c1 : index
      scf.yield %next_i, %next : index, tensor<4xf32>
    }) {tessera.autodiff.checkpoint_policy = "save",
        tessera.autodiff.max_iters = 3 : i64} :
        (index, tensor<4xf32>) -> (index, tensor<4xf32>)
    return %out : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @saved_while(
// CHECK-SAME: -> (tensor<4xf32>, index, tensor<3x4xf32>)
// CHECK-SAME: tessera.autodiff.residual_sources = ["scf.while:trip_count", "scf.while:state_tape:0"]
// CHECK: %[[LOOP:.+]]:3 = scf.while
// CHECK: tensor.insert_slice
// CHECK: tessera.autodiff.residual_owner = "scf_while"
// CHECK: tessera.autodiff.residual_result_indices = array<i64: 2>
// CHECK: return %[[LOOP]]#1, %[[LOOP]]#0, %[[LOOP]]#2
// CHECK-LABEL: func.func @saved_while__bwd(
// CHECK-SAME: %[[TRIP:[^:]+]]: index
// CHECK-SAME: %[[TAPE:[^:]+]]: tensor<3x4xf32>
// CHECK-SAME: tessera.autodiff.residual_policy = "save"
// CHECK-NOT: = scf.while
// CHECK-COUNT-1: scf.for
// CHECK: tensor.extract_slice %[[TAPE]]
// CHECK: return
