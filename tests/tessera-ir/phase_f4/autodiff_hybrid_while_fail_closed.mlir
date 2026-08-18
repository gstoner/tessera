// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// HYBRID while retains only declared interior states. Backward selects the
// nearest predecessor checkpoint and replays at most the uncovered suffix.

module {
  func.func @hybrid_while(%x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %count, %out = "scf.while"(%c0, %x) ({
    ^bb0(%i: index, %carry: tensor<4xf32>):
      %continue = arith.cmpi slt, %i, %c3 : index
      scf.condition(%continue) %i, %carry : index, tensor<4xf32>
    }, {
    ^bb0(%i: index, %carry: tensor<4xf32>):
      %next = "tessera.tanh"(%carry) :
          (tensor<4xf32>) -> tensor<4xf32>
      %next_i = arith.addi %i, %c1 : index
      scf.yield %next_i, %next : index, tensor<4xf32>
    }) {tessera.autodiff.checkpoint_indices = array<i64: 1>,
        tessera.autodiff.checkpoint_policy = "hybrid",
        tessera.autodiff.max_iters = 3 : i64,
        tessera.autodiff.residual_digest = "1111111111111111111111111111111111111111111111111111111111111111",
        tessera.autodiff.residual_schema = "tessera.region_residual_abi.v1",
        tessera.structured_cfg.digest = "0000000000000000000000000000000000000000000000000000000000000000"} :
        (index, tensor<4xf32>) -> (index, tensor<4xf32>)
    return %out : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @hybrid_while(
// CHECK-SAME: -> (tensor<4xf32>, index, tensor<1x4xf32>)
// CHECK: %[[LOOP:.+]]:3 = scf.while
// CHECK: arith.cmpi eq
// CHECK: tensor.insert_slice
// CHECK: tessera.autodiff.checkpoint_indices = array<i64: 1>
// CHECK: tessera.autodiff.residual_owner = "scf_while"
// CHECK: return %[[LOOP]]#1, %[[LOOP]]#0, %[[LOOP]]#2
// CHECK-LABEL: func.func @hybrid_while__bwd(
// CHECK-SAME: tessera.autodiff.residual_policy = "hybrid"
// CHECK-NOT: = scf.while
// CHECK: scf.for
// CHECK: arith.cmpi ule
// CHECK: tensor.extract_slice
// CHECK: scf.for
// CHECK: return
