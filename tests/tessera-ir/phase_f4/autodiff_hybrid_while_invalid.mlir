// RUN: not tessera-opt --tessera-autodiff-paired %s 2>&1 | FileCheck %s

module {
  func.func @hybrid_while_without_checkpoints(%x: tensor<4xf32>)
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
      %next = "tessera.tanh"(%carry) :
          (tensor<4xf32>) -> tensor<4xf32>
      %next_i = arith.addi %i, %c1 : index
      scf.yield %next_i, %next : index, tensor<4xf32>
    }) {tessera.autodiff.checkpoint_policy = "hybrid",
        tessera.autodiff.max_iters = 3 : i64} :
        (index, tensor<4xf32>) -> (index, tensor<4xf32>)
    return %out : tensor<4xf32>
  }
}

// CHECK: error: HYBRID scf.while requires explicit checkpoint_indices
