// RUN: tessera-opt --tessera-autodiff-paired %s -verify-diagnostics
//
// W4 admits only the tracer's bounded counted-while normal form. A while that
// does not forward and increment its trip counter by one needs an explicit
// iteration tape and therefore remains fail-closed.

module {
  func.func @noncanonical_while(%x: tensor<4xf32>, %w: tensor<4xf32>)
      -> tensor<4xf32> attributes {tessera.autodiff = "reverse"} {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    // expected-error @+1 {{[AUTODIFF_NESTED_REGION] active paired reverse-mode path contains unsupported nested-region op ('scf.while')}}
    %count, %out = scf.while (%i = %c0, %carry = %x)
        : (index, tensor<4xf32>) -> (index, tensor<4xf32>) {
      %continue = arith.cmpi slt, %i, %c4 : index
      scf.condition(%continue) %i, %carry : index, tensor<4xf32>
    } do {
    ^bb0(%i: index, %carry: tensor<4xf32>):
      %next = "tessera.mul"(%carry, %w) :
          (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %next_i = arith.addi %i, %c2 : index
      scf.yield %next_i, %next : index, tensor<4xf32>
    }
    return %out : tensor<4xf32>
  }
}
