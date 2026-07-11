// W5 (extend the middle-end to the backward graph) — native pointwise adjoints.
//
// tanh and sigmoid have compare-free closed-form derivatives, so AutodiffPass
// emits their backward NATIVELY (their forward op + mul/sub + a constant) instead
// of an opaque `tessera.custom_adjoint_call` placeholder that round-trips to the
// Python VJP. A native backward is first-class compiler IR: the middle-end can
// CSE / canonicalize / fuse it. The recomputed forward activation deliberately
// duplicates the forward value — CSE then collapses it (the concrete W5 win: the
// middle-end optimizing the backward graph, impossible through an opaque call).
// Numerics match the numpy-tape VJP exactly (dy·(1−tanh²) / dy·s·(1−s)) — see
// python/tessera/autodiff/vjp.py vjp_tanh / vjp_sigmoid.
//
// RUN: tessera-opt --tessera-autodiff %s | FileCheck %s --check-prefix=NATIVE
// RUN: tessera-opt --tessera-autodiff --cse %s | FileCheck %s --check-prefix=CSE

module {
  // tanh backward: dx = dy · (1 − tanh(x)²) — native ops, no placeholder.
  // NATIVE-LABEL: func.func @tanh_bwd
  // NATIVE-NOT:     custom_adjoint_call
  // NATIVE:         tessera.tanh
  // NATIVE:         tessera.mul
  // NATIVE:         tessera.sub
  // NATIVE:         tessera.mul
  //
  // Under CSE the recomputed tanh(x) collapses into the forward — exactly ONE
  // tanh survives for the whole (forward + backward) graph.
  // CSE-LABEL:   func.func @tanh_bwd
  // CSE:         tessera.tanh
  // CSE-NOT:     tessera.tanh
  func.func @tanh_bwd(%x: tensor<2x3xf32>) -> tensor<2x3xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.tanh"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %y : tensor<2x3xf32>
  }

  // sigmoid backward: dx = dy · s · (1 − s),  s = sigmoid(x).
  // NATIVE-LABEL: func.func @sigmoid_bwd
  // NATIVE-NOT:     custom_adjoint_call
  // NATIVE:         tessera.sigmoid
  // NATIVE:         tessera.sub
  // NATIVE:         tessera.mul
  // CSE-LABEL:   func.func @sigmoid_bwd
  // CSE:         tessera.sigmoid
  // CSE-NOT:     tessera.sigmoid
  func.func @sigmoid_bwd(%x: tensor<2x3xf32>) -> tensor<2x3xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.sigmoid"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %y : tensor<2x3xf32>
  }
}
