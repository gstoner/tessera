// W5 (extend the middle-end to the backward graph) — native pointwise adjoints.
//
// tanh and sigmoid have compare-free closed-form derivatives, so AutodiffPass
// emits their backward NATIVELY (their forward op + mul/sub + a constant) instead
// of an opaque `tessera.custom_adjoint_call` placeholder that round-trips to the
// Python VJP. A native backward is first-class compiler IR: the middle-end can
// CSE / canonicalize / fuse it. The latency path directly reuses the saved
// forward activation; a function memory budget may instead rematerialize it
// next to backward consumers.
// Numerics match the numpy-tape VJP exactly (dy·(1−tanh²) / dy·s·(1−s)) — see
// python/tessera/autodiff/vjp.py vjp_tanh / vjp_sigmoid.
//
// RUN: tessera-opt --tessera-autodiff %s | FileCheck %s --check-prefix=NATIVE
// RUN: tessera-opt --tessera-autodiff --cse %s | FileCheck %s --check-prefix=CSE
// RUN: tessera-opt --tessera-autodiff %s | FileCheck %s --check-prefix=DYN

module {
  // tanh backward: dx = dy · (1 − tanh(x)²) — native ops, no placeholder.
  // NATIVE-LABEL: func.func @tanh_bwd
  // NATIVE-NOT:     custom_adjoint_call
  // NATIVE:         tessera.tanh
  // NATIVE:         tessera.mul
  // NATIVE:         tessera.sub
  // NATIVE:         tessera.mul
  //
  // Exactly one tanh serves the whole forward + backward graph.
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

  // Dynamic (or unranked) activation type: the native form would need a dense
  // splat `1` sized to the result, which MLIR forbids for a non-static shape.
  // So the adjoint safely FALLS BACK to the opaque custom_adjoint_call placeholder
  // (the pre-W5 path) rather than asserting — native is a static-shape fast path.
  // DYN-LABEL: func.func @tanh_bwd_dynamic
  // DYN:         tessera.custom_adjoint_call "tanh"
  // DYN-NOT:     tessera.mul
  func.func @tanh_bwd_dynamic(%x: tensor<?x?xf32>) -> tensor<?x?xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.tanh"(%x) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    return %y : tensor<?x?xf32>
  }
}
