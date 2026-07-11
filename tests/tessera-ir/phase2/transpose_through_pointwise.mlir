// W4 (data-movement) — TransposeThroughPointwise: an elementwise unary op
// commutes with any permutation, so `transpose(pointwise(x))` rewrites to
// `pointwise(transpose(x))`, sinking the transpose toward the input where it can
// cancel a producer transpose or fold into a matmul — eliminating a real
// data-movement pass over the tensor.
//
// RUN: tessera-opt --tessera-canonicalize %s | FileCheck %s --check-prefix=SINK
// RUN: tessera-opt --tessera-canonicalize --canonicalize %s \
// RUN:   | FileCheck %s --check-prefix=ELIM

module attributes {tessera.ir.version = "1.0"} {

  // (1) sink: transpose(relu(x)) ⇒ relu(transpose(x)) — the transpose now runs
  // BEFORE the relu (its producer), the relu on the transposed shape.
  // SINK-LABEL: func.func @sink_through_relu
  // SINK:         tessera.transpose
  // SINK:         tessera.relu
  // SINK:         return
  func.func @sink_through_relu(%x: tensor<4x8xf32>) -> tensor<8x4xf32> {
    %0 = "tessera.relu"(%x) : (tensor<4x8xf32>) -> tensor<4x8xf32>
    %1 = "tessera.transpose"(%0) {permutation = array<i64: 1, 0>}
        : (tensor<4x8xf32>) -> tensor<8x4xf32>
    return %1 : tensor<8x4xf32>
  }

  // (2) guard: the pointwise result has a SECOND consumer (returned), so sinking
  // the transpose would keep the relu alive AND add a transpose — no reduction.
  // The pattern must NOT fire: relu stays the producer of the transpose.
  // SINK-LABEL: func.func @no_sink_when_pointwise_multi_use
  // SINK:         tessera.relu
  // SINK:         tessera.transpose
  func.func @no_sink_when_pointwise_multi_use(%x: tensor<4x8xf32>)
      -> (tensor<8x4xf32>, tensor<4x8xf32>) {
    %0 = "tessera.relu"(%x) : (tensor<4x8xf32>) -> tensor<4x8xf32>
    %1 = "tessera.transpose"(%0) {permutation = array<i64: 1, 0>}
        : (tensor<4x8xf32>) -> tensor<8x4xf32>
    return %1, %0 : tensor<8x4xf32>, tensor<4x8xf32>
  }

  // (3) elimination: transpose(relu(transpose(x))) ⇒ relu(x). The sink makes the
  // two transposes adjacent; the standard transpose(transpose)→x canonicalizer
  // then removes BOTH — two data-movement passes eliminated.
  // ELIM-LABEL: func.func @double_transpose_cancels
  // ELIM-NOT:     tessera.transpose
  // ELIM:         tessera.relu
  // ELIM-NOT:     tessera.transpose
  // ELIM:         return
  func.func @double_transpose_cancels(%x: tensor<8x4xf32>) -> tensor<8x4xf32> {
    %0 = "tessera.transpose"(%x) {permutation = array<i64: 1, 0>}
        : (tensor<8x4xf32>) -> tensor<4x8xf32>
    %1 = "tessera.relu"(%0) : (tensor<4x8xf32>) -> tensor<4x8xf32>
    %2 = "tessera.transpose"(%1) {permutation = array<i64: 1, 0>}
        : (tensor<4x8xf32>) -> tensor<8x4xf32>
    return %2 : tensor<8x4xf32>
  }
}
