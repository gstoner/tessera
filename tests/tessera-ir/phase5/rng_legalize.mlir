// RUN: tessera-opt --tessera-rng-legalize="rng-backend=philox rng-seed=42" \
// RUN:             --tessera-rng-stream-assign="num-ranks=4 global-seed=42" \
// RUN:     %s | FileCheck %s

// Test: RNGLegalizePass assigns stream IDs and backend attrs.
//       RNGStreamAssignPass finalizes stream IDs as seed * num_ranks + rank.

module {

  // CHECK-LABEL: func.func @sample_dropout
  func.func @sample_dropout(%x: tensor<128x256xbf16>) -> tensor<128x256xbf16> {

    // CHECK: rng.legalized
    // CHECK: rng.stream_id
    // CHECK: rng.backend = "philox"
    %mask = "tessera_rng.bernoulli"(%x) {p = 0.1 : f32} :
        (tensor<128x256xbf16>) -> tensor<128x256xi1>

    // CHECK: rng.stream_id
    %noise = "tessera_rng.normal"(%x) {mean = 0.0 : f32, std = 1.0 : f32} :
        (tensor<128x256xbf16>) -> tensor<128x256xbf16>

    return %x : tensor<128x256xbf16>
  }
}
