// APPLE-NTBI-2 slice 6: schedule.activation and tile.activation_kernel own the
// apple7 f16 ReLU contract; every departure from it is refused.
// RUN: tessera-opt --split-input-file --verify-diagnostics %s

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @preserved_zero(%x: tensor<5x11xf16>) -> tensor<5x11xf16> {
    // expected-error @+1 {{requires static rank-two apple7 f16/bf16 serial ReLU}}
    %0 = schedule.activation %x {accum = "f32", arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, kind = "relu", nan_mode = "propagate", schedule = "serial", storage = "f16", workgroup_size = 1 : i64, zero_mode = "preserve"} : tensor<5x11xf16> -> tensor<5x11xf16>
    return %0 : tensor<5x11xf16>
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @bf16_positive_zero(%x: tensor<5x11xbf16>) -> tensor<5x11xbf16> {
    // bf16 keeps the zero sign (the SM120 contract); "positive" is f16's.
    // expected-error @+1 {{requires static rank-two apple7 f16/bf16 serial ReLU}}
    %0 = schedule.activation %x {accum = "f32", arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, kind = "relu", nan_mode = "propagate", schedule = "serial", storage = "bf16", workgroup_size = 1 : i64, zero_mode = "positive"} : tensor<5x11xbf16> -> tensor<5x11xbf16>
    return %0 : tensor<5x11xbf16>
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @flat_schedule(%x: tensor<5x11xf16>) -> tensor<5x11xf16> {
    // expected-error @+1 {{requires static rank-two apple7 f16/bf16 serial ReLU}}
    %0 = schedule.activation %x {accum = "f32", arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, kind = "relu", nan_mode = "propagate", schedule = "flat_128", storage = "f16", workgroup_size = 1 : i64, zero_mode = "positive"} : tensor<5x11xf16> -> tensor<5x11xf16>
    return %0 : tensor<5x11xf16>
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @tile_preserved_zero(%x: !llvm.ptr, %o: !llvm.ptr) {
    %r = arith.constant 5 : i64
    %c = arith.constant 11 : i64
    // expected-error @+1 {{requires apple7 f16/bf16 serial ReLU, workgroup 1}}
    tile.activation_kernel %x, %o, %r, %c {accum = "f32", axis = -1 : i64, kind = "relu", nan_mode = "propagate", schedule = "serial", storage = "f16", tessera.schedule_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", tessera.workgroup_size = 1 : i64, zero_mode = "preserve"} : !llvm.ptr, !llvm.ptr, i64, i64
    return
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @tile_approximation(%x: !llvm.ptr, %o: !llvm.ptr) {
    %r = arith.constant 5 : i64
    %c = arith.constant 11 : i64
    // expected-error @+1 {{requires apple7 f16/bf16 serial ReLU, workgroup 1}}
    tile.activation_kernel %x, %o, %r, %c {accum = "f32", approximation = "tanh", axis = -1 : i64, kind = "relu", nan_mode = "propagate", schedule = "serial", storage = "f16", tessera.schedule_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", tessera.workgroup_size = 1 : i64, zero_mode = "positive"} : !llvm.ptr, !llvm.ptr, i64, i64
    return
  }
}
