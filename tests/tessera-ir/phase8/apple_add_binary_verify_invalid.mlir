// APPLE-NTBI-2 slice 7: schedule.binary_kernel and tile.elementwise_kernel own
// the apple7 f16 add contract; every departure from it is refused.
// RUN: tessera-opt --split-input-file --verify-diagnostics %s

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @flat_schedule(%x: tensor<5x11xf16>) -> tensor<5x11xf16> {
    // expected-error @+1 {{requires a static rank-two apple7 f16 serial add}}
    %0 = schedule.binary_kernel %x {accum = "f32", arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, kind = "add", nan_mode = "propagate", output_storage = "f16", schedule = "flat_128", storage = "f16", workgroup_size = 1 : i64} : tensor<5x11xf16> -> tensor<5x11xf16>
    return %0 : tensor<5x11xf16>
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @f32_storage(%x: tensor<5x11xf32>) -> tensor<5x11xf32> {
    // expected-error @+1 {{requires a static rank-two apple7 f16 serial add}}
    %0 = schedule.binary_kernel %x {accum = "f32", arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, kind = "add", nan_mode = "propagate", output_storage = "f32", schedule = "serial", storage = "f32", workgroup_size = 1 : i64} : tensor<5x11xf32> -> tensor<5x11xf32>
    return %0 : tensor<5x11xf32>
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @tile_wide_workgroup(%a: !llvm.ptr, %b: !llvm.ptr, %o: !llvm.ptr) {
    %n = arith.constant 55 : i64
    // expected-error @+1 {{requires the apple7 f16 serial add (a, b, o, N), workgroup 1}}
    tile.elementwise_kernel %a, %b, %o, %n {accum = "f32", axis = -1 : i64, family = "binary", kind = "add", nan_mode = "propagate", output_storage = "f16", schedule = "serial", storage = "f16", tessera.schedule_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", tessera.workgroup_size = 128 : i64} : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64
    return
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @tile_mul(%a: !llvm.ptr, %b: !llvm.ptr, %o: !llvm.ptr) {
    %n = arith.constant 55 : i64
    // expected-error @+1 {{requires the apple7 f16 serial add (a, b, o, N), workgroup 1}}
    tile.elementwise_kernel %a, %b, %o, %n {accum = "f32", axis = -1 : i64, family = "binary", kind = "mul", nan_mode = "propagate", output_storage = "f16", schedule = "serial", storage = "f16", tessera.schedule_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", tessera.workgroup_size = 1 : i64} : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64
    return
  }
}
