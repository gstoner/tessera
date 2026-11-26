// APPLE-NTBI-2 slice 11: the simdgroup_32 row-norm schedule is apple7 f16/bf16
// with a 32-lane workgroup, at Schedule and at Tile; every departure is refused.
// RUN: tessera-opt --split-input-file --verify-diagnostics %s

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @simdgroup_serial_workgroup(%x: tensor<5x64xf16>) -> tensor<5x64xf16> {
    // expected-error @+1 {{requires SM120, Zen 5, apple7 f16/bf16 serial|simdgroup_32}}
    %0 = schedule.norm %x {accum = "f32", arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, epsilon = 1.0e-05 : f32, kind = "rmsnorm", schedule = "simdgroup_32", storage = "f16", workgroup_size = 1 : i64} : tensor<5x64xf16> -> tensor<5x64xf16>
    return %0 : tensor<5x64xf16>
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @serial_simdgroup_workgroup(%x: tensor<5x64xf16>) -> tensor<5x64xf16> {
    // expected-error @+1 {{requires SM120, Zen 5, apple7 f16/bf16 serial|simdgroup_32}}
    %0 = schedule.norm %x {accum = "f32", arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, epsilon = 1.0e-05 : f32, kind = "rmsnorm", storage = "f16", workgroup_size = 32 : i64} : tensor<5x64xf16> -> tensor<5x64xf16>
    return %0 : tensor<5x64xf16>
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @simdgroup_f32(%x: tensor<5x64xf32>) -> tensor<5x64xf32> {
    // expected-error @+1 {{requires SM120, Zen 5, apple7 f16/bf16 serial|simdgroup_32}}
    %0 = schedule.norm %x {accum = "f32", arch = "apple7", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, epsilon = 1.0e-05 : f32, kind = "rmsnorm", schedule = "simdgroup_32", storage = "f32", workgroup_size = 32 : i64} : tensor<5x64xf32> -> tensor<5x64xf32>
    return %0 : tensor<5x64xf32>
  }
}

// -----

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @simdgroup_on_sm120(%x: tensor<5x64xf16>) -> tensor<5x64xf16> {
    // expected-error @+1 {{simdgroup_32 requires apple7}}
    %0 = schedule.norm %x {accum = "f32", arch = "sm_120", artifact_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", axis = -1 : i64, epsilon = 1.0e-05 : f32, kind = "rmsnorm", schedule = "simdgroup_32", storage = "f16", workgroup_size = 128 : i64} : tensor<5x64xf16> -> tensor<5x64xf16>
    return %0 : tensor<5x64xf16>
  }
}

// -----

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @tile_simdgroup_off_apple(%x: !llvm.ptr, %o: !llvm.ptr, %r: i64, %c: i64, %e: f32) {
    // expected-error @+1 {{simdgroup_32 norm requires module tessera.arch=apple7}}
    tile.norm_kernel %x, %o, %r, %c, %e {accum = "f32", affine = false, axis = -1 : i64, kind = "rmsnorm", schedule = "simdgroup_32", storage = "f16", tessera.workgroup_size = 32 : i64} : !llvm.ptr, !llvm.ptr, i64, i64, f32
    return
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @tile_simdgroup_f32(%x: !llvm.ptr, %o: !llvm.ptr, %r: i64, %c: i64, %e: f32) {
    // expected-error @+1 {{simdgroup_32 norm requires module tessera.arch=apple7}}
    tile.norm_kernel %x, %o, %r, %c, %e {accum = "f32", affine = false, axis = -1 : i64, kind = "rmsnorm", schedule = "simdgroup_32", storage = "f32", tessera.workgroup_size = 32 : i64} : !llvm.ptr, !llvm.ptr, i64, i64, f32
    return
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @tile_simdgroup_workgroup_one(%x: !llvm.ptr, %o: !llvm.ptr, %r: i64, %c: i64, %e: f32) {
    // expected-error @+1 {{and a workgroup of 32}}
    tile.norm_kernel %x, %o, %r, %c, %e {accum = "f32", affine = false, axis = -1 : i64, kind = "rmsnorm", schedule = "simdgroup_32", storage = "f16", tessera.workgroup_size = 1 : i64} : !llvm.ptr, !llvm.ptr, i64, i64, f32
    return
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @tile_simdgroup_without_workgroup(%x: !llvm.ptr, %o: !llvm.ptr, %r: i64, %c: i64, %e: f32) {
    // expected-error @+1 {{and a workgroup of 32}}
    tile.norm_kernel %x, %o, %r, %c, %e {accum = "f32", affine = false, axis = -1 : i64, kind = "rmsnorm", schedule = "simdgroup_32", storage = "f16"} : !llvm.ptr, !llvm.ptr, i64, i64, f32
    return
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @tile_serial_with_simdgroup_workgroup(%x: !llvm.ptr, %o: !llvm.ptr, %r: i64, %c: i64, %e: f32) {
    // expected-error @+1 {{an apple7 norm with a workgroup of 32 requires schedule simdgroup_32}}
    tile.norm_kernel %x, %o, %r, %c, %e {accum = "f32", affine = false, axis = -1 : i64, kind = "rmsnorm", storage = "f16", tessera.workgroup_size = 32 : i64} : !llvm.ptr, !llvm.ptr, i64, i64, f32
    return
  }
}
