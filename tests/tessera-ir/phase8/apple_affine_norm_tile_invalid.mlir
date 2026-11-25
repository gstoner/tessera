// APPLE-NTBI-2 slice 10: tile.norm_kernel carries gamma (and, for layernorm,
// beta) only on apple7, with both flags and the matching pointer count.
// RUN: tessera-opt --split-input-file --verify-diagnostics %s

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @weighted_off_apple(%x: !llvm.ptr, %g: !llvm.ptr, %o: !llvm.ptr, %r: i64, %c: i64, %e: f32) {
    // expected-error @+1 {{a weighted norm requires apple7 with affine = true}}
    tile.norm_kernel %x, %g, %o, %r, %c, %e {accum = "f32", affine = true, axis = -1 : i64, bias = false, kind = "rmsnorm", storage = "f16", tessera.workgroup_size = 1 : i64} : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f32
    return
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @beta_on_rmsnorm(%x: !llvm.ptr, %g: !llvm.ptr, %b: !llvm.ptr, %o: !llvm.ptr, %r: i64, %c: i64, %e: f32) {
    // expected-error @+1 {{a weighted norm requires apple7 with affine = true}}
    tile.norm_kernel %x, %g, %b, %o, %r, %c, %e {accum = "f32", affine = true, axis = -1 : i64, bias = true, kind = "rmsnorm", storage = "f16", tessera.workgroup_size = 1 : i64} : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f32
    return
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @missing_beta_pointer(%x: !llvm.ptr, %g: !llvm.ptr, %o: !llvm.ptr, %r: i64, %c: i64, %e: f32) {
    // expected-error @+1 {{expects source, gamma[, beta], destination, rows, columns, and epsilon}}
    tile.norm_kernel %x, %g, %o, %r, %c, %e {accum = "f32", affine = true, axis = -1 : i64, bias = true, kind = "layernorm", storage = "f16", tessera.workgroup_size = 1 : i64} : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f32
    return
  }
}

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @bias_without_affine(%x: !llvm.ptr, %o: !llvm.ptr, %r: i64, %c: i64, %e: f32) {
    // expected-error @+1 {{a weighted norm requires apple7 with affine = true}}
    tile.norm_kernel %x, %o, %r, %c, %e {accum = "f32", affine = false, axis = -1 : i64, bias = false, kind = "rmsnorm", storage = "f16", tessera.workgroup_size = 1 : i64} : !llvm.ptr, !llvm.ptr, i64, i64, f32
    return
  }
}
