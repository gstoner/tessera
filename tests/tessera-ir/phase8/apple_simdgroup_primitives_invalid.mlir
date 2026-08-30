// RUN: tessera-opt --split-input-file --verify-diagnostics %s
//
// `--verify-diagnostics` IS the check: it requires every expected-error and
// fails on any diagnostic that was not expected. It exits 0 on success, so
// this RUN line must not be wrapped in `not`.
//
// Every case below is a kernel Metal would RUN. None of them fault; they
// compute the wrong numbers, which is why the contract rejects them instead of
// trusting the emitter.

func.func @row_stride_below_matrix_width(%m: memref<64xf16>, %o: index) {
  // Metal addresses row r at base + r*leading_dim. A stride of 4 makes rows
  // 0 and 1 overlap by four elements: the MMA silently consumes the wrong
  // operand and the kernel completes normally.
  // expected-error @+1 {{`leading_dim` must be at least 8}}
  %a = tessera_apple.gpu.simdgroup_load %m, %o
      {leading_dim = 4 : i64, space = "threadgroup"}
      : memref<64xf16>, index -> !tessera_apple.simdgroup_matrix<f16>
  return
}

// -----

func.func @store_stride_below_matrix_width(%m: memref<64xf32>, %o: index,
                                           %v: !tessera_apple.simdgroup_matrix<f32>) {
  // The same arithmetic, overwriting neighbouring rows instead of reading them.
  // expected-error @+1 {{`leading_dim` must be at least 8}}
  tessera_apple.gpu.simdgroup_store %v, %m, %o
      {leading_dim = 1 : i64, space = "device"}
      : !tessera_apple.simdgroup_matrix<f32>, memref<64xf32>, index
  return
}

// -----

func.func @f16_accumulator(%a: !tessera_apple.simdgroup_matrix<f16>,
                           %c: !tessera_apple.simdgroup_matrix<f16>) {
  // The simdgroup MMA accumulates in fp32 whatever the inputs are, and
  // apple_msl.py depends on it so the fused epilogue sees full-precision
  // results. An f16 accumulator re-rounds every partial sum -- a numerics
  // change no test of the matmul alone would show.
  // expected-error @+1 {{accumulator `c` and result `d` must be f32}}
  %d = tessera_apple.gpu.simdgroup_matmul %a, %a, %c
      {storage = "f16", m = 8 : i64, n = 8 : i64, k = 8 : i64}
      : !tessera_apple.simdgroup_matrix<f16>, !tessera_apple.simdgroup_matrix<f16>,
        !tessera_apple.simdgroup_matrix<f16> -> !tessera_apple.simdgroup_matrix<f16>
  return
}

// -----

func.func @mixed_precision_operands(%a: !tessera_apple.simdgroup_matrix<f16>,
                                    %b: !tessera_apple.simdgroup_matrix<f32>,
                                    %c: !tessera_apple.simdgroup_matrix<f32>) {
  // Not a simdgroup MMA: it is a convert plus an MMA, and accepting it here
  // hides the conversion from the numerics the epilogue reasons about.
  // expected-error @+1 {{must both have element type f16}}
  %d = tessera_apple.gpu.simdgroup_matmul %a, %b, %c
      {storage = "f16", m = 8 : i64, n = 8 : i64, k = 8 : i64}
      : !tessera_apple.simdgroup_matrix<f16>, !tessera_apple.simdgroup_matrix<f32>,
        !tessera_apple.simdgroup_matrix<f32> -> !tessera_apple.simdgroup_matrix<f32>
  return
}

// -----

func.func @storage_attribute_disagrees_with_operands(
    %a: !tessera_apple.simdgroup_matrix<f16>,
    %c: !tessera_apple.simdgroup_matrix<f32>) {
  // The attribute is the declared contract; operands that contradict it mean
  // one of the two is a lie, and a reader cannot tell which.
  // expected-error @+1 {{must both have element type f32}}
  %d = tessera_apple.gpu.simdgroup_matmul %a, %a, %c
      {storage = "f32", m = 8 : i64, n = 8 : i64, k = 8 : i64}
      : !tessera_apple.simdgroup_matrix<f16>, !tessera_apple.simdgroup_matrix<f16>,
        !tessera_apple.simdgroup_matrix<f32> -> !tessera_apple.simdgroup_matrix<f32>
  return
}

// -----

func.func @non_native_shape(%a: !tessera_apple.simdgroup_matrix<f16>,
                            %c: !tessera_apple.simdgroup_matrix<f32>) {
  // Apple7 has exactly one simdgroup-matrix shape. A 16x16 request has no
  // instruction to lower to, so it fails here rather than at emission.
  // expected-error @+1 {{requires an 8x8x8 shape}}
  %d = tessera_apple.gpu.simdgroup_matmul %a, %a, %c
      {storage = "f16", m = 16 : i64, n = 16 : i64, k = 16 : i64}
      : !tessera_apple.simdgroup_matrix<f16>, !tessera_apple.simdgroup_matrix<f16>,
        !tessera_apple.simdgroup_matrix<f32> -> !tessera_apple.simdgroup_matrix<f32>
  return
}

// -----

func.func @unknown_memory_scope() {
  // The scope selects semantics (#21a): staging into threadgroup memory and
  // ordering with an unrecognised flag would compile and race.
  // expected-error @+1 {{attribute 'memory_scope' failed to satisfy constraint}}
  tessera_apple.gpu.threadgroup_barrier {memory_scope = "eventually"}
  return
}
