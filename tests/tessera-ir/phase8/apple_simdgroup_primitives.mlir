// RUN: tessera-opt %s | FileCheck %s
//
// The Apple GPU machine primitives, in the shape `emit/apple_msl.py` already
// emits: stage into threadgroup memory, barrier, load two 8x8 operands, MMA
// into an fp32 accumulator, barrier, store.
//
// Before these ops the Apple dialect could only say "call this kernel" --
// every op it declared was a dispatch container, so the MLIR pipeline could
// not express what an Apple kernel does. That is the seam CLAUDE.md names:
// the machine vocabulary lived in Python.

func.func @coopmat_inner_loop(%As: memref<512xf16>, %Bs: memref<512xf16>,
                              %Cs: memref<1024xf32>, %off: index) {
  // Stage A and B, then order the write against the reads below. `mem_none`
  // here would compile and race.
  tessera_apple.gpu.threadgroup_barrier {memory_scope = "threadgroup"}

  // `leading_dim` is the source row stride: BK for A, BN for B, exactly as the
  // MSL kernel passes them to simdgroup_load.
  %a = tessera_apple.gpu.simdgroup_load %As, %off
      {leading_dim = 16 : i64, space = "threadgroup"}
      : memref<512xf16>, index -> !tessera_apple.simdgroup_matrix<f16>
  %b = tessera_apple.gpu.simdgroup_load %Bs, %off
      {leading_dim = 32 : i64, space = "threadgroup"}
      : memref<512xf16>, index -> !tessera_apple.simdgroup_matrix<f16>

  // The accumulator is fp32 even though the inputs are f16 -- d = a*b + c.
  %zero = tessera_apple.gpu.simdgroup_load %Cs, %off
      {leading_dim = 32 : i64, space = "threadgroup"}
      : memref<1024xf32>, index -> !tessera_apple.simdgroup_matrix<f32>
  %acc = tessera_apple.gpu.simdgroup_matmul %a, %b, %zero
      {storage = "f16", m = 8 : i64, n = 8 : i64, k = 8 : i64}
      : !tessera_apple.simdgroup_matrix<f16>, !tessera_apple.simdgroup_matrix<f16>,
        !tessera_apple.simdgroup_matrix<f32> -> !tessera_apple.simdgroup_matrix<f32>

  tessera_apple.gpu.threadgroup_barrier {memory_scope = "threadgroup"}
  tessera_apple.gpu.simdgroup_store %acc, %Cs, %off
      {leading_dim = 32 : i64, space = "threadgroup"}
      : !tessera_apple.simdgroup_matrix<f32>, memref<1024xf32>, index
  return
}

// CHECK: tessera_apple.gpu.threadgroup_barrier
// CHECK: tessera_apple.gpu.simdgroup_load
// CHECK: tessera_apple.gpu.simdgroup_matmul
// CHECK: tessera_apple.gpu.simdgroup_store

// -----

// An f32 MMA is the simdgroup f32 ceiling; the accumulator is fp32 either way.
func.func @f32_inputs_same_fp32_accumulator(%m: memref<64xf32>, %o: index) {
  %a = tessera_apple.gpu.simdgroup_load %m, %o
      {leading_dim = 8 : i64, space = "device"}
      : memref<64xf32>, index -> !tessera_apple.simdgroup_matrix<f32>
  %d = tessera_apple.gpu.simdgroup_matmul %a, %a, %a
      {storage = "f32", m = 8 : i64, n = 8 : i64, k = 8 : i64}
      : !tessera_apple.simdgroup_matrix<f32>, !tessera_apple.simdgroup_matrix<f32>,
        !tessera_apple.simdgroup_matrix<f32> -> !tessera_apple.simdgroup_matrix<f32>
  tessera_apple.gpu.simdgroup_store %d, %m, %o
      {leading_dim = 8 : i64, space = "device"}
      : !tessera_apple.simdgroup_matrix<f32>, memref<64xf32>, index
  return
}
