// RUN: tessera-opt --tessera-matmul-to-apple-simdgroup --allow-unregistered-dialect %s | FileCheck %s
//
// The producer for the Apple machine primitives. Before this pass the Apple
// lowering emitted a `func.call` into `tessera_apple_gpu_mps_matmul_*` -- the
// MLIR pipeline named a symbol and the kernel lived in apple_gpu_runtime.mm.
// Here the accumulation is expressed in IR.
//
// The offsets are the part worth checking, and they are verified numerically
// against a reference matmul in tests/unit/test_apple_simdgroup_contract.py:
//   A[m,k] -> m*K + k (row stride K)   B[k,n] -> k*N + n (row stride N)

func.func @gemm_f16_storage_f32_accum(%a: tensor<16x16xf16>, %b: tensor<16x8xf16>)
    -> tensor<16x8xf32> {
  %c = "tessera.matmul"(%a, %b)
      : (tensor<16x16xf16>, tensor<16x8xf16>) -> tensor<16x8xf32>
  return %c : tensor<16x8xf32>
}

// The staging tiles are hoisted out of the loop nest -- allocated once, reused
// per tile, which is what makes the budget check a per-kernel fact.
// CHECK: tessera_apple.gpu.threadgroup_alloc {{.*}}elements = 64 : i64
//
// The accumulator is filled, not loaded from memory: its initial value must
// not depend on a buffer the compiler would then have to prove was zeroed.
// CHECK: tessera_apple.gpu.simdgroup_fill {value = 0.000000e+00 : f32} : <f32>

// The K reduction carries the accumulator as an iteration argument, so the
// dependence between K steps is explicit rather than hidden in memory.
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (!tessera_apple.simdgroup_matrix<f32>)

// The loads read the STAGED tile, so the row stride is the tile width (8), not
// the source matrix's K or N. The global strides moved to the staging copy,
// where the bounds guard lives -- that is the whole reason staging exists,
// since simdgroup_load has no bounds predicate.
// CHECK: tessera_apple.gpu.simdgroup_load {{.*}}leading_dim = 8 : i64{{.*}} -> <f16>
// CHECK: tessera_apple.gpu.simdgroup_load {{.*}}leading_dim = 8 : i64{{.*}} -> <f16>
// f16 operands, f32 accumulator -- the MMA's fixed numerical contract.
// CHECK: tessera_apple.gpu.simdgroup_matmul {{.*}}storage = "f16"{{.*}} -> <f32>
// CHECK: tessera_apple.gpu.simdgroup_store {{.*}}leading_dim = 8 : i64{{.*}} : <f32>, memref<128xf32>, index

// -----

// Ragged extents work through STAGING, not through a masked load: Metal's
// simdgroup_load has no bounds predicate, so out-of-range elements are
// substituted with zero when the tile is copied in. Zero padding is exact --
// a zero operand contributes nothing to the dot product -- and the tail rows
// of the padded accumulator are simply never copied out.
func.func @ragged_extents_stage_with_zero_padding(
    %a: tensor<17x23xf16>, %b: tensor<23x13xf16>) -> tensor<17x13xf16> {
  %c = "tessera.matmul"(%a, %b)
      : (tensor<17x23xf16>, tensor<23x13xf16>) -> tensor<17x13xf16>
  return %c : tensor<17x13xf16>
}
// CHECK-LABEL: @ragged_extents_stage_with_zero_padding
// Accumulator padded to whole tiles: 17->24 rows, 13->16 cols.
// CHECK: memref.alloc() : memref<384xf32>
// CHECK: tessera_apple.gpu.threadgroup_alloc {budget_bytes = 32768 : i64, elements = 64 : i64}
// The load is INSIDE the guard: computing the address and selecting afterwards
// would still have read out of bounds.
// CHECK: scf.if {{.*}} -> (f16)
// Orders the staging writes against the simdgroup reads.
// CHECK: tessera_apple.gpu.threadgroup_barrier {memory_scope = "threadgroup"}
// CHECK: tessera_apple.gpu.simdgroup_load {{.*}}leading_dim = 8 : i64

// -----

// An f16 result gets the rounding epilogue the MSL kernel performs: the
// accumulator tile stays f32 and each element is rounded ONCE on the way out,
// rather than at every K step. Measured, that is 1.7e-04 relative error
// against 5.8e-03 for an f16 accumulator -- 34x -- which is what the extra
// buffer buys.
func.func @f16_result_rounds_once_in_the_epilogue(
    %a: tensor<16x16xf16>, %b: tensor<16x8xf16>) -> tensor<16x8xf16> {
  %c = "tessera.matmul"(%a, %b)
      : (tensor<16x16xf16>, tensor<16x8xf16>) -> tensor<16x8xf16>
  return %c : tensor<16x8xf16>
}
// CHECK-LABEL: @f16_result_rounds_once_in_the_epilogue
// The accumulator tile is f32 even though both operands and the result are f16.
// CHECK: memref.alloc() : memref<128xf32>
// CHECK: tessera_apple.gpu.simdgroup_matmul {{.*}}storage = "f16"{{.*}} -> <f32>
// CHECK: tessera_apple.gpu.simdgroup_store {{.*}}leading_dim = 8 : i64{{.*}} : <f32>, memref<128xf32>, index
// One rounding, after the whole reduction. `arith.truncf` is
// round-to-nearest-even despite the name.
// CHECK: memref.alloc() : memref<128xf16>
// CHECK: arith.truncf
