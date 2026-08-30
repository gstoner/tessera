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

// The accumulator is filled, not loaded from memory: its initial value must
// not depend on a buffer the compiler would then have to prove was zeroed.
// CHECK: tessera_apple.gpu.simdgroup_fill {value = 0.000000e+00 : f32} : <f32>

// The K reduction carries the accumulator as an iteration argument, so the
// dependence between K steps is explicit rather than hidden in memory.
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (!tessera_apple.simdgroup_matrix<f32>)

// f16 operands, f32 accumulator -- the MMA's fixed numerical contract.
// CHECK: tessera_apple.gpu.simdgroup_load {{.*}}leading_dim = 16 : i64{{.*}} -> <f16>
// CHECK: tessera_apple.gpu.simdgroup_load {{.*}}leading_dim = 8 : i64{{.*}} -> <f16>
// CHECK: tessera_apple.gpu.simdgroup_matmul {{.*}}storage = "f16"{{.*}} -> <f32>
// CHECK: tessera_apple.gpu.simdgroup_store {{.*}} : <f32>, memref<16x8xf32>, index

// -----

// A ragged extent has no masked load yet, so the pass must DECLINE rather than
// emit an unmasked nest that reads out of bounds. The MPS lane still serves it.
func.func @ragged_extent_is_declined(%a: tensor<17x16xf16>, %b: tensor<16x8xf16>)
    -> tensor<17x8xf32> {
  %c = "tessera.matmul"(%a, %b)
      : (tensor<17x16xf16>, tensor<16x8xf16>) -> tensor<17x8xf32>
  return %c : tensor<17x8xf32>
}
// CHECK-LABEL: @ragged_extent_is_declined
// CHECK-NOT: tessera_apple.gpu.simdgroup

// -----

// An f16 result would need the convert epilogue the MSL kernel performs, since
// simdgroup_store moves raw elements and does not round. Declined, not
// reinterpreted.
func.func @f16_result_is_declined(%a: tensor<16x16xf16>, %b: tensor<16x8xf16>)
    -> tensor<16x8xf16> {
  %c = "tessera.matmul"(%a, %b)
      : (tensor<16x16xf16>, tensor<16x8xf16>) -> tensor<16x8xf16>
  return %c : tensor<16x8xf16>
}
// CHECK-LABEL: @f16_result_is_declined
// CHECK-NOT: tessera_apple.gpu.simdgroup
