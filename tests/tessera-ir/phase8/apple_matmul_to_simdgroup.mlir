// REQUIRES: tessera-apple-backend
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
  %c = "tessera.matmul"(%a, %b) {numeric_policy = {storage = "fp16", accum = "fp32"}}
      : (tensor<16x16xf16>, tensor<16x8xf16>) -> tensor<16x8xf32>
  return %c : tensor<16x8xf32>
}

// The staging tiles are hoisted out of the loop nest -- allocated once, reused
// per tile, which is what makes the budget check a per-kernel fact.
// CHECK: tessera_apple.gpu.threadgroup_alloc {{.*}}elements = 64 : i64
//
// The accumulator is the program's numeric_policy.accum (fp32 here), read by
// the pass rather than assumed (APPLE-ACCUM-1, Decision #21a).
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
  %c = "tessera.matmul"(%a, %b) {numeric_policy = {storage = "fp16", accum = "fp32"}}
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

// An f16 result with accum = "fp32" gets the rounding epilogue the MSL kernel
// performs: the accumulator tile stays f32 and each element is rounded ONCE on
// the way out, rather than at every K step.
func.func @f16_result_rounds_once_in_the_epilogue(
    %a: tensor<16x16xf16>, %b: tensor<16x8xf16>) -> tensor<16x8xf16> {
  %c = "tessera.matmul"(%a, %b) {numeric_policy = {storage = "fp16", accum = "fp32"}}
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

// -----

// bf16 is a first-class storage type, not an afterthought: the MSL synthesizer
// emits `simdgroup_matrix<bfloat, 8, 8>` natively (Metal 3.1+, Apple6 and
// later). Omitting it left the IR unable to express a route the backend
// already supported -- the exact gap these primitives exist to close.
func.func @bf16_storage_with_fp32_accumulator(
    %a: tensor<16x16xbf16>, %b: tensor<16x8xbf16>) -> tensor<16x8xbf16> {
  %c = "tessera.matmul"(%a, %b) {numeric_policy = {storage = "bf16", accum = "fp32"}}
      : (tensor<16x16xbf16>, tensor<16x8xbf16>) -> tensor<16x8xbf16>
  return %c : tensor<16x8xbf16>
}
// CHECK-LABEL: @bf16_storage_with_fp32_accumulator
// CHECK: tessera_apple.gpu.simdgroup_load {{.*}} -> <bf16>
// The accumulator stays f32: bf16 has 7 mantissa bits against f16's 10, so the
// fp32 accumulator matters at least as much for it.
// CHECK: tessera_apple.gpu.simdgroup_matmul {{.*}}storage = "bf16"{{.*}} -> <f32>
// CHECK: arith.truncf {{.*}} : f32 to bf16

// -----

// APPLE-ACCUM-1: accum = "fp16" is genuine fp16 accumulation on Apple7
// (measured bit-exact with a sequential fp16 FMA chain for f16 storage, K=4096
// max relative error 1.3e-02 to 1.8e-02 vs about 2e-06 for fp32). The accumulator tile, the
// MMA chain and the store are all f16, and an f16 result needs no epilogue
// rounding at all -- the accumulator IS the result.
func.func @f16_accumulator_is_the_declared_accumulator(
    %a: tensor<16x16xf16>, %b: tensor<16x8xf16>) -> tensor<16x8xf16> {
  %c = "tessera.matmul"(%a, %b) {numeric_policy = {storage = "fp16", accum = "fp16"}}
      : (tensor<16x16xf16>, tensor<16x8xf16>) -> tensor<16x8xf16>
  return %c : tensor<16x8xf16>
}
// CHECK-LABEL: @f16_accumulator_is_the_declared_accumulator
// CHECK: memref.alloc() : memref<128xf16>
// CHECK: tessera_apple.gpu.simdgroup_fill {value = 0.000000e+00 : f32} : <f16>
// CHECK: scf.for {{.*}} -> (!tessera_apple.simdgroup_matrix<f16>)
// CHECK: tessera_apple.gpu.simdgroup_matmul {{.*}}storage = "f16"{{.*}} -> <f16>
// CHECK: tessera_apple.gpu.simdgroup_store {{.*}} : <f16>, memref<128xf16>, index
// CHECK-NOT: arith.truncf
// CHECK: return

// -----

// An f16 accumulator under an f32 result widens exactly (arith.extf): the
// returned values are the fp16 accumulator's values, not a re-accumulation.
func.func @f16_accumulator_widens_exactly_into_an_f32_result(
    %a: tensor<16x16xbf16>, %b: tensor<16x8xbf16>) -> tensor<16x8xf32> {
  %c = "tessera.matmul"(%a, %b) {numeric_policy = {storage = "bf16", accum = "fp16"}}
      : (tensor<16x16xbf16>, tensor<16x8xbf16>) -> tensor<16x8xf32>
  return %c : tensor<16x8xf32>
}
// CHECK-LABEL: @f16_accumulator_widens_exactly_into_an_f32_result
// CHECK: tessera_apple.gpu.simdgroup_matmul {{.*}}storage = "bf16"{{.*}} -> <f16>
// CHECK: arith.extf {{.*}} : f16 to f32
