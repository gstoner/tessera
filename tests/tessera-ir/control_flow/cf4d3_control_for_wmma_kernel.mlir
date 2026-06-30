// CF4d-3 — GenerateROCMControlForWmmaKernel lowers a single-tile WMMA matmul
// recurrence control_for (carry = carry @ W, both 16x16 f16) to ONE wave gpu.func
// using rocdl.wmma.f32.16x16x16.f16, with the accumulator(f32)→input(f16)
// fragment shuffle through LDS per iteration. On-device execution on gfx1151 is
// proven by tests/unit/test_rocm_control_for_wmma_exec.py.
//
// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s -split-input-file --generate-rocm-control-for-wmma-kernel \
// RUN:   --allow-unregistered-dialect | FileCheck %s

// ─── carry (16x16 f16) @ W (16x16 f16), looped — the WMMA recurrence ────────
func.func @wb(%c: tensor<16x16xf16>, %w: tensor<16x16xf16>) -> tensor<16x16xf16> {
  %0 = "tessera.matmul"(%c, %w) : (tensor<16x16xf16>, tensor<16x16xf16>) -> tensor<16x16xf16>
  return %0 : tensor<16x16xf16>
}

// CHECK-LABEL: func.func @f
// CHECK:       gpu.func @tessera_control_for_wmma_0({{.*}}memref<?xf16>{{.*}}) workgroup({{.*}}memref<256xf16, #gpu.address_space<workgroup>>) kernel
// build B-fragment (loop-invariant) then the loop with the WMMA + LDS shuffle
// CHECK:         vector.insert
// CHECK:         scf.for
// CHECK:           rocdl.wmma.f32.16x16x16.f16
// CHECK:           gpu.barrier
// the accumulator → f16 → LDS write-back (new carry)
// CHECK:           vector.extract
// CHECK:           arith.truncf
// CHECK:           memref.store {{.*}}memref<256xf16, #gpu.address_space<workgroup>>
// CHECK:           gpu.barrier
// CHECK:         gpu.return
func.func @f(%init: tensor<16x16xf16>, %w: tensor<16x16xf16>) -> tensor<16x16xf16> {
  %r = "tessera.control_for"(%init, %w) {
    body = @wb, start = 0 : i64, stop = 3 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<16x16xf16>, tensor<16x16xf16>) -> tensor<16x16xf16>
  return %r : tensor<16x16xf16>
}

// -----
// ─── a non-16x16 carry is NOT this single-tile WMMA form — left untouched ───
func.func @wb2(%c: tensor<32x32xf16>, %w: tensor<32x32xf16>) -> tensor<32x32xf16> {
  %0 = "tessera.matmul"(%c, %w) : (tensor<32x32xf16>, tensor<32x32xf16>) -> tensor<32x32xf16>
  return %0 : tensor<32x32xf16>
}

// CHECK-LABEL: func.func @g
// CHECK:       tessera.control_for
// CHECK-NOT:   gpu.func
func.func @g(%init: tensor<32x32xf16>, %w: tensor<32x32xf16>) -> tensor<32x32xf16> {
  %r = "tessera.control_for"(%init, %w) {
    body = @wb2, start = 0 : i64, stop = 3 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<32x32xf16>, tensor<32x32xf16>) -> tensor<32x32xf16>
  return %r : tensor<32x32xf16>
}

// -----
// ─── an f32 (non-WMMA) carry is left for the GEMV/other lowering ────────────
func.func @wb3(%c: tensor<16x16xf32>, %w: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = "tessera.matmul"(%c, %w) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// CHECK-LABEL: func.func @h
// CHECK:       tessera.control_for
// CHECK-NOT:   gpu.func
func.func @h(%init: tensor<16x16xf32>, %w: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %r = "tessera.control_for"(%init, %w) {
    body = @wb3, start = 0 : i64, stop = 3 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  return %r : tensor<16x16xf32>
}
