// CF4d-4 — GenerateROCMControlForWmmaTileKernel lowers a MULTI-tile WMMA matmul
// recurrence control_for (carry = carry @ W, carry M×K / W K×K, f16) to ONE
// workgroup of MT*KT waves, each wave owning one 16x16 output tile and
// accumulating it over the shared-K dim with rocdl.wmma.f32.16x16x16.f16. The
// whole carry lives in LDS, so the per-iteration handoff is a workgroup barrier
// — no grid.sync. On-device execution on gfx1151 is proven by
// tests/unit/test_rocm_control_for_wmma_tile_exec.py.
//
// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s -split-input-file --generate-rocm-control-for-wmma-tile-kernel \
// RUN:   --allow-unregistered-dialect | FileCheck %s

// ─── 32x32 carry @ 32x32 W (2x2 = 4 tiles, 4 waves), looped ─────────────────
func.func @wb(%c: tensor<32x32xf16>, %w: tensor<32x32xf16>) -> tensor<32x32xf16> {
  %0 = "tessera.matmul"(%c, %w) : (tensor<32x32xf16>, tensor<32x32xf16>) -> tensor<32x32xf16>
  return %0 : tensor<32x32xf16>
}

// CHECK-LABEL: func.func @f
// 4 waves * 32 = 128 threads ; LDS is the whole 32x32 = 1024-elem f16 carry
// CHECK:       gpu.func @tessera_control_for_wmma_tile_0({{.*}}memref<?xf16>{{.*}}) workgroup({{.*}}memref<1024xf16, #gpu.address_space<workgroup>>) kernel
// build the KT B-fragments up front, then the loop with the WMMA accumulate
// CHECK:         vector.insert
// CHECK:         scf.for
// CHECK:           rocdl.wmma.f32.16x16x16.f16
// the accumulator → f16 → LDS write-back (new carry) then the two barriers
// CHECK:           vector.extract
// CHECK:           arith.truncf
// CHECK:           memref.store {{.*}}memref<1024xf16, #gpu.address_space<workgroup>>
// CHECK:           gpu.barrier
// CHECK:         gpu.return
func.func @f(%init: tensor<32x32xf16>, %w: tensor<32x32xf16>) -> tensor<32x32xf16> {
  %r = "tessera.control_for"(%init, %w) {
    body = @wb, start = 0 : i64, stop = 3 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<32x32xf16>, tensor<32x32xf16>) -> tensor<32x32xf16>
  return %r : tensor<32x32xf16>
}

// -----
// ─── a >8-wave carry (64x64 = 16 waves) exceeds the single-WGP envelope and is
// ─── left untouched for the grid.sync frontier / the guard ──────────────────
func.func @wb2(%c: tensor<64x64xf16>, %w: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %0 = "tessera.matmul"(%c, %w) : (tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// CHECK-LABEL: func.func @g
// CHECK:       tessera.control_for
// CHECK-NOT:   gpu.func
func.func @g(%init: tensor<64x64xf16>, %w: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %r = "tessera.control_for"(%init, %w) {
    body = @wb2, start = 0 : i64, stop = 3 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf16>
  return %r : tensor<64x64xf16>
}

// -----
// ─── a non-multiple-of-16 carry is not a tile grid — left untouched ─────────
func.func @wb3(%c: tensor<24x24xf16>, %w: tensor<24x24xf16>) -> tensor<24x24xf16> {
  %0 = "tessera.matmul"(%c, %w) : (tensor<24x24xf16>, tensor<24x24xf16>) -> tensor<24x24xf16>
  return %0 : tensor<24x24xf16>
}

// CHECK-LABEL: func.func @h
// CHECK:       tessera.control_for
// CHECK-NOT:   gpu.func
func.func @h(%init: tensor<24x24xf16>, %w: tensor<24x24xf16>) -> tensor<24x24xf16> {
  %r = "tessera.control_for"(%init, %w) {
    body = @wb3, start = 0 : i64, stop = 3 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<24x24xf16>, tensor<24x24xf16>) -> tensor<24x24xf16>
  return %r : tensor<24x24xf16>
}

// -----
// ─── an f32 (non-WMMA) carry is left for the GEMV/other lowering ────────────
func.func @wb4(%c: tensor<32x32xf32>, %w: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = "tessera.matmul"(%c, %w) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @k
// CHECK:       tessera.control_for
// CHECK-NOT:   gpu.func
func.func @k(%init: tensor<32x32xf32>, %w: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %r = "tessera.control_for"(%init, %w) {
    body = @wb4, start = 0 : i64, stop = 3 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %r : tensor<32x32xf32>
}
