// generate-rocm-batched-gemm-f32-kernel expands a tessera_rocm.batched_gemm_f32
// directive into a single-launch batched f32 GEMM gpu.func: the batch is folded
// into the 1-D grid (gid decodes batch then tile), each thread computes a 4×4
// output tile with per-batch A/B/C base offsets. On-device execution on gfx1151
// is proven by tests/unit/test_rocm_batched_gemm_f32.py.
//
// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s --generate-rocm-batched-gemm-f32-kernel | FileCheck %s

// CHECK-LABEL: gpu.module @bg_mod
// (A, B, C : memref<?xf32>, Batch, M, N, K : index) = 3 memref + 4 index
// CHECK: gpu.func @bg
// CHECK-SAME: memref<?xf32>
// CHECK-SAME: index
// batch decode (divui/remui by tilesPerBatch) + the register-blocked k-loop
// CHECK: arith.divui
// CHECK: arith.remui
// CHECK: scf.for {{.*}}iter_args({{.*}}) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: memref.store
// CHECK: gpu.return
"tessera_rocm.batched_gemm_f32"() {name = "bg"} : () -> ()
