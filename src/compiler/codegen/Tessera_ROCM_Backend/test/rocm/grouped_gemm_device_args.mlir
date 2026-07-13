// RUN: %trop --allow-unregistered-dialect --generate-rocm-moe-kernel %s | FileCheck %s

"tessera_rocm.grouped_gemm"() {name = "grouped_gemm"} : () -> ()

// One generated kernel takes X, expert weights, device offsets, output, then
// T/K/N/E. The offsets memref is read in the device-side expert-owner loop.
// CHECK: gpu.module @grouped_gemm_mod
// CHECK: gpu.func @grouped_gemm(
// CHECK-SAME: %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xi32>, %{{.*}}: memref<?xf32>, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index
// CHECK: scf.for
// CHECK: memref.load
// CHECK: arith.select
