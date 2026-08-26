// RUN: %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel %s | FileCheck %s

// Reduced-precision WMMA is deliberately opt-in. The acknowledgement string
// names the measured accuracy class; numeric_policy remains the computation's
// semantic source of truth.
"tessera_rocm.wmma_gemm"() {
  name = "f16_accum_f16", m = 16 : i64, n = 16 : i64, k = 16 : i64,
  mt = 1 : i64, nt = 1 : i64, dtype = "f16", output = "f16",
  schedule_arch = "gfx1151",
  numeric_policy = {storage = "fp16", accum = "fp16"},
  tessera.rocm.reduced_precision_accumulation = "f16_wmma_accuracy_cost_ack_v1"
} : () -> ()

// CHECK: gpu.func @f16_accum_f16({{.*}}memref<?xf16>{{.*}}memref<?xf16>
// CHECK: vector<16xf16>
// CHECK: tessera_rocm.wmma
// CHECK-SAME: vector<16xf16>
// CHECK: vector.extract {{.*}}[0]
// CHECK: vector.extract {{.*}}[2]
// CHECK: vector.extract {{.*}}[14]
