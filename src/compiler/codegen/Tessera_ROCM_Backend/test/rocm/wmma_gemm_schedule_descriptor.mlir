// RUN: %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel %s | FileCheck %s

"tessera_rocm.wmma_gemm"() {
  name = "scheduled", m = 16 : i64, n = 16 : i64, k = 16 : i64,
  mt = 3 : i64, nt = 4 : i64, dtype = "f16",
  schedule_arch = "gfx1151", schedule_pipeline_stages = 2 : i64,
  schedule_lds_layout = "swizzle", schedule_ownership = "wave",
  schedule_vgpr_estimate = 99 : i64,
  schedule_source = "gfx1151 measured GEMM macro-tile + ROCm budget models"
} : () -> ()

// CHECK: gpu.func @scheduled
// CHECK-SAME: tessera.rocm.schedule_arch = "gfx1151"
// CHECK-SAME: tessera.rocm.schedule_lds_layout = "swizzle"
// CHECK-SAME: tessera.rocm.schedule_ownership = "wave"
// CHECK-SAME: tessera.rocm.schedule_pipeline_stages = 2 : i64
// CHECK-SAME: tessera.rocm.schedule_vgpr_estimate = 99 : i64
