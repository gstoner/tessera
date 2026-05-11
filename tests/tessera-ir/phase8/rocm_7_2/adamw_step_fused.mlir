// RUN: tessera-opt --tessera-lower-to-rocm --rocm-target=gfx942 %s | FileCheck %s
// REQUIRES: tessera_opt_built
//
// Sprint H-4 — Fused AdamW step on ROCm.  Vectorized elementwise — no
// MFMA. Mixed-precision: bf16 param + fp32 accumulator/m/v.

module attributes {tessera.target = "rocm_gfx942"} {
  func.func @adamw_step_rocm(
      %param : memref<?xbf16, 1>,
      %grad : memref<?xbf16, 1>,
      %m : memref<?xf32, 1>,
      %v : memref<?xf32, 1>) {
    "tessera_rocm.optim.adamw_step"(%param, %grad, %m, %v) {
      lr = 1.0e-4 : f32,
      beta1 = 0.9 : f32,
      beta2 = 0.999 : f32,
      eps = 1.0e-8 : f32,
      weight_decay = 0.01 : f32,
      step = 1 : i64,
      param_dtype = "bf16",
      accum_dtype = "fp32",
      hipcc_arch = "gfx942"
    } : (memref<?xbf16, 1>,
         memref<?xbf16, 1>,
         memref<?xf32, 1>,
         memref<?xf32, 1>) -> ()
    return
  }
}

// CHECK: tessera_rocm.optim.adamw_step
// CHECK-SAME: param_dtype = "bf16"
// CHECK-SAME: accum_dtype = "fp32"
//
// Vectorized buffer loads — no MFMA:
// CHECK-NOT: llvm.amdgcn.mfma
// CHECK-DAG: llvm.amdgcn.buffer.load
