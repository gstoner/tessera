// RUN: tessera-opt --tessera-lower-to-gpu --gpu-target=sm_90a %s | FileCheck %s
// REQUIRES: tessera_opt_built
//
// Sprint G-4 — Fused AdamW step.  Single kernel performs:
//   m = β1 · m + (1-β1) · g
//   v = β2 · v + (1-β2) · g²
//   m_hat = m / (1 - β1^t)
//   v_hat = v / (1 - β2^t)
//   p = p - lr · (m_hat / (sqrt(v_hat) + eps) + wd · p)
//
// Mixed-precision: param stored bf16, accumulator + m + v stored fp32.
// No WGMMA — pure elementwise; vectorized over the parameter buffer.

module attributes {tessera.target = "nvidia_sm90"} {
  func.func @adamw_step_fused(
      %param : memref<?xbf16, 1>,
      %grad : memref<?xbf16, 1>,
      %m : memref<?xf32, 1>,
      %v : memref<?xf32, 1>) {
    "tessera.optim.adamw_step"(%param, %grad, %m, %v) {
      lr = 1.0e-4 : f32,
      beta1 = 0.9 : f32,
      beta2 = 0.999 : f32,
      eps = 1.0e-8 : f32,
      weight_decay = 0.01 : f32,
      step = 1 : i64,
      param_dtype = "bf16",
      accum_dtype = "fp32",
      cuda_arch_min = "sm_80"
    } : (memref<?xbf16, 1>,
         memref<?xbf16, 1>,
         memref<?xf32, 1>,
         memref<?xf32, 1>) -> ()
    return
  }
}

// CHECK: tessera.optim.adamw_step
// CHECK-SAME: param_dtype = "bf16"
// CHECK-SAME: accum_dtype = "fp32"
//
// Vectorized elementwise — no WGMMA:
// CHECK-NOT: wgmma.mma_async
//
// Cooperative vectorized load + store:
// CHECK-DAG: ld.global.v4
// CHECK-DAG: st.global.v4
