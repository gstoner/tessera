// RUN: not %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel %s \
// RUN:   --split-input-file 2>&1 | FileCheck %s

// NUMPOL-CARRIER-1 — the refusal. Accept-set:
// wmma_gemm_numeric_policy_accum.mlir.
//
// gfx1151 really does have an f16-accumulate WMMA: the in-repo ISA archive
// records V_WMMA_F16_16X16X16_F16 on RDNA 3.5. But its ROCDL form is
// `(v16f16, v16f16, v16f16) -> v16f16` with an `opsel` bit selecting a half —
// a different accumulator ABI from the v8f32 path. It is now admitted only
// behind an exact measured-cost acknowledgement; omitting that acknowledgement
// remains a named refusal rather than silently substituting fp32.

"tessera_rocm.wmma_gemm"() {
  name = "f16_accum_f16_refused", m = 16 : i64, n = 16 : i64, k = 16 : i64,
  mt = 1 : i64, nt = 1 : i64, dtype = "f16", schedule_arch = "gfx1151",
  numeric_policy = {storage = "fp16", accum = "fp16"}
} : () -> ()
// CHECK: ROCM_WMMA_ACCUM_UNSUPPORTED
// CHECK-SAME: accum="fp16"
// CHECK-SAME: opt-in accuracy class
// CHECK-SAME: 5212x

// -----

// ── an integer path asked for a float accumulator ──
"tessera_rocm.wmma_gemm"() {
  name = "int8_accum_fp32_refused", m = 16 : i64, n = 16 : i64, k = 16 : i64,
  mt = 1 : i64, nt = 1 : i64, dtype = "int8", schedule_arch = "gfx1151",
  numeric_policy = {storage = "int8", accum = "fp32"}
} : () -> ()
// CHECK: ROCM_WMMA_ACCUM_UNSUPPORTED
// CHECK-SAME: accumulates in int32
