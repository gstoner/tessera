// RUN: %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel %s \
// RUN:   --split-input-file | FileCheck %s
//
// The accept-set. Refusals live in wmma_gemm_numeric_policy_accum_invalid.mlir,
// so each file states one thing and neither can pass by accident.

// NUMPOL-CARRIER-1 (integrated-plan queue row 3b) — the declared accumulator
// gets a CONSUMER on this backend.
//
// Measured 2026-08-25: `numeric_policy` was carried faithfully all the way
// here. TileIRLoweringPass puts it on `tile.mma`; TileToROCM copies it onto
// `tessera_rocm.wmma_gemm` — and all three of its uses there are
// `copyAttrIfPresent`. Forwarded, never read. The accumulator that actually
// reached the hardware was inferred from the STORAGE dtype alone
// (`fragmentAcc = T.isInt ? "i32" : "f32"`), in a file that did not mention
// numeric_policy at all. Two sources of truth for one fact, and the declared
// one lost — they agreed only because every real program asks for fp32.
//
// That is Decision #29 (a declaration whose consumer does not exist reads as a
// closed contract while carrying nothing) on top of #21a (a semantic key
// silently defaulting). `accum` decides what the program COMPUTES.

// ── the declared accumulator matches what this path provides ──
"tessera_rocm.wmma_gemm"() {
  name = "f16_accum_fp32", m = 16 : i64, n = 16 : i64, k = 16 : i64,
  mt = 1 : i64, nt = 1 : i64, dtype = "f16", schedule_arch = "gfx1151",
  numeric_policy = {storage = "fp16", accum = "fp32"}
} : () -> ()
// CHECK: gpu.func @f16_accum_fp32

// -----

// ── integer storage, integer accumulator ──
"tessera_rocm.wmma_gemm"() {
  name = "int8_accum_int32", m = 16 : i64, n = 16 : i64, k = 16 : i64,
  mt = 1 : i64, nt = 1 : i64, dtype = "int8", schedule_arch = "gfx1151",
  numeric_policy = {storage = "int8", accum = "int32"}
} : () -> ()
// CHECK: gpu.func @int8_accum_int32

// -----

// ── no policy: unchanged, and it must stay that way ──
// Every kernel this backend generates today comes through here. A consumer
// that altered the no-policy path would not be honouring a contract, it would
// be changing every existing program.
"tessera_rocm.wmma_gemm"() {
  name = "no_policy", m = 16 : i64, n = 16 : i64, k = 16 : i64,
  mt = 1 : i64, nt = 1 : i64, dtype = "bf16", schedule_arch = "gfx1151"
} : () -> ()
// CHECK: gpu.func @no_policy
