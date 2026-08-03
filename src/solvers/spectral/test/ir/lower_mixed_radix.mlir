// RUN: ts-spectral-opt --tessera-legalize-spectral --lower-spectral-to-target-ir %s | FileCheck %s
//
// PR #496 review (P1) — statically-shaped non-power-of-two lengths must reach
// the generic radix-r kernel and the Bluestein driver.
//
// `stageSymbolFor` mapped EVERY radix other than 4 to `ts_stockham_r2_*`, with
// a comment saying "until radix-3/5/7 specializations land". They had landed,
// and nothing routed to them: N = 12 = 4x3 emitted a radix-4 call followed by a
// radix-2 call for a stage that is radix 3, producing an incorrect transform
// through the COMPILER path even though the runtime driver -- which factors N
// itself -- was correct. Direct driver tests could not see it, which is why
// this fixture exercises the passes rather than the kernel.
//
// The legalizer also used to factor N itself over radices {7,5,3,4,2}, a third
// copy of a decision the runtime drivers share via FFTPlan.h, and pushed any
// residual prime as a "stage" of that radix -- a stage nothing could execute.
// Both now come from the shared planner.

module attributes {tessera.target = "cpu"} {
  // 12 = 4 x 3: one hand-written radix-4 stage, one generic radix-3 stage.
  // CHECK-LABEL: func.func @fft_12
  func.func @fft_12(%src : memref<12xcomplex<f32>>, %dst : memref<12xcomplex<f32>>) {
    %plan = "tessera_spectral.plan"() {
      axes = [0], elem_precision = "fp16", acc_precision = "f32",
      scaling = "none", inplace = false, is_real_input = false,
      norm_policy = "backward"} : () -> !tessera_spectral.plan
    // CHECK: tessera_spectral.fft
    // CHECK-SAME: tessera.target_ir.stage_calls = ["ts_stockham_r4_cpu", "ts_stockham_rn_cpu"]
    // The radix travels with the symbol: ts_stockham_rn_* takes it as an
    // argument, and it cannot be recovered from the symbol name.
    // CHECK-SAME: tessera.target_ir.stage_radices = [4, 3]
    "tessera_spectral.fft"(%plan, %src, %dst)
      : (!tessera_spectral.plan, memref<12xcomplex<f32>>, memref<12xcomplex<f32>>) -> ()
    return
  }

  // 15 = 3 x 5: no power-of-two factor at all. Previously two radix-2 calls.
  // CHECK-LABEL: func.func @fft_15
  func.func @fft_15(%src : memref<15xcomplex<f32>>, %dst : memref<15xcomplex<f32>>) {
    %plan = "tessera_spectral.plan"() {
      axes = [0], elem_precision = "fp16", acc_precision = "f32",
      scaling = "none", inplace = false, is_real_input = false,
      norm_policy = "backward"} : () -> !tessera_spectral.plan
    // CHECK: tessera_spectral.fft
    // CHECK-SAME: tessera.target_ir.stage_calls = ["ts_stockham_rn_cpu", "ts_stockham_rn_cpu"]
    // CHECK-SAME: tessera.target_ir.stage_radices = [3, 5]
    "tessera_spectral.fft"(%plan, %src, %dst)
      : (!tessera_spectral.plan, memref<15xcomplex<f32>>, memref<15xcomplex<f32>>) -> ()
    return
  }

  // 101 is prime and above the radix bound: Bluestein is a different
  // ALGORITHM, not a longer stage list, so the op routes to the driver and
  // says so. Previously the legalizer emitted a "stage" of radix 101 and the
  // lowering turned that into a radix-2 call.
  // CHECK-LABEL: func.func @fft_101
  func.func @fft_101(%src : memref<101xcomplex<f32>>, %dst : memref<101xcomplex<f32>>) {
    %plan = "tessera_spectral.plan"() {
      axes = [0], elem_precision = "fp16", acc_precision = "f32",
      scaling = "none", inplace = false, is_real_input = false,
      norm_policy = "backward"} : () -> !tessera_spectral.plan
    // CHECK: tessera_spectral.fft
    // CHECK-SAME: tessera.target_ir.bluestein
    // CHECK-SAME: tessera.target_ir.call = "ts_fft_stockham_cpu"
    // CHECK-NOT: ts_stockham_r2_cpu
    "tessera_spectral.fft"(%plan, %src, %dst)
      : (!tessera_spectral.plan, memref<101xcomplex<f32>>, memref<101xcomplex<f32>>) -> ()
    return
  }

  // A power of two must be UNCHANGED by all of this — the regression guard on
  // the path that already worked.
  // CHECK-LABEL: func.func @fft_256
  func.func @fft_256(%src : memref<256xcomplex<f32>>, %dst : memref<256xcomplex<f32>>) {
    %plan = "tessera_spectral.plan"() {
      axes = [0], elem_precision = "fp16", acc_precision = "f32",
      scaling = "none", inplace = false, is_real_input = false,
      norm_policy = "backward"} : () -> !tessera_spectral.plan
    // CHECK: tessera_spectral.fft
    // CHECK-SAME: tessera.target_ir.stage_calls = ["ts_stockham_r4_cpu", "ts_stockham_r4_cpu", "ts_stockham_r4_cpu", "ts_stockham_r4_cpu"]
    "tessera_spectral.fft"(%plan, %src, %dst)
      : (!tessera_spectral.plan, memref<256xcomplex<f32>>, memref<256xcomplex<f32>>) -> ()
    return
  }
}
