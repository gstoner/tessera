// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(rocm-wave-lds-legality)' -split-input-file -verify-diagnostics %s
//
// Loop coverage for the wave/LDS legality (software pipelining lives in loop
// bodies). The pass is a single program-order walk that descends into scf.for
// bodies: an in-body copy/wait/mma token chain is checked by SSA, and a loop-
// carried token (an scf.for iter_arg = block argument) is assumed resident
// rather than false-rejected.

// Legal: a self-contained pipelined loop body — copy mints a token, wait
// retires it, mma consumes the retired token. No diagnostic.
func.func @loop_body_synced(%d: memref<64xf16>, %s: memref<64xf16>,
                            %lb: index, %ub: index, %step: index,
                            %x: tensor<16x16xf16>) {
  scf.for %i = %lb to %ub step %step {
    %a = "tile.async_copy"(%d, %s, %i)
        : (memref<64xf16>, memref<64xf16>, index) -> !tile.async_token
    "tile.wait_async"(%a) : (!tile.async_token) -> ()
    %c = "tile.mma"(%x, %x, %a)
        : (tensor<16x16xf16>, tensor<16x16xf16>, !tile.async_token)
          -> tensor<16x16xf16>
  }
  return
}

// -----

// Illegal: in-body copy + mma with the token never waited — a real hazard,
// caught precisely inside the loop body by the def-use check.
func.func @loop_body_unwaited(%d: memref<64xf16>, %s: memref<64xf16>,
                              %lb: index, %ub: index, %step: index,
                              %x: tensor<16x16xf16>) {
  scf.for %i = %lb to %ub step %step {
    %a = "tile.async_copy"(%d, %s, %i)
        : (memref<64xf16>, memref<64xf16>, index) -> !tile.async_token
    // expected-error @+1 {{ROCM_WAVE_LDS_MISSING_WAITCNT}}
    %c = "tile.mma"(%x, %x, %a)
        : (tensor<16x16xf16>, tensor<16x16xf16>, !tile.async_token)
          -> tensor<16x16xf16>
  }
  return
}

// -----

// Legal: the classic loop-carried double buffer — the mma at the top of the
// body consumes the PREVIOUS iteration's copy via an scf.for iter_arg (a block
// argument), while this iteration prefetches+waits the next stage and yields it.
// The block-argument token is assumed resident (its retirement is on the back-
// edge, which this single-visit walk does not follow) — must NOT be flagged.
func.func @loop_carried_token(%d: memref<64xf16>, %s: memref<64xf16>,
                              %lb: index, %ub: index, %step: index,
                              %x: tensor<16x16xf16>, %tok0: !tile.async_token) {
  %r = scf.for %i = %lb to %ub step %step
      iter_args(%tok = %tok0) -> (!tile.async_token) {
    %c = "tile.mma"(%x, %x, %tok)
        : (tensor<16x16xf16>, tensor<16x16xf16>, !tile.async_token)
          -> tensor<16x16xf16>
    %a = "tile.async_copy"(%d, %s, %i)
        : (memref<64xf16>, memref<64xf16>, index) -> !tile.async_token
    "tile.wait_async"(%a) : (!tile.async_token) -> ()
    scf.yield %a : !tile.async_token
  }
  return
}
