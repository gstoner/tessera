// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(rocm-wave-lds-legality)' -split-input-file -verify-diagnostics %s
//
// Phase C-ROCm — token def-use legality (the SSA replacement for the program-
// order heuristic). An mma's !tile.async_token operands name exactly the stages
// it consumes; each must be retired by a tile.wait_async before the mma. There
// is NO program-order re-derivation, so a live prefetch can never be mistaken
// for a dependency — software-pipelined double buffering is legal by construction.

// Double buffer via SSA tokens: the mma consumes the waited stage %a while the
// prefetch %b stays outstanding. Legal — %b is not one of the mma's operands.
func.func @token_double_buffer(%d0: memref<64xf16>, %d1: memref<64xf16>,
                               %s: memref<64xf16>, %i: index,
                               %x: tensor<16x16xf16>) {
  %a = "tile.async_copy"(%d0, %s, %i)
      : (memref<64xf16>, memref<64xf16>, index) -> !tile.async_token
  "tile.wait_async"(%a) : (!tile.async_token) -> ()
  %b = "tile.async_copy"(%d1, %s, %i)
      : (memref<64xf16>, memref<64xf16>, index) -> !tile.async_token
  %c = "tile.mma"(%x, %x, %a)
      : (tensor<16x16xf16>, tensor<16x16xf16>, !tile.async_token)
        -> tensor<16x16xf16>
  return
}

// -----

// The mma consumes a token from a copy that was never waited — a real hazard,
// caught precisely by the def-use check (the token operand %a is not retired).
func.func @token_unwaited(%d0: memref<64xf16>, %s: memref<64xf16>, %i: index,
                          %x: tensor<16x16xf16>) {
  %a = "tile.async_copy"(%d0, %s, %i)
      : (memref<64xf16>, memref<64xf16>, index) -> !tile.async_token
  // expected-error @+1 {{ROCM_WAVE_LDS_MISSING_WAITCNT}}
  %c = "tile.mma"(%x, %x, %a)
      : (tensor<16x16xf16>, tensor<16x16xf16>, !tile.async_token)
        -> tensor<16x16xf16>
  return
}

// -----

// s_barrier drains all outstanding tokens, so an mma consuming a pre-barrier
// copy's token afterwards is legal even without a targeted wait.
func.func @token_s_barrier_drains(%d0: memref<64xf16>, %s: memref<64xf16>,
                                  %i: index, %x: tensor<16x16xf16>) {
  %a = "tile.async_copy"(%d0, %s, %i)
      : (memref<64xf16>, memref<64xf16>, index) -> !tile.async_token
  "tile.s_barrier"() {tile.barrier = #tile.barrier<kind = "s_barrier", expect = 0>} : () -> ()
  %c = "tile.mma"(%x, %x, %a)
      : (tensor<16x16xf16>, tensor<16x16xf16>, !tile.async_token)
        -> tensor<16x16xf16>
  return
}
