// P2 code review (2026-08-29) — WARPSPEC_MMA_NOT_TOKEN_SYNCED failed open
// through an intervening op.
//
// resolveThroughLoopCarry stops at the first non-block-argument defining op, so
// a cast, slice, or transpose between the async copy and the mma became the
// root; isAsyncDataProducer said no, the operand was dropped from the check
// entirely, and the missing-completion-token race went unreported — the exact
// fail-open class this pass documents itself as closing.
//
// RUN: tessera-opt --tessera-warpspec-legality --allow-unregistered-dialect \
// RUN:   -split-input-file -verify-diagnostics %s

func.func @copy_hidden_behind_a_transpose(%A: tensor<64x64xbf16>,
                                          %B: tensor<64x64xbf16>)
    -> tensor<64x64xf32> {
  %tA, %tok = "tile.async_copy"(%A)
      : (tensor<64x64xbf16>) -> (tensor<64x64xbf16>, !tile.async_token)
  %t = "tile.transpose"(%tA) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  // expected-error @+1 {{WARPSPEC_MMA_NOT_TOKEN_SYNCED}}
  %C = "tile.mma"(%t, %B) {sm = 90 : i32}
      : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

// -----

// Chasing through non-producer defs must not invent a producer where there is
// none: the same transpose over a plain function argument stays legal.
func.func @transpose_of_plain_value_is_legal(%A: tensor<64x64xbf16>,
                                             %B: tensor<64x64xbf16>)
    -> tensor<64x64xf32> {
  %t = "tile.transpose"(%A) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %C = "tile.mma"(%t, %B) {sm = 90 : i32}
      : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}
