// Phase C-NV — C6 token-synchronization legality (WARPSPEC_MMA_NOT_TOKEN_SYNCED).
//
// The SSA ordering half of arrival==init: arrival==init checks the mbarrier byte
// *count* (#tile.barrier); this checks the dependency *ordering* edge via the
// !tile.async_token. A consumer tile.mma that reads a producer's async-staged
// tile must also read a completion token from that same producer, so the matrix
// op is gated on copy completion by SSA — not program order.
// WarpSpecialization auto-mints this edge from the mma's data operands.
//
// RUN: tessera-opt --tessera-warpspec-legality --allow-unregistered-dialect \
// RUN:   -split-input-file -verify-diagnostics %s

// Synced: the mma reads the copy's token alongside its data — legal.
func.func @token_synced(%A: tensor<64x64xbf16>, %B: tensor<64x64xbf16>)
    -> tensor<64x64xf32> {
  %tA, %tok = "tile.async_copy"(%A)
      : (tensor<64x64xbf16>) -> (tensor<64x64xbf16>, !tile.async_token)
  "tile.wait_async"(%tok) : (!tile.async_token) -> ()
  %C = "tile.mma"(%tA, %B, %tok) {sm = 90 : i32}
      : (tensor<64x64xbf16>, tensor<64x64xbf16>, !tile.async_token)
        -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

// -----

// Unsynced: the mma reads the copy's data tile but no completion token — the
// matrix op could run before the copy lands. Flagged.
func.func @token_unsynced(%A: tensor<64x64xbf16>, %B: tensor<64x64xbf16>)
    -> tensor<64x64xf32> {
  %tA, %tok = "tile.async_copy"(%A)
      : (tensor<64x64xbf16>) -> (tensor<64x64xbf16>, !tile.async_token)
  // expected-error @+1 {{WARPSPEC_MMA_NOT_TOKEN_SYNCED}}
  %C = "tile.mma"(%tA, %B) {sm = 90 : i32}
      : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

// -----

// A consumer mma that reads no async-staged producer tile at all has nothing to
// synchronize — legal (no false positive on pure-value mmas).
func.func @no_async_input(%A: tensor<64x64xbf16>, %B: tensor<64x64xbf16>)
    -> tensor<64x64xf32> {
  %C = "tile.mma"(%A, %B) {sm = 90 : i32}
      : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

// -----

// Held-but-unwaited: the mma carries the copy's completion token (presence is
// satisfied) but NO tile.wait_async retires it before the mma — the copy is
// still in flight when the matrix op runs. Presence-only would miss this; the
// retirement check (converging with the ROCm legality) flags it.
func.func @token_held_unwaited(%A: tensor<64x64xbf16>, %B: tensor<64x64xbf16>)
    -> tensor<64x64xf32> {
  %tA, %tok = "tile.async_copy"(%A)
      : (tensor<64x64xbf16>) -> (tensor<64x64xbf16>, !tile.async_token)
  // expected-error @+1 {{WARPSPEC_MMA_TOKEN_NOT_RETIRED}}
  %C = "tile.mma"(%tA, %B, %tok) {sm = 90 : i32}
      : (tensor<64x64xbf16>, tensor<64x64xbf16>, !tile.async_token)
        -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}
