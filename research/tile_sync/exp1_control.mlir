// Experiment 1 CONTROL (straight-line): tile.mma reads async-staged data from a
// tile.async_copy defining op with NO token edge. The warp-spec legality pass
// should fire WARPSPEC_MMA_NOT_TOKEN_SYNCED — proving the check works when the
// producer is reachable via getDefiningOp().
func.func @control_straight_line(%src: tensor<64x64xf16>) {
  %tile, %tok = tile.async_copy %src : (tensor<64x64xf16>) -> (tensor<64x64xf16>, !tile.async_token)
  %acc = tile.mma %tile, %tile : (tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf32>
  return
}
