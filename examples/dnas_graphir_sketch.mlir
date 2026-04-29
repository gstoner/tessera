// DNAS GraphIR sketch: MixedOp over attention/block alternatives.
// This is a compiler-facing example, not a complete executable module.

tessera.graph.func @search_block(%x: tensor<?x2048xbf16>) -> tensor<?x2048xbf16> {
  %alpha = tessera.graph.arch.parameter {size = 4, name = "block0.attn", init = 0.0}
  %gate = tessera.graph.arch.gumbel_softmax %alpha {temperature = 4.0, seed = 17}

  %y0 = tessera.graph.op.flash_attention(%x) {causal = true}
        : tensor<?x2048xbf16> -> tensor<?x2048xbf16>
  %y1 = tessera.graph.op.performer_attention(%x) {kernel = "relu"}
        : tensor<?x2048xbf16> -> tensor<?x2048xbf16>
  %y2 = tessera.graph.op.multi_query_attention(%x) {heads = 8}
        : tensor<?x2048xbf16> -> tensor<?x2048xbf16>
  %y3 = tessera.graph.op.gmlp(%x) {expansion = 4}
        : tensor<?x2048xbf16> -> tensor<?x2048xbf16>

  %y = tessera.graph.arch.weighted_sum %y0, %y1, %y2, %y3 by %gate
       : tensor<?x2048xbf16>, tensor<?x2048xbf16>, tensor<?x2048xbf16>, tensor<?x2048xbf16> -> tensor<?x2048xbf16>
  return %y : tensor<?x2048xbf16>
}

// After search, freeze by argmax, erase unused candidates, then lower the
// discrete graph through Schedule IR, Tile IR, and Target IR.
