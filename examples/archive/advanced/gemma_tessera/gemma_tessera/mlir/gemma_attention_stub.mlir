// --- Tessera Target IR stub for Gemma attention ---
// This is illustrative and not runnable as-is.
// Shows how a fused attention op might appear and lower to backend-specific IR.
#attrs = {
  tile_shape = [64, 64, 64],
  dtype = f16,
  causal = true
}
tessera.region @decoder_block(%x: tensor<?x?xC>)
{
  %qkv = tessera.mma.qkv_pack %x : (tensor<?x?xC>) -> tensor<?x?x(C+2*Dk)>
  %q, %k, %v = tessera.tensor.split %qkv axis = -1 sizes = [C, Dk, Dk]
  %q = tessera.rope.apply %q : (tensor<?x?xHxd>) -> tensor<?x?xHxd> {theta = 10000.0}
  %k = tessera.rope.apply %k : (tensor<?x?xKxd>) -> tensor<?x?xKxd> {theta = 10000.0}
  %out = tessera.attention.flash %q, %k, %v : (...) -> tensor<?x?xC> attributes = #attrs
  %proj = tessera.mma.proj %out : (tensor<?x?xC>) -> tensor<?x?xC>
  // MLP
  %h = tessera.norm.rms %proj : (tensor<?x?xC>) -> tensor<?x?xC> {eps = 1e-6}
  %g = tessera.mlp.swi_glu %h : (tensor<?x?xC>) -> tensor<?x?xC>
  tessera.yield %g : tensor<?x?xC>
}
