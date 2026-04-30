// Minimal illustrative lowering from high-level Tessera ops to Target IR dialect.
// Not runnable as-is; serves as a reference for implementing a real Pass.
module {
  // High-level fused attention op:
  //   %out = tessera.attention.flash %q, %k, %v {tile_shape=[64,64,16], dtype=f16, causal}
  // Target-IR sketch:
  //   %tq = ttarget.tma.load %q
  //   %tk = ttarget.tma.load %k
  //   %tv = ttarget.tma.load %v
  //   %o  = ttarget.attn.mma %tq, %tk, %tv {wgmma=true, stages=3, smem_swizzle="128b"}
  //   ttarget.tma.store %o, %out
}

