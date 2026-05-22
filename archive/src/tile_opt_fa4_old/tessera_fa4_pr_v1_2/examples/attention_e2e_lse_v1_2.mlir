// E2E Forward Attention with LSE carry (v1.2)
module {
  tessera.schedule @fa4_pipeline {
    %w_load  = tessera.schedule.warp "load", 1
    %w_mma   = tessera.schedule.warp "mma", 1
    %w_smx   = tessera.schedule.warp "softmax", 8
    %w_corr  = tessera.schedule.warp "correction", 4
    %w_epi   = tessera.schedule.warp "epilogue", 2
    tessera.schedule.pipe %w_load, %w_mma, %w_smx, %w_corr, %w_epi { buffering = {K=3, V=3, S=2, O=2} }
    tessera.schedule.policy "persistent", 1, "static"
  }
  tessera.numerics.softmax "poly3", 2.0e-3
  %acc = "tessera.tile.mma.tcgen05"(%q, %k -> %accbuf, 2) : (memref<128x64xbf16>, memref<128x64xbf16>, memref<128x64xf32>) -> memref<128x64xf32>
  %lse = "tessera.attn.lse.save"(%acc) : (memref<128x64xf32>) -> memref<128xf32>
  // backward later: %lse_in = "tessera.attn.lse.load"() : () -> memref<128xf32>
}
