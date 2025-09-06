// RUN: tessera-opt %s -tessera-kv-cache-blockify | FileCheck %s
// Verify block windows and boundary stripes handling.
module {
  %c = "tessera.graph.kv_cache.block_init"() {blocks=64, B_tok=16, approx="fp8_e4m3+fp16_stripe"} : () -> !tessera.cache.handle
  %q,%k,%v = "tessera.graph.attn_io"() : () -> (tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>)
  %o = "tessera.tile.attn_bidir"(%q,%k,%v) {window=16} : (...) -> tensor<?x?x?xf16>
  "tessera.graph.return"(%o) : (tensor<?x?x?xf16>) -> ()
}

// CHECK: tessera.tile.kv_block_read
// CHECK-SAME: window = 16
// CHECK: tessera.tile.kv_block_pack
// CHECK: tessera.tile.attn_bidir
