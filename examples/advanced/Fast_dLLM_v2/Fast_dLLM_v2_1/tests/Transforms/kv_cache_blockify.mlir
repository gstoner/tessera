// RUN: tessera-opt %s -tessera-kv-cache-blockify | FileCheck %s
// CHECK: tessera.tile.kv_block_read
// CHECK: tessera.tile.attn_bidir
// CHECK: tessera.tile.kv_block_pack
module {
  %c = "tessera.graph.kv_cache.block_init"() {blocks=64, B_tok=16} : () -> !tessera.cache.handle
  %q,%k,%v = "tessera.graph.attn_io"() : () -> (tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>)
  %o = "tessera.tile.attn_bidir"(%q,%k,%v) {window=16} : (...) -> tensor<?x?x?xf16>
  "tessera.graph.return"(%o) : (tensor<?x?x?xf16>) -> ()
}
