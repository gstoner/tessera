// RUN: tessera-opt %s -tessera-kv-cache-blockify -tessera-parallel-decode-expand | FileCheck %s
module {
  // Graph IR: high-level decode step with kv cache handle.
  %c = "tessera.graph.kv_cache.block_init"() {blocks = 64, B_tok = 16, approx = "fp8_e4m3+fp16_stripe"} : () -> !tessera.cache.handle
  %st0 = "tessera.graph.start_state"() : () -> !tessera.state

  // Parallel decoding (K=4)
  %branches = "tessera.graph.parallel_decode"(%st0, %c) {K = 4} : (!tessera.state, !tessera.cache.handle) -> !tessera.branches

  "tessera.graph.region"(%branches) ({
    // Inside a branch
    %q,%k,%v = "tessera.graph.attn_io"() : () -> (tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>)
    // After blockify pass we expect kv_block_read/pack around attn.
    %o = "tessera.tile.attn_bidir"(%q,%k,%v) {window = 16} : (...) -> tensor<?x?x?xf16>
    "tessera.tile.confidence_stats"(%o) : (tensor<?x?x?xf16>) -> tensor<?xf32>
    "tessera.graph.yield"() : () -> ()
  }) : (!tessera.branches) -> ()

  %st1 = "tessera.graph.validate_and_merge"(%branches) {tau = 0.75, window = 8} : (!tessera.branches) -> !tessera.state
}

// CHECK: tessera.tile.kv_block_read
// CHECK: tessera.tile.attn_bidir
// CHECK: tessera.tile.confidence_stats
// CHECK: tessera.graph.validate_and_merge
