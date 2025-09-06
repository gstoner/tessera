// RUN: tessera-opt %s | FileCheck %s
// CHECK: tessera.tile.confidence_stats
// CHECK: tessera.tile.prefix_lcp
module {
  %logits = "mock.logits"() : () -> tensor<?x?xf32>
  %s = "tessera.tile.confidence_stats"(%logits) : (tensor<?x?xf32>) -> tensor<?xf32>
  %tok = "mock.tokens"() : () -> tensor<?x?xi32>
  %lcp = "tessera.tile.prefix_lcp"(%tok) : (tensor<?x?xi32>) -> i32
  "tessera.graph.return"() : () -> ()
}
