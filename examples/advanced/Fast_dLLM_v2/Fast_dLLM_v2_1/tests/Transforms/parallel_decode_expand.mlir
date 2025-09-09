// RUN: tessera-opt %s -tessera-parallel-decode-expand='K=4' | FileCheck %s
// CHECK: tessera.graph.branch
// CHECK: tessera.graph.join
module {
  %c = "tessera.graph.kv_cache.block_init"() {blocks=64, B_tok=16} : () -> !tessera.cache.handle
  %st0 = "tessera.graph.start_state"() : () -> !tessera.state
  %b = "tessera.graph.parallel_decode"(%st0, %c) {K=4} : (!tessera.state, !tessera.cache.handle) -> !tessera.branches
  "tessera.graph.use"(%b) : (!tessera.branches) -> ()
}
