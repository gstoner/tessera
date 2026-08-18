// RUN: tessera-opt %s -verify-diagnostics

// W1.1b: topology kind selects static versus mutable/replanned semantics. A
// typo must not silently become a static topology, and substring matches such
// as "not_dynamic" must not become mutable.
func.func @unknown_topology_kind() {
  // expected-error@+1 {{NEIGHBORS_TOPOLOGY_UNKNOWN_KIND}}
  %topology = "tessera.neighbors.topology.create"() {
      kind = "2d_mseh"
  } : () -> !tessera.neighbors.topology
  return
}
