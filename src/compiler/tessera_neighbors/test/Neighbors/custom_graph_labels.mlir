\
// RUN: tessera-opt %s -tessera-legalize-neighbors | FileCheck %s
// CHECK: kind = "custom_graph"
%topo = "tessera.neighbors.topology.create"() {kind = "custom_graph", defaults = "labels"} : () -> !tessera.neighbors.topology
