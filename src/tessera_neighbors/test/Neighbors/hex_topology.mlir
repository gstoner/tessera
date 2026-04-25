\
// RUN: tessera-opt %s -tessera-legalize-neighbors | FileCheck %s
// CHECK: kind = "hex_2d"
%topo = "tessera.neighbors.topology.create"() {kind = "hex_2d", axes = "q,r", defaults = "hex_axial"} : () -> !tessera.neighbors.topology
