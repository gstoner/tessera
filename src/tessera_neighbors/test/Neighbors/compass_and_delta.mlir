\
// RUN: tessera-opt %s -tessera-legalize-neighbors | FileCheck %s
// CHECK: delta_array<[0, +1]>
%topo = "tessera.neighbors.topology.create"() {kind = "2d_mesh"} : () -> !tessera.neighbors.topology
%r = "tessera.neighbors.neighbor.read"(%arg0, %topo, #tessera.neighbors.delta_array<[0,+1]>) : (tensor<?x?xf32>, !tessera.neighbors.topology, attr) -> tensor<?xf32>
