\
// RUN: tessera-opt %s -tessera-legalize-neighbors -tessera-halo-infer | FileCheck %s
// CHECK: halo.width = [1, 1, 1]
%topo = "tessera.neighbors.topology.create"() {kind = "3d_mesh", defaults = "faces"} : () -> !tessera.neighbors.topology
%a = "tessera.neighbors.neighbor.read"(%arg0, %topo, #tessera.neighbors.delta_array<[+1,0,0]>) : (tensor<?x?x?xf32>, !tessera.neighbors.topology, attr) -> tensor<?x?xf32>
