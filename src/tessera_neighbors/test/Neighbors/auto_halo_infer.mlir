\
// RUN: tessera-opt %s -tessera-legalize-neighbors -tessera-halo-infer | FileCheck %s
// CHECK: halo.width = [1, 1]
%topo = "tessera.neighbors.topology.create"() {kind = "2d_mesh", defaults = "von_neumann"} : () -> !tessera.neighbors.topology
%st   = "tessera.neighbors.stencil.define"() {taps = [#tessera.neighbors.delta_array<[0,0]>, #tessera.neighbors.delta_array<[+1,0]>, #tessera.neighbors.delta_array<[-1,0]>, #tessera.neighbors.delta_array<[0,+1]>, #tessera.neighbors.delta_array<[0,-1]>]} : () -> !tessera.stencil
%out  = "tessera.neighbors.stencil.apply"(%st, %arg0, %topo) : (!tessera.stencil, tensor<?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?xf32>
