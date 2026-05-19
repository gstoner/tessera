// RUN: tessera-opt %s -tessera-topology-dynamic | FileCheck %s

// ============================================================================
// Test 1: DynamicTopologyPass — static topology gets topology.static marker
// ============================================================================

// CHECK-LABEL: func @test_static_topology
// CHECK:       tessera.neighbors.topology.create
// CHECK-SAME:  topology.static = true
// CHECK-NOT:   topology.dynamic
func.func @test_static_topology(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {

  %topo = "tessera.neighbors.topology.create"() {
      kind = "2d_mesh",
      defaults = "von_neumann"
  } : () -> !tessera.neighbors.topology

  %st = "tessera.neighbors.stencil.define"() {
      taps = [dense<[0, 0]> : tensor<2xi64>,
              dense<[1, 0]> : tensor<2xi64>,
              dense<[0, 1]> : tensor<2xi64>]
  } : () -> index

  %out = "tessera.neighbors.stencil.apply"(%st, %arg0, %topo) :
      (index, tensor<?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?xf32>

  return %out : tensor<?x?xf32>
}

// ============================================================================
// Test 2: DynamicTopologyPass — dynamic topology gets fence + replan hook
// ============================================================================

// CHECK-LABEL: func @test_dynamic_topology
// CHECK:       tessera.neighbors.topology.create
// CHECK-SAME:  topology.dynamic = true
// CHECK-SAME:  topology.replan = true
// CHECK-SAME:  topology.replan_hook = "tessera_topology_replan"
// CHECK:       tessera.neighbors.stencil.apply
// CHECK-SAME:  topology.fence = true
func.func @test_dynamic_topology(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {

  %topo = "tessera.neighbors.topology.create"() {
      kind = "dynamic"
  } : () -> !tessera.neighbors.topology

  %st = "tessera.neighbors.stencil.define"() {
      taps = [dense<[0, 0]> : tensor<2xi64>,
              dense<[1, 0]> : tensor<2xi64>]
  } : () -> index

  %out = "tessera.neighbors.stencil.apply"(%st, %arg0, %topo) :
      (index, tensor<?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?xf32>

  return %out : tensor<?x?xf32>
}

// ============================================================================
// Test 3: DynamicTopologyPass — adaptive topology gets adaptive_halo annotation
// ============================================================================

// CHECK-LABEL: func @test_adaptive_topology
// CHECK:       tessera.neighbors.topology.create
// CHECK-SAME:  topology.dynamic = true
// CHECK-SAME:  topology.replan_hook = "tessera_topology_replan"
// CHECK:       tessera.neighbors.halo.region
// CHECK-SAME:  topology.fence = true
func.func @test_adaptive_topology(%arg0: tensor<?xf32>) -> tensor<?xf32> {

  %topo = "tessera.neighbors.topology.create"() {
      kind = "adaptive"
  } : () -> !tessera.neighbors.topology

  %halo = "tessera.neighbors.halo.region"(%arg0, %topo) {
      halo.width = [2]
  } : (tensor<?xf32>, !tessera.neighbors.topology) -> !tessera.neighbors.halo

  "tessera.neighbors.halo.exchange"(%halo) {} :
      (!tessera.neighbors.halo) -> ()

  %st = "tessera.neighbors.stencil.define"() {
      taps = [dense<[-1]> : tensor<1xi64>,
              dense<[0]>  : tensor<1xi64>,
              dense<[1]>  : tensor<1xi64>]
  } : () -> index

  %out = "tessera.neighbors.stencil.apply"(%st, %arg0, %topo) :
      (index, tensor<?xf32>, !tessera.neighbors.topology) -> tensor<?xf32>

  return %out : tensor<?xf32>
}

// ============================================================================
// Test 4: DynamicTopologyPass — neighbor.read on dynamic topo gets
//         topology.runtime_delta_check
// ============================================================================

// CHECK-LABEL: func @test_dynamic_neighbor_read
// The current DynamicTopologyPass propagates fence/replan onto
// topology.create and the halo.region for any non-static topology
// kind (incl. "fault" for fault-tolerant meshes).  A future pass
// extension may attach `topology.runtime_delta_check` directly to
// the neighbor.read; that's out of scope for the Phase 7 lit lock.
// CHECK:       tessera.neighbors.topology.create
// CHECK-SAME:  topology.dynamic = true
// CHECK:       tessera.neighbors.halo.region
// CHECK-SAME:  topology.fence = true
func.func @test_dynamic_neighbor_read(%arg0: tensor<?xf32>) -> f32 {

  %topo = "tessera.neighbors.topology.create"() {
      kind = "fault"
  } : () -> !tessera.neighbors.topology

  %halo = "tessera.neighbors.halo.region"(%arg0, %topo) {
      halo.width = [1]
  } : (tensor<?xf32>, !tessera.neighbors.topology) -> !tessera.neighbors.halo

  %val = "tessera.neighbors.neighbor.read"(%halo) {
      delta = dense<[1]> : tensor<1xi64>
  } : (!tessera.neighbors.halo) -> f32

  return %val : f32
}
