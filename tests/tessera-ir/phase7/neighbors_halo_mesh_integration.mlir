// RUN: tessera-opt %s --allow-unregistered-dialect -tessera-stencil-lower -tessera-boundary-condition-lower -tessera-halo-mesh-integration | FileCheck %s

// ============================================================================
// Test 1: Sharded 5-point periodic stencil — halo.exchange is inserted
//         before stencil.apply, threading through the field operand.
//
// The schedule.mesh.region is pre-built (DistributionLoweringPass would
// produce equivalent IR; the current pass has a separate dominance issue
// when the func returns the stencil result, so the fixture wraps the
// region inline to focus on the integration semantics under test).
// ============================================================================

// CHECK-LABEL: func @test_halo_mesh_periodic_stencil
// CHECK-DAG:   "tessera.neighbors.halo.exchange"
// CHECK-DAG:   inserted_by = "halo-mesh-integration"
// CHECK-DAG:   halo.mesh_integrated = true
// The default mesh policy is "open"; periodic BC against an open mesh
// is a conflict and the pass must record the diagnostic on the apply op.
// CHECK-DAG:   mesh.bc_conflict = "axis 0: stencil BC 'periodic' incompatible with mesh axis policy 'open'
func.func @test_halo_mesh_periodic_stencil(%arg0: tensor<?x?xf32>) {

  "schedule.mesh.define"() {axis_names = ["dp", "tp"], dims = [2, 2]} : () -> ()
  "schedule.mesh.region"() ({
    %topo = "tessera.neighbors.topology.create"() {
        kind = "2d_mesh"
    } : () -> !tessera.neighbors.topology

    %st = "tessera.neighbors.stencil.define"() {
        taps = [dense<[0, 0]>  : tensor<2xi64>,
                dense<[1, 0]>  : tensor<2xi64>,
                dense<[-1, 0]> : tensor<2xi64>,
                dense<[0, 1]>  : tensor<2xi64>,
                dense<[0, -1]> : tensor<2xi64>],
        bc = "periodic"
    } : () -> index

    %h = "tessera.neighbors.halo.region"(%arg0) {
        halo.width = [1, 1]
    } : (tensor<?x?xf32>) -> tensor<?x?xf32>

    %out = "tessera.neighbors.stencil.apply"(%st, %h, %topo) :
        (index, tensor<?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?xf32>

    "schedule.yield"() : () -> ()
  }) {mesh = @dp, axis = "dp"} : () -> ()

  return
}

// ============================================================================
// Test 2: Reflect BC — no periodic / open conflict, so no diagnostic.
// ============================================================================

// CHECK-LABEL: func @test_halo_mesh_reflect_stencil
// CHECK-DAG:   "tessera.neighbors.halo.exchange"
// CHECK-DAG:   halo.mesh_integrated = true
// CHECK-NOT:   mesh.bc_conflict
func.func @test_halo_mesh_reflect_stencil(%arg0: tensor<?x?xf32>) {

  "schedule.mesh.define"() {axis_names = ["dp"], dims = [2]} : () -> ()
  "schedule.mesh.region"() ({
    %topo = "tessera.neighbors.topology.create"() {
        kind = "2d_mesh"
    } : () -> !tessera.neighbors.topology

    %st = "tessera.neighbors.stencil.define"() {
        taps = [dense<[0, 0]> : tensor<2xi64>,
                dense<[1, 0]> : tensor<2xi64>],
        bc = "reflect,reflect"
    } : () -> index

    %h = "tessera.neighbors.halo.region"(%arg0) {
        halo.width = [1, 1]
    } : (tensor<?x?xf32>) -> tensor<?x?xf32>

    %out = "tessera.neighbors.stencil.apply"(%st, %h, %topo) :
        (index, tensor<?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?xf32>

    "schedule.yield"() : () -> ()
  }) {mesh = @dp, axis = "dp"} : () -> ()

  return
}

// ============================================================================
// Test 3: Periodic stencil + periodic mesh axis policy → no conflict.
// ============================================================================
//
// Pass option: -tessera-halo-mesh-integration='mesh-axis-policy=periodic'.
// (Each RUN line below re-runs the same input with periodic policy.)

// RUN: tessera-opt %s --allow-unregistered-dialect -tessera-stencil-lower -tessera-boundary-condition-lower -tessera-halo-mesh-integration='mesh-axis-policy=periodic' | FileCheck %s --check-prefix=PERIODIC

// PERIODIC-LABEL: func @test_halo_mesh_periodic_stencil
// PERIODIC:       halo.mesh_integrated = true
// Under periodic mesh policy the periodic stencil BC is compatible.
// PERIODIC-NOT:   mesh.bc_conflict
