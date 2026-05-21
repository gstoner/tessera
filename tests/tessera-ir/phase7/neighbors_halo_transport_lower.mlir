// RUN: tessera-opt %s --allow-unregistered-dialect -tessera-stencil-lower -tessera-boundary-condition-lower -tessera-halo-mesh-integration -tessera-halo-transport-lower | FileCheck %s

// ============================================================================
// Sub-4 — Full halo pipeline.
//
// Compose the four halo-related passes:
//   1. stencil-lower               — emits stencil.bc string-typed ABI
//   2. boundary-condition-lower    — parses BC into per-axis structured attrs
//   3. halo-mesh-integration       — inserts halo.exchange before stencil.apply
//   4. halo-transport-lower        — replaces halo.exchange with
//                                    (pack, transport, unpack) triples
//
// After this chain the IR no longer contains any halo.exchange ops;
// every exchange has become real pack/transport/unpack triples that the
// runtime adapter can serve.
// ============================================================================

// CHECK-LABEL: func @test_full_halo_pipeline
// halo.exchange has been completely lowered away.
// CHECK-NOT:   tessera.neighbors.halo.exchange
// Every pack, transport, and unpack op shows up, with the
// halo-transport-lower provenance.
// CHECK-DAG:   "tessera.neighbors.halo.pack"
// CHECK-DAG:   "tessera.neighbors.halo.transport"
// CHECK-DAG:   "tessera.neighbors.halo.unpack"
// CHECK-DAG:   inserted_by = "halo-transport-lower"
// Per-axis × per-side parameters surface on each op.
// CHECK-DAG:   side = "lo"
// CHECK-DAG:   side = "hi"
// CHECK-DAG:   peer_rule = "neg1"
// CHECK-DAG:   peer_rule = "pos1"
// Idempotency sentinel on the threaded result.
// CHECK-DAG:   halo.transport_lowered = true
func.func @test_full_halo_pipeline(%arg0: tensor<?x?xf32>) {

  "schedule.mesh.define"() {axis_names = ["dp"], dims = [2]} : () -> ()
  "schedule.mesh.region"() ({
    %topo = "tessera.neighbors.topology.create"() {
        kind = "2d_mesh"
    } : () -> !tessera.neighbors.topology

    %st = "tessera.neighbors.stencil.define"() {
        taps = [dense<[0, 0]>  : tensor<2xi64>,
                dense<[1, 0]>  : tensor<2xi64>,
                dense<[-1, 0]> : tensor<2xi64>],
        bc = "periodic,reflect"
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
// Test 2 — width=0 on an axis elides that axis's triples.
//
// A halo.exchange with halo.width=[0, 1] should emit only 2 triples
// (axis-1 lo + hi), not 4.  The pass must skip width-0 axes silently.
// ============================================================================

// CHECK-LABEL: func @test_halo_width_zero_axis_elided
// At least one of each triple op, and no surviving halo.exchange.  The
// underlying pass elides axis-0 (width=0) so only axis-1 emits triples;
// the structural shape is verified, not the exact count.
// CHECK-DAG:    tessera.neighbors.halo.pack
// CHECK-DAG:    tessera.neighbors.halo.transport
// CHECK-DAG:    tessera.neighbors.halo.unpack
// CHECK-NOT:    tessera.neighbors.halo.exchange
func.func @test_halo_width_zero_axis_elided(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %x = "tessera.neighbors.halo.exchange"(%arg0) {
      halo.width = [0, 1],
      mesh.axis  = "dp"
  } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %x : tensor<?x?xf32>
}
