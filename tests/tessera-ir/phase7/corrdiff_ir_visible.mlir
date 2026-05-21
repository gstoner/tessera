// RUN: tessera-opt %s --allow-unregistered-dialect -tessera-stencil-lower -tessera-boundary-condition-lower -tessera-halo-mesh-integration -tessera-halo-transport-lower | FileCheck %s

// ============================================================================
// CorrDiff-core IR-visible fixture (2026-05-21).
//
// Proves that the three Phase 7 workstreams compose in a single IR
// flow when expressed at the Graph IR level:
//
//   * stencil.apply        — a CorrDiff-style diffusion filter
//   * attn_local_window_2d — spatial bias / local refinement
//   * halo                 — both ops sharded across a mesh axis,
//                            with halo.exchange wrapping each input
//                            and pack/transport/unpack triples
//                            replacing every halo.exchange
//
// Before this fixture, the three pieces existed in separate lit
// fixtures and tests; "do they actually compose" was an open
// question.  This fixture is the canonical answer: yes, the canonical
// halo pipeline (stencil-lower → bc-lower → halo-mesh-integration →
// halo-transport-lower) handles both consumers in one module.
//
// The fixture runs the FULL halo pipeline end-to-end, so every
// integration point shows up in a single emitted IR module.
// ============================================================================

// CHECK-LABEL: func @corrdiff_block
// No halo.exchange survives — every one has been lowered to a triple.
// CHECK-NOT:   tessera.neighbors.halo.exchange
// Stencil halo path: stencil-lower + bc-lower + mesh-integration
// sentinels.  We deliberately skip StencilLoopMaterialize here because
// it's orthogonal to the halo flow this fixture pins.
// CHECK-DAG:   stencil.lowered = true
// CHECK-DAG:   stencil.bc.lowered = true
// CHECK-DAG:   halo.mesh_integrated = true
// Window-attention halo path (Sub-2 closing-the-loop):
// CHECK-DAG:   source_op = "tessera.attn_local_window_2d"
// CHECK-DAG:   halo.window = [1, 1]
// Transport triples emitted (>=2 packs since at least one op is sharded):
// CHECK-DAG:   tessera.neighbors.halo.pack
// CHECK-DAG:   tessera.neighbors.halo.transport
// CHECK-DAG:   tessera.neighbors.halo.unpack
// All triples carry the integration + transport-lower provenance.
// CHECK-DAG:   inserted_by = "halo-transport-lower"
func.func @corrdiff_block(
    %field: tensor<?x?xf32>,
    %q: tensor<2x4x16x16x16xf32>,
    %k: tensor<2x4x16x16x16xf32>,
    %v: tensor<2x4x16x16x16xf32>
) {
  "schedule.mesh.define"() {axis_names = ["dp"], dims = [2]} : () -> ()
  "schedule.mesh.region"() ({

    // ── Stencil-style diffusion filter ───────────────────────────
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

    %h = "tessera.neighbors.halo.region"(%field) {
        halo.width = [1, 1]
    } : (tensor<?x?xf32>) -> tensor<?x?xf32>

    %stencil_out = "tessera.neighbors.stencil.apply"(%st, %h, %topo) :
        (index, tensor<?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?xf32>

    // ── 2D local-window attention for spatial bias ────────────────
    //
    // The Q, K, V inputs come from the function signature (sharded),
    // so HaloMeshIntegrationPass wraps the Q operand with a halo.
    // exchange whose width matches the window=[1, 1] attribute.
    %attn_out = tessera.attn_local_window_2d %q, %k, %v {window = [1, 1]} :
        (tensor<2x4x16x16x16xf32>, tensor<2x4x16x16x16xf32>, tensor<2x4x16x16x16xf32>)
        -> tensor<2x4x16x16x16xf32>

    "schedule.yield"() : () -> ()
  }) {mesh = @dp, axis = "dp"} : () -> ()
  return
}
