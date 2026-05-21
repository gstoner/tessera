// RUN: tessera-opt %s --allow-unregistered-dialect -tessera-halo-mesh-integration | FileCheck %s

// ============================================================================
// HaloMeshIntegrationPass closing-the-loop on attn_local_window_2d.
//
// The pass now recognises tessera.attn_local_window_2d as a halo-aware
// consumer (matching the Python-side _HALO_AWARE_OPS registry).  When
// the op appears inside a schedule.mesh.region and consumes a sharded
// input (a func.func argument), the pass wraps the Q operand with a
// halo.exchange whose width = the op's `window` attribute.
// ============================================================================

// CHECK-LABEL: func @sharded_attn_local_window_2d
// CHECK-DAG:   "tessera.neighbors.halo.exchange"
// CHECK-DAG:   halo.width = [1, 1]
// CHECK-DAG:   inserted_by = "halo-mesh-integration"
// Provenance distinguishes attn-driven exchanges from stencil-driven ones.
// CHECK-DAG:   source_op = "tessera.attn_local_window_2d"
// CHECK-DAG:   tessera.attn_local_window_2d
// CHECK-DAG:   halo.mesh_integrated = true
// CHECK-DAG:   halo.window = [1, 1]
func.func @sharded_attn_local_window_2d(
    %q: tensor<2x4x8x8x16xf32>,
    %k: tensor<2x4x8x8x16xf32>,
    %v: tensor<2x4x8x8x16xf32>
) {
  "schedule.mesh.define"() {axis_names = ["dp"], dims = [2]} : () -> ()
  "schedule.mesh.region"() ({
    %o = tessera.attn_local_window_2d %q, %k, %v {window = [1, 1]} :
        (tensor<2x4x8x8x16xf32>, tensor<2x4x8x8x16xf32>, tensor<2x4x8x8x16xf32>)
        -> tensor<2x4x8x8x16xf32>
    "schedule.yield"() : () -> ()
  }) {mesh = @dp, axis = "dp"} : () -> ()
  return
}

// ============================================================================
// Asymmetric window threads through to halo.width unchanged.
// ============================================================================

// CHECK-LABEL: func @sharded_attn_local_window_2d_asymmetric
// CHECK-DAG:   "tessera.neighbors.halo.exchange"
// CHECK-DAG:   halo.width = [2, 3]
// CHECK-DAG:   halo.window = [2, 3]
func.func @sharded_attn_local_window_2d_asymmetric(
    %q: tensor<1x2x6x6x8xf32>,
    %k: tensor<1x2x6x6x8xf32>,
    %v: tensor<1x2x6x6x8xf32>
) {
  "schedule.mesh.define"() {axis_names = ["dp"], dims = [2]} : () -> ()
  "schedule.mesh.region"() ({
    %o = tessera.attn_local_window_2d %q, %k, %v {window = [2, 3]} :
        (tensor<1x2x6x6x8xf32>, tensor<1x2x6x6x8xf32>, tensor<1x2x6x6x8xf32>)
        -> tensor<1x2x6x6x8xf32>
    "schedule.yield"() : () -> ()
  }) {mesh = @dp, axis = "dp"} : () -> ()
  return
}

// ============================================================================
// Op outside a mesh.region is NOT halo-wrapped.  The op still gets the
// sentinel so a re-run skips it cleanly.
// ============================================================================

// CHECK-LABEL: func @unsharded_attn_local_window_2d
// CHECK-NOT:   "tessera.neighbors.halo.exchange"
// CHECK-DAG:   tessera.attn_local_window_2d
// CHECK-DAG:   halo.mesh_integrated = true
func.func @unsharded_attn_local_window_2d(
    %q: tensor<1x1x4x4x4xf32>,
    %k: tensor<1x1x4x4x4xf32>,
    %v: tensor<1x1x4x4x4xf32>
) -> tensor<1x1x4x4x4xf32> {
  %o = tessera.attn_local_window_2d %q, %k, %v {window = [1, 1]} :
      (tensor<1x1x4x4x4xf32>, tensor<1x1x4x4x4xf32>, tensor<1x1x4x4x4xf32>)
      -> tensor<1x1x4x4x4xf32>
  return %o : tensor<1x1x4x4x4xf32>
}
