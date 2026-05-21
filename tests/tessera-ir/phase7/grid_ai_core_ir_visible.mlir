// RUN: tessera-opt %s --allow-unregistered-dialect -tessera-stencil-lower -tessera-boundary-condition-lower -tessera-halo-mesh-integration -tessera-halo-transport-lower | FileCheck %s

// Generic grid-AI core IR-visible fixture.
//
// This is intentionally not weather-specific.  It pins the compiler-facing
// substrate a library needs for gridded regional AI:
//   * stencil.apply local feature
//   * attn_local_window_2d spatial neighborhood
//   * conv2d + fused/pointwise block remains visible as ordinary Graph IR
//   * deterministic RNG op remains visible
//   * mesh-region halo lowering emits pack/transport/unpack triples

// CHECK-LABEL: func @grid_ai_core_block
// CHECK-NOT:   tessera.neighbors.halo.exchange
// CHECK-DAG:   stencil.lowered = true
// CHECK-DAG:   stencil.bc.lowered = true
// CHECK-DAG:   halo.mesh_integrated = true
// CHECK-DAG:   source_op = "tessera.attn_local_window_2d"
// CHECK-DAG:   halo.window = [1, 1]
// CHECK-DAG:   tessera.conv2d_nhwc
// CHECK-DAG:   tessera.relu
// CHECK-DAG:   tessera_rng.normal
// CHECK-DAG:   tessera.neighbors.halo.pack
// CHECK-DAG:   tessera.neighbors.halo.transport
// CHECK-DAG:   tessera.neighbors.halo.unpack
func.func @grid_ai_core_block(
    %field: tensor<?x?xf32>,
    %x: tensor<1x16x16x3xf32>,
    %w0: tensor<3x3x3x8xf32>,
    %b0: tensor<8xf32>,
    %q: tensor<1x2x16x16x4xf32>,
    %k: tensor<1x2x16x16x4xf32>,
    %v: tensor<1x2x16x16x4xf32>
) {
  "schedule.mesh.define"() {axis_names = ["tp"], dims = [2]} : () -> ()
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

    %h = "tessera.neighbors.halo.region"(%field) {
        halo.width = [1, 1]
    } : (tensor<?x?xf32>) -> tensor<?x?xf32>

    %feature = "tessera.neighbors.stencil.apply"(%st, %h, %topo) :
        (index, tensor<?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?xf32>

    %conv = "tessera.conv2d_nhwc"(%x, %w0) {
        dilations = [1, 1],
        strides = [1, 1]
    } : (tensor<1x16x16x3xf32>, tensor<3x3x3x8xf32>) -> tensor<1x14x14x8xf32>

    %act = "tessera.relu"(%conv) :
        (tensor<1x14x14x8xf32>) -> tensor<1x14x14x8xf32>

    %attn = tessera.attn_local_window_2d %q, %k, %v {window = [1, 1]} :
        (tensor<1x2x16x16x4xf32>, tensor<1x2x16x16x4xf32>, tensor<1x2x16x16x4xf32>)
        -> tensor<1x2x16x16x4xf32>

    %noise = "tessera_rng.normal"(%act) {
        mean = 0.0 : f32,
        seed = 123 : i64,
        std = 0.05 : f32
    } : (tensor<1x14x14x8xf32>) -> tensor<1x14x14x8xf32>

    "schedule.yield"() : () -> ()
  }) {mesh = @tp, axis = "tp"} : () -> ()
  return
}
