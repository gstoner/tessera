// RUN: tessera-opt %s -tessera-halo-infer | FileCheck %s
// RUN: tessera-opt %s -tessera-halo-infer --mlir-print-debuginfo | FileCheck %s --check-prefix=DEBUG

// ============================================================================
// Test 1: HaloInferPass annotates stencil.apply with halo.width from taps
// ============================================================================

// CHECK-LABEL: func @test_stencil_halo_infer
// CHECK:       tessera.neighbors.stencil.apply
// CHECK-SAME:  halo.width = [1, 1]
func.func @test_stencil_halo_infer(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {

  %topo = "tessera.neighbors.topology.create"() {
      kind = "2d_mesh",
      defaults = "von_neumann"
  } : () -> !tessera.neighbors.topology

  // Von Neumann 5-point stencil: max |Δ| per axis = 1
  %st = "tessera.neighbors.stencil.define"() {
      taps = [dense<[0, 0]> : tensor<2xi64>,
              dense<[1, 0]> : tensor<2xi64>,
              dense<[-1, 0]> : tensor<2xi64>,
              dense<[0, 1]> : tensor<2xi64>,
              dense<[0, -1]> : tensor<2xi64>],
      bc = "periodic"
  } : () -> index

  // CHECK: stencil.lowered = true
  %out = "tessera.neighbors.stencil.apply"(%st, %arg0, %topo) :
      (index, tensor<?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?xf32>

  return %out : tensor<?x?xf32>
}

// ============================================================================
// Test 2: HaloInferPass — wide stencil (3-point radius-2)
// ============================================================================

// CHECK-LABEL: func @test_wide_stencil
// CHECK:       tessera.neighbors.stencil.apply
// CHECK-SAME:  halo.width = [2]
func.func @test_wide_stencil(%arg0: tensor<?xf32>) -> tensor<?xf32> {

  %topo = "tessera.neighbors.topology.create"() {
      kind = "1d_mesh"
  } : () -> !tessera.neighbors.topology

  // Radius-2 stencil: max |Δ| = 2
  %st = "tessera.neighbors.stencil.define"() {
      taps = [dense<[-2]> : tensor<1xi64>,
              dense<[-1]> : tensor<1xi64>,
              dense<[0]>  : tensor<1xi64>,
              dense<[1]>  : tensor<1xi64>,
              dense<[2]>  : tensor<1xi64>]
  } : () -> index

  %out = "tessera.neighbors.stencil.apply"(%st, %arg0, %topo) :
      (index, tensor<?xf32>, !tessera.neighbors.topology) -> tensor<?xf32>

  return %out : tensor<?xf32>
}

// ============================================================================
// Test 3: HaloInferPass — 3D 7-point stencil
// ============================================================================

// CHECK-LABEL: func @test_3d_stencil
// CHECK:       tessera.neighbors.stencil.apply
// CHECK-SAME:  halo.width = [1, 1, 1]
func.func @test_3d_stencil(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {

  %topo = "tessera.neighbors.topology.create"() {
      kind = "3d_mesh"
  } : () -> !tessera.neighbors.topology

  %st = "tessera.neighbors.stencil.define"() {
      taps = [dense<[0, 0, 0]>  : tensor<3xi64>,
              dense<[1, 0, 0]>  : tensor<3xi64>,
              dense<[-1, 0, 0]> : tensor<3xi64>,
              dense<[0, 1, 0]>  : tensor<3xi64>,
              dense<[0, -1, 0]> : tensor<3xi64>,
              dense<[0, 0, 1]>  : tensor<3xi64>,
              dense<[0, 0, -1]> : tensor<3xi64>]
  } : () -> index

  %out = "tessera.neighbors.stencil.apply"(%st, %arg0, %topo) :
      (index, tensor<?x?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?x?xf32>

  return %out : tensor<?x?x?xf32>
}
