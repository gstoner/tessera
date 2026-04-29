// RUN: tessera-opt %s -tessera-stencil-lower | FileCheck %s

// ============================================================================
// Test 1: StencilLowerPass sets stencil.lowered and all phase annotations
// ============================================================================

// CHECK-LABEL: func @test_stencil_lower_basic
// CHECK:       tessera.neighbors.stencil.apply
// CHECK-SAME:  stencil.lowered = true
// CHECK-SAME:  stencil.pack_phase = true
// CHECK-SAME:  stencil.compute_phase = true
// CHECK-SAME:  stencil.tap_count = 5
// CHECK-SAME:  stencil.bc = "periodic"
func.func @test_stencil_lower_basic(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {

  %topo = "tessera.neighbors.topology.create"() {
      kind = "2d_mesh",
      defaults = "von_neumann"
  } : () -> !tessera.neighbors.topology

  %st = "tessera.neighbors.stencil.define"() {
      taps = [dense<[0, 0]>  : tensor<2xi64>,
              dense<[1, 0]>  : tensor<2xi64>,
              dense<[-1, 0]> : tensor<2xi64>,
              dense<[0, 1]>  : tensor<2xi64>,
              dense<[0, -1]> : tensor<2xi64>],
      bc = "periodic"
  } : () -> index

  // CHECK: stencil.halo_width = [1, 1]
  // CHECK: stencil.exchange_policy = "blocking"
  %out = "tessera.neighbors.stencil.apply"(%st, %arg0, %topo) :
      (index, tensor<?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?xf32>

  return %out : tensor<?x?xf32>
}

// ============================================================================
// Test 2: StencilLowerPass — async exchange when pipeline.config is present
// ============================================================================

// CHECK-LABEL: func @test_stencil_lower_async
// CHECK:       tessera.neighbors.stencil.apply
// CHECK-SAME:  stencil.halo_async = true
// CHECK-SAME:  stencil.exchange_policy = "async"
func.func @test_stencil_lower_async(%arg0: tensor<?xf32>) -> tensor<?xf32> {

  // Async pipeline config triggers async halo exchange
  "tessera.neighbors.pipeline.config"() {
      overlap = "full",
      depth    = 2 : i64
  } : () -> ()

  %topo = "tessera.neighbors.topology.create"() {
      kind = "1d_mesh"
  } : () -> !tessera.neighbors.topology

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
// Test 3: StencilLowerPass — already-lowered op is not re-processed
// ============================================================================

// CHECK-LABEL: func @test_stencil_lower_idempotent
// CHECK:       tessera.neighbors.stencil.apply
// CHECK-SAME:  stencil.lowered = true
// CHECK-NOT:   stencil.pack_phase
func.func @test_stencil_lower_idempotent(%arg0: tensor<?xf32>) -> tensor<?xf32> {

  %topo = "tessera.neighbors.topology.create"() {
      kind = "1d_mesh"
  } : () -> !tessera.neighbors.topology

  %st = "tessera.neighbors.stencil.define"() {
      taps = [dense<[0]> : tensor<1xi64>]
  } : () -> index

  // Pre-annotated as already lowered — pass must leave it alone
  %out = "tessera.neighbors.stencil.apply"(%st, %arg0, %topo) {
      stencil.lowered = true
  } : (index, tensor<?xf32>, !tessera.neighbors.topology) -> tensor<?xf32>

  return %out : tensor<?xf32>
}
