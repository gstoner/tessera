// RUN: tessera-opt %s -tessera-stencil-lower -tessera-boundary-condition-lower | FileCheck %s

// ============================================================================
// Test 1: periodic — single token broadcasts to every axis
// ============================================================================

// CHECK-LABEL: func @test_bc_periodic
// CHECK:       tessera.neighbors.stencil.apply
// CHECK-SAME:  stencil.bc.has_value = [false, false]
// CHECK-SAME:  stencil.bc.lowered = true
// CHECK-SAME:  stencil.bc.modes = ["periodic", "periodic"]
// CHECK-SAME:  stencil.bc.values = [0.000000e+00, 0.000000e+00]
func.func @test_bc_periodic(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {

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

  return %out : tensor<?x?xf32>
}

// ============================================================================
// Test 2: per-axis mix — periodic on x, reflect on y
// ============================================================================

// CHECK-LABEL: func @test_bc_periodic_reflect
// CHECK:       tessera.neighbors.stencil.apply
// CHECK-SAME:  stencil.bc.has_value = [false, false]
// CHECK-SAME:  stencil.bc.modes = ["periodic", "reflect"]
func.func @test_bc_periodic_reflect(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {

  %topo = "tessera.neighbors.topology.create"() {
      kind = "2d_mesh"
  } : () -> !tessera.neighbors.topology

  %st = "tessera.neighbors.stencil.define"() {
      taps = [dense<[1, 0]>  : tensor<2xi64>,
              dense<[-1, 0]> : tensor<2xi64>,
              dense<[0, 1]>  : tensor<2xi64>,
              dense<[0, -1]> : tensor<2xi64>],
      bc = "periodic,reflect"
  } : () -> index

  %h = "tessera.neighbors.halo.region"(%arg0) {
      halo.width = [1, 1]
  } : (tensor<?x?xf32>) -> tensor<?x?xf32>

  %out = "tessera.neighbors.stencil.apply"(%st, %h, %topo) :
      (index, tensor<?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?xf32>

  return %out : tensor<?x?xf32>
}

// ============================================================================
// Test 3: dirichlet(v) + neumann(v) — scalar payloads parsed and recorded
// ============================================================================

// CHECK-LABEL: func @test_bc_dirichlet_neumann
// CHECK:       tessera.neighbors.stencil.apply
// CHECK-SAME:  stencil.bc.has_value = [true, true]
// CHECK-SAME:  stencil.bc.modes = ["dirichlet", "neumann"]
// CHECK-SAME:  stencil.bc.values = [2.500000e+00, -1.000000e+00]
func.func @test_bc_dirichlet_neumann(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {

  %topo = "tessera.neighbors.topology.create"() {
      kind = "2d_mesh"
  } : () -> !tessera.neighbors.topology

  %st = "tessera.neighbors.stencil.define"() {
      taps = [dense<[0, 0]> : tensor<2xi64>,
              dense<[1, 0]> : tensor<2xi64>],
      bc = "dirichlet(2.5),neumann(-1.0)"
  } : () -> index

  %h = "tessera.neighbors.halo.region"(%arg0) {
      halo.width = [1, 1]
  } : (tensor<?x?xf32>) -> tensor<?x?xf32>

  %out = "tessera.neighbors.stencil.apply"(%st, %h, %topo) :
      (index, tensor<?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?xf32>

  return %out : tensor<?x?xf32>
}

// ============================================================================
// Test 4: no stencil.bc declared — defaults to periodic on every axis
// ============================================================================

// CHECK-LABEL: func @test_bc_default
// CHECK:       tessera.neighbors.stencil.apply
// CHECK-SAME:  stencil.bc.lowered = true
// CHECK-SAME:  stencil.bc.modes = ["periodic"]
func.func @test_bc_default(%arg0: tensor<?xf32>) -> tensor<?xf32> {

  %topo = "tessera.neighbors.topology.create"() {
      kind = "1d_mesh"
  } : () -> !tessera.neighbors.topology

  %st = "tessera.neighbors.stencil.define"() {
      taps = [dense<[-1]> : tensor<1xi64>,
              dense<[0]>  : tensor<1xi64>,
              dense<[1]>  : tensor<1xi64>]
  } : () -> index

  %h = "tessera.neighbors.halo.region"(%arg0) {
      halo.width = [1]
  } : (tensor<?xf32>) -> tensor<?xf32>

  %out = "tessera.neighbors.stencil.apply"(%st, %h, %topo) :
      (index, tensor<?xf32>, !tessera.neighbors.topology) -> tensor<?xf32>

  return %out : tensor<?xf32>
}
