// RUN: tessera-opt %s -tessera-stencil-lower -tessera-boundary-condition-lower -tessera-stencil-loop-materialize | FileCheck %s

// ============================================================================
// Test 1: 5-point periodic stencil materializes to scf.for + arith.remsi
// ============================================================================
//
// Periodic BC requires a signed-remainder transform on the raw index plus
// a "negative-remainder → add N" correction, then a single tensor.extract
// at the fixed index per tap.  The five-point stencil sums five
// arith.addf in sequence into the accumulator.

// CHECK-LABEL: func @test_materialize_periodic_5pt
// CHECK-DAG:     scf.for {{.*}} iter_args
// CHECK-DAG:     arith.remsi
// CHECK-DAG:     tensor.extract
// CHECK-DAG:     arith.addf
// CHECK-DAG:     tensor.insert
// CHECK-DAG:     stencil.materialized = true
func.func @test_materialize_periodic_5pt(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {

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
// Test 2: Dirichlet BC emits the constant + select for OOB short-circuit
// ============================================================================
//
// Dirichlet uses clamp on the index (maxsi + minsi) and a select that
// replaces the extracted value with the BC constant when the axis is OOB.

// CHECK-LABEL: func @test_materialize_dirichlet
// CHECK-DAG:     scf.for {{.*}} iter_args
// CHECK-DAG:     arith.constant 2.500000e+00 : f32
// CHECK-DAG:     arith.maxsi
// CHECK-DAG:     arith.minsi
// CHECK-DAG:     arith.select
// CHECK-DAG:     tensor.extract
// CHECK-DAG:     tensor.insert
// CHECK-DAG:     stencil.materialized = true
func.func @test_materialize_dirichlet(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {

  %topo = "tessera.neighbors.topology.create"() {
      kind = "2d_mesh"
  } : () -> !tessera.neighbors.topology

  %st = "tessera.neighbors.stencil.define"() {
      taps = [dense<[0, 0]>  : tensor<2xi64>,
              dense<[1, 0]>  : tensor<2xi64>],
      bc = "dirichlet(2.5),dirichlet(2.5)"
  } : () -> index

  %h = "tessera.neighbors.halo.region"(%arg0) {
      halo.width = [1, 1]
  } : (tensor<?x?xf32>) -> tensor<?x?xf32>

  %out = "tessera.neighbors.stencil.apply"(%st, %h, %topo) :
      (index, tensor<?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?xf32>

  return %out : tensor<?x?xf32>
}

// ============================================================================
// Test 3: Neumann BC adds the constant offset to extracted reads
// ============================================================================
//
// Neumann uses the same index clamp as Dirichlet but the value-side rule
// is extract + bc.values[a] when OOB on a neumann axis.

// CHECK-LABEL: func @test_materialize_neumann
// CHECK-DAG:     scf.for {{.*}} iter_args
// CHECK-DAG:     arith.constant -1.000000e+00 : f32
// CHECK-DAG:     arith.maxsi
// CHECK-DAG:     arith.minsi
// CHECK-DAG:     tensor.extract
// CHECK-DAG:     arith.addf
// CHECK-DAG:     stencil.materialized = true
func.func @test_materialize_neumann(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {

  %topo = "tessera.neighbors.topology.create"() {
      kind = "2d_mesh"
  } : () -> !tessera.neighbors.topology

  %st = "tessera.neighbors.stencil.define"() {
      taps = [dense<[0, 0]>  : tensor<2xi64>,
              dense<[1, 0]>  : tensor<2xi64>,
              dense<[0, 1]>  : tensor<2xi64>],
      bc = "neumann(-1.0),neumann(-1.0)"
  } : () -> index

  %h = "tessera.neighbors.halo.region"(%arg0) {
      halo.width = [1, 1]
  } : (tensor<?x?xf32>) -> tensor<?x?xf32>

  %out = "tessera.neighbors.stencil.apply"(%st, %h, %topo) :
      (index, tensor<?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?xf32>

  return %out : tensor<?x?xf32>
}
