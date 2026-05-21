// RUN: tessera-opt %s -tessera-stencil-lower -tessera-boundary-condition-lower -tessera-stencil-loop-materialize | FileCheck %s

// ============================================================================
// Rank-3 (vertical-level / z-time) 7-point stencil — the recursive loop-
// nest builder must emit three nested scf.for ops and tensor.extract /
// tensor.insert with a 3-index list.
// ============================================================================
//
// Taps: 7-point cross stencil in 3D — center + ±1 along each of three
// axes (typical for CFD on a uniform grid).  BC is periodic on every
// axis; the same modular-remainder fixup applies to all three.

// CHECK-LABEL: func @test_materialize_rank3_periodic_7pt
// CHECK-DAG:     scf.for {{.*}} iter_args
// CHECK-DAG:     arith.remsi
// CHECK-DAG:     tensor.extract
// CHECK-DAG:     tensor.insert
// CHECK-DAG:     stencil.materialized = true
// CHECK-DAG:     stencil.rank = 3
func.func @test_materialize_rank3_periodic_7pt(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {

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
              dense<[0, 0, -1]> : tensor<3xi64>],
      bc = "periodic"
  } : () -> index

  %h = "tessera.neighbors.halo.region"(%arg0) {
      halo.width = [1, 1, 1]
  } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

  %out = "tessera.neighbors.stencil.apply"(%st, %h, %topo) :
      (index, tensor<?x?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?x?xf32>

  return %out : tensor<?x?x?xf32>
}

// ============================================================================
// Rank-3 mixed BC — vertical level is dirichlet (closed BC at top/bottom),
// horizontal axes are periodic.  Common for atmospheric models.
// ============================================================================

// CHECK-LABEL: func @test_materialize_rank3_mixed_bc
// CHECK-DAG:     scf.for
// Periodic axes use arith.remsi; dirichlet axis uses arith.maxsi+minsi clamp.
// CHECK-DAG:     arith.remsi
// CHECK-DAG:     arith.maxsi
// CHECK-DAG:     arith.minsi
// Dirichlet BC constant.
// CHECK-DAG:     arith.constant 0.000000e+00 : f32
// CHECK-DAG:     stencil.materialized = true
// CHECK-DAG:     stencil.rank = 3
func.func @test_materialize_rank3_mixed_bc(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {

  %topo = "tessera.neighbors.topology.create"() {
      kind = "3d_mesh"
  } : () -> !tessera.neighbors.topology

  // bc list axes: (z=dirichlet(0), x=periodic, y=periodic).
  %st = "tessera.neighbors.stencil.define"() {
      taps = [dense<[0, 0, 0]> : tensor<3xi64>,
              dense<[1, 0, 0]> : tensor<3xi64>,
              dense<[-1, 0, 0]> : tensor<3xi64>],
      bc = "dirichlet(0.0),periodic,periodic"
  } : () -> index

  %h = "tessera.neighbors.halo.region"(%arg0) {
      halo.width = [1, 1, 1]
  } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

  %out = "tessera.neighbors.stencil.apply"(%st, %h, %topo) :
      (index, tensor<?x?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?x?xf32>

  return %out : tensor<?x?x?xf32>
}

// ============================================================================
// Rank-4 (time × z × y × x) — exercises the recursion at one level deeper.
// ============================================================================

// CHECK-LABEL: func @test_materialize_rank4_periodic
// CHECK-DAG:     scf.for
// CHECK-DAG:     arith.remsi
// CHECK-DAG:     tensor.extract
// CHECK-DAG:     tensor.insert
// CHECK-DAG:     stencil.materialized = true
// CHECK-DAG:     stencil.rank = 4
func.func @test_materialize_rank4_periodic(%arg0: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {

  %topo = "tessera.neighbors.topology.create"() {
      kind = "4d_mesh"
  } : () -> !tessera.neighbors.topology

  %st = "tessera.neighbors.stencil.define"() {
      taps = [dense<[0, 0, 0, 0]> : tensor<4xi64>,
              dense<[1, 0, 0, 0]> : tensor<4xi64>,
              dense<[0, 1, 0, 0]> : tensor<4xi64>],
      bc = "periodic"
  } : () -> index

  %h = "tessera.neighbors.halo.region"(%arg0) {
      halo.width = [1, 1, 1, 1]
  } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

  %out = "tessera.neighbors.stencil.apply"(%st, %h, %topo) :
      (index, tensor<?x?x?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?x?x?xf32>

  return %out : tensor<?x?x?x?xf32>
}
