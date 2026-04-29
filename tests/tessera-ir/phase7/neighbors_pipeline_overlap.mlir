// RUN: tessera-opt %s -tessera-pipeline-overlap | FileCheck %s

// ============================================================================
// Test 1: PipelineOverlapPass — basic two-stream annotation
// ============================================================================

// CHECK-LABEL: func @test_pipeline_overlap_basic
// CHECK:       tessera.neighbors.halo.exchange
// CHECK-SAME:  comm.stream_id = 0
// CHECK-SAME:  comm.async = true
// CHECK:       tessera.neighbors.stencil.apply
// CHECK-SAME:  compute.stream_id = 1
// CHECK-SAME:  compute.depends_comm = true
func.func @test_pipeline_overlap_basic(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {

  "tessera.neighbors.pipeline.config"() {
      overlap = "full",
      depth    = 2 : i64
  } : () -> ()

  %topo = "tessera.neighbors.topology.create"() {
      kind = "2d_mesh"
  } : () -> !tessera.neighbors.topology

  %halo = "tessera.neighbors.halo.region"(%arg0, %topo) {
      halo.width = [1, 1]
  } : (tensor<?x?xf32>, !tessera.neighbors.topology) -> !tessera.neighbors.halo

  "tessera.neighbors.halo.exchange"(%halo) {} :
      (!tessera.neighbors.halo) -> ()

  %st = "tessera.neighbors.stencil.define"() {
      taps = [dense<[0, 0]>  : tensor<2xi64>,
              dense<[1, 0]>  : tensor<2xi64>,
              dense<[-1, 0]> : tensor<2xi64>,
              dense<[0, 1]>  : tensor<2xi64>,
              dense<[0, -1]> : tensor<2xi64>]
  } : () -> index

  %out = "tessera.neighbors.stencil.apply"(%st, %arg0, %topo) :
      (index, tensor<?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?xf32>

  return %out : tensor<?x?xf32>
}

// ============================================================================
// Test 2: PipelineOverlapPass — double-buffer index assignment
// ============================================================================

// CHECK-LABEL: func @test_pipeline_overlap_double_buf
// CHECK:       tessera.neighbors.stencil.apply
// CHECK-SAME:  pipeline.buffer_idx = 0
// CHECK:       tessera.neighbors.stencil.apply
// CHECK-SAME:  pipeline.buffer_idx = 1
// CHECK:       tessera.neighbors.stencil.apply
// CHECK-SAME:  pipeline.buffer_idx = 0
func.func @test_pipeline_overlap_double_buf(
    %a0: tensor<?xf32>, %a1: tensor<?xf32>, %a2: tensor<?xf32>
) -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {

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

  // Three consecutive applies — buffer_idx alternates 0, 1, 0
  %o0 = "tessera.neighbors.stencil.apply"(%st, %a0, %topo) :
      (index, tensor<?xf32>, !tessera.neighbors.topology) -> tensor<?xf32>
  %o1 = "tessera.neighbors.stencil.apply"(%st, %a1, %topo) :
      (index, tensor<?xf32>, !tessera.neighbors.topology) -> tensor<?xf32>
  %o2 = "tessera.neighbors.stencil.apply"(%st, %a2, %topo) :
      (index, tensor<?xf32>, !tessera.neighbors.topology) -> tensor<?xf32>

  return %o0, %o1, %o2 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
}

// ============================================================================
// Test 3: PipelineOverlapPass — pipeline.config marked resolved
// ============================================================================

// CHECK-LABEL: func @test_pipeline_config_resolved
// CHECK:       tessera.neighbors.pipeline.config
// CHECK-SAME:  pipeline.resolved = true
// CHECK-SAME:  pipeline.comm_stream = 0
// CHECK-SAME:  pipeline.compute_stream = 1
func.func @test_pipeline_config_resolved(%arg0: tensor<?xf32>) -> tensor<?xf32> {

  "tessera.neighbors.pipeline.config"() {
      overlap = "full",
      depth    = 2 : i64
  } : () -> ()

  %topo = "tessera.neighbors.topology.create"() {
      kind = "1d_mesh"
  } : () -> !tessera.neighbors.topology

  %st = "tessera.neighbors.stencil.define"() {
      taps = [dense<[0]> : tensor<1xi64>]
  } : () -> index

  %out = "tessera.neighbors.stencil.apply"(%st, %arg0, %topo) :
      (index, tensor<?xf32>, !tessera.neighbors.topology) -> tensor<?xf32>

  return %out : tensor<?xf32>
}
