// RUN: tessera-opt %s -verify-diagnostics

// CORE-COMPILER-1: representative failures from the shared declarative table.
// The verifier-coverage drift gate separately proves that all eleven formerly
// open operations own real verify() bodies.

func.func @bad_topology(%x: tensor<4xf32>) {
  // expected-error@+1 {{expects 0 operand(s); got 1}}
  %bad_topology = "tessera.neighbors.topology.create"(%x)
      : (tensor<4xf32>) -> index
  return
}

func.func @bad_halo(%x: tensor<4xf32>) {
  // expected-error@+1 {{requires the data result shape and element type to match its input}}
  %bad_halo = "tessera.neighbors.halo.region"(%x)
      : (tensor<4xf32>) -> tensor<5xf32>
  return
}

func.func @bad_stencil(%x: tensor<4xf32>, %stencil: index,
                       %topology: index) {
  // expected-error@+1 {{requires the data result shape and element type to match its input}}
  %bad_apply = "tessera.neighbors.stencil.apply"(%stencil, %x, %topology)
      : (index, tensor<4xf32>, index) -> tensor<5xf32>
  return
}

func.func @bad_pipeline_config() {
  // expected-error@+1 {{expects 0 result(s); got 1}}
  %bad_config = "tessera.neighbors.pipeline.config"() : () -> index
  return
}
