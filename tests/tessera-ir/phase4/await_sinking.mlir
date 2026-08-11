// RUN: tessera-opt --tessera-await-sinking %s | FileCheck %s

// A pure independent chain may overlap the collective. The await moves to the
// first SSA consumer, immediately after that chain.
func.func @sink_across_pure(%future: !tessera_collective.future<tensor<4xf32>>,
                            %x: tensor<4xf32>) -> tensor<4xf32> {
  %ready = tessera_collective.await %future : !tessera_collective.future<tensor<4xf32>> -> tensor<4xf32>
  %c0 = arith.constant 0.0 : f32
  %zero = tensor.splat %c0 : tensor<4xf32>
  %out = arith.addf %ready, %zero : tensor<4xf32>
  return %out : tensor<4xf32>
}
// CHECK-LABEL: func.func @sink_across_pure
// CHECK: tessera.await_sinking.moved = 1
// CHECK: arith.constant
// CHECK: tensor.splat
// CHECK: %[[READY:.*]] = tessera_collective.await
// CHECK-NEXT: %{{.*}} = arith.addf %[[READY]]

// Unknown/effectful mutation is a hard barrier.
func.func @mutation_barrier(%future: !tessera_collective.future<tensor<4xf32>>,
                            %buffer: memref<4xf32>) -> tensor<4xf32> {
  %ready = tessera_collective.await %future : !tessera_collective.future<tensor<4xf32>> -> tensor<4xf32>
  %c0 = arith.constant 0.0 : f32
  %i0 = arith.constant 0 : index
  memref.store %c0, %buffer[%i0] : memref<4xf32>
  return %ready : tensor<4xf32>
}
// CHECK-LABEL: func.func @mutation_barrier
// CHECK: tessera.await_sinking.moved = 0
// CHECK: tessera_collective.await
// CHECK: memref.store

// Alias-producing views are barriers even though constructing the view does
// not itself read or write memory.
func.func @alias_barrier(%future: !tessera_collective.future<tensor<4xf32>>,
                         %buffer: memref<4xf32>) -> tensor<4xf32> {
  %ready = tessera_collective.await %future : !tessera_collective.future<tensor<4xf32>> -> tensor<4xf32>
  %view = memref.cast %buffer : memref<4xf32> to memref<?xf32>
  return %ready : tensor<4xf32>
}
// CHECK-LABEL: func.func @alias_barrier
// CHECK: tessera.await_sinking.moved = 0
// CHECK: tessera_collective.await
// CHECK: memref.cast

// Stateful RNG is an ordering barrier regardless of its memory-effect model.
func.func @rng_barrier(%future: !tessera_collective.future<tensor<4xf32>>) -> tensor<4xf32> {
  %ready = tessera_collective.await %future : !tessera_collective.future<tensor<4xf32>> -> tensor<4xf32>
  %sample = "tessera.rng_uniform"() {shape = [4], seed = 7 : i64,
      tessera.effect_kind = "random", tessera.stochastic_identity = "implicit_stream"}
      : () -> tensor<4xf32>
  return %ready : tensor<4xf32>
}
// CHECK-LABEL: func.func @rng_barrier
// CHECK: tessera.await_sinking.moved = 0
// CHECK: tessera_collective.await
// CHECK: tessera.rng_uniform

// Ordered collectives cannot be crossed.
func.func @collective_barrier(
    %first: !tessera_collective.future<tensor<4xf32>>,
    %second: !tessera_collective.future<tensor<4xf32>>) -> tensor<4xf32> {
  %ready = tessera_collective.await %first : !tessera_collective.future<tensor<4xf32>> -> tensor<4xf32>
  %ordered = tessera_collective.await %second : !tessera_collective.future<tensor<4xf32>> -> tensor<4xf32>
  return %ready : tensor<4xf32>
}
// CHECK-LABEL: func.func @collective_barrier
// CHECK: tessera.await_sinking.moved = 0
// CHECK: tessera_collective.await %arg0
// CHECK-NEXT: tessera_collective.await %arg1

// x86/no-async input is intentionally a no-op apart from provenance count.
func.func @x86_noop(%x: tensor<4xf32>) -> tensor<4xf32> {
  return %x : tensor<4xf32>
}
// CHECK-LABEL: func.func @x86_noop
// CHECK: tessera.await_sinking.moved = 0

// A discardable attribute cannot erase the registered write effect.
func.func @effect_spoof_is_barrier(
    %future: !tessera_collective.future<tensor<4xf32>>,
    %x: tensor<4xf32>) -> tensor<4xf32> {
  %ready = tessera_collective.await %future : !tessera_collective.future<tensor<4xf32>> -> tensor<4xf32>
  %unused = "tessera.dropout"(%x) {p = 0.0 : f64,
      tessera.effect_kind = "pure"} : (tensor<4xf32>) -> tensor<4xf32>
  return %ready : tensor<4xf32>
}
// CHECK-LABEL: func.func @effect_spoof_is_barrier
// CHECK: tessera.await_sinking.moved = 0
