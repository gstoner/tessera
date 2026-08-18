// RUN: tessera-opt --tessera-graph-dataflow --allow-unregistered-dialect %s | FileCheck %s

func.func @facts(%x: tensor<4x?xf32>, %shape: tensor<2xi64>) -> tensor<4x?xf32> {
  %dead = arith.addf %x, %x : tensor<4x?xf32>
  %view = "test.view"(%x, %shape) {tessera.aliasing = "operand_0", tessera.effect_kind = "pure"}
      : (tensor<4x?xf32>, tensor<2xi64>) -> tensor<4x?xf32>
  %unknown = "test.external"(%view) : (tensor<4x?xf32>) -> tensor<4x?xf32>
  return %unknown : tensor<4x?xf32>
}

// CHECK-LABEL: func.func @facts
// CHECK-SAME: tessera.dataflow.schema_version = 1
// CHECK: %{{.*}} = arith.addf {{.*}}tessera.dataflow.activity = "inactive"
// CHECK-SAME: tessera.dataflow.alias_roots = [1]
// CHECK-SAME: tessera.dataflow.live = [false]
// CHECK-SAME: tessera.dataflow.shape = {{\[\[4, "\?"\]\]}}
// CHECK: test.view
// CHECK-SAME: tessera.dataflow.activity = "active"
// CHECK-SAME: tessera.dataflow.alias_roots = [1]
// CHECK-SAME: tessera.dataflow.aliases_operands = {{\[\[0\]\]}}
// CHECK: test.external
// CHECK-SAME: tessera.dataflow.activity = "active"
// CHECK-SAME: tessera.dataflow.alias_roots = ["top"]
// CHECK-SAME: tessera.dataflow.aliases_operands = ["top"]
// CHECK-SAME: tessera.dataflow.live = [true]

func.func @memory_activity(%buffer: memref<1xf32>, %value: f32,
                           %index: index) -> f32 {
  // A result-less write is part of the active program because the returned
  // load may observe it. SSA-only reverse activity used to omit this store.
  memref.store %value, %buffer[%index] : memref<1xf32>
  %loaded = memref.load %buffer[%index] : memref<1xf32>
  return %loaded : f32
}

// CHECK-LABEL: func.func @memory_activity
// CHECK: memref.store
// CHECK-SAME: tessera.dataflow.activity = "active"
// CHECK: memref.load
// CHECK-SAME: tessera.dataflow.activity = "active"
