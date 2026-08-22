// RUN: %tnv %s --generate-nvidia-philox-kernel | FileCheck %s

module {
  "tessera_nvidia.philox"() {name = "philox_uniform", mode = "uniform_core"} : () -> ()
}

// CHECK-NOT: tessera_nvidia.philox
// CHECK: gpu.module @philox_uniform_module
// CHECK: gpu.func @philox_uniform(%{{.*}}: memref<?xf32>, %{{.*}}: index, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32) kernel
// CHECK: arith.constant -766435501 : i32
// CHECK: arith.constant -845247145 : i32
// CHECK: gpu.block_id x
// CHECK: gpu.thread_id x
// CHECK-COUNT-4: memref.store
