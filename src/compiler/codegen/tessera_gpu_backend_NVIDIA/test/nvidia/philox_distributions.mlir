// RUN: %tnv %s --generate-nvidia-philox-kernel | FileCheck %s

module {
  "tessera_nvidia.philox"() {name = "uniform_range", mode = "uniform_range"} : () -> ()
  "tessera_nvidia.philox"() {name = "normal", mode = "normal"} : () -> ()
  "tessera_nvidia.philox"() {name = "dropout", mode = "dropout"} : () -> ()
}

// CHECK-NOT: tessera_nvidia.philox
// CHECK: gpu.module @uniform_range_module
// CHECK: gpu.func @uniform_range(%{{.*}}: memref<?xf32>, %{{.*}}: index, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: f32, %{{.*}}: f32) kernel
// CHECK: arith.subf
// CHECK: gpu.module @normal_module
// CHECK: gpu.func @normal(%{{.*}}: memref<?xf32>, %{{.*}}: index, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: f32, %{{.*}}: f32) kernel
// CHECK: math.log
// CHECK: math.sqrt
// CHECK: math.sin
// CHECK: math.cos
// CHECK: gpu.module @dropout_module
// CHECK: gpu.func @dropout(%{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: index, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: f32) kernel
// CHECK: arith.cmpf oge
// CHECK: arith.divf
// CHECK: memref.load
