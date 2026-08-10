// RUN: %trop %s --generate-rocm-philox-kernel | FileCheck %s

module {
  "tessera_rocm.philox"() {name = "uniform_range", mode = "uniform_range"} : () -> ()
  "tessera_rocm.philox"() {name = "normal", mode = "normal"} : () -> ()
  "tessera_rocm.philox"() {name = "dropout", mode = "dropout"} : () -> ()
}

// CHECK: gpu.module @uniform_range_mod
// CHECK: gpu.func @uniform_range(%{{.*}}: memref<?xf32>, %{{.*}}: index, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: f32, %{{.*}}: f32) kernel
// CHECK: arith.mulf
// CHECK: gpu.module @normal_mod
// CHECK: gpu.func @normal(%{{.*}}: memref<?xf32>, %{{.*}}: index, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: f32, %{{.*}}: f32) kernel
// CHECK: math.log
// CHECK: math.sqrt
// CHECK: math.sin
// CHECK: math.cos
// CHECK: gpu.module @dropout_mod
// CHECK: gpu.func @dropout(%{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: index, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: f32) kernel
// CHECK: arith.cmpf oge
// CHECK: memref.load
// CHECK: memref.store
