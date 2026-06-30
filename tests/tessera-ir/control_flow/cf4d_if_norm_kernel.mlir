// CF4d-if — GenerateROCMControlIfNormKernel lowers a CROSS-ELEMENT control_if
// (O = flag>0 ? rmsnorm(x) : layer_norm(x), 1xK carry) to one
// cooperative-workgroup gpu.func: x in LDS, the shape-(1) flag read once and
// uniform, a uniform scf.if selecting which cooperative norm runs. On-device
// execution on gfx1151 is proven by tests/unit/test_rocm_control_if_norm_exec.py.
//
// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s -split-input-file --generate-rocm-control-if-norm-kernel \
// RUN:   | FileCheck %s

func.func @t(%x: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %0 = "tessera.rmsnorm"(%x) {eps = 1.000000e-05 : f64} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %0 : tensor<1x8xf32>
}
func.func @e(%x: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %0 = "tessera.layer_norm"(%x) {eps = 1.000000e-05 : f64} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %0 : tensor<1x8xf32>
}

// CHECK-LABEL: func.func @f
// kernel ABI (X, FLAG, O : memref<?xf32>, K : index); x → LDS, barrier, then the
// uniform flag[0] > 0 predicate selecting the cooperative norm, store O.
// CHECK:       gpu.func @tessera_control_if_norm_0(%{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: index) workgroup({{.*}}memref<256xf32, #gpu.address_space<workgroup>>) kernel
// CHECK:         gpu.barrier
// CHECK:         arith.cmpf ogt
// CHECK:         scf.if
// CHECK:           math.sqrt
// CHECK:         memref.store
// CHECK:         gpu.return
func.func @f(%x: tensor<1x8xf32>, %flag: tensor<1xf32>) -> tensor<1x8xf32> {
  %o = "tessera.control_if"(%x, %flag) {
    then_branch = @t, else_branch = @e, flag_arg_index = 1 : i64
  } : (tensor<1x8xf32>, tensor<1xf32>) -> tensor<1x8xf32>
  return %o : tensor<1x8xf32>
}

// -----
// ─── a non-norm (elementwise relu) branch is NOT a cross-element norm — left
// ─── untouched for the CF4c elementwise control_if / the guard ──────────────
func.func @t2(%x: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %0 = "tessera.relu"(%x) : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %0 : tensor<1x8xf32>
}
func.func @e2(%x: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %0 = "tessera.relu"(%x) : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %0 : tensor<1x8xf32>
}

// CHECK-LABEL: func.func @g
// CHECK:       tessera.control_if
// CHECK-NOT:   gpu.func
func.func @g(%x: tensor<1x8xf32>, %flag: tensor<1xf32>) -> tensor<1x8xf32> {
  %o = "tessera.control_if"(%x, %flag) {
    then_branch = @t2, else_branch = @e2, flag_arg_index = 1 : i64
  } : (tensor<1x8xf32>, tensor<1xf32>) -> tensor<1x8xf32>
  return %o : tensor<1x8xf32>
}
