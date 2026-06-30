// CF4c — GenerateROCMControlForKernel lowers an elementwise-branch
// tessera.control_if to ONE gpu.func device kernel: grid over the data elements;
// per thread, r = (FLAG[0] > 0) ? then_scalar(x) : else_scalar(x). One dispatch;
// the shape-(1) flag selects the branch. (On-device execution on gfx1151 is
// proven by tests/unit/test_rocm_control_if_exec.py.)
//
// RUN: tessera-opt %s -split-input-file --generate-rocm-control-for-kernel \
// RUN:   --allow-unregistered-dialect | FileCheck %s

// ─── then = relu(x), else = x + x ───────────────────────────────────────────
func.func @tb(%x: tensor<8xf32>) -> tensor<8xf32> {
  %0 = "tessera.relu"(%x) : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
func.func @eb(%x: tensor<8xf32>) -> tensor<8xf32> {
  %0 = "tessera.add"(%x, %x) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK:       gpu.func @tessera_control_if_0(%{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: index) kernel
// CHECK:         %[[X:.*]] = memref.load
// CHECK:         %[[F:.*]] = memref.load
// CHECK:         %[[S:.*]] = arith.cmpf ogt, %[[F]]
// CHECK:         %[[R:.*]] = scf.if %[[S]] -> (f32) {
// CHECK:           arith.maximumf %[[X]]
// CHECK:           scf.yield
// CHECK:         } else {
// CHECK:           arith.addf %[[X]], %[[X]]
// CHECK:           scf.yield
// CHECK:         }
// CHECK:         memref.store %[[R]]
// CHECK:         gpu.return
func.func @f(%flag: tensor<1xf32>, %x: tensor<8xf32>) -> tensor<8xf32> {
  %r = "tessera.control_if"(%flag, %x) {
    then_branch = @tb, else_branch = @eb, flag_arg_index = 0 : i64
  } : (tensor<1xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}

// -----
// ─── a non-elementwise branch (matmul) is NOT lowered — left for the guard ──
func.func @tb2(%x: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = "tessera.matmul"(%x, %x) : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
func.func @eb2(%x: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = "tessera.add"(%x, %x) : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func.func @g
// CHECK:       tessera.control_if
// CHECK-NOT:   gpu.func
func.func @g(%flag: tensor<1xf32>, %x: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %r = "tessera.control_if"(%flag, %x) {
    then_branch = @tb2, else_branch = @eb2, flag_arg_index = 0 : i64
  } : (tensor<1xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}

// -----
// ─── a SHARED then/else stub is NOT lowered (distinct symbols required) ─────
func.func @shared(%x: tensor<8xf32>) -> tensor<8xf32> {
  %0 = "tessera.relu"(%x) : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func.func @h
// CHECK:       tessera.control_if
// CHECK-NOT:   gpu.func
func.func @h(%flag: tensor<1xf32>, %x: tensor<8xf32>) -> tensor<8xf32> {
  %r = "tessera.control_if"(%flag, %x) {
    then_branch = @shared, else_branch = @shared, flag_arg_index = 0 : i64
  } : (tensor<1xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}

// -----
// ─── an EXTRA non-flag operand the (X,FLAG,O,N) kernel can't realize → skip ──
// control_if(%flag, %x, %w): the flat kernel would ignore %w, so the op is left
// for the SCF lowering / guard.
func.func @tb3(%x: tensor<8xf32>) -> tensor<8xf32> {
  %0 = "tessera.relu"(%x) : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
func.func @eb3(%x: tensor<8xf32>) -> tensor<8xf32> {
  %0 = "tessera.add"(%x, %x) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func.func @k
// CHECK:       tessera.control_if
// CHECK-NOT:   gpu.func
func.func @k(%flag: tensor<1xf32>, %x: tensor<8xf32>, %w: tensor<8xf32>)
    -> tensor<8xf32> {
  %r = "tessera.control_if"(%flag, %x, %w) {
    then_branch = @tb3, else_branch = @eb3, flag_arg_index = 0 : i64
  } : (tensor<1xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}
