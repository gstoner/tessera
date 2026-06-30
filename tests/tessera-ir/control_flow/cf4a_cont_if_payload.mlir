// CF4a-cont — decode the tessera.control_if op-list payload (then_opcodes /
// else_opcodes / ...) into real @then / @else func.funcs of tessera.* ops, so
// CF2's LowerControlFlowToSCF can lower the branch to scf.if.
//
// Op-list ABI for control_if: ids 0..n-1 = operands, each branch op j = id n+j
// (no carry). CF2 calls the branches with the NON-flag operands, so the
// materialized @then/@else take the non-flag operands and a payload id k maps to
// arg (k < flag ? k : k-1). A branch referencing the flag id is left untouched.
//
// RUN: tessera-opt %s -split-input-file --tessera-materialize-control-payload \
// RUN:   --allow-unregistered-dialect | FileCheck %s

// ─── then = silu(x), else = mul(x, x); flag at index 0, data x at index 1 ────
// ids: 0 = flag, 1 = x; then/else op0 → id 2. The flag is NOT a branch arg, so
// the materialized branches take just %x (the non-flag operand).
func.func private @tb()
func.func private @eb()

// CHECK-LABEL: func.func private @tb(%arg0: tensor<8xf32>) -> tensor<8xf32>
// CHECK:       tessera.silu %arg0
// CHECK-LABEL: func.func private @eb(%arg0: tensor<8xf32>) -> tensor<8xf32>
// CHECK:       tessera.mul %arg0, %arg0
// CHECK-LABEL: func.func @f
// CHECK:       tessera.control_if %{{.*}}, %{{.*}} {else_branch = @eb, flag_arg_index = 0 : i64, then_branch = @tb}
// CHECK-NOT:   then_opcodes
func.func @f(%flag: tensor<1xf32>, %x: tensor<8xf32>) -> tensor<8xf32> {
  %r = "tessera.control_if"(%flag, %x) {
    then_branch = @tb, else_branch = @eb, flag_arg_index = 0 : i64,
    then_opcodes = array<i32: 23>, then_in0 = array<i32: 1>,
    then_in1 = array<i32: -1>, then_iattr = array<i32: 0>,
    then_fattr = array<f32: 1.0e-05>, then_out_id = 2 : i64,
    else_opcodes = array<i32: 3>, else_in0 = array<i32: 1>,
    else_in1 = array<i32: 1>, else_iattr = array<i32: 0>,
    else_fattr = array<f32: 1.0e-05>, else_out_id = 2 : i64
  } : (tensor<1xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}

// -----
// ─── a branch that references the FLAG id (0) is left untouched ─────────────
// then_in0 = [0] → the flag; the flag isn't a branch arg, so the op-list can't
// resolve it. The payload stays for the guard / a future lowering.
func.func private @tb2()
func.func private @eb2()

// CHECK-LABEL: func.func @g
// CHECK:       tessera.control_if
// CHECK-SAME:  then_opcodes
func.func @g(%flag: tensor<8xf32>, %x: tensor<8xf32>) -> tensor<8xf32> {
  %r = "tessera.control_if"(%flag, %x) {
    then_branch = @tb2, else_branch = @eb2, flag_arg_index = 0 : i64,
    then_opcodes = array<i32: 20>, then_in0 = array<i32: 0>,
    then_in1 = array<i32: -1>, then_iattr = array<i32: 0>,
    then_fattr = array<f32: 1.0e-05>, then_out_id = 2 : i64,
    else_opcodes = array<i32: 20>, else_in0 = array<i32: 1>,
    else_in1 = array<i32: -1>, else_iattr = array<i32: 0>,
    else_fattr = array<f32: 1.0e-05>, else_out_id = 2 : i64
  } : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}

// -----
// ─── then_branch == else_branch (SHARED stub) is left untouched ─────────────
// Materializing the second arm would overwrite the first's body, so both arms
// would call one body — a silent semantic change. Leave it for the guard.
func.func private @shared()

// CHECK-LABEL: func.func @s
// CHECK:       tessera.control_if
// CHECK-SAME:  then_opcodes
func.func @s(%flag: tensor<1xf32>, %x: tensor<8xf32>) -> tensor<8xf32> {
  %r = "tessera.control_if"(%flag, %x) {
    then_branch = @shared, else_branch = @shared, flag_arg_index = 0 : i64,
    then_opcodes = array<i32: 23>, then_in0 = array<i32: 1>,
    then_in1 = array<i32: -1>, then_iattr = array<i32: 0>,
    then_fattr = array<f32: 1.0e-05>, then_out_id = 2 : i64,
    else_opcodes = array<i32: 20>, else_in0 = array<i32: 1>,
    else_in1 = array<i32: -1>, else_iattr = array<i32: 0>,
    else_fattr = array<f32: 1.0e-05>, else_out_id = 2 : i64
  } : (tensor<1xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}
