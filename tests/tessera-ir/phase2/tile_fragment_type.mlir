// RUN: tessera-opt %s | tessera-opt | FileCheck %s
//
// W1.1 step 1 — `!tile.fragment` carries its own instruction contract.
//
// Piped through `tessera-opt` TWICE on purpose: the second parse consumes the
// first print, so this asserts an actual round-trip rather than that the
// printer emits something FileCheck likes. A custom parser/printer pair that
// disagrees is the classic way a parameterized type ships broken, and printing
// alone would not catch it.
//
// The type was `Opaque`, so `!tile.fragment == !tile.fragment` was ALWAYS TRUE
// and MLIR's type equality checked nothing — every compatibility rule was
// hand-written in `MMAOp::verify()` by chasing each operand's producing op.
// That chase cannot cross a block-argument edge, and a K-loop accumulator is an
// `scf.for` iter-arg by construction, so the typed form was unusable by every
// real GEMM (W1_1_TYPING_DESIGN.md §2).

// The BARE spelling stays legal and means all-unknown. Every fixture in the
// tree and every Python text emitter writes it, and producers migrate one at a
// time in steps 3-4 — making the parameters mandatory now would break all of
// them at once for no gain.
// CHECK-LABEL: func.func @bare_form_still_parses
// CHECK-SAME: !tile.fragment)
// CHECK-NOT: !tile.fragment<
func.func @bare_form_still_parses(%a: !tile.fragment) -> !tile.fragment {
  return %a : !tile.fragment
}

// -----

// CHECK-LABEL: func.func @typed_form_round_trips
// CHECK-SAME: !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "a", layout = "row_major", family = "auto">
func.func @typed_form_round_trips(
    %a: !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32",
                       role = "a", layout = "row_major", family = "auto">)
    -> !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32",
                      role = "a", layout = "row_major", family = "auto"> {
  return %a : !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16",
                             acc = "f32", role = "a", layout = "row_major",
                             family = "auto">
}

// -----

// Decision #15a's storage/accumulator split, expressed in the type where
// codegen selects the instruction. `elem` is the storage dtype and `acc` the
// accumulator; a fused single dtype cannot be written here even by accident.
// CHECK-LABEL: func.func @storage_and_accumulator_are_separate
// CHECK-SAME: elem = "fp8_e4m3", acc = "f32"
func.func @storage_and_accumulator_are_separate(
    %a: !tile.fragment<m = 16, n = 8, k = 32, elem = "fp8_e4m3", acc = "f32",
                       role = "acc", layout = "col_major",
                       family = "mma_sync">)
    -> !tile.fragment<m = 16, n = 8, k = 32, elem = "fp8_e4m3", acc = "f32",
                      role = "acc", layout = "col_major", family = "mma_sync"> {
  return %a : !tile.fragment<m = 16, n = 8, k = 32, elem = "fp8_e4m3",
                             acc = "f32", role = "acc", layout = "col_major",
                             family = "mma_sync">
}
