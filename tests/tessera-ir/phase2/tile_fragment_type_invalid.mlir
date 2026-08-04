// RUN: not tessera-opt --split-input-file %s 2>&1 | FileCheck %s
//
// W1.1 step 1 — what the parameterized `!tile.fragment` now REJECTS.
//
// Decision #10a: a type that only ever accepts proves nothing. Every case here
// was legal before this step, because `!tile.fragment` was `Opaque` and so
// every fragment compared equal to every other one.
//
// None of these needs a line of verifier code — they are MLIR's own type
// equality, which is the whole point of moving the contract into the type.

// The one that matters most, and the one an earlier draft of the design would
// have allowed: `family` selects a PHYSICAL REGISTER ABI, not just an
// instruction. `ROCMFragmentLayout.h` resolves it to a descriptor whose wave
// size differs by family (32 for RDNA3/RDNA4/gfx125x WMMA, 64 for CDNA MFMA),
// and `TileToROCM.cpp` says those descriptors "are intentionally
// non-interchangeable". The draft put `family` on the attribute, reasoning that
// instruction selection is an operation-local choice; that would have let a
// loop init packed for mma_sync feed a body tile.mma selecting wgmma, with the
// types comparing equal — weakening an existing contract exactly at the
// block-argument edge W1.1 exists to open.
func.func @family_is_not_interchangeable(
    %a: !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32",
                       role = "a", layout = "row_major", family = "mma_sync">)
    -> !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32",
                      role = "a", layout = "row_major", family = "wgmma"> {
  // CHECK: error: type of return operand 0
  return %a : !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16",
                             acc = "f32", role = "a", layout = "row_major",
                             family = "mma_sync">
}

// -----

// A mismatched ACCUMULATOR. This is Decision #15a's storage/accum split, and
// the exact fact W1.3's boundary verifier carries down from Graph IR — an
// fp32 accumulator silently becoming fp16 is the instruction-selection
// corruption both items exist to prevent.
func.func @accumulator_width_must_match(
    %a: !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32",
                       role = "acc", layout = "row_major", family = "auto">)
    -> !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f16",
                      role = "acc", layout = "row_major", family = "auto"> {
  // CHECK: error: type of return operand 0
  return %a : !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16",
                             acc = "f32", role = "acc", layout = "row_major",
                             family = "auto">
}

// -----

// An operand ROLE mismatch — a B fragment where an A fragment is required.
func.func @role_must_match(
    %a: !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32",
                       role = "a", layout = "row_major", family = "auto">)
    -> !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32",
                      role = "b", layout = "row_major", family = "auto"> {
  // CHECK: error: type of return operand 0
  return %a : !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16",
                             acc = "f32", role = "a", layout = "row_major",
                             family = "auto">
}

// -----

// The BARE form is not a wildcard. `!tile.fragment` means all-unknown, and an
// all-unknown fragment is not interchangeable with a stated one — otherwise the
// legacy spelling would be a hole straight through the new contract, and every
// unmigrated producer would launder any fragment into any use.
func.func @bare_is_not_a_wildcard(%a: !tile.fragment)
    -> !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32",
                      role = "a", layout = "row_major", family = "auto"> {
  // CHECK: error: type of return operand 0
  return %a : !tile.fragment
}
