// RUN: not tessera-opt --split-input-file %s 2>&1 | FileCheck %s
//
// W1.1 step 2 — what the TYPE-based `tile.mma` contract rejects.
//
// The legacy path recovered these facts by chasing producers and comparing
// whole `#tile.mma_desc` attributes. Reading them from the operand types has to
// be no weaker, or step 2 would trade a working contract for an expressible
// one. Every case below was caught by descriptor equality before; each is now
// caught by the types, and reported per-field so the diagnostic names the
// disagreement instead of "descriptors differ".

!fa = !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "a", layout = "row_major", family = "auto">

// `family` selects a physical register ABI — wave 32 for RDNA/WMMA vs 64 for
// CDNA/MFMA — which is why it is a type parameter at all (PR #501 review).
func.func @family_mismatch(
    %x: !fa,
    %y: !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "b", layout = "col_major", family = "wgmma">,
    %z: !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">)
    -> !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto"> {
  // CHECK: TILE_MMA_FAMILY_MISMATCH
  %r = tile.mma %x, %y, %z : (!fa, !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "b", layout = "col_major", family = "wgmma">, !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">) -> !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">
  return %r : !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">
}

// -----

!ga = !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "a", layout = "row_major", family = "auto">

// One accumulator contract per MMA (Decision #15a). This is the same fact W1.3's
// boundary verifier carries down from Graph IR — an fp32 accumulator quietly
// becoming fp16 is the instruction-selection corruption both items exist to stop.
func.func @accumulator_dtype_mismatch(
    %x: !ga,
    %y: !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f16", role = "b", layout = "col_major", family = "auto">,
    %z: !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">)
    -> !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto"> {
  // CHECK: TILE_MMA_ACCUM_MISMATCH
  %r = tile.mma %x, %y, %z : (!ga, !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f16", role = "b", layout = "col_major", family = "auto">, !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">) -> !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">
  return %r : !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">
}

// -----

!ha = !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "a", layout = "row_major", family = "auto">

// The accumulator's own element type IS the accumulator dtype. Without this a
// producer could hand over a bf16-element fragment while every operand agreed
// `acc = "f32"` — the accumulator-width confusion wearing a correct label.
func.func @accumulator_element_is_not_the_accumulator(
    %x: !ha,
    %y: !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "b", layout = "col_major", family = "auto">,
    %z: !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "acc", layout = "row_major", family = "auto">)
    -> !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "acc", layout = "row_major", family = "auto"> {
  // CHECK: TILE_MMA_ACCUM_ELEMENT
  %r = tile.mma %x, %y, %z : (!ha, !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "b", layout = "col_major", family = "auto">, !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "acc", layout = "row_major", family = "auto">) -> !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "acc", layout = "row_major", family = "auto">
  return %r : !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "acc", layout = "row_major", family = "auto">
}

// -----

!ia = !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "a", layout = "row_major", family = "auto">

// Operand roles are positional: A, B, accumulator. A B-fragment in the
// accumulator slot is a swapped-operand bug, and it type-checks under any
// contract that does not state the role.
func.func @role_in_the_wrong_slot(
    %x: !ia,
    %y: !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "b", layout = "col_major", family = "auto">,
    %z: !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "b", layout = "row_major", family = "auto">)
    -> !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "b", layout = "row_major", family = "auto"> {
  // CHECK: TILE_MMA_OPERAND_ROLE
  %r = tile.mma %x, %y, %z : (!ia, !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "b", layout = "col_major", family = "auto">, !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "b", layout = "row_major", family = "auto">) -> !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "b", layout = "row_major", family = "auto">
  return %r : !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "b", layout = "row_major", family = "auto">
}

// -----

!ja = !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "a", layout = "row_major", family = "auto">

// The typed form is all-or-nothing. A half-migrated op would otherwise get the
// weaker contract on whichever operand still carried the bare type — the same
// "legacy spelling is a hole through the new contract" hazard the bare-form
// wildcard case guards at the type level.
func.func @mixed_typed_and_bare(
    %x: !ja,
    %y: !tile.fragment,
    %z: !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">)
    -> !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto"> {
  // CHECK: TILE_MMA_MIXED_FRAGMENT_FORMS
  %r = tile.mma %x, %y, %z : (!ja, !tile.fragment, !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">) -> !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">
  return %r : !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">
}
