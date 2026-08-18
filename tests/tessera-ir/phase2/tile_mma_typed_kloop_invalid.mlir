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

// W1.1 step 5 removed the bare-fragment producer-chasing contract entirely.
// Stored historical IR still parses so it can receive this named migration
// diagnostic rather than an opaque parser failure.
func.func @mixed_typed_and_bare(
    %x: !ja,
    %y: !tile.fragment,
    %z: !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">)
    -> !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto"> {
  // CHECK: TILE_MMA_BARE_FRAGMENT_REMOVED
  %r = tile.mma %x, %y, %z : (!ja, !tile.fragment, !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">) -> !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">
  return %r : !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">
}

// -----

// ── Descriptor / type disagreement (PR #503 review) ───────────────────────
//
// The first cut compared only m/n/k/family/accType, which let a retained
// descriptor contradict the type on exactly the fields codegen reads. Both
// `NVIDIALowering.cpp` and `TileToROCM.cpp` select the instruction variant AND
// the physical register layout from the DESCRIPTOR — so such IR verified while
// codegen followed a different contract than the type stated. The comparison is
// role-dependent because the descriptor has always carried A and B separately.

#bad_elem = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16,
                           a = "f16", b = "bf16", acc = "f32",
                           a_layout = "row_major", b_layout = "col_major",
                           k_blocks = 1>
!ka = !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "a", layout = "row_major", family = "auto">
!kb = !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "b", layout = "col_major", family = "auto">
!kc = !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">

// A bf16 A-fragment paired with `a = "f16"`.
func.func @descriptor_element_contradicts_type(%x: !ka, %y: !kb, %z: !kc) -> !kc {
  // CHECK: TILE_MMA_DESC_DISAGREES
  // CHECK-SAME: contradicts the fragment type's elem
  %r = tile.mma %x, %y, %z {mma = #bad_elem} : (!ka, !kb, !kc) -> !kc
  return %r : !kc
}

// -----

#bad_layout = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16,
                             a = "bf16", b = "bf16", acc = "f32",
                             a_layout = "col_major", b_layout = "col_major",
                             k_blocks = 1>
!la = !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "a", layout = "row_major", family = "auto">
!lb = !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32", role = "b", layout = "col_major", family = "auto">
!lc = !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "auto">

// A row_major A-fragment paired with `a_layout = "col_major"`. This one decides
// the physical register layout, so the wrong answer is a transposed operand.
func.func @descriptor_layout_contradicts_type(%x: !la, %y: !lb, %z: !lc) -> !lc {
  // CHECK: TILE_MMA_DESC_DISAGREES
  // CHECK-SAME: contradicts the fragment type's layout
  %r = tile.mma %x, %y, %z {mma = #bad_layout} : (!la, !lb, !lc) -> !lc
  return %r : !lc
}
