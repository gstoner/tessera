/-
LEAN-PILOT-1, theorem 2 of 3: the paged-KV logical→physical slot map is
in-bounds and injective.

Models (Tessera source; the anchors are drift-checked by
`tests/unit/test_formal_lean_pilot.py`, which fails if the modeled text moves):

tessera-anchor: src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateROCMPagedKVReadKernel.cpp :: Value logicalPage = b.create<arith::DivUIOp>(loc, logical, pageSize);
tessera-anchor: src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateROCMPagedKVReadKernel.cpp :: Value pageOffset = b.create<arith::RemUIOp>(loc, logical, pageSize);
tessera-anchor: src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateROCMPagedKVReadKernel.cpp :: mul(add(mul(physical, pageSize), pageOffset), elementsPerToken),
tessera-anchor: src/compiler/programming_model/lib/NativePagedKV.h :: builder.getNamedAttr("table_bounds", builder.getStringAttr("runtime_checked_physical_page_indices")),
tessera-anchor: python/tessera/models/ar_decode.py :: ``t`` lives at page ``t // page_size``, slot ``t % page_size``.

`NativePagedKV.h` owns the Schedule contract for `tessera.paged_kv_read`
(`LP` logical pages, `P` physical pages, page size `PS`); the ROCm generator
computes `logicalPage = pos / PS`, `pageOffset = pos % PS`, then the compact
physical token slot `table[logicalPage] * PS + pageOffset` (scaled by the
per-token element count, which is a separate, unmodeled multiply).
`ar_decode.paged_kv_pages` lays tokens out the same way with an identity table
(it writes through a row-major reshape, i.e. `page * PS + slot = t`).

What this proves, over `Nat` (unbounded naturals):
* `page_in_bounds`   — `pos < LP * PS → pos / PS < LP` (the table read is in range);
* `offset_in_bounds` — `pos < LP * PS → pos % PS < PS`;
* `reconstruct`      — `(pos / PS) * PS + pos % PS = pos` (the identity-table
  layout of `paged_kv_pages` is the flat token index);
* `slot_map_injective` — two positions with the same `(page, offset)` are equal;
* `physical_in_bounds` — if every table entry for a logical page `< LP` is
  `< P` (the property the contract names `runtime_checked_physical_page_indices`),
  the physical slot is `< P * PS`;
* `physical_injective` — if, additionally, the table is injective on `[0, LP)`,
  distinct in-range positions land in distinct physical slots.

What this does NOT cover:
* Machine integers. The kernel uses MLIR `index` arithmetic, the table is
  `i32`, and the host ABI passes `i64`; overflow of `table * PS`,
  `* elementsPerToken`, or a negative `i32` entry is not modeled — `Nat` cannot
  overflow or go negative.
* Where or whether the runtime actually checks the table bounds: the theorem
  *assumes* the bound the contract string declares. Table injectivity is not
  declared or checked anywhere; a read-only gather does not need it, and
  `physical_injective` is stated so a future paged *write* has the lemma it
  would need.
* The strided layout (`page_layout = "strided"`, source element strides), the
  head/feature split of `tokenOffset`, `start + token` offsetting, or any
  generated GPU code (ROCm, the SM120 sibling in `NVIDIALowering.cpp`).
-/

namespace TesseraFormal.PagedKV

/-- `logicalPage = pos / pageSize` (GenerateROCMPagedKVReadKernel.cpp). -/
def logicalPage (ps pos : Nat) : Nat := pos / ps

/-- `pageOffset = pos % pageSize` (GenerateROCMPagedKVReadKernel.cpp). -/
def pageOffset (ps pos : Nat) : Nat := pos % ps

/-- Compact physical token slot `table[logicalPage] * pageSize + pageOffset`. -/
def physicalSlot (table : Nat → Nat) (ps pos : Nat) : Nat :=
  table (logicalPage ps pos) * ps + pageOffset ps pos

/-- A non-empty logical range forces a positive page size. -/
theorem pageSize_pos {lp ps pos : Nat} (hpos : pos < lp * ps) : 0 < ps := by
  apply Nat.pos_of_ne_zero
  intro h
  subst h
  rw [Nat.mul_zero] at hpos
  exact Nat.not_lt_zero pos hpos

theorem page_in_bounds {lp ps pos : Nat} (hpos : pos < lp * ps) :
    logicalPage ps pos < lp :=
  (Nat.div_lt_iff_lt_mul (pageSize_pos hpos)).2 hpos

theorem offset_in_bounds {lp ps pos : Nat} (hpos : pos < lp * ps) :
    pageOffset ps pos < ps :=
  Nat.mod_lt pos (pageSize_pos hpos)

theorem reconstruct (ps pos : Nat) :
    logicalPage ps pos * ps + pageOffset ps pos = pos := by
  unfold logicalPage pageOffset
  rw [Nat.mul_comm]
  exact Nat.div_add_mod pos ps

theorem slot_map_injective {ps a b : Nat}
    (hp : logicalPage ps a = logicalPage ps b)
    (ho : pageOffset ps a = pageOffset ps b) : a = b := by
  have ha := reconstruct ps a
  have hb := reconstruct ps b
  rw [hp, ho] at ha
  exact ha.symm.trans hb

/-- `t * ps + o < np * ps` whenever `t < np` and `o < ps`. -/
private theorem slot_lt {t o np ps : Nat} (ht : t + 1 ≤ np) (ho : o < ps) :
    t * ps + o < np * ps := by
  have h1 : (t + 1) * ps ≤ np * ps := Nat.mul_le_mul ht (Nat.le_refl ps)
  rw [Nat.add_mul, Nat.one_mul] at h1
  generalize t * ps = a at *
  generalize np * ps = b at *
  omega

theorem physical_in_bounds {table : Nat → Nat} {np lp ps pos : Nat}
    (htab : ∀ p, p < lp → table p < np) (hpos : pos < lp * ps) :
    physicalSlot table ps pos < np * ps :=
  slot_lt (htab _ (page_in_bounds hpos)) (offset_in_bounds hpos)

/-- Division and remainder recover `(t, o)` from `t * ps + o` when `o < ps`. -/
private theorem decode_slot {t o ps : Nat} (ho : o < ps) :
    (t * ps + o) / ps = t ∧ (t * ps + o) % ps = o := by
  have hps : 0 < ps := Nat.lt_of_le_of_lt (Nat.zero_le o) ho
  rw [Nat.add_comm]
  constructor
  · rw [Nat.add_mul_div_right o t hps, Nat.div_eq_of_lt ho, Nat.zero_add]
  · rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ho]

theorem physical_injective {table : Nat → Nat} {lp ps a b : Nat}
    (hinj : ∀ p q, p < lp → q < lp → table p = table q → p = q)
    (ha : a < lp * ps) (hb : b < lp * ps)
    (h : physicalSlot table ps a = physicalSlot table ps b) : a = b := by
  unfold physicalSlot at h
  have da := decode_slot (t := table (logicalPage ps a)) (offset_in_bounds ha)
  have db := decode_slot (t := table (logicalPage ps b)) (offset_in_bounds hb)
  rw [h] at da
  have ht : table (logicalPage ps a) = table (logicalPage ps b) :=
    da.1.symm.trans db.1
  have ho : pageOffset ps a = pageOffset ps b := da.2.symm.trans db.2
  have hp := hinj _ _ (page_in_bounds ha) (page_in_bounds hb) ht
  exact slot_map_injective hp ho

end TesseraFormal.PagedKV
