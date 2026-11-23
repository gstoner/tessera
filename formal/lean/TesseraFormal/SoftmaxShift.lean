/-
LEAN-PILOT-1, theorem 3 of 3: softmax is invariant under a constant shift,
`softmax(x - c) = softmax(x)`.

Models (Tessera source; the anchors are drift-checked by
`tests/unit/test_formal_lean_pilot.py`, which fails if the modeled text moves):

tessera-anchor: src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAFlashAttnKernel.cpp :: rescale corr = exp(m_old - m_new), p = exp(s - m_new), sum l.
tessera-anchor: src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAFlashAttnKernel.cpp :: b.create<arith::SubFOp>(loc, s, mnew));

The gfx1151/gfx1201 WMMA flash-attention generator subtracts the running row
maximum `m_new` from every score before `exp`, and the final `O = acc / l`
normalizes by the shifted sum. That is only the softmax of the *unshifted*
scores because of the identity proven here, with `c` the (arbitrary) shift.

What this proves, over the real numbers `ℝ`, for a finite vector `x : Fin n → ℝ`
and any real `c`: `softmax (x - c) i = softmax x i`. The proof holds for every
`c`, so in particular for the row maximum.

What this does NOT cover:
* IEEE-754 arithmetic. The point of the max shift in the kernel is to avoid
  `exp` overflow, which does not exist in `ℝ`; the theorem shows the shift is
  harmless, not that it is sufficient. `-inf` masking, the `moldSmall` select,
  `nan` handling, and rounding of `exp`/`sum`/division are not modeled.
* The online recurrence itself: that rescaling a partial sum and accumulator by
  `corr = exp(m_old - m_new)` per key tile reproduces the one-pass softmax. That
  is a separate theorem (a candidate follow-on), which this lemma would support.
* `P @ V`, the scale factor, causal/ragged masking, dropout, or any generated
  code on any backend (ROCm, NVIDIA, Apple `emit/apple_msl.py`).
-/
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Analysis.SpecialFunctions.Exp

open Finset

namespace TesseraFormal.Softmax

/-- Real-valued softmax of a finite vector. -/
noncomputable def softmax {n : Nat} (x : Fin n → ℝ) (i : Fin n) : ℝ :=
  Real.exp (x i) / ∑ j, Real.exp (x j)

theorem softmax_shift {n : Nat} (x : Fin n → ℝ) (c : ℝ) (i : Fin n) :
    softmax (fun j => x j - c) i = softmax x i := by
  have he : Real.exp c ≠ 0 := (Real.exp_pos c).ne'
  unfold softmax
  simp only [Real.exp_sub]
  rw [← Finset.sum_div]
  exact div_div_div_cancel_right₀ he _ _

end TesseraFormal.Softmax
