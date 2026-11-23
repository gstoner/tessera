/-
LEAN-PILOT-1, theorem 1 of 3: the registered `exp` VJP rule is the pullback of
the derivative of `exp`.

Models (Tessera source; the anchors are drift-checked by
`tests/unit/test_formal_lean_pilot.py`, which fails if the modeled text moves):

tessera-anchor: python/tessera/autodiff/vjp.py :: @_vjp("exp")
tessera-anchor: python/tessera/autodiff/vjp.py :: return (np.asarray(dout) * np.exp(np.asarray(x)),)

`vjp_exp(dout, x)` returns `dout * exp(x)` elementwise. `vjpExp` below is that
expression on one scalar element; `exp` has no broadcasting, so the rule has no
`_sum_to_shape` reduction to model.

What this proves, over the real numbers `ℝ`:
* `vjpExp dout x = dout * deriv Real.exp x` — the rule is the scalar pullback
  `dout ↦ dout * f'(x)` of `f = exp`;
* `HasDerivAt Real.exp (vjpExp 1 x) x` — the seed-1 cotangent is the derivative.

What this does NOT cover:
* IEEE-754 arithmetic. NumPy evaluates `np.exp` and `*` in float64 (or the
  array's dtype) with rounding, overflow to `inf`, and `nan` propagation; none
  of that is modeled. A proof over `ℝ` is not a proof about float results.
* Any generated code. The compiled AD routes (native tapes, `native_*_jvp`,
  device kernels) and `math.exp` lowerings (libdevice, ROCDL, MSL) are separate
  implementations; this theorem says nothing about them.
* The tape machinery that calls the rule (`autodiff/tape.py`), dtype
  promotion, or the `_vjp` registration itself — only the rule's formula.
-/
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

namespace TesseraFormal.VJPExp

/-- The scalar formula of `vjp_exp` in `python/tessera/autodiff/vjp.py`. -/
noncomputable def vjpExp (dout x : ℝ) : ℝ := dout * Real.exp x

/-- The registered rule is the pullback of the derivative of `exp`. -/
theorem vjpExp_eq_pullback (dout x : ℝ) :
    vjpExp dout x = dout * deriv Real.exp x := by
  simp [vjpExp, Real.deriv_exp]

/-- With a unit cotangent the rule returns the derivative itself. -/
theorem vjpExp_hasDerivAt (x : ℝ) : HasDerivAt Real.exp (vjpExp 1 x) x := by
  simpa [vjpExp] using Real.hasDerivAt_exp x

end TesseraFormal.VJPExp
