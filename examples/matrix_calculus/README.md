# Matrix Calculus in Tessera

A runnable tour of Tessera's differentiation surface, taught through
**Bright, Edelman & Johnson, _Matrix Calculus (for Machine Learning and Beyond)_**
([arXiv:2501.14787](https://arxiv.org/abs/2501.14787) — the MIT 18.S096/18.063
notes).

```bash
PYTHONPATH=python python3 examples/matrix_calculus/matrix_calculus_tutorial.py
```

Runs on **any host**: pure Python/numpy reference lane, no device, no build.

---

## The one idea

A derivative is not a number and not a matrix. It is the **linear operator**
`f'(x)` that maps a small change `dx` in the input to the first-order change
in the output:

```
df = f(x + dx) − f(x) = f'(x)[dx]
```

Gradients, Jacobians, forward vs reverse mode, adjoint methods and Hessians are
all that one statement specialized to a particular vector space, inner product,
or association order. The tutorial walks each specialization and **checks it
numerically** — every number it prints was computed, not asserted.

## What each section shows

| § | Notes | On the Tessera surface |
|---|---|---|
| 1 | §2.2, §2.6 | `jvp` on `f(A) = A²` returns `dA·A + A·dA` — and *not* `2A·dA`, because `dA` and `A` do not commute |
| 2 | §2.2.1, §4 | the directional derivative **is** `f'(A)[V]`; two function evaluations check the entire operator, basis-free |
| 3 | §4.4–4.6 | the truncation-vs-roundoff table: forward differences fall like `s`, central like `s²`, both blow up below `s ≈ √ε` |
| 4 | §5.1 | `∇‖A‖_F = A/‖A‖_F` via the Frobenius inner product — and why a weighted metric gives a *different gradient for the same differential* |
| 5 | §2.5.1, §8.4 | measured cost: `jacrev` takes 1 forward evaluation on `R³²→R`, `jacfwd` takes 33. This asymmetry is why backpropagation exists |
| 6 | §6.3 | the notes' own tridiagonal adjoint problem, run on `ops.tridiagonal_solve` |
| 7 | §12.2 | second derivatives are *symmetric bilinear maps*; a second-difference oracle that uses no AD at all |

## Start with §6

The notes' §6.3.3 problem is `g(p) = (cᵀ A(p)⁻¹ b)²` for a symmetric tridiagonal
`A(p)`, and they derive by hand that the whole gradient costs **two** solves —
one forward, one transposed — giving

```
∂g/∂p_k = v_k · x_{k+1} + v_{k+1} · x_k
```

Tessera already ships that adjoint. `python/tessera/solvers_ops.py` documents
its VJP as *"the transpose IS another tridiagonal solve, which is what makes
the VJP O(n)"*. The tutorial builds `A(p)` out of `ops.roll` +
`ops.tridiagonal_solve`, calls `tessera.autodiff.grad`, and shows the result
matching the textbook derivation to ~1e-17 — with a directional finite
difference as an independent third opinion.

## Related

- Review of this text against the Tessera surface, including the gaps it
  exposes: [`docs/audit/compiler/MATRIX_CALCULUS_REVIEW.md`](../../docs/audit/compiler/MATRIX_CALCULUS_REVIEW.md)
- Autodiff design and spec: [`docs/spec/AUTODIFF_SPEC.md`](../../docs/spec/AUTODIFF_SPEC.md),
  [`docs/audit/compiler/AUTODIFF_NEXTGEN_PLAN.md`](../../docs/audit/compiler/AUTODIFF_NEXTGEN_PLAN.md)
- Programming guide chapter 7 (Autodiff):
  [`docs/programming_guide/Tessera_Programming_Guide_Chapter7_Autodiff.md`](../../docs/programming_guide/Tessera_Programming_Guide_Chapter7_Autodiff.md)
