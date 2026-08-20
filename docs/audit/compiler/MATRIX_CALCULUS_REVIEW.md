---
last_updated: 2026-08-20
audit_role: reference
scope: python/tessera/ops (linalg family), python/tessera/autodiff/{vjp,jvp,grad,transforms,tape}.py, python/tessera/debug.py, python/tessera/compiler/primitive_coverage.py, python/tessera/ga/manifold.py, python/tessera/ebm/geo_sampling.py
companions: DIFFERENTIABLE_PROGRAMMING_REVIEW.md (the sibling book review this is a delta against) · AUTODIFF_NEXTGEN_PLAN.md · AUTODIFF_ARCHITECTURE_REVIEW.md · RIEMANNIAN_OT_PLAN.md · ../../spec/AUTODIFF_SPEC.md
source_text: Bright, Edelman & Johnson, "Matrix Calculus (for Machine Learning and Beyond)" (arXiv:2501.14787v1, MIT 18.S096/18.063)
example: ../../../examples/matrix_calculus/
---

# Matrix Calculus — Book Review Against the Tessera Surface

> **Routing:** start at [`README.md`](README.md). This is a delta/reference
> review; it orders nothing. Global ordering lives only in
> [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md), and
> `MASTER_AUDIT.md` + `docs/audit/generated/` stay status truth (Decision #26).
> Every observation below is labelled with how it was produced and when.

**Why this text, given we already have a book review.**
[`DIFFERENTIABLE_PROGRAMMING_REVIEW.md`](DIFFERENTIABLE_PROGRAMMING_REVIEW.md)
covers Blondel & Roulet, which is a *differentiable-programming* book: AD
machinery, relaxations, implicit differentiation, stochastic gradients. Edelman
& Johnson is a *matrix* calculus book: what the derivative of `det`, `A⁻¹`,
`A½`, a QR factor, or a symmetric eigendecomposition **is**, what vector space
it lives in, and which inner product turns a differential into a gradient. The
delta between the two source texts is almost exactly the delta this review
reports.

**The organizing idea, and why it is not decoration.** The notes never define a
derivative as a matrix. They define it as the linear operator `f'(x)` with
`df = f'(x)[dx]`, and then observe that a Jacobian is one *representation* of
that operator — often an expensive and structure-destroying one. That is the
same position Tessera already took twice: `implicit.py` carries `A`/`Aᵀ` as
matvecs rather than matrices, and
[`AUTODIFF_NEXTGEN_PLAN.md`](AUTODIFF_NEXTGEN_PLAN.md) §3.5 promotes that to an
`OperatorTangent` type. The notes supply the mathematical vocabulary for a
decision the repo has already made on engineering grounds.

---

## Already owned — confirmed by this text, not re-litigated here

| Existing finding | Owner | Confirmation in the notes |
|---|---|---|
| B4 — eager `hvp` is finite differences | `AUTODIFF_ARCHITECTURE_REVIEW.md` | §8.4.1 makes forward-over-reverse *the* HVP algorithm |
| B1/B2 — `jacrev`/`jacfwd` sweep one row/column | `AUTODIFF_ARCHITECTURE_REVIEW.md` | §8.4 forward cost ∝ inputs, reverse ∝ outputs |
| C1 — one derivative datum, two registries | `DIFFERENTIABLE_PROGRAMMING_REVIEW.md`, `AUTODIFF_NEXTGEN_PLAN.md` §2.1 | §2.5 chain rule is associativity; the mode is the association order, not the datum |
| C2 — nonsmooth selection is a semantic key | `DIFFERENTIABLE_PROGRAMMING_REVIEW.md` | §14 distributional/weak derivatives; generalized gradients at crossings |
| C3 — estimator legality per node (pathwise vs score-function) | `DIFFERENTIABLE_PROGRAMMING_REVIEW.md` | §11.2–11.4 reparameterization vs discrete jumps |
| T3 — implicit differentiation via IFT | `DIFFERENTIABLE_PROGRAMMING_REVIEW.md`, `implicit.py` | §6.3.1 IFT + one adjoint solve; §9.2.2 the ODE adjoint |
| §3.5 operator tangents | `AUTODIFF_NEXTGEN_PLAN.md` | §2.6 `d(A⁻¹) = −A⁻¹ dA A⁻¹` is a rule, not a matrix |
| §3.7 stochastic estimators | `AUTODIFF_NEXTGEN_PLAN.md` | §11.4 stochastic triples for discrete randomness |

---

## Summary — the delta findings

Ordered by confidence × payoff. Costs are rough and **not** a schedule.

| ID | Finding | Notes ref | Kind | Governance hook |
|---|---|---|---|---|
| **MC1** | ~~The matrix-function / factorization derivative family is absent from the op surface~~ — **fixed 2026-08-20**: ten primitives added with both modes, law-swept. `expm`/`pinv` deferred with reasons | §3, §7, §13 | stdlib + AD | #24 (both registries) |
| **MC2** | ~~`svd`'s VJP returns **NaN** at repeated singular values, and its `eps` guard is dead code~~ — **fixed 2026-08-20**, see below | §13.2.1, §14 | correctness defect | #21a, "no silent NaN" |
| **MC3** | ~~The gradient is `metric⁻¹ ∘ differential`~~ — **fixed 2026-08-20**: `tessera.metric` + `grad(metric=)` as the consumer. MLIR-side `manifold` lowering still open | §5.1, §13.1–13.2 | design | #29, #30 |
| **MC4** | Kronecker/`vec` is a cost trap: `(B⊗C)vec(Y) = vec(CYBᵀ)` — **partly closed 2026-08-20**: the ops and the identity exist and are pinned; the Graph IR rewrite is **not** written | §3.3.3 | canonicalization | #28 (never cap the ceiling) |
| **MC5** | ~~`check_grad` is coordinate-wise, fixed-ε, fixed-tolerance~~ — **fixed 2026-08-20**: scale-aware step, `check_grad_directional`, `check_order_of_accuracy` | §2.2.1, §4.4–4.6 | test methodology | #26 (evidence quality) |
| **MC6** | Forward-over-reverse does not compose — **and neither does reverse-over-forward, for the same reason**. Diagnostic fixed 2026-08-20; the composition itself is blocked on AD-WEIL-1 | §8.4.1 | sharpens B4 | — |
| **MC7** | ~~`count_primitive_executions` does not count JVP-mode executions~~ — **fixed 2026-08-20**: the increment sat after the dispatch that returns | §2.5.1 | measurement defect | #29 (consumer sees wrong number) |
| **MC8** | ~~`ops.stack([Parameter, …])` silently returns a `dtype=object` array; `ops.cat` raises a raw numpy error~~ — **fixed 2026-08-20**, and it was the smaller half of a larger defect; see below | — (found while writing the tutorial) | fail-open defect | fail-closed discipline |
| **MC9** | ~~Second-derivative symmetry and the second-difference formula are free laws~~ — **fixed 2026-08-20**: added to the sweep as `hessian_symmetry` and `hessian_second_difference` | §12.2 | law harness | AD-LAW-1 Law set |

---

## Status after the 2026-08-20 remediation pass

Seven of the nine findings are closed and two are partly closed. Everything
below is code plus an enforcing test; the numbers are from the run recorded in
the commit message, not from this prose (Decision #26).

| ID | State | What landed | Enforcer |
|---|---|---|---|
| MC1 | ✅ closed | `python/tessera/linalg_ops.py` — `det`, `logdet`, `inv`, `solve`, `trace`, `eigh`, `kron`, `vec`, `matrix_power`, `norm`, each with a real VJP **and** JVP, registered in `op_catalog` and (derived from it) `primitive_coverage` | `test_linalg_matrix_functions.py` + `LAW_INPUT_SPECS` entries so all ten are swept by the adjoint and chain laws |
| MC2 | ✅ closed | `degeneracy_policy` semantic key (see above) | `test_factorization_degeneracy.py` |
| MC3 | ✅ closed | `python/tessera/metric.py` — `Metric` protocol with `Euclidean`/`Weighted`/`Sphere`/`Orthogonal`, consumed by `grad(..., metric=)` | `test_metric_gradients.py` |
| MC4 | 🟡 partial | `kron`/`vec` exist with **column-major** `vec`, so `(B⊗C)vec(Y) = vec(CYBᵀ)` holds as written; the identity is pinned by test | `test_linalg_matrix_functions.py::test_kronecker_vec_identity` |
| MC5 | ✅ closed | `debug.fd_step`, `check_grad_directional`, `check_order_of_accuracy`; `check_grad`'s default step is now scale-aware | `test_derivative_methodology.py` |
| MC6 | 🟡 partial | the diagnostic now names the real cause; `hvp` is scale-aware | `test_derivative_methodology.py` |
| MC7 | ✅ closed | the counter increment moved above the forward-mode dispatch | `test_derivative_methodology.py::test_jacfwd_cost_scales_with_the_input_dimension` |
| MC8 | ✅ closed | transitive unwrap + sequence operands on the tape (see above) | `test_sequence_operand_unwrap.py` |
| MC9 | ✅ closed | Laws `hessian_symmetry` and `hessian_second_difference` in `run_law_sweep()` | `test_derivative_methodology.py` |

### What is deliberately **not** done, and why

* **MC4's Graph IR rewrite.** `matmul(kron(B, C), vec(Y))` →
  `vec(matmul(matmul(C, Y), transpose(B)))` is a canonicalization on the C++
  MLIR side, needs a `tessera-opt` build plus a lit fixture, and per Decision
  #10a needs a **negative** fixture too (a case where the rewrite must not fire
  — `B⊗C` consumed elsewhere, where materializing once may genuinely be
  cheaper). Writing it blind and unverified would be worse than leaving it
  named. The Python-level ops now exist, which is what the rewrite would need
  to match against.
* **MC6's actual composition.** Sharper than the original finding: **both**
  orders are blocked, and for one structural reason — VJP and JVP rule bodies
  are numpy functions, not `ops.*` calls, so neither mode can trace through the
  other's rules. `jvp(grad(f), …)` and `grad(lambda x: jvp(f, x, v)[1])` fail
  identically. That is not fixable by a deeper tape; it is exactly the problem
  AD-WEIL-1 exists to solve (one derivative datum evaluated in a higher-order
  algebra). Upstream landed `derivative_contract.py` on 2026-08-19, but that is
  a *declaration* registry — 136 lines, no evaluator — so the substrate is not
  there yet. What did land here is that the failure now says so instead of
  blaming raw-numpy loss math.
* **`expm` and `pinv`.** Both are real work rather than a transcription:
  `expm` needs scaling-and-squaring plus the block-matrix Fréchet-derivative
  trick (`expm([[A, E], [0, A]])`'s upper-right block), and `pinv`'s VJP
  carries two extra projector terms for the non-square case. They are named in
  MC1's table and remain open.
* **The MLIR-side `manifold` key.** MC3 gives it a Python consumer; making the
  attribute reach a backend is a lowering concern that this review does not own.

---

## MC1 — the matrix-function / factorization family is absent

**Observed 2026-08-20**, by enumerating the live surface rather than grepping:

```bash
PYTHONPATH=python python3 -c "import tessera, re; n=[x for x in dir(tessera.ops) if not x.startswith('_')]; \
print(sorted(x for x in n if re.search(r'det|inv|solve|eig|kron|trace|expm|pinv|chol|qr|svd|lu', x)))"
```

Of 379 public `tessera.ops` attributes the linear-algebra family is
`cholesky`, `cholesky_solve`, `lu`, `qr`, `svd`, `tri_solve`,
`tridiagonal_solve`. The 505-row `primitive_coverage` registry agrees — the
same seven, plus nothing. **Absent:** `det`, `logdet`/`slogdet`, `inv`,
general dense `solve`, `trace`, `eigh`/`eigvalsh`, `expm`, `kron`, `vec`,
`pinv`, `lstsq`, `matrix_power`, `norm`.

AD coverage inside the family is also uneven:

| op | VJP | JVP |
|---|---|---|
| `cholesky`, `qr`, `svd`, `tri_solve` | ✅ | ✅ |
| `tridiagonal_solve` | ✅ | ❌ |
| `cholesky_solve`, `lu` | ❌ | ❌ |

**Why this matters more than "add some ops."** These are the primitives whose
derivatives are *not* elementwise and *not* multilinear — the exact cases
`linear.py` cannot derive and the pointwise-ODE family of
[`AUTODIFF_NEXTGEN_PLAN.md`](AUTODIFF_NEXTGEN_PLAN.md) §3.2 cannot reach. They
are the third derivative family, and the notes hand them over closed-form:

| primitive | differential (notes §7, §13) | gradient of the scalar case |
|---|---|---|
| `det` | `d(det A) = det(A) · tr(A⁻¹ dA)` | `∇ det = det(A) · A⁻ᵀ` |
| `logdet` | `d log det A = tr(A⁻¹ dA)` | `∇ logdet = A⁻ᵀ` |
| `inv` | `d(A⁻¹) = −A⁻¹ dA A⁻¹` | — (operator, see below) |
| `solve` | `dx = −A⁻¹ (dA) x + A⁻¹ db` | one adjoint solve (§6.3) |
| `eigh` (simple λ) | `dλᵢ = qᵢᵀ dS qᵢ` | `∇λᵢ = qᵢ qᵢᵀ` |
| `eigh` (vectors) | `(Qᵀ dQ)ᵢⱼ = (Qᵀ dS Q)ᵢⱼ / (λⱼ − λᵢ)`, `i ≠ j` | — |

`inv` is the poster child for §3.5's `OperatorTangent`: written as a Jacobian
it is `−(A⁻ᵀ ⊗ A⁻¹)`, an `n²×n²` object costing `Θ(n⁴)` to apply; written as
the rule it is two triangular solves. **Adding `inv` without an operator
tangent is how you build the slow version.** So MC1 is not
independent of AD-OPERATOR-1 — it is that slice's most persuasive consumer,
which matters under Decision #29.

**Where it lands.** `logdet`/`solve`/`eigh` are not decorative: `logdet`
is the normalizing constant of every Gaussian likelihood and the EBM partition
term; `eigh` derivatives are what a spectral-norm penalty, a Riemannian
optimizer, or a curvature diagnostic needs; `expm` is the exponential map
[`RIEMANNIAN_OT_PLAN.md`](RIEMANNIAN_OT_PLAN.md) writes as `exp_x(·)`.
Tessera already ships `spectral_norm` with a VJP but has no `eigh` under it.

---

## MC2 — `svd`'s VJP NaNs at repeated singular values

> **Fixed 2026-08-20.** The `degeneracy_policy` semantic key now exists at
> [`python/tessera/autodiff/degeneracy.py`](../../../python/tessera/autodiff/degeneracy.py),
> is consumed by every factorization rule in `vjp.py`/`jvp.py`, and is enforced
> by [`tests/unit/test_factorization_degeneracy.py`](../../../tests/unit/test_factorization_degeneracy.py)
> (40 cases, negative fixtures included). The diagnosis below is kept as the
> record of what was wrong; the fix is described after it.


`python/tessera/autodiff/vjp.py:5327` builds the coupling matrix

```python
F = 1.0 / (s2[None, :] - s2[:, None] + np.eye(len(s)) * eps)   # eps = 1e-12
np.fill_diagonal(F, 0.0)
```

The `eps` term is added **only on the diagonal**, and the very next line sets
the diagonal to zero — so the guard protects exactly the entries it then
erases, and the off-diagonal `1/(sⱼ² − sᵢ²)` is left completely unregularized.
Repro (2026-08-20, Mac/Homebrew python3):

```bash
PYTHONPATH=python python3 -c "
import numpy as np
from tessera.autodiff.vjp import _VJPS
A = np.eye(3) * 2.0                       # singular values 2, 2, 2
d = (np.full((3,3), .1), np.full(3, .1), np.full((3,3), .1))
print(_VJPS['svd'](d, A)[0])"
```

emits three bare numpy warnings (`divide by zero`, `invalid value`) and returns
an all-`NaN` gradient. Near-degenerate spectra are worse than the exactly
degenerate case, because no warning fires at all — the gradient is merely
enormous and wrong.

The notes state the cause precisely (§14): at eigenvalue/singular-value
crossings the factors **cease to be differentiable**, and for a defective
matrix `dλ` scales like `√‖dA‖`, so no finite generalized derivative exists.
This is a semantic condition, and Decision #21a says a semantic key never
defaults silently.

**The repo already knows how to do this.** `solvers_ops.py` documents its own
contract as *"a singular solve fails closed rather than emitting NaNs"* and
raises on a zero pivot. `svd`'s VJP is the same situation with the opposite
behaviour, in the same tree.

### What shipped (2026-08-20)

`degeneracy.py` declares the key, mirroring `nonsmooth.py`'s shape — one
declared policy per op, helpers that every rule routes through, and a test file
as the enforcer (Decision #29).

| policy | behaviour | offered for |
|---|---|---|
| `fail_closed` | **default.** Raise `TesseraDegeneracyError` naming the op, the coinciding index pair, the measured gap, and the cluster membership | all four |
| `generalized` | admit the part that survives the degeneracy, refuse the rest: the cotangent is accepted iff its antisymmetric `U`/`V` couplings vanish on degenerate pairs **and** its `ds` weights are constant across each cluster — exactly when the limit exists | `svd` |
| `damped:<τ>` | Tikhonov form `g/(g²+τ²)`, τ relative to `s_max²`; an approximation the caller has explicitly accepted | `svd` |
| `unchecked` | run with no guard — the pre-fix behaviour, kept reachable *by explicit request*, which is what distinguishes it from a silent default | all four |

Selecting a policy an op does not implement raises rather than degrading to
something else, so the table is a contract and not a suggestion.

**Two thresholds, not one.** Conflating them was the temptation: the coupling
loses half its digits at `gap ≈ √ε`, but the derivative only ceases to *exist*
at `gap ≈ n·ε`. Refusing at `√ε` would reject gradients that exist and still
carry eight good digits, so the guard splits them — `existence_tolerance()`
(`n·eps`, numpy's own `matrix_rank` criterion) fails closed, and the band above
it up to `conditioning_tolerance()` (`√eps`) proceeds while emitting
`TesseraDegeneracyWarning`. Non-existence is semantic and fails closed (#21a);
poor conditioning is accuracy, and may proceed *provided it says so*. That is
the same truncation-vs-roundoff split the notes draw in §4.6, applied to
conditioning instead of step size.

**The dead `eps` is gone.** What it was accidentally doing — keeping the
diagonal out of the division — is now structural: the denominator's diagonal is
set to 1 before the reciprocal and zeroed after, and under `generalized` the
degenerate blocks are masked out of the *denominator* rather than divided and
patched, so no `1/0` is ever evaluated and no numpy warning is emitted.

**`qr`, `cholesky`, `tri_solve` had the same shape and are fixed too.** All
three invert a triangular factor; `check_factor_rank` gates each on
`min|diag| / max|diag|`, in both modes. `cholesky` additionally converts
numpy's `LinAlgError` on a non-PD input into the Tessera diagnostic, preserving
the numpy error as `__cause__`.

**Worked case that now succeeds.** `grad(‖A‖_*)` — the nuclear norm — is
genuinely differentiable at a degenerate full-rank `A`, and equals `U Vᵀ`.
Before, it returned `NaN` (the `ds`-only path multiplied `inf` by `0`); under
`fail_closed` it refuses with an explanation; under `generalized` it returns
`U Vᵀ` exactly. The three outcomes are all defensible and the caller picks
which one they meant.

**Both of this section's "still open" items were closed upstream while this
was in flight** (rebased onto `1d3009e`, 2026-08-20). An earlier draft recorded
that `jvp_svd` returned zero `U`/`Vᵀ` tangents and that the reduced-SVD
backward dropped the orthogonal-complement term for non-square `A`. PR #594's
AD-LAW-1 work fixed both independently — the forward rule now carries the full
singular-vector tangent with projector terms, and the backward rule carries the
matching adjoints — and PR #594 review made the whole `svd` pair **batch-aware**
over leading dimensions.

That is a better outcome than this review's own plan, and the guard was
rebased *onto* it rather than over it: `svd_coupling` is now batch-aware
(`s` of shape `[..., k]` → `F` of shape `[..., k, k]`, each batch element
judged on its own spectrum and named in the diagnostic), and both modes build
their coupling matrix through it. Forward mode passes
`admit_generalized=False`: every output it returns is per-component — each
`s_i`, each singular vector — and none of those exist inside a degenerate
cluster, so there is no restricted form for `generalized` to fall back on and
it refuses there too.

Note the independent convergence: upstream's rewrite *also* dropped the dead
`1e-12` term, replacing it with the same structural `(1 - I) / (den + I)`
construction used here. It did not, however, add a degeneracy guard — the
off-diagonal `1/(s_j² - s_i²)` was still unbounded, so a repeated singular
value still produced `inf`/`NaN`. The eps was the visible half of the defect;
the missing contract was the other half.

---

## MC3 — the gradient is `metric⁻¹ ∘ differential`

§5.1 is blunt about it: `f'(x)` is a linear **form**; the gradient is whatever
you take an inner product with to recover `df`, so *changing the inner product
changes the gradient*. `∇^(W) f = W⁻¹ ∇f`. The differential is invariant; the
gradient is not.

Three things in-tree become one thing under that sentence:

1. **`manifold` is declared and unconsumed.** Decision #29 flags it; §5.1 +
   §13 name the consumer. On a constraint surface the differential is
   restricted to the tangent space, and the gradient is the projection:
   `x ᵀx = 1 ⇒ xᵀ dx = 0 ⇒ ∇_S f = (I − xxᵀ) ∇f` (§13.1.2); on the orthogonal
   group `QᵀdQ` is antisymmetric (§13.2), which is the whole parameter count
   `n(n−1)/2`.
2. **Two independent local implementations already exist.**
   `python/tessera/ebm/geo_sampling.py` does tangent projection + retraction
   for sphere sampling; `python/tessera/ga/manifold.py` defines
   `Euclidean`/`Sphere`/`SOn` — but as *quadrature domains*
   (`sample_points()` + `weights()`), with no `project_tangent` / `egrad2rgrad`
   / `retract`. Neither knows about the other. That is a sixth instance of the
   pattern [`AUTODIFF_NEXTGEN_PLAN.md`](AUTODIFF_NEXTGEN_PLAN.md) §1 tabulates:
   the same abstraction paid for repeatedly because it was never named.
3. **Muon is already a non-Euclidean gradient method** and nothing says so.
   It is steepest descent under a spectral-norm geometry; it is registered as
   an ordinary op with a VJP. Same for every preconditioner and any future
   natural-gradient work.

**Fix shape:** a small `Metric` protocol — `inner(u,v)`, `sharp(covector)`
(i.e. `W⁻¹`), `project_tangent(x, v)`, `retract(x, v)` — with `Euclidean` as
the default instance, consumed by (a) `grad(..., metric=)`, (b) the optimizer
surface, (c) the `manifold` key's lowering. Small, host-free, and it converts a
#29 violation into a contract. It is also the substrate
[`RIEMANNIAN_OT_PLAN.md`](RIEMANNIAN_OT_PLAN.md) assumes but does not define.

---

## MC4 — Kronecker/`vec` is a rewrite rule, never a materialization

§3.3.3 is the whole finding: the identity

```
(B ⊗ C) vec(Y) = vec(C Y Bᵀ)
```

turns an `n²×n²` matvec (`Θ(n⁴)`, plus `Θ(n⁴)` storage) into two GEMMs
(`Θ(n³)`, no extra storage). Tessera has no `kron` and no `vec` today, so
there is nothing to fix — which is exactly why this belongs in the record
**before** MC1 lands. The moment `vec`/`kron` exist, a user (or a generated
Jacobian path) will write the `Θ(n⁴)` form, and it will be correct, which is
what makes it dangerous.

Concretely: a canonicalization on Graph IR matching `matmul(kron(B, C), vec(Y))`
→ `vec(matmul(matmul(C, Y), transpose(B)))`, with a negative fixture per
Decision #10a — a case where the rewrite must **not** fire (e.g. `B ⊗ C` also
consumed elsewhere, where materializing once may genuinely be cheaper). The
notes' §3.2.1 4×4 Jacobian of the 2×2 matrix-square function is a ready-made
lit fixture: small enough to write out by hand, structured enough that a wrong
rewrite is visible.

---

## MC5 — `check_grad`'s methodology is weaker than the notes'

`python/tessera/debug.py:326` loops `np.ndindex(arr.shape)`, perturbs one
element at a time by a **hardcoded `eps=1e-4`**, and compares against a fixed
`atol`/`rtol`. Three problems the notes address directly:

1. **Coordinate-wise is the wrong unit** (§2.2.1). `f'(x)[v]` for one random
   `v` tests the whole operator in **two** evaluations; `n` coordinate probes
   test the same operator in `2n`. For matrix- or operator-valued inputs
   "one coordinate" is not even the natural object.
2. **A fixed `eps` ignores scale** (§4.6). The notes' rule of thumb is
   `‖δx‖ ≈ √ε ‖x‖` — about half the significant digits. `1e-4` is right only
   when `‖x‖ ≈ 10⁴`; on well-scaled inputs it sits four orders above the
   optimum, and on fp32 it sits below the roundoff floor entirely.
3. **A tolerance is not the strongest available test** (§4.4–4.5). Sweeping
   `s` and checking that the relative error falls like `s` (forward) or `s²`
   (central) before the roundoff floor is *scale-free* and *dtype-honest*: a
   rule that is wrong by a constant factor still passes a loose tolerance but
   flattens the slope immediately.

The tutorial (below, §3) prints the sweep; the table it produces is the shape
`check_grad` should be asserting on. This is small, host-free work that
strengthens an oracle the whole AD-LAW-1 lane leans on.

---

## MC6 — forward-over-reverse does not compose in the eager lane

`AUTODIFF_ARCHITECTURE_REVIEW.md` B4 records that eager `hvp` is a central
difference of `grad`. The sharper statement, verified 2026-08-20: the
composition that would replace it **raises**.

```bash
PYTHONPATH=python python3 -c "
import numpy as np, tessera
from tessera import ops
from tessera.autodiff import grad, jvp
f = lambda z: ops.reduce(ops.mul(ops.sin(z), z), op='sum')
jvp(grad(f), np.arange(4.0), np.ones(4))"
# TesseraAutodiffError: backward target is not a tape-recorded output.
```

The JVP trace does not see through the reverse tape, so §8.4.1's canonical
algorithm — reverse for `∇f`, forward for `∂/∂α` — is unavailable at the Python
surface even though both halves exist. `grad.py` is honest about this and
points at `JitFn.compiled_hvp_ir` for the exact path; the gap is that the
reference lane, which is the oracle everything else is tested against, cannot
express the algorithm it is supposed to be an oracle for.

Cheap interim: make the JVP trace transparent to a nested tape, or expose
`hvp(..., mode="forward_over_reverse")` implemented as `jvp` over a
tape-recorded gradient. Either way, MC9's symmetry law becomes the test.

---

## MC7 — the cost oracle cannot see forward mode

`count_primitive_executions` (`tape.py:226`) is R1's Baur–Strassen guard. It
increments in the **tape** wrapper, so JVP-traced executions are invisible.
Verified 2026-08-20 on `f: R³² → R`:

| transform | `count_primitive_executions` | actual forward evaluations |
|---|---|---|
| `jacrev` | 4 | 1 (+ 1 backward sweep) |
| `jacfwd` | 4 | **33** |

`jacfwd` runs one sample evaluation plus one `jvp` per input dimension
(`transforms.py:266`), so the true ratio is ~33× and the oracle reports 1×.
Any cost-ratio law evaluated in forward mode is therefore measuring nothing.
The fix is one counter increment on the JVP path plus a fixture that pins the
`jacfwd` count to `in_size + 1`.

---

## MC8 — sequence operands were broken end to end

> **Fixed 2026-08-20.** Enforced by
> [`tests/unit/test_sequence_operand_unwrap.py`](../../../tests/unit/test_sequence_operand_unwrap.py)
> — 23 cases, of which **22 fail against unpatched `HEAD`** and all pass after.
> The diagnosis below is the record of what was wrong; what shipped follows it.


Found while writing the tutorial, not while reading the notes. `_unwrap` in
`python/tessera/__init__.py:3942` peels exactly one `_data` level; a
`nn.Parameter` needs two. The tape wrapper unwraps top-level arguments, so this
never surfaces for `mul`/`matmul` — but ops taking a **list** of tensors bypass
it:

```bash
PYTHONPATH=python python3 -c "
import numpy as np
from tessera import ops
from tessera.nn.module import Parameter
p, q = Parameter(np.arange(3.)), Parameter(np.arange(3.)+10)
print(repr(ops.stack([p, q], axis=0)))   # dtype=object array — no error
print(repr(ops.cat([p, q], axis=0)))     # raw numpy ValueError"
```

`stack` returns a `dtype=object` array of `DistributedArray`s and reports
success; `cat` raises a numpy shape error with no Tessera diagnostic. Both sit
on a gradient path. This is the failure shape the repo's own rules are written
against — a silent wrong answer beats a loud one only for the person who
shipped it.

### The larger defect underneath it (found while fixing this)

Item 4 of the fix — "assert gradients reach each Parameter" — turned out to be
impossible without a second, independent repair, because **the tape could not
carry a sequence operand at all**. `_route_positional` recorded a list argument
as neither an operand nor a kwarg, so `vjp_cat` was invoked with no `xs`:

```
TypeError: vjp_cat() missing 1 required positional argument: 'xs'
```

That is not a `Parameter` problem. Reverse mode through `ops.cat`/`ops.stack`
raised for **any** input, plain ndarrays included, and it reproduces on
unpatched `HEAD`. Forward mode was worse than raising: `_JVPTrace.record_op`
looked up `id(primal)` for a list, never matched, left the trace inactive, and
returned a silently **zero** tangent.

So the reported symptom was the visible third of a defect that also included a
hard failure nobody had hit and a silent wrong answer nobody could see.

### What shipped (2026-08-20)

| Layer | Change |
|---|---|
| `_unwrap` ([`python/tessera/__init__.py`](../../../python/tessera/__init__.py)) | peels `._data` **transitively** to a depth limit, and fails closed unless the result has a numeric dtype (`biufc`) — an object array, a string, or a cyclic wrapper is a diagnostic, never a return value |
| `_unwrap_sequence` | new: unwraps each element of a sequence operand and names *the op and the offending index* when one is not a tensor. `ops.cat`/`ops.stack` route through it |
| `TapeEntry.input_groups` | new field recording how flat `inputs` regroup into the rule's positional arguments: `None` per plain operand, `k` for a sequence built from the next `k` inputs. `None` overall keeps today's fast path untouched |
| `_describe_sequence` | recognises a list/tuple whose elements are **all non-literal array-likes**, so a configuration list (`ops.pad(x, [(1, 1)])`, whose elements describe as literals) is never mistaken for operands |
| `Tape.backward` | regroups arguments before calling the rule and scatters a sequence rule's per-element cotangents back to one per input, so accumulation (including `cat([a, a])`) needs no special case |
| `_JVPTrace.record_op` | looks up tangents **per element** of a sequence operand, with an explicit zero for untracked elements, so forward mode can no longer report an all-zero tangent by omission |

`E_TENSOR_UNWRAP` is registered in the central diagnostic-code registry and
drift-gated. Verified after the fix: `grad` through `cat`/`stack` matches a
directional finite difference, `jacrev` through `cat([a, a])` is the stacked
identity, repeated operands accumulate, and forward-mode tangents are exact.

**Method note for the audit.** Only `cat` and `stack` take a sequence operand
today — established by signature-scanning the live `tessera.ops` namespace, not
by grep. Any op added with that shape needs `input_groups` and a per-element
VJP, or it will fail the same three ways.

---

## MC9 — free second-derivative laws for the AD-LAW-1 harness

§12.2 proves `f''(x)[u,v] = f''(x)[v,u]` from nothing more than commutativity
of `+`, and derives an oracle that uses **no AD at all**:

```
f''(x)[u,v] ≈ [ f(x+u+v) + f(x) − f(x+u) − f(x+v) ] / (s²)
```

Two consequences for the law harness of
[`AUTODIFF_NEXTGEN_PLAN.md`](AUTODIFF_NEXTGEN_PLAN.md) §4:

- **Symmetry is a free law.** Any HVP path — eager FD, `compiled_hvp_ir`, a
  future forward-over-reverse, a `TruncatedJet(2)` — must satisfy
  `⟨v, Hu⟩ = ⟨u, Hv⟩`. It needs no reference implementation, which is exactly
  the property that makes Law 3 useful there; unlike Law 3 it is not
  satisfiable by a matched-wrong pair *and* a nonzero probe, because the
  second-difference oracle pins the magnitude independently.
- **The oracle is external.** §3 of that plan wants derivative correctness
  carried by independent oracles rather than by the mode pair agreeing with
  itself. This is one, for order 2, at the cost of four function evaluations.

The tutorial's §7 runs both.

---

## The tutorial — `examples/matrix_calculus/`

The second half of the ask: a worked sample that teaches Tessera *through* the
notes. [`examples/matrix_calculus/matrix_calculus_tutorial.py`](../../../examples/matrix_calculus/matrix_calculus_tutorial.py)
runs on any host (pure reference lane, no device) and prints computed numbers,
never asserted ones:

| § | Notes | What it shows on the Tessera surface |
|---|---|---|
| 1 | §2.2, §2.6 | `jvp` on `f(A)=A²` gives `dA·A + A·dA`, and **not** `2A·dA` |
| 2 | §2.2.1, §4 | the directional derivative *is* `f'(A)[V]`; two evaluations check the whole operator |
| 3 | §4.4–4.6 | the truncation/roundoff table: forward `O(s)`, central `O(s²)`, floor at `√ε` |
| 4 | §5.1 | `∇‖A‖_F = A/‖A‖_F` via the Frobenius inner product; the weighted-metric gradient (MC3) |
| 5 | §2.5.1, §8.4 | measured `jacrev` (1 eval) vs `jacfwd` (33 evals) on `R³²→R` |
| 6 | §6.3 | the notes' own tridiagonal adjoint problem, on `ops.tridiagonal_solve`, matching the hand derivation to `~1e-17` |
| 7 | §12.2 | Hessian symmetry + the AD-free second-difference oracle |

§6 is the one to read first. The notes' §6.3.3 problem — `g(p) = (cᵀA(p)⁻¹b)²`
with `A(p)` symmetric tridiagonal — is *already* implemented in
`python/tessera/solvers_ops.py`, whose VJP is documented as "the transpose IS
another tridiagonal solve, which is what makes the VJP O(n)." The tutorial
shows Tessera's reverse mode reproducing the textbook's hand-derived
`∂g/∂p_k = v_k x_{k+1} + v_{k+1} x_k` to `5.6e-17`. That is a rare thing to be
able to demonstrate, and it is the strongest single argument the example makes.

**Suggested follow-on** (not done here): promote §§1–3 and §7 into
`tests/unit/` as executable laws rather than printed output, and cite the
tutorial from `docs/programming_guide/Tessera_Programming_Guide_Chapter7_Autodiff.md`.

---

## What not to do

- **Do not add `det`/`inv`/`eigh` as numpy passthroughs with numeric VJPs.**
  `_numeric_vjp` exists in `vjp.py` and flips a coverage axis to complete; on
  this family it would encode a `Θ(n⁴)` central difference as the contract and
  make MC2's class of defect invisible. These primitives have closed forms; use
  them, and register the operator form where one exists.
- **Do not "fix" a degeneracy by raising `eps`.** Damping produces a finite,
  plausible, wrong gradient — the exact failure Decision #21a's
  `manifold`-defaulting scar records. MC2 ships damping only as a policy the
  caller must name, never as a default. The same rule applies to `eigh` and
  `expm` when MC1 lands.
- **Do not build a `Metric` abstraction with no consumer.** MC3 is worth doing
  only with `grad(..., metric=)` and at least one optimizer wired to it in the
  same change (#29).
- **Do not treat the notes' §11 as superseding C3.** They agree; the stochastic
  triple is one construction, and C3's node classification is the thing that
  decides *which* estimator is legal.
