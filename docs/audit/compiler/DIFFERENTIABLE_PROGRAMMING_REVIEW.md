---
last_updated: 2026-08-08
audit_role: reference
scope: python/tessera/autodiff, python/tessera/losses.py, python/tessera/rng.py, python/tessera/arch.py, python/tessera/custom.py, python/tessera/compiler/{primitive_coverage,op_catalog,evaluator,rematerialization_cost}.py, src/transforms/lib/{EffectAnnotationPass,ActivationRematerializationPass}.cpp
companions: AUTODIFF_ARCHITECTURE_REVIEW.md (the primary autodiff review this extends) · SEQUENCE_MIXER_ENGINEERING_PLAN.md · RIEMANNIAN_OT_PLAN.md · ../../spec/AUTODIFF_SPEC.md
source_text: Blondel & Roulet, "The Elements of Differentiable Programming" (arXiv:2403.14606v4, 2024)
---

# Differentiable Programming — Book Review Against the Tessera Surface

> **Routing:** start at [`README.md`](README.md). This is a delta/reference
> review. Its Python implementations are semantic oracles; compiler work is
> ordered only by [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md).
>
A capability review of Tessera's differentiation surface, TSOL operator set,
IR passes, and helper libraries against **Blondel & Roulet, "The Elements of
Differentiable Programming"** (arXiv:2403.14606v4). The book is a
first-principles treatment of AD, smoothing/relaxation, implicit
differentiation, second-order structure, and stochastic-program gradients — the
theory surface a standalone compiler must eventually implement.

**This is a *delta* review.** It does **not** re-derive what
[`AUTODIFF_ARCHITECTURE_REVIEW.md`](AUTODIFF_ARCHITECTURE_REVIEW.md) already owns
— the book independently confirms every one of those findings, and where it
does the confirmation is noted inline. Everything below is content the book
covers that the existing autodiff plan does **not**.

Already owned by `AUTODIFF_ARCHITECTURE_REVIEW.md` (confirmed by the book, not
repeated here):

| Existing finding | Book confirmation |
|---|---|
| A4 / D2 — no compiler forward mode | §4 "forward mode needs no tape"; the mode a tile compiler is best at |
| A5 / D3 — no activity analysis | §4.4 reverse-mode complexity depends on the active set |
| A3 / D4 — reverse pass rejects control flow | Ch. 4 §"Control flows" — for/scan/while adjoints |
| B6 / D5 — greedy-interval checkpointing, not Revolve | §4.6 recursive halving + treeverse DP |
| B4 — `hvp` is finite differences | §8.1 forward-on-reverse is the preferred HVP |
| B1 / B2 — `jacrev`/`jacfwd` re-run forward | §4.3 Jacobian one row/column at a time |
| B3 — `vmap` is a Python loop | — |
| B8 — implicit diff scaffolded, not built | Ch. 10 (this review extends it — see T3) |

**Status truth stays with the generated dashboards** (Decision #26). Counts
quoted here are point-in-time reads from the source tree on 2026-08-06 and are
labelled as such; they are evidence for *gaps*, not status claims.

---

## Summary — the delta findings

| ID | Finding | Book ref | Cost | Governance hook |
|---|---|---|---|---|
| **C1** | Automatic linear transposition — VJP/JVP hand-maintained twice; `transpose_rule` axis has no consumer | §4.5.4 | ~2 wk | #29, and *reduces* D2 |
| **C2** | Nonsmooth (Clarke) selection is undeclared and inconsistent across ops | §2.7 | days | #21a |
| **C3** | Stochastic computation graphs give the effect lattice a fail-closed structure | §11.5 | ~2 wk | #5, #30 |
| **C4** | Semirings unify attention / scan / sequence-mixer; backward comes free | §10.9 | ~4 wk | #21a |
| **C5** | Cost-weighted treeverse (better than uniform Revolve); online + reversible | §4.6–4.7 | folds into D5 | #28 |
| **C6** | GGN / Fisher / IHVP / Hessian-diagonal — a 13th contract axis | Ch. 8 | folds into D6 | — |
| **T1** | Smoothing/relaxation family absent (sparsemax, gumbel, soft-topk, perturbed) | Ch. 4, 12, 13 | ~3 wk | PB-3 shape |
| **T2** | Fenchel-Young losses collapse a chunk of `losses.py` | Ch. 15 §4 | ~2 wk | — |
| **T3** | Python `custom_root`/IHVP oracle plus value-producing compiler IFT IR landed; physical solver consumption remains open | Ch. 10 | landing | — |
| **R1** | Baur–Strassen cost-ratio oracle | §4.4.3 | days | catches B1/B2 |
| **R2** | Randomized forward-mode gradient (memory-free lane) | §4.8 | folds into D2 | #28 |

---

## Implementation status (built 2026-08-07)

Seven of the eleven findings are **implemented and tested in the Python
reference lane**. This is not native Graph/Schedule/Tile support. Each
row below is code + a passing test file; counts and details live in the tests,
not here.

| ID | Status | Modules | Tests |
|---|---|---|---|
| **C2** | ✅ Python oracle | `autodiff/nonsmooth.py`; refactored `autodiff/vjp.py` | `test_nonsmooth_selection.py` |
| **R1** | ✅ forward-reexecution guard | `autodiff/tape.py` (`count_primitive_executions`), `compiler/evaluator.py` | `test_baur_strassen_oracle.py` |
| **C1** | ✅ Python oracle | `autodiff/linear.py`; `custom.py` (`transpose_rule` consumer) | `test_linear_transposition.py` |
| **T3** | ✅ Python oracle | `autodiff/implicit.py` (`cg_solve`/`ihvp`/`custom_root`/`adjoint_state_grad`) | `test_implicit_diff.py` |
| **T1** | ✅ Python/reference catalog | `relaxation.py` (sparsemax/entmax15/soft_top_k/gumbel_softmax/perturbed_argmax); `rng.py` (`gumbel`) | `test_relaxation_ops.py` |
| **T2** | ✅ Python helper | `losses.py` (`fenchel_young_loss`/`fy_loss_and_grad`/`sparsemax_loss`/`softmax_fy_loss`) | `test_fenchel_young_losses.py` |
| **C3** | ✅ Python trace analysis | `compiler/stochastic_graph.py` (analysis + `certify_deterministic`) | `test_stochastic_graph.py` |
| **C4** | ⏳ open | semirings — larger, rides the sequence-mixer track | — |
| **C5** | landing | complete-backward/residual-memory measurement and measured-step treeverse candidate pruning landed; executable treeverse and exact family packets remain open | `compiler/residual_evaluator.py` |
| **C6** | ⏳ open | GGN/Fisher/IHVP-optimizer/Hessian-diagonal — IHVP primitive landed in T3; the second-order *estimators* remain | — |
| **R2** | ⏳ open | randomized forward-mode — folds into the planned D2 | — |

**Scope of what landed.** These are correctness- and surface-level slices in the
numpy reference lane: a declared nonsmooth policy, a cost oracle, a JVP-derivation
consumer for `transpose_rule`, an implicit-diff surface, the relaxation operator
family, the Fenchel-Young loss template, and a fail-closed stochastic-graph
analysis. They do **not** by themselves rewire the C++ MLIR passes (the effect
lattice, `AutodiffPass`), which is the W2 work C3's analysis is a substrate for.

---

## Compiler

### C1. Automatic linear transposition — one derivative registry, not two (§4.5.4)

**Observed (2026-08-06):** `python/tessera/autodiff/vjp.py` is 5,413 lines /
292 `@_vjp` registrations; `jvp.py` is 3,805 lines / 259. For the linear
primitives the two files are the same logic written twice — `vjp_transpose` /
`jvp_transpose`, and the `gemm` / `matmul` / `cast` / `reshape` / `cat` / `pad`
/ `slice` pairs.

**Book result.** For a linear map `l`, the JVP is `l` applied to the tangent
(`∂l(w)[v] = l(v)`) and the VJP is its adjoint (`∂l(w)*[u] = l*(u)`), both
independent of the primal point. The consequence the book states explicitly:
**the VJP of a VJP is the JVP.** So for the linear subset you declare *one*
thing — linearity plus the adjoint — and both modes are recoverable by
transposition.

**The consumer gap this closes.** `transpose_rule` is a declared
`primitive_coverage` axis and a field on `@custom_primitive`, set at
[`custom.py:61`](../../../python/tessera/custom.py) and reported at
[`custom.py:217`](../../../python/tessera/custom.py) — and **nothing consumes
it.** That is a live Decision #29 violation (a declaration must have a
consumer). An automatic-transposition engine is that consumer, so the fix and
the payoff are the same work.

**Interaction with D2.** This makes the planned forward mode *cheaper* than the
`AUTODIFF_ARCHITECTURE_REVIEW.md` §D2 estimate: `buildTangent` is only needed
for genuinely nonlinear primitives; the linear families fall out of transposing
`buildAdjoint`.

### C2. Nonsmooth (Clarke) selection is a semantic choice, currently undeclared (§2.7)

**Book result.** At a kink, any element of the Clarke subdifferential is a
valid generalized gradient. Which one you pick is therefore a **semantic**
decision — and Decision #21a says semantic keys never default silently.

**Observed (2026-08-06):** Tessera picks differently in each place, with
nothing written down.

| Op | Selection at kink/tie | Reference |
|---|---|---|
| `relu` | hard `0` (`x > 0`) | `vjp.py:480` |
| `maximum` / `minimum` | even `0.5 / 0.5` split | `vjp.py:2680` |
| `amax` / `amin` | `1/count` split | `vjp.py:2335` |
| `sign` | `0` (correct, documented) | `vjp.py:5096` |

All four are legal Clarke selections, but elementwise-max splits ties while
`relu` does not, and no `math_semantics` row states the policy — so a backend
kernel is free to disagree with the numpy oracle at exactly the input where the
selections differ. That is the PB "gate green while the guarded thing is
broken" failure shape from [`../MASTER_AUDIT.md`](../MASTER_AUDIT.md).
Separately, `abs` (`vjp.py:5089`) and `absolute` (`vjp.py:2666`) are two
registrations of the same math, and one forces `float64` while the other does
not.

**Fix.** A `nonsmooth_selection` key on the `math_semantics` axis, plus one
differential fixture per nonsmooth op that probes *exactly at* the kink (the
only input where a wrong selection and a right one differ).

### C3. Stochastic computation graphs give the effect lattice its missing structure (§11.5)

At review time Decision #5 admitted `EffectLattice` walked the Python source
AST and failed open. **W2.2 closed this on 2026-08-10:** registered Graph
effects and the concrete trace certificate now detect aliased/local/helper RNG,
while unresolved operations fail closed.

**Book result.** A stochastic program is a DAG with two node kinds —
**function nodes** and **distribution nodes** — where a node's output is a
random variable iff its random-parent set is non-empty. That is a trivially
derivable *forward dataflow* property on Graph IR: no AST, no dotted-name
matching, and it fails **closed** by construction (an unclassified source of
randomness cannot be silently absorbed into `pure`).

Two payoffs beyond fixing #5:

1. It gives `@jit(deterministic=True)` a real proof rather than a name-match.
2. It tells the compiler **which gradient estimator is legal** per node —
   all-function-nodes ⇒ pathwise / reparameterization; any distribution node ⇒
   score-function estimator. That is a compiler decision Tessera has no
   vocabulary for today.

Fits the PA W2 analysis layer directly, and is a Decision #30 ("derive, don't
ask") instance.

### C4. Semirings unify attention, scan, and the sequence-mixer track (§10.9)

Tessera already ships `logsumexp` as a first-class op with a `stable_reduction`
lowering (`op_catalog.py:274`) — which is the book's own argument for why
log-sum-exp deserves primitive status (§4.4.1). The missing generalization:
sum-product, max-plus (Viterbi), and log-sum-exp-plus are **the same algorithm
over three semirings**, and the book's "inference as backpropagation" result
(§10 "Inference as differentiation") gives the backward pass *for free* from
the forward one — backtracking is reverse-mode with soft backpointers.

Lands on the [`SEQUENCE_MIXER_ENGINEERING_PLAN.md`](SEQUENCE_MIXER_ENGINEERING_PLAN.md)
`linear_recurrence` normal form, on `associative_scan`
([`control.py:139`](../../../python/tessera/control.py)), and on the attention
family. A `semiring` attribute is a **semantic** key — Decision #21a requires
it to fail closed on absence.

### C5. Cost-weighted treeverse — do better than uniform Revolve (§4.6–4.7)

`AUTODIFF_ARCHITECTURE_REVIEW.md` §D5 proposes Revolve. The book notes the DP
form `C*(k,s) = min_l { C*(k−l, s−1) + C*(l, s) + l }` "could a priori
incorporate varying computational costs," which the closed-form Griewank scheme
cannot. Tessera **already has** per-device measured recompute costs in
[`rematerialization_cost.py`](../../../python/tessera/compiler/rematerialization_cost.py)
and `tessera.remat_cost_ns` in
[`ActivationRematerializationPass.cpp`](../../../src/transforms/lib/ActivationRematerializationPass.cpp).
So cost-weighted treeverse is a strictly better target than uniform-cost
Revolve at no extra research risk.

Two more §4.6.3–4.7 items D5 omits:

- **Online checkpointing** — required when the trip count is unknown (while
  loops), a hard prerequisite for the D4 control-flow adjoints.
- **Reversible layers** — optimal memory with *zero* recompute when `f⁻¹` is
  available. Belongs as a third candidate in the Decision #28 arbiter alongside
  store-all and recompute, not as a footnote.

### C6. Second-order structure beyond HVP (Ch. 8)

`hvp` is central finite differences
([`grad.py:120`](../../../python/tessera/autodiff/grad.py)); there is no
Gauss-Newton, Fisher, IHVP, or Hessian-diagonal anywhere in the tree. The
book's §8.1 complexity analysis confirms D2's premise (forward-on-reverse is
the preferred HVP) and adds three items D6 does not name:

- **GGN / Fisher** — `GN(ℓ∘f)[v] = ∂f* [∇²ℓ [∂f[v]]]`, 2 forward + 1 backward
  pass, PSD when `ℓ` is convex, and *equal* to the Fisher for exponential-family
  losses. The principled preconditioner for `optim.py`.
- **Block-diagonal / diagonal Hessian backprop** (§8.6–8.7), generalized from
  feedforward nets to arbitrary DAGs. One extra oracle per primitive beyond its
  VJP — i.e. a **13th contract axis** that composes with the existing registry.
- **Girard–Hutchinson / Bartlett estimators** (§8.8) — trace and diagonal from
  matvecs only, which is what a GPU wants when it cannot touch matrix entries.

---

## TSOL and helper libraries

### T1. The smoothing / relaxation family is absent (Ch. 4, 12, 13)

**Observed (2026-08-06), repo-wide excluding `archive/`:** no `sparsemax`, no
`entmax`, no `straight_through`, no soft-sort / soft-topk, no perturbed-optimizer
surface, and **`gumbel` is not among `rng.py`'s 12 samplers**. The one thing
present is scoped wrong: `tessera.arch.gumbel_softmax` / `arch.hard_concrete`
are declared in
[`TesseraOps.td:2507`](../../../src/compiler/ir/TesseraOps.td) over
`ArchParamType` (NAS logits, not tensors), and the Python side
([`arch.py:142`](../../../python/tessera/arch.py)) operates on
`Sequence[float]`. They appear in neither `op_catalog.py` nor
`primitive_coverage.py` — a reachability island of the PB-3 shape.

**Where it bites.** `vjp_top_k_routing` explicitly documents that "the
selection (argmax set) is treated as constant" (`vjp.py:933`). That is honest,
and it is the book's §4.3.2 result — the predicate's derivative is well-defined
and *uninformative*. MoE routing currently recovers the missing signal
indirectly via `z_loss` / `load_balance_loss`. A perturbed / Gumbel top-k gives
the direct path.

### T2. Fenchel-Young losses collapse a chunk of `losses.py` (Ch. 15 §4)

`losses.py` has 34 hand-written losses. The FY construction —
`L(θ,y) = Ω*(θ) + Ω(y) − ⟨θ,y⟩`, gradient exactly `ŷ(θ) − y` — generates
cross-entropy, sparsemax loss, structured / CRF losses, and perceptron loss
from one template, each with a closed-form *exact* gradient rather than a
hand-derived VJP. Fewer rules to maintain, and fewer to get wrong at the kink
(see C2).

### T3. Implicit differentiation — value-producing shared IR; physical consumption open (Ch. 10)

**Current split:** `custom_vjp` is exported from
[`custom.py`](../../../python/tessera/custom.py), and the Python reference lane
now provides `custom_root`, IHVP, and adjoint-state helpers in
[`implicit.py`](../../../python/tessera/autodiff/implicit.py). The compiler now
validates an explicit residual function and emits registered, value-producing
`residual` → matrix-free `linear_solve` → `residual_adjoint` VJP values plus an
optional JVP function. Architecture-owned lowering and execution of that chain
remain open. The book's §10.4 gives the recipe: the JVP solves
`A t = B v`, the VJP solves `A* r = u` then `B* r`, where
`A` / `B` are the JVPs of the residual `F` and `A*` / `B*` its VJPs — built
entirely from machinery already present, plus a matrix-free solver.
`solver_config.py` already names CG and GMRES. The shared AD wiring now exists;
physical matrix-free solver selection and execution are the remaining seam.

Downstream consumers, all already in-tree: the EBM / Langevin samplers in
`rng.py`, [`NewtonAutodiff.cpp`](../../../src/solvers/core/passes/NewtonAutodiff.cpp)
(which now emits the shared IFT value chain), the
[`RIEMANNIAN_OT_PLAN.md`](RIEMANNIAN_OT_PLAN.md), and any bilevel / hyperparameter
work. The same CG instantiation also yields IHVP — the missing piece for a real
Newton / natural-gradient path in `optim.py`.

Add alongside: **envelope theorems** (§10.2, Danskin / Bertsekas / Rockafellar)
— when the outer and inner objectives coincide you need only `max`
differentiation, not `argmax` differentiation. Naming that case stops callers
paying for implicit diff they do not need.

---

## Runtime / evaluator

### R1. Forward-reexecution guard; full Baur–Strassen accounting remains open (§4.4.3)

The theorem bounds `S(∇f) ≤ 5·S(f)`. The current guard counts forward
`tessera.ops` re-execution and catches the B1/B2 redundant-forward class. It
does **not** count raw-NumPy work performed inside backward VJPs, so it is not a
complete gradient-cost or Baur–Strassen conformance measurement. That complete
accounting belongs to AD-RESIDUAL-EVAL-1.

> **Correction learned while building this (2026-08-07).** The in-tree `jacrev`
> is **already fixed** — W0.4 rewrote it to record one forward pass and reuse
> the tape (`retain_graph=True`), so it passes the oracle at ratio ≈1. R1 is
> therefore a **regression guard** that the fix stays, plus a general detector
> for any primitive whose gradient path recomputes the forward — not a catch of
> a live B1 bug. The implemented counter measures forward-primitive
> *re-execution* specifically (backward VJPs are raw numpy, not `ops.*` calls),
> which is exactly the B1/B2 signature: a redundant-recompute Jacobian returns
> the *right* values expensively, so the numerical oracles stay green.

### R2. Randomized forward-mode gradient — a memory-free lane (§4.8)

`∇f(w) = E_Z[∂f(w)[Z] · Z]` — unbiased, no tape, no residual storage. High
variance, so not a default; but a real lane for the activation-memory-bound
regime, and it costs nothing once D2 lands. The book is explicit about the
variance/dimension trade-off, so budget it as an arbiter candidate (Decision
#28), not a replacement.

---

## Current route

| # | Item | State | Next compiler boundary |
|---|---|---|---|
| 1 | **C1** linear transposition | Compiler interface and paired CPU proof complete | `LinearTransposeInterface` owns the migrated Graph families; Python remains the oracle |
| 2 | TSOL spectral adjoints | Compiler Graph/Schedule/Tile slice complete; native compound-backward packages open | FFT/IFFT/RFFT/IRFFT/DCT have numerical compiler proof; x86/gfx1151 native package work stays architecture-owned |
| 3 | **C3** stochastic/effect typing plus `stop_gradient` | Compiler Graph/pass slice complete | C++ activity/effects and fail-closed regions are direct-tested; residual save policy remains separately owned |
| 4 | **T3** implicit differentiation | Python oracle, value-producing shared solver IR, and a bounded diagonal-sqrt AVX-512/gfx1151 physical pilot with compiled packets landed | Extend the same artifact path to general residuals and iterative/Krylov matrix-free solves; add Apple/NVIDIA consumers |
| 5 | **R1/C5** cost and residual policy | Measurement/selection boundary landed | Record exact family SAVE/RECOMPUTE/HYBRID packets; execute and measure region-adjoint treeverse schedules |
| 6 | **C4/C6/R2/T1/T2** breadth | reference or open | Bind separate integrated IDs only after the spine above is executable |

The global order and stop-the-line gates live in
[`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md); this table maps the
book findings onto that route and does not create another queue.

---

## Sources

- Blondel & Roulet, ["The Elements of Differentiable Programming"](https://arxiv.org/abs/2403.14606) (arXiv:2403.14606v4, 2024) — the reviewed text
- [`AUTODIFF_ARCHITECTURE_REVIEW.md`](AUTODIFF_ARCHITECTURE_REVIEW.md) — the primary autodiff review this extends (findings A1–B8, moves M1–M3, plan D1–D7)
- [`../MASTER_AUDIT.md`](../MASTER_AUDIT.md) — PA/PB governance program and Decisions #21a/#28/#29/#30
