---
last_updated: 2026-09-07
audit_role: reference
---

# Differentiable-programming status — August 2026

Historical record, not a current queue or execution claim. Current ownership is
[in the integrated plan](../INTEGRATED_COMPILER_LOG.md#2026-09-07--capability-plan-reconciliation). Mathematical contracts retained here
are not new device or performance evidence.

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
[`INTEGRATED_COMPILER_PLAN.md`](../INTEGRATED_COMPILER_PLAN.md); this table maps the
book findings onto that route and does not create another queue.

---
