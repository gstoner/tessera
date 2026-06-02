---
status: Normative (live specification)
classification: Spec
authority: Defines the EBM (Energy-Based Models) primitive surface across all four Tessera IR levels
last_updated: 2026-05-16
provenance: Adapted from `archive/examples/advanced/EBT/Tessera_EBT_Package_v1/docs/EBT_in_Tessera.md` (archived)
---

# Tessera EBM Specification

This document is the normative spec for Tessera's **Energy-Based Model**
primitive surface. It covers the energy primitive contract, the
inner-loop / self-verify schedule pattern, training losses, and the
mapping to all four Tessera IR levels.

The spec is **broader than EBT**: it covers Restricted Boltzmann
Machines (RBMs), Energy-Based Transformers (EBTs), score-matching
diffusion, and the geometric-Langevin demo (EBM7). The `tessera.ebm`
namespace is the home for all of them.

Provenance: adapted from the archived `Tessera_EBT_Package_v1` design
([EBT_in_Tessera.md](../../archive/examples/advanced/EBT/Tessera_EBT_Package_v1/docs/EBT_in_Tessera.md)),
with the EBM0 revisions documented in [`docs/audit/domain/DOMAIN_AUDIT.md`](../audit/domain/DOMAIN_AUDIT.md):
functional state, explicit `RNGKey`, broader namespace, MLP energy head.

## 1. Concept

An **energy-based model** learns a scalar `E(x, y; θ)` that scores
candidate outputs `y` given input `x` (or just `E(y; θ)` for
unconditional models). Three operational modes:

- **Inference by minimization** — `y* = argmin_y E(x, y)`. Run an
  iterative inner loop (gradient descent / Langevin / coordinate
  updates) starting from an initial `y₀`.
- **Inference by sampling** — draw `y ~ p(y|x) ∝ exp(-E(x, y))` via
  Langevin / MALA / HMC / Gibbs.
- **Self-verify** — generate `K` candidate trajectories, evaluate the
  final energy of each, return `argmin_k E(x, y_K^k)`.

Training objectives (covered in EBM4): contrastive divergence (CD),
persistent CD, score matching, denoising score matching, noise-
contrastive estimation (NCE).

## 2. Primitive contract

The EBM1 surface ships five primitives, all pure functions:

### 2.1 `tessera.ebm.energy(model_fn, x, y) -> Tensor`

Calls a user-provided `model_fn(x, y) -> scalar | per-token tensor`.
Energy is the scalar (or per-token reduction target) — never a softmax,
never normalized. The user owns the energy head; Tessera owns the
inner loop, sampling, and losses around it.

Contract:
- `model_fn` is any callable `model_fn(x, y, *, params) -> Tensor`
  where the output is reducible to a scalar per (batch, candidate).
- Returns shape `(B,)` or `(B, K)` depending on whether candidates are
  explicit.
- Differentiable in `y` (used by `langevin_step` and `inner_step`).
- Differentiable in `params` (used by training losses).

### 2.2 `tessera.ebm.inner_step(y, grad, eta, *, rng_key=None, noise_scale=0.0) -> y'`

Pluggable inner-loop update. Defaults to gradient descent
(`y' = y - eta * grad`). When `rng_key` + `noise_scale > 0`, adds
Gaussian noise — this is the Langevin special case.

Contract:
- Pure function. No mutation of `y`.
- Returns the same shape / dtype / device as `y`.
- When `rng_key` is provided, the function is deterministic in the
  key per S4 conventions.

### 2.3 `tessera.ebm.langevin_step(y, energy_fn, eta, temperature, rng_key) -> (y', rng_key')`

Convenience composition of `tessera.autodiff.vjp` (to compute
`∂E/∂y`) plus `inner_step` with the Langevin noise scale
`sqrt(2 * eta * temperature)`. Returns the consumed key.

Contract:
- Always consumes one RNG key.
- `temperature=0` collapses to pure gradient descent.
- Differentiable through the energy gradient (for higher-order training).

### 2.4 `tessera.ebm.self_verify(energies, candidates, *, beta=None) -> y_best`

Reduce K candidates to the lowest-energy one. When `beta is None`,
hard `argmin`. When `beta` is a positive scalar, soft-min
(softmax over `-beta * energies` then weighted sum of candidates) —
this is differentiable for end-to-end training.

Contract:
- `energies.shape == (B, K)`, `candidates.shape == (B, K, ...)`.
- Returns `(B, ...)`.
- `beta=None` is non-differentiable in `candidates` (just selection);
  `beta` positive makes it fully differentiable.

### 2.5 `tessera.ebm.decode_init(x, *, K, init_strategy="noise") -> y0`

Initialize K candidate trajectories. Strategies: `"noise"` (Gaussian
from RNG), `"base_model"` (call a user-provided initializer),
`"copy"` (broadcast `x` itself when shapes match).

Contract:
- Pure; takes an explicit `RNGKey` for `"noise"`.
- Returns shape `(B, K, ...)` matching the user-declared output shape.

## 3. Mapping to Tessera IR

| Level | What lives here |
|---|---|
| **Graph IR** | `ebm.energy`, `ebm.inner_step`, `ebm.langevin_step`, `ebm.self_verify`, `ebm.decode_init`, `ebm.partition_z`. New dialect lands in EBM5. |
| **Schedule IR** | The K-candidates × T-steps loop nest. Inner loop is a `tessera.control.scan` (S5), not a hand-rolled `scf.for`. Candidates map across streams / devices via existing parallelization policies. |
| **Tile IR** | Energy-head kernels (MLP or attention-conditioned bilinear). `grad_y` reuses backward paths of the underlying ops. Sampler kernels (Langevin / MALA / HMC) are vectorized; bf16 storage, fp32 accumulators (Decision #15a). |
| **Target IR** | Each backend follows Decision #19's hardware-free layer (per Q4 lock, x86 → Apple CPU → Apple GPU → NVIDIA after Phase G). |

### 3.1 Canonical Schedule IR

```mlir
// Inference: K candidates, T inner steps.
%y0 = ebm.decode_init %x, %K, %rng_key
%E_all, %Y_final = tessera.control.scan {
  iters = %K,
  body = ^bb(%k_idx, %y_k):
    %y_T = tessera.control.scan {
      iters = %T,
      body = ^bb_inner(%t_idx, %y_t):
        %grad = ebm.grad_y %x, %y_t
        %y_next = ebm.inner_step %y_t, %grad, %eta
        tessera.control.scan.yield %y_next
    } init = %y_k
    %E_k = ebm.energy %x, %y_T
    tessera.control.scan.yield (%E_k, %y_T)
} init = (init_E, init_Y)
%y_best = ebm.self_verify %E_all, %Y_final
```

### 3.2 Training-time variant

For training, energies and candidates feed into a loss
(EBM4 — CD / score matching / NCE) instead of `self_verify`.
The gradient flows back through `params` via the existing
`tessera.autodiff` machinery; multivector-aware autodiff (GA6)
is opt-in for geometric energies.

## 4. Pass pipelines (EBM5 + EBM6)

- **`-ebm-canonicalize`** (EBM5):
  - Inline `decode_init` when the strategy is `"copy"` or simple noise.
  - Normalize `self_verify` to a min-reduce + soft-min in the `beta`
    branch.
  - Hoist loop-invariants (context `x`) and rematerialize light ops.

- **`-ebm-lower`** (EBM5):
  - Graph → Schedule: materialize the K × T scan nest.
  - Schedule → Tile: tile the energy-head MLP / attention; generate
    `grad_y` kernels.
  - Tile → Target: dispatch via existing backend pipelines.

- **`-ebm-fuse-energy-grad`** (EBM6): fuse `energy` evaluation with
  its `grad_y` to reuse activations; target ≥ 30% memory-traffic
  reduction.

- **`-ebm-checkpoint-inner-loop`** (EBM6): apply Phase F2 rematerialization
  to the T-step inner-loop trajectory; tunable budget.

- **`-ebm-pipeline-candidates`** (EBM6): map K across streams / devices
  for `self_verify`.

## 5. Training losses (EBM4 — in `tessera.losses`)

All four register VJP + JVP per S11 conventions.

- `contrastive_divergence_loss(energy_fn, x, y_pos, n_steps, rng_key)` —
  k-step CD: alternate Gibbs / Langevin to produce `y_neg`, return
  `E(x, y_pos) - E(x, y_neg)`.
- `persistent_cd_loss(energy_fn, x, y_pos, persistent_state, rng_key)` —
  PCD; carries the negative-sample chain across batches as part of
  module state (S3 — `STATE_COLLECTION_SPECS["memory_state"]`).
- `score_matching_loss(energy_fn, x, y)` — `½‖∇_y E‖² + tr(∇²_y E)`.
  Hutchinson trace estimator for the Hessian.
- `denoising_score_matching_loss(energy_fn, x, y, sigma, rng_key)` —
  Vincent (2011); noisy-y target with a closed-form denoiser objective.

## 6. Runtime expectations

EBM1 ships pure-Python reference implementations on numpy arrays —
same shape as the S2–S15 reference surface. EBM5 brings the IR
dialect; EBM6 brings fusion; backend lowering follows Q4.

The archived runner ([`ebt_runner.h`](../../archive/examples/advanced/EBT/Tessera_EBT_Package_v1/models/ebt/runtime/ebt_runner.h))
is a useful reference for the C++ runtime shim signatures that will
land alongside EBM5. The functional revisions: the live shim takes
an explicit `RNGKey` and returns the consumed key, no mutable state.

## 7. Open choices (deferred to later sprints)

- Energy-head implementation pattern (bilinear vs. MLP vs. attention-
  conditioned): all are valid; the user owns `model_fn`. Tessera
  ships example heads in EBM8 conformance.
- Manifold-aware variants (EBM7): Q5 (manifold set) is deferred.
- Persistent-state lifecycle: how PCD chains compose with checkpoint /
  resume (S12). Will be resolved during EBM4.

## 8. Out of scope

- Symbolic computation of partition functions for non-trivial models —
  EBM3 ships exact (small discrete), Monte Carlo, and AIS only.
- Non-EBM "energy-like" architectures (e.g., Hopfield networks) — those
  belong in `tessera.nn`, not `tessera.ebm`.
- Wrapping any external EBM library — per Decision #23.

---

## Appendix A — Conformance demos (EBM8)

EBM8 ships three tiny standalone models proving the stack:

1. **RBM on MNIST-tiny** — classical contrastive divergence training;
   proves the EBM4 loss + sampler integration.
2. **EBT-tiny** — encoder + inner-loop "thinking" + self-verify on a
   tiny seq2seq task; resurrects the archived EBT pattern as a live
   conformance model.
3. **SO(3) score-matching diffusion** — orientation diffusion using
   EBM7's geometric Langevin; proves the GA + EBM stack composition.

## Appendix B — References

- EBT paper: <https://arxiv.org/abs/2507.02092>
- EBT project page: <https://energy-based-transformers.github.io/>
- LeCun et al., "A Tutorial on Energy-Based Learning" (2006)
- Vincent (2011) — "A Connection Between Score Matching and Denoising
  Autoencoders"
- Du & Mordatch (2019) — "Implicit Generation and Modeling with
  Energy-Based Models"
