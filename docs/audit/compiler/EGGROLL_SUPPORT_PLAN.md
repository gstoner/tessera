---
last_updated: 2026-08-09
audit_role: plan
plan_state: open
---

# EGGROLL / Evolution-Strategies Support Plan

> **Status:** engineering plan (gradient-free / evolution-strategies track).
> Extends Decision #28 (three-tier / measured arbiter) and Decision #23
> (standalone — no JAX/vLLM at runtime). `MASTER_AUDIT.md` + generated
> dashboards stay status truth (Decision #26); this plan is the build sequence
> and the design rationale, not a status claim.
>
> **Routing:** start at [`README.md`](README.md). This document owns the
> Evolution-Strategies track scope only; cross-plan sequencing (what lands next
> across the compiler) is owned by
> [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md), not here.
>
> **Provenance.** Derived from EGGROLL (*Evolution Strategies at the Hyperscale*,
> arXiv:2511.16652v2) + the reference implementations `ESHyperscale/HyperscaleES`
> and `ESHyperscale/nano-egg` (read as reference vocabulary only, Decision #23).
> Every algebraic and statistical claim below is **verified numerically** — see
> §Oracles. The verification harness and drop-in pytest fixture live at
> `scratchpad/eggroll_oracle.py` and `scratchpad/test_es_low_rank_correction.py`
> (target: `tests/unit/test_es_low_rank_correction.py`).

---

## 1. What EGGROLL is, and why it is a *compiler* problem

Evolution Strategies (ES) is a zeroth-order, gradient-free optimizer: perturb
the weights, evaluate a scalar *fitness* per population member, move the mean
toward high-fitness perturbations. Naïvely this materializes a full-rank
perturbation `E_i ∈ ℝ^{m×n}` per member, so evaluating the population is a
batched matmul with **arithmetic intensity < 1** — hopelessly memory-bound.

EGGROLL's contribution is an **arithmetic-intensity rewrite**: structure each
perturbation as low-rank `E_i = (1/√r) A_i B_iᵀ`, so the forward becomes

```
u(M + σE_i)ᵀ  =  u·Mᵀ  +  (σ/√r)·(u·B_i)·A_iᵀ
```

— one shared high-AI GEMM plus a cheap per-member low-rank correction
(batched-LoRA / SGMV). This is a *fusion/rewrite decision*, exactly what
Decision #28's arbiter owns. Two structural facts make it a good fit for Tessera:

- **It is gradient-free** → exercises only the inference path (no autodiff, no
  backward), and trains *at the inference dtype* — Tessera's int8/low-precision
  story (Decision #15a/#32).
- **It rides existing tracks** — RWKV/minGRU recurrence sits on the Sequence
  Mixer `linear_recurrence` op (Track L); its fitness normalization is GRPO's
  `normalize_group_advantages` (`rl.py:54`).

---

## 2. The single new op (proof-backed)

The whole algorithm reduces to **one** new Graph-IR node plus existing ops.
Per improvement **I1 (add, don't fuse)** the base `u·Mᵀ` stays a separate
`matmul`; per **I2** the update is `rng.normal → slice → scale → reshape →
matmul`. The only genuinely new node is the batched low-rank correction:

```
forward = tessera.matmul(x, M, transposeB)                    # existing, tensor-core
        + tessera.es_low_rank_correction(x, member_ids, key)  # NEW
```

### 2.1 `tessera.es_low_rank_correction` — ODS sketch

```tablegen
def Tessera_ESLowRankCorrectionOp : Op<Tessera_Dialect, "es_low_rank_correction",
  [Pure, DeclareOpInterfaceMethods<Tessera_AdjointInterface>]> {
  let summary = "Batched low-rank ES correction (σ/√r)·sign·(x·B)·Aᵀ";
  let arguments = (ins
    TensorType:$x,               // [P, ..., n]  population on leading axis
    TensorType:$member_ids,      // [P] i64      the chunk this worker owns
    TensorType:$key,             // RNGKey        MEMBER-keyed, NOT rank-keyed (G2)
    I64Attr:$out_dim,            // m  (n = x last dim)
    I64Attr:$rank,               // r  — required (bias is O(σ²/r), see §3)
    I64Attr:$epoch,
    F64Attr:$sigma,              // SEMANTIC, required — fails closed (G4)
    Tessera_ESScoreAttr:$score,  // {gaussian} — ONLY legal value (§3, dropped I4)
    DefaultValuedAttr<BoolAttr,"false">:$antithetic,
    Tessera_NumericPolicyAttr:$numeric_policy);   // accum REQUIRED f32/s32 (I6)
  let results = (outs TensorType:$result);        // [P, ..., m]
  let hasVerifier = 1;
}
```

Non-`DefaultValued` attrs (`sigma`, `rank`, `score`, `numeric_policy`) make
absence a parse error → **fail-closed**, enforcing G4 and Decision #21a.

### 2.2 Semantics (verifier + every emitter must preserve)

For population row `p`, `member = member_ids[p]`:
```
pair  = member >> 1
sign  = (member & 1) ? -1 : +1                         # antithetic (I5)
[B;A] = reshape(normal(key, epoch, pair, rank), (n+m, r))   # B=[:n], A=[n:]
out[p]= sign · (σ/√r) · (x[p] @ B) @ Aᵀ
```

### 2.3 The three contracts the proof forced out

1. **Member-keyed RNG (G2).** The perturbation stream is keyed by
   `(seed, epoch, pair)` and is **rank-invariant** — the opposite of Decision #18's
   per-rank device stream. Every worker reconstructing member `m` must obtain
   bit-identical `(A,B)`; the counter base carries **no rank term**. The two RNG
   axes coexist (rank-keyed device stream ⟂ member-keyed perturbation stream).
2. **`σ` is a `σ√d`-regime semantic (G4).** `‖σE‖_F ~ σ√(mn)`; consistency needs
   `σ√d = o(1)` (paper's `σ_d = o(d^{-1/2})`). `σ` fails closed and the verifier
   emits a diagnostic when `σ√(mn)` leaves the validated band.
3. **`accum = f32/s32` mandatory (I6 / Decision #32).** The update sums over a
   large population; bf16 accumulation loses the signal to cancellation
   (verified: >0.1% error, oracle A5).

### 2.4 The update needs no new op (Thm 2)

`ΔM = Σ_i f_i·sign_i·A_iB_iᵀ / √r / N` is, after scaling `A` rows by fitness, a
single GEMM with contraction dim `N·r` (`Ā B̄ᵀ`) — `rng.normal → slice → mul →
reshape → matmul`. A fused streaming variant (reconstruct `A` in tiles) is a
perf option, not new semantics.

---

## 3. Corrections the verification forced

Doing the proof numerically (not just on paper) changed three things:

| Item | Original claim | Corrected finding |
|---|---|---|
| **Theorem 3 (rank)** | aggregate rank `= min(Nr, m, n)` | Under antithetic pairing there are only `⌈N/2⌉` distinct directions ⇒ rank `= min(⌈N/2⌉·r, m, n)`. Verified (oracle A3). Update is still full-rank at scale ⇒ **G3 stands: no optimizer-state saving**. |
| **I4 (mean-field score)** | offer `{gaussian, meanfield_ggd}` as a semantic key | **Deleted.** Measured cosine-to-gradient at r=1: gaussian **+0.97**, mean-field **−0.04** (orthogonal). A nonlinear score destroys the rank-r structure that *carries* the gradient, and breaks the factorization (2048× memory). **Score linearity is a requirement, not a choice** — `score` has one legal value. |
| **G6 (int8 threshold)** | Gaussian quantile mis-calibrated on Bessel-tailed `E` | **Downgraded — not a gap.** The *aggregate* update sums over the population ⇒ CLT ⇒ Gaussian to KS=0.009 (even N=16); the Gaussian-quantile threshold hits the target update fraction dead-on (0.098–0.100 vs 0.10). Bessel tails only affect a *single* perturbation, never the update. |

Statistical facts confirmed: the low-rank bias is **`O(σ²/r)`** (not `O(1/r)`),
unbiased for linear/quadratic `f` at any rank, and empirically ≲ the Monte-Carlo
noise floor at r=1 (~1% of the gradient norm) — which is why rank-1 works.

---

## 4. Oracles (verify each op the way the audit split demands)

Exact identities are **unit** oracles; statistical equivalence is an
**expectation** oracle — conflating them (asserting the ES-limit as equality) is
the G1 mistake. Both families are implemented and green (17/17 in
`test_es_low_rank_correction.py`).

**Family A — EXACT (per-sample, any r):** A1 forward identity, A2 update
three-forms, A3 rank (antithetic-aware), A4 RNG member-keying, A5 accum-fp32.
Backend-gated like `verify_synthesized_gated`.

**Family B — STATISTICAL (expectation, seeded):** B1 moment match, B2
unbiased-on-linear, B3 bias-needs-`∇³f`, C1 moment-free convergence. Metamorphic
lane; tolerances `∝ 1/√samples`.

---

## 5. Workstreams

| WS | Scope | Gate |
|---|---|---|
| **W1 — reference tier (host-free)** ✅ **LANDED** | `python/tessera/stdlib/es.py`: `low_rank_perturbation`, `population_forward`, `es_update`, `fitness_shaping`/`centered_rank`/`antithetic_sign` (reusing `rl.normalize_group_advantages`, O6). Member-keyed via `rng.RNGKey.fold_in` (G2 by construction). Oracles: `tests/unit/test_es_reference.py` (16/16 green; A1–A5 exact + G4 fail-closed, B1–B3 statistical). mypy clean. | oracles green ✅ |
| **W2 — op + emitters** | `tessera.es_low_rank_correction` ODS + verifier + member-keyed RNG contract; Apple MSL emitter first (reference), then ROCm/CUDA where a hand SGMV/punica kernel is a Tier-3 arbiter candidate (Decision #28). Rank-1 bucket first (I3). | A-oracles on a real runner |
| **W3 — moment-free update path (G3)** | ES estimator → pseudo-gradient → existing `optim.py` (Adam), **and** a moment-free sign-threshold optimizer (the 14B path). C1 verified. | C1 + optax parity |
| **W4 — distributed (Phase G/H)** | coordinator-worker topology; scalar `all_gather` of fitnesses; member-keyed reconstruction on every worker. Base-3 ternary packing is optional (repo uses plain scalar all-gather). | multi-GPU box |

**Non-goals:** the pure-integer EGG stack (bit-shift, saturating requantize,
minGRU, L1-norm, LUTs) is a *separate optional demonstration track* — great for
the int8 lane but orthogonal to the core ES win. Its operator needs are catalogued
in §6 as cross-cutting improvements, not EGGROLL prerequisites.

**Cross-backend sync key:** `EGGROLL-ES-LOWRANK-2026-08-09` — the shared
`es_low_rank_correction` contract + fp32/s32 accumulation policy (I6) is recorded
in each backend queue (`docs/audit/backend/{apple,nvidia,rocm,x86}/todo.md`) with
its per-architecture outcome (follow-up required / not applicable), per AGENTS.md.

---

## 6. Operator-improvement catalog — where EGGROLL drives *overall* wins

EGGROLL support is a forcing function for improvements to **existing** Tessera
operators, most of which pay off far beyond ES. Ranked by cross-cutting value.

### P0 — cross-cutting, high value beyond EGGROLL

| # | Operator | Current state | Improvement | Also benefits |
|---|---|---|---|---|
| O1 | **`numeric_policy` carrier below Graph IR** | vanishes above the MMA; "no carrier below Graph IR" (Decision #29/#32 note) | thread storage/accum (int8→s32, bf16→f32) through the update op to codegen; boundary verifier fails on silent loss | **every quantized matmul** (int8/fp8/nvfp4), the entire quant lane — the #1 systemic win |
| O2 | **saturating requantize** (int32→int8, clip [-127,127]) | MISSING (`ops.cast` floats only) | first-class `saturating_requantize(x_i32, scale) → i8`; it is also EGG's nonlinearity (`clipped_add`) | all int8 inference / QAT-free quantization, nano-egg, RWKV int8 distill |
| O3 | **shared-operand batched GEMM** (`matmul`/`batched_gemm`) | `BatchedGemmOp` gates out broadcasting + transpose (`TesseraOps.td:198-201`) | a batched GEMM with one operand broadcast across the batch (population = N adapters, 1 base) | multi-LoRA serving (punica/SGMV), batched inference with shared weights, MoE |

### P1 — strong secondary wins

| # | Operator | Current state | Improvement | Also benefits |
|---|---|---|---|---|
| O4 | **bit-shift ops** (`shl`/`shr`) | MISSING (only bitwise and/or/xor/not) | elementwise `shift_left`/`shift_right` | fixed-point arithmetic, divide-by-pow2, any int quant codepath |
| O5 | **`tessera_rng` member-keyed derivation** | `RNGStreamAssign` is rank-keyed (Decision #18) | a rank-**invariant**, member-keyed `fold_in` stream (orthogonal axis) | reproducible-across-workers noise: dropout replay, shared-seed sampling, deterministic aug |
| O6 | **fitness/advantage shaping utility** | `normalize_group_advantages` is GRPO-specific (`rl.py:54`) | shared centered-rank / z-score / whitening utility for ES **and** RL | GRPO/PPO/CISPO + ES — converges RL and ES scoring |
| O7 | **`optim.py` pseudo-gradient + moment-free** | 9 gradient-consuming optimizers; `.step(grads=None)` reads `.grad` | (a) accept an externally-supplied estimator gradient; (b) add a moment-free sign-threshold update (G3) | memory-constrained training generally (sign-SGD/Lion-like), not just ES |

### P2 — enabling / track-aligned

| # | Operator | Current state | Improvement | Also benefits |
|---|---|---|---|---|
| O8 | **L1 / mean-abs norm** | RMS + mean-subtract only | mean-absolute norm (no sqrt) | int-friendly / low-precision normalization, EGG |
| O9 | **minGRU facet** | GRU/LSTM only; no minGRU | minGRU on the `linear_recurrence` op (Track L) | recurrent-model family, EGG, RWKV |
| O10 | **`exp2`/`log2` / first-class LUT** | gather/embedding as LUT vehicle only | `exp2`/`log2` ops or a first-class LUT op | int softmax, fixed-point activations, EGG fitness |
| O11 | **scalar / sub-byte collective** | tensor-sized collectives only | scalar `all_gather`; optional base-3 ternary packing | any small-scalar reduction (metrics, losses, fitness), gradient-free distributed |
| O12 | **`GroupedGemmOp` contract reuse** | MoE-ragged only (`TesseraOps.td:223`) | reuse grouped-layout + `numeric_policy` + `scale_layout` for the population axis | unifies MoE expert-routing and ES population-routing under one contract |

**Decision #23 hygiene.** vLLM `WorkerExtension`, Optax, and JAX PRNG in the
reference repos are **vocabulary only** — reimplement multi-LoRA population
serving on `runtime.launch()`; do not wrap vLLM.

---

## 7. Coverage axes (Decision #24/#29) for the new op

`op_catalog._SPECS` += `OpSpec("es_low_rank_correction", ...)`; a
`PrimitiveCoverage` entry over all 12 axes. Notable: **`vjp`/`jvp` = terminal
`non_differentiable`** (ES is zeroth-order — the op has no adjoint; a clean use
of the terminal status, and `_VJPS` auto-flip correctly does not fire). Every
attr has a named consumer (Decision #29): `score`→sampler+emitter,
`member_ids`/`key`→RNG derivation, `rank`→factor shape, `numeric_policy`→accum.

---

## 8. Suggested `MASTER_AUDIT.md` queue row

> **[P1] Gradient-free / Evolution-Strategies track (EGGROLL).** New primitive
> `es_low_rank_correction` + reference tier + moment-free optimizer. Op contract
> proof-backed (`EGGROLL_SUPPORT_PLAN.md`); oracles green. Drives P0 operator
> wins O1 (numeric_policy carrier), O2 (saturating requantize), O3 (shared-operand
> batched GEMM). W1 host-free first; W4 distributed is Phase G/H.
