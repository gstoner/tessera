---
last_updated: 2026-08-15
audit_role: plan
plan_state: open
---

# FORGE assessment — fusing the optimizer into the weight-gradient epilogue

**Date:** 2026-08-15 · **Status:** external-paper assessment + proposed
workstreams (direction, not status truth — Decision #26).
[`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md) is the sole
cross-domain compiler queue and owns global order; nothing here reprioritizes
it. Read [`README.md`](README.md) for the authority chain before using any
section below as a work queue. · **Charter:** Decision #28's three-tier/arbiter
model ([`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md)),
the `numeric_policy` carrier gap named in Decision #32, and the
`TilingInterface`-without-a-consumer gap in
[`TilingInterface_NOTES.md`](../../../src/compiler/ir/TilingInterface_NOTES.md).

**Subject:** FORGE — *Fused On-Register Gradient Elimination for Memory-Efficient
LLM Training*, arXiv:2606.22932v2, code at
[dk4248/FORGE](https://github.com/dk4248/FORGE).

---

## 0. What FORGE is, and how much of it we can trust

FORGE fuses the optimizer's state update into the weight-gradient GEMM's
epilogue, one register tile at a time. Each tile of `∇_W L = ΔYᵀX` is
accumulated in fp32 registers, consumed by AdamW, and discarded; the full
`grad_W` tensor never reaches HBM. On Llama-3.1-8B/H200 they report peak memory
62.04 → 48.36 GB and step time 167.1 → 110.2 ms.

**Claim hygiene.** The measurement discipline is above average: ceilings measured
on-device rather than from datasheets, three runs × median-of-20, per-cell
clock/power/temperature/source-hash records, arms matched within one precision
recipe, a 20-step fp32-parity gate before any cell is recorded, OOM treated as a
capability boundary, and implementation costs charged *against* FORGE. They
report where it loses (FSDP2 parity-and-slower, PCIe composed 2–2.6× slower, no
saving at `BT ≥ 4096`).

**But the artifact does not back the paper.** The public repo (created
2026-08-11) ships **AdamW and SGD only**. There is no distributed code at all —
zero occurrences of `torch.distributed`, `all_reduce`, or a bucket coordinator —
and none of Muon/LAMB/Lion/RMSprop/Adafactor/SM3/Adam-mini. So §4 plus
Appendices K–N (two 8-GPU nodes) and Appendix I (thirteen families) are not
reproducible from the release. `CONTRIBUTING.md` states "Cross-element
preconditioners (Muon, Shampoo) are out of scope for the fused path," which sits
badly against §2's regime-2 Muon result. What *is* released matches the paper's
description exactly and is verifiable: fp32 register accumulator, correct
decoupled-AdamW epilogue with bias correction, and `grad_input = grad_output @
weight` computed before the in-place update.

By our own standards: the single-GPU mechanism is an evidence row; the
distributed and multi-optimizer results are not.

---

## 1. Mathematical verification

Re-derived and checked numerically. Harness:
[`tests/unit/test_fused_wgrad_optimizer_contract.py`](../../../tests/unit/test_fused_wgrad_optimizer_contract.py).

> **Harness determinism pin (2026-08-15).** The P1 bitwise assertions
> originally computed chunk partials with a BLAS GEMM, whose per-element
> reduction order is a shape- and hardware-dependent kernel choice — the
> full-matrix and tile GEMMs legitimately diverged in ULPs on GitHub's
> heterogeneous runners (observed intermittently on PR #566; green on rerun).
> The harness now accumulates per-token outer products so every element has an
> identical sequential fp32 chain on both sides, on any hardware. The
> *contract* is unchanged; the pin is exactly the accumulation-order
> discipline the fused epilogue itself must declare (W2/W3).
> Cross-referenced by the substrate view, which folds this assessment as the
> seventh paper ([`CORE_SUBSTRATE_VIEW.md`](CORE_SUBSTRATE_VIEW.md) S9).

### 1.1 What holds

| Claim | Result |
|---|---|
| Prop. 1 per-tile exactness | **Bit-identical** to one-shot AdamW on `(W, m, v)`, and independent of tile partition |
| Prop. 2(ii) data-parallel obstruction | Confirmed — stepping on per-rank partials errs 0.70 relative in `v` |
| Prop. 2(iii) affine reduce-into-state | Exact to fp32 reassociation (1.6e-7) |
| App. B int8 state bounds | max `|e|` 2.56 vs bound `s/(2(1−β₁))` = 5.0; RMS 0.671 vs predicted `s/√(12(1−β₂²))` = 0.662 |
| §2 bandwidth 16 → 12 B/param | Self-consistent for bf16 weights + bf16 moments; bound 1.333× |
| Eq. 1 memory accounting | 15.0 GB deleted at 8.03 B params; matches Table 1 within activations |

The transferable idea is **Def. 1/2 — the state/weight (`A`/`U`) decomposition**.
Writing a step as `S_t = A(S_{t−1}, G)` then `W_t = U(W_{t−1}, S_t)` and
observing that only `A` is schedule-constrained (because `U` reads resident
state) is strictly stronger than the usual "coordinate-wise optimizer" framing.
It is what admits Muon (Newton–Schulz reads `B`, never `G`) and LAMB (norms of
`W` and of the update, never of `G`). Prop. 2(iii) — seed `𝒜(S)` on one rank
and `0` on the others so a plain all-reduce of the *state* reconstructs the
exact update — is the second good idea and is reusable independent of tiling.

An internal-consistency check that *passes*: Qwen3-32B int8 (137.0 GB) vs fp8
(134.4 GB) differ by ≈2.6 GB, which is exactly the per-64-block fp32 absmax
scales int8 carries and per-tensor fp8 does not. The numbers are measured.

### 1.2 Where it overreaches

1. **"The error does not average away" (§Precision) is wrong as stated.** The
   appendix hedges correctly — a *systematic* ε passes through undamped — but
   bf16 round-to-nearest-even round-off is zero-mean (measured mean/rms = −0.003
   on a real weight gradient), and for zero-mean ε the first moment damps it by
   `√((1−β₁)/(1+β₁))` = 0.229. Measured: 1.0000 systematic, 0.2314 zero-mean.

2. **Global-norm gradient clipping is the missing limitation.** `‖G‖₂` across all
   parameters is a cross-layer gradient statistic — the paper's own regime 4,
   `O(P)` residency, entire saving gone. A commented-out §2 draft says clipping
   is "treated in §Limitations"; the shipped §Limitations does not mention it.
   What survives is two appendix protocol lines: "global-norm clipping off on
   every arm" (App. F) and "no clipping" (App. K). Every mainline LLM recipe
   clips at 1.0, so convergence parity is established in a configuration that is
   not the standard recipe. See §2.2 — there is no safe workaround.

3. **App. I contradicts its own table.** "Both arms hold fp32 optimizer state
   throughout this appendix," yet the FORGE AdamW cell reads 48.8 GB, ≈identical
   to the 48.36 GB bf16-state headline. fp32 `m`+`v` at 8 B params is ~64 GB of
   state alone. The table's own per-state-tensor delta (SGD 20.9 → 1-state 34.9
   → AdamW 48.8, ≈14 GB ≈ 2 B/param) says bf16. Sentence or numbers is wrong.

4. **"No exact scheme can materialize less than the reduction granule"** is not a
   lower bound — bucket size is an implementation choice ("matched to the
   framework's"). The genuine data-parallel floor is a tile. Does not affect any
   measurement.

### 1.3 The finding the paper's own table obscures

Isolating gradient rounding against an fp64 reference, 20 steps:

```
fp32 master W, fp32 states    standard 2.970e-04   FORGE 3.254e-07    913x
fp32 master W, bf16 states    standard 8.053e-04   FORGE 7.508e-04      1.1x
bf16 W, bf16 states           standard 7.371e-03   FORGE 7.380e-03      1.0x   <- paper's recipe
```

bf16 **state** rounding swamps the gradient rounding entirely. The paper measures
its own precision claim inside a recipe that masks it and reports 4.4%. Its own
fidelity table corroborates this: the int8 and fp8 state rows are 4× and 44×
*worse* than baseline — so in the memory-saving configurations FORGE actually
advertises, the precision argument inverts.

This is a compiler-shaped fact. Whether the fusion's numerical benefit is
realizable is a function of `numeric_policy.accum` × state dtype. A compiler can
decide and report that; a hand-written kernel cannot. It is the strongest single
argument for closing the Decision #32 carrier gap, and it makes Decision #15a
("storage dtype on the tensor, accumulator in `numeric_policy`") load-bearing
rather than a naming convention.

---

## 2. Verified propositions for the plan

Each row is a proposition the workstreams in §4 depend on, with its verdict.
Tests carry the same identifiers.

| # | Proposition | Verdict |
|---|---|---|
| P1 | `producer = matmul ∧ consumer = optimizer ∧ single-use` subsumes FORGE Rem. 2 (tied weights) | **Holds** |
| P2 | Exact global-norm clipping can be fused | **False, provably** |
| P2a | "Clip the update" (scalar into `U`, regime 2) substitutes for it | **No** |
| P2b | Delayed (previous-step) norm substitutes for it | **No** |
| P2c | Clipping is exact under SGD | **Holds** |
| P3 | Prop. 2(iii) composes with ZeRO-2 reduce-scatter (Decision #16) | **Holds** |
| P4 | Micro-batch accumulation into momentum removes the accumulator | **Affine `A` only** |
| P5 | Folding state decay into the GEMM's `β` is safe | **False for MoE** |
| P6 | The fp32-accumulator precision benefit is realizable | **Conditional** — see §1.3 |

### 2.1 P1 — the legality predicate is structural

A tied weight's `%g1` is consumed by an `add`, not by `adamw`, so a
`producer = matmul, consumer = optimizer, one-use` match cannot fire. Fusing
anyway errs 2.3e-2 relative. No aliasing analysis is required.

Tessera has a structural advantage the paper's PyTorch implementation does not:
optimizer ops are **value-semantic and `Pure`**
([`TesseraOps.td:2571`](../../../src/compiler/ir/TesseraOps.td)), so `%W` and
`%W'` are distinct SSA values. FORGE's ΔX-before-update hazard — which its repo
guards with a comment and a correctness script — **cannot occur by
construction** at Graph IR. It reduces to a scheduling/bufferization concern.

### 2.2 P2 — clipping has no safe workaround

`m_t(c) − c·m_t(1) = (1−c)β₁m_{t−1}`. Recovering the clipped state from the
unclipped one needs `G` or `m_{t−1}`, both `O(P)`. Exact fusion is impossible.

Both candidate approximations fail, measured on a single 50× gradient spike:

| policy | max\|ΔW\| on the spike | vs exact | √v̄ ten steps later | vs exact |
|---|---|---|---|---|
| exact clip | 8.94e-4 | 1.00× | 5.17e-1 | 1.00× |
| clip-the-update | 4.03e-5 | 0.05× | 1.645 | **3.18×** |
| delayed (prev-step) | 1.74e-3 | 1.95× | 1.644 | **3.18×** |
| no clipping | 1.74e-3 | 1.95× | 1.645 | 3.18× |

`clip-the-update` bounds the displacement (Adam is approximately scale-invariant,
so it already was) but lets the spike into `v`, where the damage persists
`~1/(1−β₂)` steps — *identical to no clipping*. `delayed` does not even bound the
arriving step. Under SGD, clipping `G` and scaling `lr` are the same operation
(7.5e-9), which is the exception that names the cause: the obstruction is `A`'s
nonlinearity, not the fusion.

**Consequence for the design:** `grad_clip_scope` must be a semantic key that
**fails closed** (Decision #21a). `global` is incompatible with the fusion and
the verifier rejects it; `layer` is regime 3 (residency falls to the largest
layer); `none` fuses. Silently dropping clipping — which is what a user gets
today if they adopt FORGE — is the failure this prevents.

### 2.3 P5 — the MoE hazard

Folding state decay into the GEMM's `β` applies `α` only where a contribution
arrives (FORGE Rem. 1). After 200 steps at 50% routing, `β`-folded `m` is
**2.24×** the correct value: an unrouted expert's state stops decaying while
every other expert's decays. Tessera has `tessera.moe_swiglu_block` and MoE
routing in `distributed/moe.py`, so this must be a verifier condition, not a
comment.

---

## 3. What Tessera already has

Four things checked in tree, three of which are further along than a first pass
suggests.

| Piece | State |
|---|---|
| **The fusion target** | `Tessera_MatmulOp` implements `LinearTransposeInterface`; `MatmulOp::buildLinearTranspose` emits `dRhs = matmul(lhs, dy, transposeA=true)` = `XᵀΔY` — the weight-gradient GEMM — consumed by `AutodiffPass`, with a lit fixture asserting `transposeA = true` ([`autodiff_paired_matmul.mlir`](../../../tests/tessera-ir/phase2_autodiff/autodiff_paired_matmul.mlir)). It exists today. `MatmulOp` does **not** implement `AdjointInterface`; the route is the transpose interface. |
| **The fusion pattern** | [`TrainingStepFusionPass.cpp`](../../../src/transforms/lib/TrainingStepFusionPass.cpp) already implements the identical shape one level down: match `loss_backward → sgd/adamw`, check `hasOneUse()`, rewrite to `tessera.training.loss_adamw`. Registered, in `addGraphIRPreLoweringPasses`, with a **negative fixture** (`@shared_gradient`) and a `LOWER-COUNT-1: linalg.generic` check that statically proves the intermediate is never materialized. |
| **The legality condition** | `MatmulOp::getLoopIteratorTypes()` → `{parallel, parallel}` over (M, N) with the K reduction kept whole inside the op, stamped `tessera.full_k`. That *is* FORGE Prop. 1's condition. `TilingInterface_NOTES.md` §"Why no urgent consumer" records that nothing drives it — a live Decision #29 issue this work closes. |
| **The numeric carrier** | `numeric_policy` is an ODS attribute on `MatmulOp` itself, with consumers below Graph IR (`DtypeLegalizePass` stamps `accum`, `IRContractLegalityPass` verifies storage/accum coupling). `MatmulLowering` already does "bf16 accumulate in f32 then `truncf`" — **that `truncf` is exactly the bf16 gradient rounding FORGE deletes**, in the linalg lane, testable on CPU. |

The `LOWER-COUNT-1` idiom is the important one: Tessera can assert FORGE's
central claim as a **compiler property on any host**, where the paper can only
report `torch.cuda.max_memory_allocated()` on a 141 GB H200. No machine in the
fleet needs to hold an 8 B model for this work to be gated.

---

## 4. Proposed workstreams

Do not port FORGE. FORGE is one instance of a class Tessera is equipped to
generalize:

> **Fuse a consumer into its producer's tiled epilogue when the consumer's
> read-locality is no coarser than the producer's tile partition, and prove
> statically that the intermediate never materializes.**

FORGE does one consumer × one producer × one backend and can only *measure* the
outcome. The eight items below give N×M, declared legality, static proof,
cross-backend admission via the Decision #28 arbiter, and a compiler that reports
when the win is not there.

### W1 — Locality lattice (P0, foundational)

Generalize the `A`/`U` decomposition into declared operand metadata:

```
coordinate ⊏ row/column ⊏ block ⊏ tensor ⊏ layer ⊏ global
```

Fusion into a tiled producer is legal iff
`consumer.read_locality(operand) ⊑ producer.tile_partition`. This subsumes
FORGE's four regimes as lattice positions and generalizes past optimizers. On the
optimizer ops, split it FORGE's way: `state_locality` on `A`,
`weight_update_locality` on `U`. Both are semantic keys — absence is a
diagnostic, never a default to `coordinate` (Decision #21a). Consumer: W3. Drift
gate: `test_governance_declarations.py` (Decision #29).

### W2 — Residency contract + static materialization proof (P0)

Declare `tessera.residency ∈ {tile, layer, full}` on a value; a boundary verifier
fails if the lowering materializes above it. The `LOWER-COUNT-1: linalg.generic`
idiom already in tree is the prototype — generalize it into a pass so every
fusion ships its own residency proof.

Highest-leverage item: it makes the whole class **host-free testable**, which is
Decision #19's discipline applied to a memory property.

### W3 — `matmul → optimizer` fusion (P0, the FORGE instance)

```mlir
%gW = tessera.matmul(%X, %dY) {transposeA = true, numeric_policy = {accum = "fp32"}}
%W', %m', %v' = tessera.adamw(%W, %gW, %m, %v)
  ⟹  tessera.training.wgrad_adamw(%X, %dY, %W, %m, %v)
```

Guards, all structural: `%gW` single-use; consumer is an optimizer op with
`state_locality = coordinate`; `full_k` intact (the token axis is not tiled).
Negative fixtures required — tied weight, `state_locality = row` (Adafactor),
MoE-routed weight (W4). The lowering removes the `truncf` in `MatmulLowering`,
which is the fp32-accumulator win in the linalg lane, provable on CPU.

**Decision #31 risk to fold into acceptance, not defer:** W3 creates a second
lowering for `matmul → optimizer`. Per the ordering caveat in
[`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md), the fused path must
carry everything the unfused path carries — `numeric_policy`, `distribution`,
`layout` — *before* anything is collapsed.

### W4 — Semantic keys that fail closed (P0)

- **`grad_clip_scope ∈ {global, layer, tile, none}`** — per P2, `global` is
  incompatible and the verifier rejects it rather than approximating. A
  correctness feature the paper does not have and PyTorch cannot express: today a
  user silently gets no clipping.
- **`routing ∈ {dense, conditional}`** on any fused weight — per P5, a
  conditionally-routed weight needs a separate gradient-free decay pass or its
  state stops decaying. 2.24× error, silent.
- **`accum_dtype`** carried to the epilogue, or the drop named (Decision #32).

### W5 — Precision-realizability oracle (P1, novel)

Per §1.3, report at compile time whether the fusion's numerical benefit is
realizable given `numeric_policy.accum` × state dtype, as a diagnostic:
*"wgrad fusion removes bf16 gradient rounding, but `state_dtype = bf16` masks it;
expected weight-error improvement ≈1.1×."* This stops a developer from believing
the paper's 4.4% applies to their int8 run, where the fidelity is 4× worse.

### W6 — Distribution: affine reduce-into-state (P1, standalone)

Prop. 2(iii), verified at P3 including the ZeRO-2 reduce-scatter form. For affine
`A` — SGD, heavy-ball, Muon's `B ← μB + G` — reduce the *momentum*, seeding
`𝒜(S)` on the owner rank and `0` elsewhere. Same wire volume, no staging buffer,
no rank ever forms a gradient. Lands in `GPUCollectiveInsertionPass` /
`OptimizerShardPass`; needs no fused kernel. P4 gives the same trick across
*time* for micro-batch accumulation.

### W7 — Tier-2 epilogue seam (P1, Decision #28)

The arbiter constraint is that the fused epilogue must not cap ROCm/CUDA. The
accumulator is already in registers after `mma.sync`/WMMA/`simdgroup_matrix`, so
the epilogue is natural; tile geometry is arch-specific (register file on
NVIDIA/AMD, threadgroup budget on Apple), which is what the autotuner exists to
pick. `FusedRegion`
([`fusion_core.py:226`](../../../python/tessera/compiler/fusion_core.py)) needs a
new region class: **stateful, multi-output, side-effecting epilogue** — extra
(M, N) operands read at tile granularity, multiple in-place stores, discarded
`D`. The existing `residual` field is the precedent for the operand; the
multi-store and the discard are new. Arbiter admission is trivial: the candidate
is *exact*, so any accuracy budget is satisfied.

### W8 — The other consumers (P2, where the richness is)

Same machinery, different pairs:

| producer | consumer | what collapses |
|---|---|---|
| logits GEMM | loss VJP | LM-head logits (128 k vocab) — **larger than the gradient pool** in real LLMs; partially shipped as `training.loss_adamw` |
| wgrad GEMM | optimizer | FORGE (W3) |
| wgrad GEMM | norm reduction | enables an exact two-pass clipping policy for W4 |
| wgrad GEMM | all-reduce bucket | FORGE's bucket-transient schedule |
| any GEMM | quantize/dequantize | `DequantMatmulOp` already exists |
| optimizer | EMA / weight averaging | the EMA copy |

### Build order and gates

**P0** — W1 → W2 → W3 → W4, in that order (W2 before W3, or W3 has no gate). All
pure IR/pass work, all host-free, all routed to the Strix Halo box per
[`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md) §6a. Gate: lit
fixtures with residency proofs and negative cases, plus the numeric contract
tests in §2.

**P1** — W5, W6, W7. W6 is independent and may run in parallel.

**P2** — W8, one pair at a time, each reusing W1/W2.

**Explicitly out of scope:** reproducing the paper's memory numbers. No machine
in the fleet has the gradient pool at the peak (62 GB Strix Halo, 16 GB RTX
5070 Ti, M1 Max), and W2 makes it unnecessary — the claim is proven in the IR.

---

## 5. Relationship to existing plans

- **Decision #28 / [`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md)**
  — W7 is a Tier-1 region kind with a trivially satisfiable accuracy budget.
- **Decision #29** — W1 gives `TilingInterface` its first in-tree consumer.
- **Decision #32** — §1.3 supplies the measured motivation for the
  `numeric_policy` carrier work, with a published error target.
- **Decision #16 (ZeRO-2)** — P3 shows W6 composes with optimizer sharding.
- **Decision #23** — FORGE's measurements come from a PyTorch autograd
  `Function` on HF models. None of that transfers; only the math and the IR
  shape do.
