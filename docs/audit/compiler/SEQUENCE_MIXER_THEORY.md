---
last_updated: 2026-09-08
audit_role: reference
---

> Current ownership review (2026-09-07): use the
> [live compiler queue](INTEGRATED_COMPILER_PLAN.md#live-queue)
> for live residuals and dependency order. Historical absence claims, timings,
> fleet instructions and effort estimates below describe their dated review;
> reference implementation is not native execution or promotion evidence.


# Sequence Mixer Theory of Operation

> **Status:** design paper (Track L / compiler direction). Pairs with
> [`SEQUENCE_MIXER_ENGINEERING_PLAN.md`](SEQUENCE_MIXER_ENGINEERING_PLAN.md).
> Extends Decision #28 (three-tier / measured-arbiter) into the sequence-mixing
> layer. This is *direction*; `MASTER_AUDIT.md` + generated dashboards stay
> status truth (Decision #26).
>
> **Terminology.** *TSOL* = **Tessera Standard Operator Library** — the
> reference/primitive surface in `python/tessera/stdlib/` + `tessera.ops` +
> the coverage registry. (If the intended expansion differs, the mapping in
> §9 is what matters, not the name.)

---

## 0. Why this document exists

Shipping open models in mid-2026 no longer agree on how attention should scale,
and the ones that ship long context all do it by **mixing cheap sequence
mixers with periodic expensive ones**. Two concrete data points bracket the
design space:

* **Kimi Linear** — 3 KDA (linear-recurrence) layers : 1 MLA (latent full
  attention), NoPE, constant recurrent state, ~75% KV reduction, ~6× decode at
  1M. Reaches long context by *never forming the S×S matrix* on 75% of layers.
* **Inkling** (Thinking Machines) — 66 layers, `local_layer_ids` sliding-window
  (`window=512`) + global, **GQA 8:1** (64 heads / 8 KV), a **short causal conv**
  (`use_sconv`, `k=4`), MoE (6/256 + 2 shared), **NVFP4/MXFP8** W4A4/W4A16.
  Reaches the *same* 1M context by *bounding the window + quantizing the cache*,
  with **zero** linear recurrence.

These are two different branches of one tree — "mostly-cheap layers + periodic
global mixing" — and Tessera already has scattered pieces of both
(`kimi_delta_attention`, `gated_deltanet`, `selective_ssm`, `attn_sliding_window`,
`stdlib/delta_rule.py`, `stdlib/hybrid.py`, `DeltaNetStateHandle`,
`SSMStateHandle`). What is missing is the **single abstraction** that makes them
one family the compiler can reason about, lower, cache-plan, and arbiter-select
uniformly. This paper defines that abstraction; the companion plan builds it.

The task is **unification and faithful completion**, not greenfield.

---

## 1. The unifying abstraction: the Sequence Mixer

> A **sequence mixer** `M` is a stateful causal map over a token span:
>
> ```
> M : (x[B, T, D], state_in) → (y[B, T, D], state_out)
> ```
>
> such that **streaming ≡ recompute**: composing `M` over single-token spans
> carrying `state` represents the same causal operator as one whole-span call,
> with identical initial state, masks, positions and RNG assignment. Floating
> results require a declared error budget; bitwise equality additionally requires
> preserving the evaluation order. This contract is not performance evidence.

Every mixer in the family (full attention, sliding-window attention, GLA,
RetNet, DeltaNet, Gated DeltaNet, **KDA**, Mamba-2 SSD, short causal conv, MLA)
is characterized by **four orthogonal facets** plus **two lowerings**:

| Facet | Question it answers | §  |
|-------|---------------------|----|
| **A. Transition structure** | How does the carried state evolve per step? | §2 |
| **B. Carried-state type** | What object holds history, and how big is it? | §3 |
| **C. Reassociation form** | Is the S×S matrix formed, avoided, or bounded? | §4 |
| **D. Numeric policy** | Accumulation dtype, chunk bound, scale metadata? | §5 |
| **Lowerings** | Sequential (decode) vs chunk-parallel (prefill) | §6 |

These facets organize contracts, not independent optimization knobs. Transition
structure constrains reassociation and layout; precision and decay range constrain
chunk length. A common interface can dispatch to distinct typed operation families
without replacing every family with one attribute-heavy op.

---

## 2. Facet A — the transition-structure lattice

Model the per-step state update as a typed **transition** with a structure tag.
The proposed tag describes transition structure; lowering additionally needs
shape, numerical and ownership contracts. The table is a family taxonomy,
not a proved lattice or a hardware cost ordering:

| Tag | Per-step state update | Model instance | Chunk cost |
|-----|----------------------|----------------|-----------|
| `identity` | `S_t = S_{t-1} + k_t v_tᵀ` | linear attention | cumsum |
| `scalar_diagonal` | `S_t = α_t S_{t-1} + k_t v_tᵀ`, `α_t∈ℝ` | scalar-decay attention / Mamba-2 SSD core | +scalar decay |
| `channel_diagonal` | `S_t = Diag(α_t) S_{t-1} + k_t v_tᵀ`, `α_t∈ℝ^{d_k}` | **KDA gate** | fold-into-K |
| `identity_minus_rank1` | `S_t = (I − β_t k_t k_tᵀ) S_{t-1} + β_t k_t v_tᵀ` | DeltaNet | UT solve |
| `dplr_bound` | `S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t-1} + β_t k_t v_tᵀ` | **KDA (full)** | fold + UT solve |
| `dplr_general` | `S_t = (Diag(a_t) + u_t w_tᵀ) S_{t-1} + b_t c_tᵀ` | general DPLR; not Mamba-2 SSD | structure-dependent scan |
| `conv(k)` | `y_t = Σ_{i<k} w_i x_{t-i}` (state = last `k−1`) | short causal conv | depthwise |
| `none` | (no finite-rank linear transition; carried KV history) | full / windowed attn | online softmax attention |

**The KDA insight, stated structurally.** KDA's transition is
`A_t = (I − β_t k_t k_tᵀ) Diag(α_t)` — a **Diagonal-Plus-Low-Rank** matrix whose
low-rank factor is *bound to the key* `k_t`. A *general* DPLR (`dplr_general`)
has independent low-rank factors; `dplr_bound` exploits the binding to specialize
the delta-rule algorithm and its WY/UT machinery. This does not by itself prove
floating-point stability or an exact factor-of-two reduction in work. This
is a cost-model distinction the arbiter should see. With `D = Diag(α)`,
`(I − βkkᵀ)D = D + uwᵀ` has `u = −βk` and `w = Dk`: the factors are related,
not generally equal. Recognition must prove that relation, update coefficients
and multiplication order. No universal speed ordering follows from the tag.
[Kimi Linear report](https://arxiv.org/abs/2510.26692).

Mamba-2 SSD instead restricts its transition to scalar-times-identity and belongs
with that specialization, not generic DPLR.
[SSD authors' model explanation](https://goombalab.github.io/blog/2024/mamba2-part1-model/).

**Why channel-diagonal is the load-bearing generalization.** Scalar decay
(`α_t∈ℝ`) lets the chunkwise form factor the pairwise decay as a scalar
`γ_t/γ_j`. Channel decay (`α_t∈ℝ^{d_k}`) does not factor — but it **absorbs into
the operands**: with cumulative channel decay `Γ_t = Π_{i≤t} α_i`,

```
k̂_t = k_t ⊙ Γ_t ,   k̃_j = k_j ⊘ Γ_j   ⇒   k̂_tᵀ k̃_j = k_tᵀ Diag(Γ_t/Γ_j) k_j
```

This exact-arithmetic identity requires nonzero cumulative decay and does not
alone derive the full erase/write recurrence. Zero gates invalidate division;
underflow, overflow and conditioning can invalidate a floating implementation
even with fp32 accumulation. A proposed `AbsorbChannelDecayIntoKeys` legalization
needs domain, finite-intermediate and error-budget checks, reset handling and
chunk-local scaling. Retain a safe recurrence when these cannot be proved. The
pass name is a proposal, not a registered executable pass.

---

## 3. Facet B — carried-state types and the N-way cache

Each mixer carries a distinct state object. The compiler's memory planner must
allocate the right type per layer. Shared size classes are an allocator policy
subject to alignment, lifetime, capacity and fragmentation measurements; one
uniform physical block is not a mathematical requirement.

| State type | Shape | Growth | Tessera handle today |
|------------|-------|--------|----------------------|
| `growing_kv` | `[B, H_kv, S, d]` | **linear in S** | `KVCacheHandle` |
| `windowed_kv(W)` | `[B, H_kv, W, d]` ring | **bounded** | *(gap — see plan)* |
| `latent_kv` (MLA) | compressed `[B, S, d_c]` | linear, compressed | *(gap)* |
| `recurrent_matrix` | `[B, H, d_k, d_v]` | **constant** | `DeltaNetStateHandle` |
| `ssm_state` | `[B, D, N]` | **constant** | `SSMStateHandle` |
| `conv_state(k)` | `[B, k−1, C]` | **constant** | `_causal_dwconv` carry |

Two consequences:

1. **The dual-cache is really an N-way cache contract.** A hybrid stack mixes
   these freely (Kimi: `recurrent_matrix` + `latent_kv`; Inkling: `windowed_kv`
   + `growing_kv` + `conv_state`). The planner allocates per-layer and proves the
   whole mix satisfies streaming ≡ recompute. `stdlib/hybrid.py` already encodes
   this for `recurrent_matrix`/`ssm_state`/`conv_state`; the gap is
   `windowed_kv`/`latent_kv` and the *uniform-block* normalization.
2. **Quantized state is a first-class variant.** Inkling's NVFP4 checkpoint
   stores `growing_kv`/`windowed_kv` at 4-bit + microscale metadata. The planner
   must treat a quantized cache block as the same block class at a smaller
   physical size — this is what makes 975B fit in 600 GB.

---

## 4. Facet C — the reassociation normal form

For dense, unmasked linear attention in exact arithmetic:

```
(Q Kᵀ) V   =   Q (Kᵀ V)          [dense, unmasked, exact arithmetic]
   S×S matrix         d×d state
```

Computing `KᵀV` first produces a fixed `d_k×d_v` state. Causal attention needs
prefix state rather than the final all-token state. Masks, bias, dropout,
normalization and data-dependent erase/write transitions need separate proofs;
convolution is not simply this reassociation. A proposed common interface is:

* `tessera.linear_recurrence` is the **normal form** that a reassociation
  rewrite lowers `softmax-free-attention → running-state recurrence` into. GLA,
  DeltaNet, KDA, Mamba all become this op + a transition tag.
* Softmax attention follows its own online reduction algorithm. FlashAttention
  need not materialize a global score matrix to preserve softmax semantics.

Absence of softmax is insufficient for general reassociation, and floating
reassociation needs numerical permission. NoPE does not make arbitrary MHA/GQA
weights equivalent to MQA: head sharing requires equality of the projections and
readout, or an explicitly admitted approximation. Absence of positional encoding
alone proves neither.

---

## 5. Facet D — numeric policy and the low-precision track

Every mixer carries a `numeric_policy` (Decision #15a): **storage dtype on the
operand, accumulator separate, plus mixer-specific constraints**:

* **Accumulation.** Delta/DPLR state is numerically sensitive. FP32 accumulation
  can be a minimum implementation envelope but does not prove stability or finite
  inverse decay. Require a recurrence-specific domain and error budget, including
  long-horizon drift, near-zero gates and nonnormal transient amplification.
* **Quantized state.** Scale values are runtime data; format, block size,
  packing and addressing are typed layout contracts. Scales may be explicit
  operands or part of a packed representation. Metadata cannot replace their
  storage, ownership and propagation.
* **Target admission.** Consult canonical dtype capabilities and backend queues
  for exact formats/instructions: SM100 and SM120 do not inherit each other's
  instruction or schedule proof. Model checkpoint storage does not establish
  a KV-cache format or a native quantized recurrence.
* **Promotion.** Measure storage, accumulation and state error together for the
  actual target and shape. The arbiter compares admissible implementations of
  the same semantics; changing head sharing or quantization policy requires a
  separate equality or approximation proof.

Precision work can proceed as a separate track, but its numerical constraints
couple to chunk size, recurrence conditioning and state representation.

---

## 6. The two lowerings and symbolic-dim policy

Each eligible family can expose recurrence and chunked candidates once their
equivalence is proved. For delta-rule families, the candidate forms include:

* **Sequential recurrence** — state-update work per token, decode / `S=1`
  bucket, state reused (often `O(S·d_k·d_v)` for the rank-one update).
  (Tessera: `gated_delta_rule_recurrent`, `_ssm_scan`, `DeltaNetStateHandle`.)
* **Chunk-parallel scan (WY/UT transform)** — prefill / large-S bucket. Rank-1
  sequential updates compress into a block-dense form: **everything is GEMM
  except one within-chunk unit-lower-triangular solve** `(I + Ã)⁻¹`. (Tessera:
  `gated_delta_rule_chunked` + `_forward_substitution`.)

Shared compiler infrastructure includes loops, contractions, reductions and
verified state ownership. Delta-rule families can share justified WY/triangular
machinery; scalar SSD needs its own derivation. Do not force a triangular solve
into every mixer or assume that a common interface implies a common physical
schedule. Specialize only after semantic and numerical legality, then measure
on the owning target.

The arbiter selects eligible candidates by bucket after checking the complete
state and numerical contract; decode is an `S=1` specialization and prefill is
the larger-span workload, not a guarantee of one universal lowering pair.
Symbolic-dim awareness is from day one: `S` is `bucket`-policy, window `W` and
chunk `C` are static/tuning params.

---

## 7. Correctness discipline — host-free oracles

Host-free oracles establish bounded mathematical/reference claims. Native
execution, race freedom and performance additionally require artifact and
exact-device evidence. The mixer abstraction needs these equivalence oracles:

| Oracle | Claim | Guards |
|--------|-------|--------|
| **chunk ≡ recurrent** | prefill lowering == decode lowering | the scan pre-fold |
| **streaming ≡ recompute** | N-cache decode == full recompute | the cache planner |
| **scalar-reduction** | `channel_diagonal` with broadcast scalar == `scalar_diagonal` | KDA generalizes GDN |
| **structure-canon** | `dplr_bound` == `dplr_general` on bound inputs | the arbiter fork §2 |
| **metamorphic low-precision** | W4A4/W4A16 within accuracy budget of bf16 | Facet D / arbiter |
| **DESIL cross-path** | backend kernel == numpy reference | per-backend bring-up |

These are the acceptance gates in the plan; each new mixer tag ships with its
row filled.

---

## 8. The schedule and the hybrid stack

A **schedule** assigns a mixer to each layer. `stdlib/hybrid.py`'s
`HybridSchedule` models the periodic case (`period`, `full_offset` → Kimi/Qwen
3:1). Inkling's `local_layer_ids` is an **explicit set**, so the schedule
abstraction needs both modes:

```
Schedule = periodic(period, offset) | explicit(layer_ids → mixer_tag)
```

The config space is three orthogonal axes:

```
(schedule)  ×  (cheap mixer ∈ {linear_recurrence[tag], sliding_window, short_conv})
            ×  (global mixer ∈ {full_attention, mla})
```

with the FFN (dense / MoE) a **fourth, independent** axis — attention scaling and
MoE sparsity are separate levers and must not be cross-credited (`hybrid.py`
already keeps `ffn` orthogonal to `linear_mixer`).

---

## 9. Layer map — where each concept lives

The abstraction is realized across the stack the user named
(abstraction → framework → library/TSOL → IR → runtime → arbiter):

```
┌── ABSTRACTION (this paper) ──────────────────────────────────────────────┐
│ Sequence Mixer contract · 4 facets · reassociation normal form · oracles  │
└───────────────────────────────────────────────────────────────────────────┘
        │
┌── FRAMEWORK (cross-cutting machinery) ───────────────────────────────────┐
│ HybridSchedule (periodic+explicit) · N-way cache planner (uniform block)  │
│ streaming≡recompute verifier · measured accuracy-budgeted arbiter          │
└───────────────────────────────────────────────────────────────────────────┘
        │
┌── TSOL (Tessera Standard Operator Library — reference + primitives) ──────┐
│ stdlib/delta_rule (→ channel-wise KDA) · sliding_window · short_conv · mla │
│ selective_ssm · registered ops (op_catalog) + coverage rows (Decision #24) │
└───────────────────────────────────────────────────────────────────────────┘
        │
┌── GRAPH IR ──────────────────────────────────────────────────────────────┐
│ tessera.linear_recurrence {transition, numeric_policy} · reassociation     │
│ canonicalization · schedule/mixer metadata                                 │
└───────────────────────────────────────────────────────────────────────────┘
        │
┌── TILE IR / TARGET IR (compiler enhancement) ────────────────────────────┐
│ chunkwise-scan pass (tag-parameterized) · AbsorbChannelDecayIntoKeys       │
│ triangular-solve tile primitive · windowed-attn lowering · FP4/FP8 emit    │
└───────────────────────────────────────────────────────────────────────────┘
        │
┌── RUNTIME / ABI ─────────────────────────────────────────────────────────┐
│ N cache-handle types (recurrent/ssm/conv/windowed/latent/kv) · uniform     │
│ physical block alloc · quantized (NVFP4/MXFP8) cache blocks                 │
└───────────────────────────────────────────────────────────────────────────┘
        │
┌── ARBITER (Decision #28) ────────────────────────────────────────────────┐
│ mixer-kernel candidates per (op, shape-bucket, dtype, target)              │
│ dplr_bound vs dplr_general fork · W4A4/W4A16 precision-budget selection     │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 10. What Tessera has vs. the gap

| Capability | Today | Gap to close |
|------------|-------|--------------|
| Delta rule + erase | `stdlib/delta_rule.py` (scalar) | **channel-wise `dplr_bound`** |
| Chunkwise scan | `_chunked` + `_forward_substitution` | **tag-parameterized, shared** |
| Registered ops | `kimi_delta_attention`, `gated_deltanet`, `selective_ssm` | faithful reference; **`linear_recurrence` normal form** |
| Windowed attention | `attn_sliding_window`, `lsa.py` | **first-class mixer + `windowed_kv` handle** |
| Short conv | `hybrid.py` LIV `_causal_dwconv` | **standalone `short_conv` primitive** |
| Cache handles | `DeltaNetStateHandle`, `SSMStateHandle`, `KVCacheHandle` | **windowed/latent + uniform-block planner** |
| Schedule | `HybridSchedule` (periodic) | **explicit-set mode** |
| Graph IR op | `attention`-lowering strings | **`linear_recurrence` + transition attr** |
| Low precision | `planned_gated` (Decision #15a) | **NVFP4/MXFP8 operand type + W4A4/W4A16 + FP4 emit** |
| Arbiter | Decision #28 direction | **mixer + precision candidates wired** |

The abstraction turns a dozen would-be bespoke ops into **one op × four facet
knobs**, and turns "add KDA / add Inkling-style hybrids" into "register a
transition tag / a cheap-mixer tag" — each a small, oracle-gated increment.
The engineering plan sequences that build.

## 10. Current compiler ownership (2026-09-08)

This reference owns algebra and acceptance contracts, not readiness or priority.
Earlier model timings and inventory tables are dated design context. The
[integrated plan](INTEGRATED_COMPILER_PLAN.md) owns sequencing.

| Boundary | Reusable foundation | Remaining mixer obligation |
| --- | --- | --- |
| Source effects — W4-PRODUCT-1 | Explicit source-state SSA, bounded completion and builtin error results, opt-in CPU JIT | Typed cache fields/views, positions and reset semantics; arbitrary objects remain unsupported |
| Lifetime — W2.4a | Scoped generations and bounded checked status fan-in | Register every reader/writer, including failed steps, before reuse |
| Native binding — F2 | Artifact projection and device next-state results | First-class tiled SSD/mixer producer, state ABI and backend consumers |
| Optimization — F3 / MSW-9 | Recipe identity and measured admission contracts | Prove the recurrence, masks and initial state, then measure candidates |
| Numerical legality — FA-4 | Scoped recurrence-stability obligation in the engineering plan | Certificates consumed by chunking/lowering, not detached declarations |

For an effectful step, specify `(values, next_state, completion)` and ownership
separately. CPU error transport preserves preceding declared writes. A device
next-state generation is not automatic in-place mutation. Define whether a failed
token commits state and how completion gates the next token before claiming
streaming/prefill equivalence. Test nonzero initial state, resets, uneven chunks,
zero gates, overlapping views, failure after writes and pending readers at reuse.
