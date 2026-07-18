---
last_updated: 2026-07-17
audit_role: reference
---

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
> carrying `state` equals one call over the whole span. This equivalence is the
> mixer's defining correctness contract, not an implementation detail.

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

The power of the decomposition: **the facets are independent knobs**. KDA and
Gated DeltaNet differ *only* in Facet A (channel-diagonal vs scalar-diagonal
gate); they share Facets B/C/D and both lowerings. Inkling's local layer and a
global layer differ *only* in Facet B (windowed vs growing KV) and the window
bound in C. Getting the compiler to treat these as **one op with attributes**
instead of a dozen bespoke ops is the entire thesis.

---

## 2. Facet A — the transition-structure lattice

Model the per-step state update as a typed **transition** with a structure tag.
The tag is an IR attribute; the chunkwise-scan lowering (§6) is a *function of
the tag*. The lattice, from cheapest to most general:

| Tag | Per-step state update | Model instance | Chunk cost |
|-----|----------------------|----------------|-----------|
| `identity` | `S_t = S_{t-1} + k_t v_tᵀ` | linear attention | cumsum |
| `scalar_diagonal` | `S_t = α_t S_{t-1} + k_t v_tᵀ`, `α_t∈ℝ` | GLA / RetNet | +scalar decay |
| `channel_diagonal` | `S_t = Diag(α_t) S_{t-1} + k_t v_tᵀ`, `α_t∈ℝ^{d_k}` | **KDA gate** | fold-into-K |
| `identity_minus_rank1` | `S_t = (I − β_t k_t k_tᵀ) S_{t-1} + β_t k_t v_tᵀ` | DeltaNet | UT solve |
| `dplr_bound` | `S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t-1} + β_t k_t v_tᵀ` | **KDA (full)** | fold + UT solve |
| `dplr_general` | `S_t = (Diag(a_t) + u_t w_tᵀ) S_{t-1} + b_t c_tᵀ` | Mamba-2 / SSD | full DPLR scan |
| `conv(k)` | `y_t = Σ_{i<k} w_i x_{t-i}` (state = last `k−1`) | short causal conv | depthwise |
| `none` | (no recurrent state; S×S softmax) | full / windowed attn | not reassociable |

**The KDA insight, stated structurally.** KDA's transition is
`A_t = (I − β_t k_t k_tᵀ) Diag(α_t)` — a **Diagonal-Plus-Low-Rank** matrix whose
low-rank factor is *bound to the key* `k_t`. A *general* DPLR (`dplr_general`)
has independent low-rank factors and a heavier scan; `dplr_bound` exploits the
binding to (a) halve the low-rank work and (b) stay algebraically identical to
the classical delta rule (numerically stable, reuses the WY/UT machinery). This
is a real cost-model fork the arbiter should see: **`dplr_bound` is a strictly
cheaper, exact specialization of `dplr_general` recognizable by structural
equality of the two low-rank factors** — a canonicalization, not a heuristic.

**Why channel-diagonal is the load-bearing generalization.** Scalar decay
(`α_t∈ℝ`) lets the chunkwise form factor the pairwise decay as a scalar
`γ_t/γ_j`. Channel decay (`α_t∈ℝ^{d_k}`) does not factor — but it **absorbs into
the operands**: with cumulative channel decay `Γ_t = Π_{i≤t} α_i`,

```
k̂_t = k_t ⊙ Γ_t ,   k̃_j = k_j ⊘ Γ_j   ⇒   k̂_tᵀ k̃_j = k_tᵀ Diag(Γ_t/Γ_j) k_j
```

so a **single pre-pass reweighting of K by `Γ` and `1/Γ`** turns the
channel-diagonal / `dplr_bound` chunk form back into "all GEMM + one triangular
solve." That pre-pass is a nameable compiler legalization
(`AbsorbChannelDecayIntoKeys`, §6), shared by KDA, GLA and Mamba. Its cost is
numerical: `1/Γ` grows as decay → 0, which **bounds chunk size and mandates
fp32 accumulation** (Facet D) — a compiler-enforceable precondition, not a
kernel footnote.

---

## 3. Facet B — carried-state types and the N-way cache

Each mixer carries a distinct state object. The compiler's memory planner must
allocate the right type per layer **and normalize heterogeneous types onto one
physical block class** (the vLLM lesson: size the logical KV block so linear and
full layers share a physical footprint, avoiding fragmentation).

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

The root legality under every linear mixer is a **matmul-chain reassociation**:

```
(Q Kᵀ) V   →   Q (Kᵀ V)          [legal iff no softmax between QKᵀ and V]
   S×S matrix         d×d state
```

Computing `KᵀV` first produces a fixed `d_v×d_k` state; the sequence dimension
never materializes as a pairwise matrix. **Every mixer with a recurrent state
(Facet A ≠ `none`) is an instance of this reassociation**, distinguished only by
the transition structure inserted between the running `KᵀV` accumulation and the
`Q` read. This gives the compiler one canonical form:

* `tessera.linear_recurrence` is the **normal form** that a reassociation
  rewrite lowers `softmax-free-attention → running-state recurrence` into. GLA,
  DeltaNet, KDA, Mamba all become this op + a transition tag.
* Mixers with Facet A = `none` (full / windowed / MLA) are the
  **non-reassociable branch** — they *do* form an (S×S) or (S×W) score matrix
  and lower through the attention path (`FlashAttn` family, windowed variant).

The legality predicate is exactly "no nonlinearity (softmax) sits on the score
matrix" — the same shape of reasoning as `EffectLattice`: a rewrite is legal
because a specific structure is *absent*. **NoPE composes here**: because Kimi's
linear layers carry no positional encoding, the *global* anchor layers can be
legally collapsed to MQA at inference — an "absence-of-effect unlocks a rewrite"
pattern worth modeling positional encoding as a tracked effect to make sound.

---

## 5. Facet D — numeric policy and the low-precision track

Every mixer carries a `numeric_policy` (Decision #15a): **storage dtype on the
operand, accumulator separate, plus mixer-specific constraints**:

* **Accumulation.** Delta/DPLR state is numerically sensitive (erase +
  rank-update); fp32 accumulation over bf16 storage is mandatory regardless of
  the storage dtype. The `AbsorbChannelDecayIntoKeys` pre-pass (§2) makes this a
  *precondition*, not advice: `1/Γ` conditioning bounds chunk size.
* **Low-precision as operand-type metadata.** Inkling forces
  MXFP8 / NVFP4 from "planned_gated" (Decision #15a) to a required path. The
  correct model — consistent with the DeepGEMM extraction ("scale layout as an
  IR operand type") — is:
  * **NVFP4** = FP4 (E2M1) values + per-16 **UE4M3** (unsigned E4M3, bias 7)
    block scale (+ optional per-tensor FP32). **MXFP4** uses an **E8M0**
    (power-of-two) block scale instead; **MXFP8** = FP8 + per-32 E8M0. The block
    scale is *operand-type metadata carried in the IR*, not a side tensor — and
    its physical layout is load-bearing: on consumer Blackwell the interleaved
    UE4M3 scale layout (CuTe atom `((32,4),(16,4))`) must be exact or ~10% of
    outputs corrupt. This is the concrete case for "scale layout as an IR operand
    type" (the DeepGEMM extraction), not a hand-fudged side buffer.
  * **W4A4 vs W4A16** is an accuracy/VRAM tradeoff the **measured,
    accuracy-budgeted arbiter** (Decision #28) selects per
    `(op, shape-bucket, dtype, target)` — Inkling ships both checkpoints because
    it *is* a budget decision. W4A4 requires SM100+ (Blackwell `tcgen05` FP4
    MMA); on the fleet, the sm_120 box exercises the FP4 path.

Low-precision is **orthogonal to the mixer math** — it plugs into Facet D and
the arbiter, and can be built as a parallel track (see plan W6).

---

## 6. The two lowerings and symbolic-dim policy

One mixer op carries **two algebraically-equivalent lowerings**, selected by the
sequence-length bucket (Decision #28's `static | bucket | dynamic` policy):

* **Sequential recurrence** — `O(S·d²)`, decode / `S=1` bucket, state reused.
  (Tessera: `gated_delta_rule_recurrent`, `_ssm_scan`, `DeltaNetStateHandle`.)
* **Chunk-parallel scan (WY/UT transform)** — prefill / large-S bucket. Rank-1
  sequential updates compress into a block-dense form: **everything is GEMM
  except one within-chunk unit-lower-triangular solve** `(I + Ã)⁻¹`. (Tessera:
  `gated_delta_rule_chunked` + `_forward_substitution`.)

The **chunkwise scan is a single shared Tile-IR lowering parameterized by the
Facet-A tag** — not one per mixer. The tag changes only the pre-fold
(`AbsorbChannelDecayIntoKeys` for diagonal decay; the `k k̂ᵀ` erase for delta);
the scan body — chunk the sequence, build `Ã`/`W̃`, triangular-solve, carry
state — is common to DeltaNet, KDA, GLA, and Mamba-2. The triangular solve is
"the triangular-solve tile primitive" a real kernel specializes.

Decode and prefill are thus **two lowerings of one op**, and the arbiter picks
by bucket — decode is the `S=1` specialization, prefill the large-S one.
Symbolic-dim awareness is from day one: `S` is `bucket`-policy, window `W` and
chunk `C` are static/tuning params.

---

## 7. Correctness discipline — host-free oracles

Everything is provable without hardware (Tessera's oracle discipline). The
mixer abstraction comes with a fixed set of equivalence oracles:

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
