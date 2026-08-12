---
last_updated: 2026-08-12
audit_role: plan
plan_state: landing
---

# Block AttnRes — Model, Algorithms, and a ROCm-First Execution Plan

> **Routing:** start at [`README.md`](README.md). This document owns the Block
> AttnRes mathematical contract and ROCm-first acceptance criteria; global
> sequencing remains owned by
> [`INTEGRATED_COMPILER_PLAN.md`](INTEGRATED_COMPILER_PLAN.md).

**Status:** plan (2026-08-12). **Source paper:** Attention Residuals, arXiv 2603.15031
(Kimi Team / MoonshotAI). Official repo ships no code; no faithful public
implementation of Block AttnRes exists (survey in §5 of this doc's provenance
review, summarized in Appendix B). Gaps the paper leaves open are filled here
and marked **[GAP-n]** with the choice justified.

**Why ROCm first:** core compiler work is routed to the Strix Halo box
(`INTEGRATED_COMPILER_PLAN.md` §6a), which is the only fleet machine with an
executing non-Apple GPU lane (gfx1151 matmul + flash-attention family via
`runtime.launch()`), and ROCm is a lead performance target whose ceiling shared
infra must not cap (Decision #28). The mathematical contract lands host-free
first (Phases 0–2 run anywhere); the first hardware proof lands on gfx1151.

**Verification:** every derived result below is numerically checked by
`tests/unit/test_block_attnres_model.py` (pure numpy, no new deps): the VJP
matches central finite differences to ≤ 5e-10; the softmax-merge lemma is exact
(atol 1e-12) over random partition trees; properties P2–P4 hold as stated.

---

## Part I — Mathematical Model

### I.1 Setting and notation

All depth-attention computation is **per token**: every vector below carries
implicit batch/sequence indices `(β, t)` and the operator acts elementwise over
them. Only the sub-layer functions `f_l` mix tokens. This factorization is what
makes the construction well-defined: depth attention commutes with the token
axis.

- `d` — model width. `L` — number of **sub-layers** (each self-attention and
  each MLP counts as one; a transformer block contributes 2).
- Partition `{1..L} = B_1 ⊔ … ⊔ B_N` into contiguous, nonempty blocks, requiring
  `1 ≤ N ≤ L`. Let `(q,r) = divmod(L,N)` and define boundaries
  `a_n = (n−1)q + min(n−1,r) + 1` and
  `z_n = nq + min(n,r)`, so `B_n = {a_n,…,z_n}`. The first `r` blocks have
  `q+1` sub-layers and the rest have `q`; define `S = max_n |B_n| = ⌈L/N⌉`.
  **[GAP-1: the paper says only "the last block contains the remaining
  layers"; quotient/remainder balancing gives exactly `N` nonempty blocks and
  avoids an empty tail when `L` is not divisible by `N`.]** Write `n(l)` for
  the block containing `l`.
- `x ∈ R^d` — token embedding. `h_l ∈ R^d` — input to sub-layer `l`.
  `f_l` — sub-layer transform **without any internal residual**; its output is
  `v_l = f_l(Norm_l(h_l))` where `Norm_l` is the usual PreNorm RMSNorm (this
  norm belongs to the sub-layer, not to AttnRes).
- Learned AttnRes parameters: pseudo-queries `w_1, …, w_L, w_{L+1} ∈ R^d`, all
  **initialized to zero**. No other parameters (the key-normalization is
  weightless — see P4).

**Normalization.** `RMS(v) = √(‖v‖²/d + ε)`, `k̂(v) = v / RMS(v)`, so
`‖k̂(v)‖ ≈ √d`. A learnable key-norm gain `g` would satisfy
`w·(g⊙k̂) = (g⊙w)·k̂` and is therefore absorbed into the pseudo-query — omitted
WLOG. This resolves the paper's ambiguity about whether the key-norm carries a
gain.

### I.2 The depth-attention operator

For a query `w ∈ R^d` and an ordered tuple of sources `V = (v_1, …, v_m)`:

```
s_j    = w · k̂(v_j)                        (logits)                      (1)
α_j    = exp(s_j) / Σ_{j'} exp(s_{j'})     (weights, softmax over j)     (2)
A_w(V) = Σ_j α_j v_j                        (output)                      (3)
```

Properties (each load-bearing later; all except P1/P5/P6 machine-checked):

- **P1 (convexity).** `α` is on the simplex ⇒ `‖A_w(V)‖ ≤ max_j ‖v_j‖`.
  Depth attention cannot amplify magnitude — the mechanism behind the paper's
  bounded-periodic magnitude dynamics.
- **P2 (uniformity at init).** `w = 0 ⇒ α_j = 1/m ⇒ A_0(V) = mean(V)` — the
  **average**, not the residual **sum**. Standard residual is *not* the init
  point. Tests must assert uniform `α`, not equality with `Σ v_j`.
- **P3 (epsilon-qualified key scale response).** For `c > 0`, key *direction*
  is invariant, while fixed `ε` makes its norm only asymptotically invariant:
  `k̂(c v) = v / √(‖v‖²/d + ε/c²)`. Thus `k̂(c v) → k̂(v)` only when
  `‖v‖²/d ≫ max(ε, ε/c²)` (or exactly when `ε=0`/`c=1`). Near the epsilon
  floor, rescaling legitimately changes logits and weights; kernels and tests
  must preserve this behavior rather than claiming exact scale invariance.
- **P4 (gain absorption).** A key-norm gain is absorbable into `w` — the
  operator's only per-layer degree of freedom is one `d`-vector.
- **P5 (temperature).** `|s_j| ≤ ‖w‖√d`: the learned query norm is a per-layer
  temperature; zero-init starts at infinite temperature, training anneals it.
- **P6 (content-only addressing).** `A_w` is equivariant to source reordering —
  there is **no depth-position encoding**. Positional preference is realizable
  only through per-layer queries specializing to depth-specific content
  statistics (the paper's diagonal-dominance analysis shows this suffices).
  The only structurally distinguished slot is the partial sum (last position).

### I.3 Forward semantics (the complete recurrence)

State per token: completed block representations `b_0, b_1, …` and the
intra-block partial sum `p`.

```
b_0 = x                                                  (embedding is source 0)
For n = 1..N:
  p ← (absent)
  For i = 1..|B_n|:  let l be the i-th sub-layer of B_n
    σ(l) = (b_0, …, b_{n−1})          if i = 1            (4)  ← partial EXCLUDED
         = (b_0, …, b_{n−1}, p)       if i ≥ 2            (5)  ← partial is LAST
    h_l  = A_{w_l}( σ(l) )                                (6)
    v_l  = f_l( Norm_l(h_l) )                             (7)
    p    ← v_l  if absent  else  p + v_l                  (8)  ← standard recurrence
  b_n ← p                                                 (9)  block rep = raw sum
Output aggregation:
  h_out  = A_{w_{L+1}}( (b_0, …, b_N) )                   (10)
  logits = Head( Norm_out(h_out) )
```

- **[GAP-2] First sub-layer of a block** attends only over `(b_0..b_{n−1})` —
  never a zero placeholder. A zero source would receive weight
  `exp(w·k̂(0)) = exp(0)`: an artificial uniform-share sink. (One public
  implementation gets this right, one gets it wrong; see Appendix B.)
- **[GAP-3] Output aggregation.** The paper says only "the final output layer
  aggregates all N block representations." We model it as one more
  depth-attention with its own zero-init `w_{L+1}` (10). A plain sum — as one
  community repo does — reintroduces `O(N)` magnitude growth exactly at the
  head and abandons selectivity where it costs one `d`-vector.
- **[GAP-4] Numeric policy** (paper silent; fixed in Decision-#15a terms):
  storage dtype of `b_n, p, v_l` = model storage dtype (bf16); logits,
  softmax, weights, RMS, and the weighted-sum **accumulator** in fp32. I.e.
  `numeric_policy = {storage: bf16, softmax: fp32, accum: fp32}` — never
  compressed into the storage dtype string.

**Recovered limits.** `N = L, S = 1` ⇒ Full AttnRes with
`σ(l) = (v_0..v_{l−1})`. `N = 1` ⇒ every sub-layer sees exactly `(b_0, p)`:
a two-source convex gate between embedding and accumulated sum.

### I.4 Depth-mixing-matrix form

Unrolling (4)–(9): `h_l = Σ_{j<l} M_{j→l}(x)·v_j + M_{0→l}(x)·x` with
**row-stochastic** `M`. Sources in a completed block share their block's
weight; sources accumulated into the current partial share the partial's
weight. Hence `rank(M) ≤ N + S` and `M` is block-semiseparable. Standard
residual is the all-ones-triangular limit; Full AttnRes the dense rank-L
limit. **Mixing-matrix rank = number of depth-states that must stay live** —
the liveness bound the memory planner can consume (§III.4).

### I.5 Backward pass (VJP) — **[GAP-5]**, derived (the paper gives none)

Let `g = ∂ℒ/∂h` be the cotangent of `A_w(V)`. Write `r_j = RMS(v_j)`,
`k̂_j = v_j/r_j`, `u_j = g·v_j`, `ū = Σ_j α_j u_j`.

```
∂ℒ/∂w   = Σ_j α_j (u_j − ū) k̂_j                                        (11)
∂ℒ/∂v_j = α_j g                                        (value path)     (12)
        + α_j (u_j − ū) · (1/r_j)(I − k̂_j k̂_jᵀ/d) w   (key path)       (13)
```

Derivation of (13): RMSNorm Jacobian `∂k̂/∂v = (1/r)(I − k̂k̂ᵀ/d)` (using
`v vᵀ/‖v‖² = k̂k̂ᵀ/d`), then the softmax VJP `∂ℒ/∂s_j = α_j(u_j − ū)`.
Machine-checked against central finite differences (≤ 5e-10).

**Gradient-highway reading.** The value path (12) replaces the residual
identity path `g` with `α_j g` — a *weighted* highway. At init, a source in
the current block receives `Σ_l (1/|σ(l)|) g_l`: attenuated by the source
count. This is the price of the average-not-sum init, and why zero-init +
warmup matter for stability (the paper validates empirically).

Backward through (8)–(9) is ordinary fan-out accumulation; block reps collect
(12)+(13) from every consumer sub-layer in later blocks plus (10).

### I.6 Stats form and the merge lemma (exactness of two-phase)

Define attention-with-stats over sources `V`:

```
AttnStats_w(V) = (o, m, ℓ):  m = max_j s_j,  ℓ = Σ_j e^{s_j−m},  o = Σ_j e^{s_j−m} v_j   (14)
```

so `A_w(V) = o/ℓ`. For any partition `V = V₁ ⊔ V₂` with per-part stats:

**Lemma (merge exactness).** With `m* = max(m₁, m₂)`:

```
o = e^{m₁−m*}o₁ + e^{m₂−m*}o₂,   ℓ = e^{m₁−m*}ℓ₁ + e^{m₂−m*}ℓ₂,   m = m*   (15)
```

`A_w(V) = o/ℓ` **exactly**, and (15) is associative and commutative over any
partition refinement. *Proof:* both sides equal
`Σ_j e^{s_j−m*} v_j / Σ_j e^{s_j−m*}` after rescaling each part by
`e^{m_i−m*}`; associativity because max and shifted-sum are each
associative/commutative. ∎ (Machine-checked over random partition trees,
atol 1e-12.)

This lemma is the entire legality argument for the two-phase schedule, for
sequence-sharded merging, and for ring/context-parallel attention generally.
In sequence-mixer terms: **`(o, m, ℓ)` is the associative state that makes
softmax attention reassociable** — the `reassociable` facet, stated as an op
trait.

### I.7 Cost model (per token; matches the paper's accounting)

- Live depth-state: `N+1` vectors — the liveness bound. Full AttnRes: `l`,
  unbounded in `L`.
- Residual-mechanism I/O per sub-layer, two-phase: `(N/S + 5)d ≈ 5.5d` at
  `N=8, S=16` (standard residual `3d`; mHC m=4 `34d`).
- Prefill memory: `(N+1)·T·d` live across the depth sweep (shard over `T`).
  Decode: `T=1`, negligible — depth state is per-token and dies with the
  token's forward.
- Pipeline comm (interleaved 1F1B, P physical, V virtual, N_p blocks/stage):
  naïve `(PV)(PV−1)/2·N_p d` → cached `P(P−1)/2·N_p d + (V−1)P²N_p d`.

---

## Part II — Algorithms

Tensors carry `[B,T,d]`; source stacks `[m,B,T,d]`; all per-token math
vectorizes over `(B,T)`.

### A1 — Depth-attention operator (training form)

```
DEPTH_ATTN(w[d], V[m,B,T,d]) -> h[B,T,d]:
  K = V / sqrt(mean(V², axis=d) + ε)         # fp32 math          [m,B,T,d]
  s = einsum('d,mbtd->mbt', w, K)            # logits, fp32       [m,B,T]
  α = softmax(s, axis=0)                     # fp32               [m,B,T]
  h = einsum('mbt,mbtd->btd', α, V)          # fp32 accum → bf16
```

### A2 — Training forward (Block AttnRes)

```
BLOCK_ATTNRES_FORWARD(x, {f_l}, {w_l}, partition {B_n}):
  blocks = [x]                               # b_0
  for n = 1..N:
    p = None
    for i-th sub-layer l in B_n:
      V   = stack(blocks if p is None else blocks + [p])
      h   = DEPTH_ATTN(w_l, V)
      out = f_l(Norm_l(h))                   # f_l has NO internal residual
      p   = out if p is None else p + out
    blocks.append(p)                         # b_n
  return Norm_out(DEPTH_ATTN(w_{L+1}, stack(blocks)))   # [GAP-3]
```

### A3 — ATTN_WITH_STATS / SOFTMAX_MERGE / FINALIZE (the primitive pair)

```
ATTN_WITH_STATS(Q[q,d], K[m,d], V[m,d]) -> (O[q,d], M[q], L[q]):   # per token
  s = Q @ RMSNorm(K)ᵀ                # depth-attn form; general form takes K pre-normed
  M = max(s, axis=1);  P = exp(s − M[:,None]);  L = sum(P, axis=1);  O = P @ V
  # O is UNNORMALIZED and max-shifted

SOFTMAX_MERGE((O₁,M₁,L₁), (O₂,M₂,L₂)) -> (O,M,L):                  # elementwise
  M = max(M₁,M₂);  c₁,c₂ = exp(M₁−M), exp(M₂−M)
  O = c₁O₁ + c₂O₂;  L = c₁L₁ + c₂L₂

FINALIZE(O, L) -> O / L
```

Contract: `FINALIZE(SOFTMAX_MERGE over any partition) == softmax attention`
exactly (Lemma I.6); `SOFTMAX_MERGE` is associative/commutative →
tree-reducible across shards, ranks, phases.

### A4 — Two-phase inference for block `n` (paper Alg. 1, completed)

```
TWO_PHASE_BLOCK(n, {w_l}_{l∈B_n}, blocks=(b_0..b_{n−1})):
  Q = stack([w_l for l in B_n])                       # [S,d] — parameters, known ahead
  (O¹,M¹,L¹) = ATTN_WITH_STATS(Q, blocks, blocks)     # Phase 1: KV read ONCE
  p = None
  for i-th sub-layer l in B_n:                        # Phase 2: sequential lookback
    if p is None: h = O¹[i] / L¹[i]
    else:
      (O²,M²,L²) = ATTN_WITH_STATS(w_l[None], p[None], p[None])
      h = FINALIZE(SOFTMAX_MERGE((O¹[i],M¹[i],L¹[i]), (O²,M²,L²)))
    out = f_l(Norm_l(h));  p = out if p is None else p + out
  return p                                            # b_n
```

Exact (Lemma I.6) — training and inference produce identical `h_l`. Phase 1
overlaps the block's first sub-layer; the Phase-2 merge is elementwise → fuses
into `Norm_l`.

### A5 — VJP of DEPTH_ATTN (for the tape)

```
DEPTH_ATTN_VJP(w, V, α, K, r, g) -> (∂w, ∂V):
  u  = einsum('btd,mbtd->mbt', g, V);  ū = sum(α·u, axis=0)
  δs = α · (u − ū)                                    # softmax VJP
  ∂w = einsum('mbt,mbtd->d', δs, K)                   # (11)
  ∂V = α[...,None]·g[None]                            # value path (12)
     + δs[...,None] · (w[None] − K·(K@w)/d) / r       # key path (13)
```

Saved-for-backward: `(α, K)` — or recompute `K` from `V` under the remat
budget.

### A6 — Sequence-sharded prefill (TP integration)

Depth state `[N+1, T, d]` shards over `T` across P ranks (depth attention is
per-token ⇒ embarrassingly parallel over T). Phase 1 runs on local shards.
Where a sub-layer's TP output is produced as partial sums (row-parallel
matmul), the elementwise `SOFTMAX_MERGE`/`FINALIZE` + `Norm_l` ride the
existing reduce-scatter → local-op → all-gather path — no new collective, one
fusion. Per-device state: `(N+1)·(T/P)·d`.

---

## Part III — Tessera Mapping (ROCm-first)

### III.1 Op-set (Graph IR)

| Op | Signature | Role |
|---|---|---|
| `tessera.softmax_merge` | `(o₁,m₁,ℓ₁,o₂,m₂,ℓ₂) → (o,m,ℓ)`, elementwise | **The** new primitive. Named consumers (Decision #29): two-phase schedule, sharded prefill, ring/context-parallel attention, split-KV decode. Carries the associative+commutative trait (= the `reassociable` facet as an op trait). |
| `tessera.flash_attn_stats` | `flash_attn` + results `(m, ℓ)` (equiv. `lse = m + log ℓ`) | An attr `return_stats` on `FlashAttnOp` growing its results. FA-4 Tile IR already carries LSE as a backward checkpoint (`attn.lse.save/load`; see `LSE_CHECKPOINT_CONTRACT.md`) — this promotes it to a first-class forward value. |
| `tessera.depth_attn` | `(V[m,B,T,d], w[d]) → h[B,T,d]`, attr `sources: m` (symbolic, `bucket` policy) | Thin op with a **canonical decomposition** (A1). The decomposition is the declared oracle with a differential test; any fused kernel is an arbiter candidate (Decision #31). |

Contracts: `numeric_policy = {storage: bf16, softmax: fp32, accum: fp32}`
carried as an attribute and boundary-verified (Decision #32 — survive or record
a named drop reason); no semantic defaults (Decision #21a) — absence of the
partial-sum source is represented by `m`, never a zero placeholder;
`block_attn_res` itself is **TSOL sugar** in `python/tessera/stdlib/`, a peer
of `stdlib/delta_rule.py` — the same "bespoke ops are sugar" policy as the
sequence-mixer plan.

### III.2 Autodiff

Register (11)–(13) as the `depth_attn` VJP in `autodiff.vjp._VJPS` (JVP the
directional analog); `primitive_coverage` auto-flips the (V/J)VP axes
(Decision #24). The decomposed path composes existing tape rules for free —
giving a standing **differential test: analytic VJP vs traced-decomposition
VJP**. Update `op_catalog.py` **and** `primitive_coverage.py` together
(Decision #24); `depth_attn` enters as `reference` with backend axes `planned`
until the gfx1151 proof flips them.

### III.3 ROCm execution lane (the first hardware proof)

The gfx1151 lane already executes a compiler-generated matmul +
flash-attention family via `runtime.launch()`; the deltas are small and stay
inside existing seams:

1. **Stats-emitting attention kernel.** The online-softmax emitters hold
   `run_max/run_sum` in registers at exactly the point they would be stored —
   the change in `emit/rocm_hip.py` is "also write `(m, ℓ)`", not a new kernel
   family. Depth-attn Phase 1 is attention with tiny KV
   (`Nk = n ≤ N+1 ≈ 9`, `q = S ≈ 6–16`, one head, batched over `B·T` tokens):
   small GEMM + row softmax on **vector ALUs — no WMMA needed** (shapes are far
   below the 16×16×16 WMMA tile; RDNA 3.5's lack of FP8 WMMA is irrelevant
   here).
2. **Merge/finalize kernel.** `SOFTMAX_MERGE + FINALIZE + Norm` is a
   pointwise-reduce region — the `pointwise-graph`/`pointwise-reduce` seams in
   `fusion_core.py`; fuses into the sub-layer's PreNorm.
3. **Target IR discipline (Decision #19).** Both ops lower through
   `tessera_rocm.*` hardware-free ops before AMDGCN emission; lit fixtures in
   the ROCm backend suite (`check-tessera-rocm` — note this suite runs under
   `tessera-rocm-opt` and is NOT covered by `lit tests/tessera-ir/`).
4. **Hardware verification on Strix Halo.** F4-verify the fused kernels
   against the numpy references (A1/A3/A4); WSL specifics apply
   (`/dev/dxg` not `/dev/kfd`; no torch — numpy-only tests).
5. **Benchmark row** (stable JSON schema, Decision #12): measured
   residual-mechanism I/O vs the `(N/S+5)d` model; `op = "depth_attn"`,
   `backend = "rocm"`.

### III.4 Schedule IR (after the ROCm lane executes)

1. **Query hoisting (Phase 1).** `depth_attn`'s query is a parameter ⇒
   loop-invariant w.r.t. the sub-layer loop. A schedule transform batches the
   S per-layer ops of a block into one `flash_attn_stats` + per-layer
   `softmax_merge` (A4). Legality is *derived* from the reassociable trait
   (Decision #30), never assumed.
2. **Liveness.** Annotate `tessera.depth_state = N+1` on the region. Block
   reps are pinned for `InsertRecomputePass` (recomputing one = re-running S
   sub-layers — never in budget); AttnRes intermediates (`α`, `K`) are cheap
   recompute targets. Ship the Decision-#10a negative fixture: correct output
   marks **no** block rep rematerializable.
3. **Pipeline / TP.** Cross-stage caching enters `pipeline_planner.py`'s
   interleaved-1F1B cost model (`O(P)` vs `O(PV)` per transition); the
   sharded-prefill merge rides the existing collective path
   (`GPUCollectiveInsertionPass` after `EffectAnnotationPass`, unchanged
   ordering).

### III.5 Other backends (after ROCm)

| Target | Path |
|---|---|
| Apple GPU | Same two deltas in `emit/apple_msl.py` (`synthesize_attention_online_msl` stats variant + merge kernel); F4-verify. Mac-routed work. |
| x86 | Reference lane via `tessera.cpu.reference`; AVX-512 vectorized softmax/merge later through `emit/x86_llvm.py`. |
| NVIDIA sm_120 | Deferred until the pattern is proven on ROCm; `mma.sync` unnecessary at these shapes. |

### III.6 Verification plan (oracles + tests)

| Oracle / test | Checks | Kind |
|---|---|---|
| `two_phase == naive` (A4 vs A2, per block) | Lemma I.6 end-to-end | metamorphic, exact-to-fp |
| random partition trees → identical `(o,m,ℓ) → h` | reassociability | metamorphic |
| `w=0 ⇒ α uniform ⇒ h = mean(V)` | P2 | unit |
| source rescale preserves key direction; fixed-`ε` magnitude/weight response matches the exact formula, including near-zero sources | P3 | property |
| key-norm gain vs `g⊙w` identical | P4 | property |
| analytic VJP vs finite differences / traced decomposition | (11)–(13) | differential |
| limits `N=L` ≡ Full; first-of-block excludes partial | I.3 / GAP-2 | unit |
| fidelity checklist (per-sub-layer queries, raw `f(h)` sources, zero-init, attention-not-sum at head) | GAP-2/3, P2 | unit |
| measured I/O vs `(N/S+5)d` | I.7 | benchmark |

The model-level rows land now as `tests/unit/test_block_attnres_model.py`
(pure numpy — the executable contract for every later kernel). Kernel F4
verification reuses the same reference functions.

### III.7 Phasing (ROCm-first order)

| Phase | Deliverable | Where it runs |
|---|---|---|
| 0 *(this PR)* | Model doc + numpy contract tests (VJP, merge lemma, P2–P4) | any box |
| 1 | `softmax_merge` + numpy `ATTN_WITH_STATS` reference in `tessera.ops`; metamorphic tests | any box |
| 2 | `stdlib/attn_res.py` (A1/A2/A4), zero-init, GAP-2/3 semantics + fidelity tests → **first faithful public Block AttnRes** | any box |
| 3 | `depth_attn` Graph IR op + registered VJP + catalog/coverage rows + differential autodiff test | any box |
| 4 | ROCm stats-attention + merge kernels; Target IR fixtures (`check-tessera-rocm`); F4 hardware verify; benchmark row | **Strix Halo** |
| 5 | Schedule transforms (hoisting, liveness annotation + negative fixture); Apple parity; pipeline/TP integration | Strix Halo / Mac |

Phases 0–3 are pure Python and land the entire mathematical contract; 4 makes
it execute on gfx1151; 5 makes it scale.

---

## Appendix A — Relation to the sequence-mixer direction

The paper's structured-matrix analysis (§I.4) instantiates the
`SEQUENCE_MIXER_THEORY.md` facet algebra along the **depth** axis: standard
residual = depth cumsum (vanilla linear attention), Highway =
1-semiseparable, HC/mHC = m-semiseparable, DDL = depth DeltaNet, AttnRes =
depth softmax attention. Recommendation carried from the review: make the
recurrence **axis** (time | depth) a facet of `linear_recurrence` before the
W3 freeze, and expose the mixing-matrix semiseparable rank to the analysis
layer as a liveness bound.

## Appendix B — Public implementation survey (2026-08-12)

- `MoonshotAI/Attention-Residuals` (official): README + PDF only, **no code**.
- `kyegomez/attn_res` (PyTorch): faithful core operator (zero-init, weightless
  key-norm, sub-layer granularity, correct GAP-2 handling); final aggregation
  is a **sum** (GAP-3 violation), Algorithm 1 docstring-only, no tests.
- `nktkt/attention-residuals` (PyTorch, tested): off-spec — AttnRes once per
  block boundary with per-*block* queries; sources include internal residuals
  (`h + f(h)`); normal-init queries (contradicts the paper's "crucially,
  zero-init").
- `AbdelStark/attnres` (Rust/burn): only public two-phase implementation;
  alpha; Full variant only.

No faithful, complete Block AttnRes implementation exists publicly; the
pipeline cross-stage caching has zero public implementations. Phase 2 above
would be the first.
