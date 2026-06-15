# Frontier-Model Contract Survey (2026 H1) — what the next wave asks the compiler for

> Status: **survey + grounded re-ranked roadmap**. No code landed yet.
> Companion to [`MODEL_CLASS_ROADMAP.md`](MODEL_CLASS_ROADMAP.md) (which stood up
> DeepSeek-V3.2 / GLM-5 / Kimi-K2, M0–M5). This doc covers the *next* wave — hybrid
> linear-attention models, dual-stream world models, native MTP, agent serving —
> and defines **Track L** (the linear-mixer keystone) as the M6-equivalent.
> Last updated: 2026-06-15.

This is a survey of ~12 recent model/system papers read at primary-source depth,
each distilled to the **specific execution contract** a compiler must model, then
**grounded against Tessera's actual code** (not the CLAUDE.md prose — see the
doc-drift note below). The output is a re-ranked roadmap whose ordering falls out
of a primitive dependency graph, not a wish list.

## Meta-finding: the contracts mostly *exist*; they don't *execute*

Tessera already has Graph IR ops + Python reference + "complete" contract-axis
status for nearly the whole surveyed frontier (`gated_deltanet`,
`deepseek_sparse_attention`, the reasoning-attention family, `moe_swiglu_block`,
`NativeSparseAttention`, `MixtureOfRecursions`). **Almost none execute on a
backend.** So the dominant lever is **promote reference → executable lowering**,
plus a smaller set of genuinely-absent *stateful/temporal* contracts. Tessera
already shipped the right validation tool for promotion: the evaluator's DESIL
cross-path + metamorphic oracles (`python/tessera/compiler/evaluator.py`) — a
fused kernel is auto-provable against its reference.

### Grounding corrections (verified at source)

Two claims that floated through the audit, both checked against the tree:

1. **The false "Mamba2 op landed" claim was real — in `primitive_coverage.py`,
   not CLAUDE.md.** CLAUDE.md proper says the op is "*pending*"; but the coverage
   registry's comment asserted *"dedicated Mamba2 Graph IR op landed (2026-05-18)
   as `tessera.selective_ssm`"* and set `graph_ir_lowering="registered"` while **no
   ODS op existed** in `TesseraOps.td` — registry intent that outran the compiler
   surface (Decision #25/#26). **L4 closed this for real** (see below): the op is
   now materialized + lit-proven, so `registered` is finally honest. (`gated_deltanet`
   always did have its ODS op at `TesseraOps.td:1109`.)

2. **The real correctness finding (now test-proven):** the shipped
   `gated_deltanet` / `kimi_delta_attention` / `modified_delta_attention`
   reference (`__init__.py::_delta_attention_impl`, lines 1215-1225) computes
   `Ŝ_t = α_t·Ŝ_{t-1} + β_t·k_t v_tᵀ` — **gated linear attention, missing the
   DeltaNet `(I − β_t k_t k_tᵀ)` erase term**. The ODS summary ("Gated DeltaNet
   recurrent attention") and the runtime comment ("the delta recurrence is
   algebraically the quadratic form `(QKᵀ⊙mask)@V`", `runtime.py:6428`) describe
   the delta rule, but the math is linear attention — and every parity test passes
   because the GPU path faithfully matches the mislabeled reference.
   `tessera.stdlib.delta_rule` (Track L L1/L2, landed below) adds the genuine rule
   and the oracle that locks the distinction
   (`tests/unit/test_stdlib_delta_rule.py`).

## Per-model contract → Tessera status (grounded)

| Model | Verified compiler contract | Tessera today (file) |
|---|---|---|
| **Cosmos 3** (NVIDIA) [2606.02800] | **Two-launch varlen attention** with *independent q/k `cu_seqlens`* (causal launch + rectangular-block launch) — explicitly *not* a mask/bias (they beat FlexAttention 22% by moving structure into packing metadata). Plus **positional dual-tower weight binding** (segment→weightset, not MoE routing). | ❌ zero varlen surface; `attn_bias` is the wrong substrate. |
| **Nemotron-3 Super** (NVIDIA) [2604.12374] | Mamba2+**LatentMoE** paired blocks + sparse GQA anchors. LatentMoE = down-proj `ℓ=1024` → **route in latent** → `ℓ→ℓ` experts → up-proj (d/ℓ=4; 512 experts/top-22). **2 shared-weight recursive MTP** heads. NVFP4 per-layer dtype map (16-elem micro-blocks + BF16/MXFP8 islands). | SSM ref-only; no LatentMoE; MTP = Python orchestration; nvfp4 planned-gated dtype only. |
| **Qwen3.6-35B-A3B** (Alibaba) | `layer_types=[linear_attention×3, full_attention]×10`. **Gated DeltaNet dual-form**: chunked UT-transform GEMM (C=64, S∈[128,128], only non-GEMM step = C×C `(I−A)⁻¹`) for prefill / rank-1 gated state update for decode. 256e/top-8+1shared. 1 MTP layer. | `gated_deltanet` ODS exists (`TesseraOps.td:1071`); recurrent `linear_attn_f32` MSL kernel exists but **decay/state/β unwired** (`LinearAttnToAppleGPU.cpp:130`); **no chunked path anywhere**. |
| **Mellum2** (JetBrains) [2605.31268] | 3:1 SWA (window 1024) : full; **layer-selective YaRN** (only global layers); MTP self-speculation. | `attn_sliding_window` composes via bias; no per-layer-type RoPE table. |
| **LFM2.5-8B-A1B** (Liquid) | **LIV** double-gated short conv (depthwise causal k=3) in 18/24 layers; 32e/top-4. | `depthwise_conv1d` exists — closest existing primitive; not wired as a mixer. |
| **GLM-5.1** (Z.ai) [2602.15763] | DSA: **deterministic** top-k=2048 token gather, **all layers**, over MLA latent. Consistent-hash rollout→rank affinity; keep-recent-5 + discard-all@32k KV folding. (Non-deterministic top-k "caused drastic performance degradation" in RL.) | sparse-attn ref-only; no deterministic-topk primitive; no KV folding. |
| **MiniMax-M2** [2605.26494] | Dense GQA + **prefix-tree "compute-prefix-once, fork-branches" attention** + DFS radix global KV cache + windowed-FIFO (W=0.3N). MoE-under-branch dispatch undocumented (open Q). | scheduler single-request, resets cache each call (`dflash_serve.py:780`). |
| **DeepSeek-V4** [official report — REAL] | CSA: block-compress (m=4) → **FP4 lightning-indexer** → det top-k=1024 block gather → MQA, **interleaved with dense HCA** (m′=128). **Two-tier paged + SSM-state heterogeneous cache** keyed on compression-block boundaries. Per-*dim* FP8/BF16 KV split. | M4 `dsa_*` stdlib exists (block-index/select/sparse); paging metadata-only; no FP4 indexer; no heterogeneous cache. |
| **DiffusionGemma** (Google) | Block-AR + within-block parallel diffusion; **non-monotonic** accept/re-noise; `entropy_bound=0.1`; KV promote at block boundary only. | sampler already emits `accepted_mask`+`renoise_mask`, computes entropy then **discards it** (`models/sampler.py:77`). |
| **MemTrace** [2605.28732] | Bipartite **provenance DAG** (Variables w/ timestamps + Operations w/ In/Out sets) → "decisive error set" = earliest minimal faulty cut. | `memory_read/write/evict` + effect tracking exist; no In/Out provenance edges. |
| **YOLO26** (Ultralytics) [2606.03748] | Static-K=300, NMS-free (dual-head topk=7→1), DFL-free (`reg_max=1`) **fused decode → `f32[N,300,6]`**. | only `conv2d`/depthwise; zero detection tail. |
| **Holo3.1** (H Co.) | **Unpinnable** from primary sources (140ms / action-schema are third-party and contradict vendor's own 3.3s figure). | n/a — needs `config.json` inspection before any claim. |

## The cross-model convergence (where to bet)

**Architectural insight:** the SSM/linear-mixer recurrent state and the KV cache
are *the same heterogeneous-cache problem*. Nemotron's "constant Mamba state, KV
only on anchors," Qwen3.6's "dual recurrent-S + KV," and DeepSeek-V4's "two-tier
paged + SSM-state pool" are three statements of one contract. **Designing the
dual-form mixer's decode state correctly *is* designing the heterogeneous cache.**

Primitive dependency graph (build order falls out of this):

```
deterministic top-k ──────────► DSA/CSA sparse attention  +  RL bit-parity
varlen / packing attention ───► Cosmos dual-stream  +  CSA gather  +  prefix-fork
dual-form linear mixer ───────► Nemotron · Qwen3.6 · Mellum2 · LFM2.5   (4 flagships)
   └─ decode state == cache ──► heterogeneous cache (SSM-state ∪ windowed ∪ compressed)
MTP draft-head graph object ──► near-universal (every model here ships one)
```

Two tracks the original 8-item backlog **missed**: **deterministic top-k** (load-
bearing for all sparse attention + RL parity) and **sub-tensor mixed precision**
(NVFP4 micro-blocks + per-layer BF16/MXFP8 islands; per-dim KV split) — a concrete
`numeric_policy` extension, not a new dtype-by-fiat.

## Re-ranked roadmap

Validation rule (inherited from `MODEL_CLASS_ROADMAP.md`): a contract cell flips
to `complete` only when an oracle independently re-derives it. The natural oracle
for each promotion is named.

**Tier 0 — cheap, well-specified, Tessera already ~80% there:**
- **Diffusion commit-trace.** Sampler already produces `accepted_mask` +
  `renoise_mask` + per-position entropy — just retain a per-step
  `(step, position, entropy, accepted, renoised)` buffer. Non-monotonic semantics
  already modeled. Oracle: replay determinism. *Days.*
- **Memory provenance DAG.** MemTrace gives the exact schema; map In/Out sets +
  variable timestamps onto existing effect/dependency tracking + `MemoryStateHandle`.
  Hard rule: evict/overwrite must be an op with *both* In and Out edges or
  information-loss is un-attributable.
- **Deterministic top-k** (NEW, foundational). Unblocks all sparse attention + RL
  bit-parity. Oracle: metamorphic permutation-invariance of the selected set.

**Tier 1 — the keystone (4-flagship coverage): Track L, see below.**

**Tier 2 — new primitive classes:**
- **Varlen / two-launch attention** (Cosmos) — independent q/k `cu_seqlens`,
  causal + rectangular block. Same primitive is the substrate for CSA gather and
  prefix-fork → pays for itself 3×. Plus positional dual-tower weight binding.
- **LatentMoE** (precise spec) + close the `moe_swiglu_block` VJP/JVP gap.
- **MTP draft-head as a graph contract** — promote DFlash Python orchestration to
  an internal shared-weight, recursively-applied graph object. Near-universal.

**Tier 3 — serving + sparse + numerics (after their deps):**
- **Heterogeneous cache + prefix-shared/branch-fork attention** — common
  denominator of GLM sticky-hash / MiniMax radix-tree / DeepSeek two-tier. Couples
  with Track L's decode state.
- **DSA/CSA sparse attention executable** (block-compress → indexer → det-topk →
  varlen-gather → MQA). Depends on Tier-0 det-topk + Tier-2 varlen. Extends M4.
- **Sub-tensor mixed-precision `numeric_policy`** (NVFP4 micro-blocks, MXFP8
  islands, per-dim KV split).

**Scope decisions (explicit, not silent backlog):** **YOLO26** is a clean static-
shape NMS-free decode but the weakest strategic fit for an LLM compiler — decide
yes/no, don't drift. **Holo3.1 / LFM2.5-offload** contracts are unpinnable from
primary sources — don't plan against them until `config.json`/inference code is
inspected.

---

## Track L — dual-form linear mixer + hybrid schedule (the keystone)

Milestone ladder, mirroring `MODEL_CLASS_ROADMAP.md`. Unblocks Nemotron-3,
Qwen3.6, Mellum2 (schedule), LFM2.5 (LIV variant). "Definition of done" =
the two provable claims: full-config artifact (lit + verifier + conformance) and
scaled execution on Apple GPU gated vs numpy.

### Accurate starting state (grounded)

| Piece | Status | Anchor |
|---|---|---|
| `gated_deltanet` ODS op (q/k/v/gate/β/decay/state, return_state, state_dtype) | ✅ exists | `TesseraOps.td:1071` (base `Tessera_DeltaAttentionOp`) |
| Python emit path | ✅ | `runtime.py:6408` (`_gated_deltanet`) |
| Recurrent Apple GPU kernel `linear_attn_f32` | 🟡 partial | `apple_gpu_runtime.mm:8224`; pass `LinearAttnToAppleGPU.cpp` |
| — but **β/decay/state unwired** (→ it's plain linear attn, not the gated delta rule) | ❌ gap | `LinearAttnToAppleGPU.cpp:130` |
| — f32 / rank-4 / `D_qk·D_v ≤ 256` / causal only | constraint | `LinearAttnToAppleGPU.cpp:84-125` |
| **Chunked UT-transform prefill path** | ❌ absent | none in tree (prefill is O(S) sequential) |
| `selective_ssm` ODS op (Mamba2) | ❌ absent | python ref + JVP only (`autodiff/jvp.py`) |
| Hybrid `layer_types` as first-class schedule attr | ❌ absent | none |

### Ladder

| L | Title | Definition of done | Oracle |
|---|---|---|---|
| **L0** ✅ | Grounding correction + contract lock | Correct the propagated misquote; document the *real* finding (delta family = linear attn, no erase); lock it with an oracle | `test_existing_gated_deltanet_is_linear_attention_not_delta` |
| **L1** ✅ | **Genuine gated delta recurrence** (decode form) | `gated_delta_rule_recurrent` adds the `(v_t − α_t v̂_t)` erase, fp32 state, return_state; **not** "wire β/decay" (those were already wired) — the existing reference was missing the erase entirely | vs independent brute-force `(I−βkkᵀ)` recurrence; `erase=False` ≡ existing ref; state-carry |
| **L2** ✅ | **Chunked UT-transform prefill (the keystone)** | `gated_delta_rule_chunked`: chunk C, `Ã=tril(β·γ-ratio·KKᵀ,−1)`, `(I+Ã)⁻¹` via explicit forward substitution (`_forward_substitution`), WY/output as GEMM, γ-decay folding, cross-chunk state carry | **chunk ≡ recurrent** across ungated/β/fully-gated/output-gated + chunk-size-invariant (the make-or-break proof) |
| **L1.1** ✅ | Genuine delta rule on Metal (decode form) | `tessera_apple_gpu_gated_delta_rule_f32` — per-(b,h) sequential MSL scan with the erase; `backend="apple_gpu"` on the recurrent reference | **Metal ≡ numpy** (DESIL) + Metal ≡ L2 chunked (independent routes) |
| **L2.1** ✅ | Chunked UT-transform on Metal (prefill form) | `tessera_apple_gpu_gated_delta_rule_chunked_f32` — one threadgroup per (b,h), the within-chunk `(I+Ã)⁻¹` solve on-device; `backend="apple_gpu"` on the chunked reference | **Metal chunked ≡ numpy** (all chunk sizes incl. partial) + **Metal chunked ≡ Metal recurrent** |
| **L2.2** ✅ | Cooperative-parallel chunk kernel | **Measured headroom found** at high occupancy (256+ threadgroups), where L2.1's lane-0 advantage shrinks to ~1.3× over recurrent. Key insight: the within-chunk solve's **d_v columns are independent chains** → each thread owns columns and solves **barrier-free**; state carry parallelizes over cells. `coop=True` (default). | **L2.2 ≡ L2.1 ≡ numpy** (correctness) + **2.3× over L2.1 lane-0, 2.9–3.1× over recurrent** (measured, high occupancy) |
| **L3** ✅ | **Hybrid layer schedule** as a first-class attribute | `HybridSchedule` lowers `layer_types` literally; reference stack threads the **dual cache** (recurrent Ŝ for linear layers, KV for full layers) | **streaming dual-cache decode ≡ full recompute** + Qwen3.6 full-config schedule check |
| **L3.1** ✅ | `gated_deltanet` shipped-numerics decision | **Decision: opt-in, don't flip.** Added `erase=False` (default = current linear attn, backward-compatible) to `gated_deltanet`/`kimi_delta`/`modified_delta`; `erase=True` is the genuine rule. Flipping the default would break every caller's numerics — a future major version may. ODS `erase` attr deferred until graph→kernel honors it (no-op attr = drift) | `erase=True` ≡ `stdlib.delta_rule`; default ≡ existing; no regression |
| **L4** ✅ | `selective_ssm` (Mamba2) ODS op | Materialize the op the registry falsely claimed: `Tessera_SelectiveSsmOp` + verifier; close the drift. Chunk-scan (`_mamba_ssd.py`) + chunk≡sequential oracle already existed | lit roundtrip/verifier + chunk-scan ≡ sequential-scan |
| **L4.1** ✅ | Hybrid SSM mixer (Nemotron) | `linear_mixer="ssm"` adds a Mamba SSM mixer to the L3 hybrid stack; SSM state `h[D,N]` carried in the dual cache alongside attention-anchor KV | dual-cache decode ≡ recompute with SSM layers + `_ssm_scan` ≡ shipped `selective_ssm` |
| **L5** ✅ | LFM2.5 LIV mixer variant | `linear_mixer="liv"` — `Linear→(B⊙x̃)→depthwise-causal-conv(k=3)→(C⊙z)→Linear_out`; constant conv state (last k-1) in the dual cache | dual-cache decode ≡ recompute + conv causality |
| **Full blocks** ✅ | MoE FFN + MTP draft head | `ffn="moe"` (routed experts + shared, exact per-token routing); `HybridLM` + graph-level MTP head + lossless self-speculation | decode ≡ recompute (MoE); spec == AR (MTP, lossless) |

### Landed 2026-06-15 — L0–L2 (reference tier, host-free)

`python/tessera/stdlib/delta_rule.py` + `tests/unit/test_stdlib_delta_rule.py`
(18 oracles, all green). Mirrors the M-series house pattern: numpy reference +
oracle first, fused MSL kernel (L1.1/L2.1) as the hardware-gated follow-up. What
is proven host-free: the genuine gated delta rule (recurrent ≡ independent
brute-force in the paper's `(I−βkkᵀ)` layout), the chunk-parallel UT-transform
(`chunk ≡ recurrent` across all gating modes, chunk-size-invariant), the
triangular-solve primitive, cross-chunk state carry, and the L0 lock that the
shipped `gated_deltanet` is linear attention (`erase=False`), materially distinct
from the true rule when keys correlate.

### Landed 2026-06-15 — L1.1 (genuine delta rule on Metal)

`tessera_apple_gpu_gated_delta_rule_f32` (`apple_gpu_runtime.mm` + non-Darwin
stub parity) — a per-(b,h) sequential MSL scan carrying the `(v_t − α·v̂_t)`
erase, registered in `_apple_gpu_backend` and reachable via
`gated_delta_rule_recurrent(..., backend="apple_gpu")`.
`tests/unit/test_apple_gpu_gated_delta_rule.py` (7 oracles, all run on Metal):
**Metal ≡ numpy** for true delta / β+decay / output-gate / non-square head dims,
`erase=False` ≡ the shipped linear reference, and **Metal recurrent ≡ the L2
chunked UT-transform** (the genuine rule reached by two fully independent routes).

**Numerics finding (realism, → `numeric_policy`):** the delta rule is only
well-conditioned with **L2-normalized keys** — then `β·‖k‖²=β<1` makes
`(I−βkkᵀ)` a contraction and f32≡f64. With unnormalized keys (`‖k‖²≫1`) the
recurrence *expands* (eigenvalue `1−β‖k‖²<0`) and f32 legitimately diverges from
f64 (~10% here) — genuine ill-conditioning, not a kernel defect. A production
`gated_delta_rule` op should carry key-normalization in its contract (real models
do), with fp32 state accumulation.

### Landed 2026-06-15 — L2.1 (chunked UT-transform on Metal)

`tessera_apple_gpu_gated_delta_rule_chunked_f32` — one threadgroup per (b,h), C
threads (one per token-in-chunk), state Ŝ in threadgroup memory across the chunk
loop.  The GEMM-shaped rows (A, W̃, output) parallelize across threads; the
`(I+Ã)⁻¹` forward substitution + rank-1 state carry run on lane 0 (cooperative
parallelization of those is L2.2, a perf follow-up).  Reachable via
`gated_delta_rule_chunked(..., backend="apple_gpu")`.  Oracles (in the same test
file): **Metal chunked ≡ numpy** across chunk sizes 1/4/8/16/32 (S=20 exercises a
partial last chunk), **Metal chunked ≡ Metal recurrent** (two independent
on-device kernels), output-gate, and `erase=False` ≡ the shipped linear ref.

### Landed 2026-06-15 — L3 (hybrid layer schedule + dual cache)

`tessera.stdlib.hybrid` — `HybridSchedule` makes `layer_types` first-class
(`qwen3_6_schedule` = `[lin,lin,lin,full]·N`; `nemotron_schedule` = sparse
anchors) + a reference stack that threads the **dual cache**: constant-size
recurrent Ŝ for linear (genuine gated-delta) layers, growing KV for full-attention
layers.  Linear layers L2-normalize keys (the L1.1 conditioning finding).
`tests/unit/test_stdlib_hybrid.py` (9 oracles): the headline **streaming
dual-cache decode ≡ full recompute** across prefill points and schedules
(all-linear, every-other-anchor), schedule validation, and the Qwen3.6 full-config
check (30 linear / 10 full at 40 layers).

**Not yet done:** L2.2 (cooperative-parallel chunk kernel — perf), L3.1 (promote
the `gated_deltanet` ODS op to the true rule — a shipped-numerics decision), and
MoE/MTP composition into the hybrid stack (`stdlib.moe` exists; wiring is additive).

### Landed 2026-06-15 — L4 (`selective_ssm` Mamba2 ODS op)

`Tessera_SelectiveSsmOp` (`src/compiler/ir/TesseraOps.td`) + `SelectiveSsmOp::verify`
(`TesseraOps.cpp`) — `tessera.selective_ssm` is now a genuine Graph IR op
(rank-checked: rank-3 x / shape-equal delta / matching b,c / A rank-1|2 / optional
`gate` shape-equal x / `init` state rank-3). `tessera-opt` rebuilds clean (MLIR
22.1.6); lit fixture `tests/tessera-ir/model_class/selective_ssm.mlir` passes
(model_class sweep 5/5). The **drift is closed**: the coverage registry's
`graph_ir_lowering="registered"` for `selective_ssm` is now backed by a real op
(comment corrected in `primitive_coverage.py`). The chunked-parallel SSD lowering
(`_mamba_ssd.py::selective_ssm_parallel`) and its **chunk ≡ sequential** oracle
(`test_mamba_ssd_gpu.py`, 12 tests) already existed and stay green.

### Landed 2026-06-15 — L4.1 (hybrid SSM mixer — Nemotron expressible)

`tessera.stdlib.hybrid` now takes `linear_mixer = "delta" | "ssm"`. The stack was
refactored to per-mixer **span functions** so `hybrid_forward` (one span) and
`hybrid_decode` (streamed spans) run identical per-layer code — the dual-cache
oracle is meaningful for all three mixer types (delta Ŝ, SSM h[D,N], attention
KV).  `_ssm_scan` reproduces the shipped `tessera.ops.selective_ssm` (the L4 op's
reference) **and returns the carried state** (which the public reference does
not), so streaming SSM decode is exact.  Nemotron is now the second flagship
(after Qwen3.6) expressible end-to-end: `nemotron_schedule` + `linear_mixer="ssm"`
= Mamba layers + sparse attention anchors.  `tests/unit/test_stdlib_hybrid.py`
(+5 L4.1 oracles): `_ssm_scan ≡ selective_ssm`, Nemotron-shaped dual-cache decode
≡ recompute (across prefill points), and an all-SSM stack.

### Landed 2026-06-15 — L5 + MoE/MTP (full model blocks) + L3.1

**L5 (LIV):** `linear_mixer="liv"` — the LFM2.5 double-gated causal short-conv
(`Linear→B⊙x̃→depthwise-conv k=3→C⊙z→Linear_out`), constant conv state (last k-1
inputs) in the dual cache. Third hybrid family expressible. **MoE FFN:** `ffn="moe"`
wires `stdlib.moe` (routed experts + optional shared) with **exact, no-capacity-drop
routing** so it stays per-token → decode≡recompute holds; `top_k` proven
load-bearing; a Nemotron SSM+MoE full block runs. **MTP:** `HybridLM` (tied-embed
LM head + a shared-weight MTP draft head predicting t+2 from `h_t` + `embed(t+1)`)
+ `mtp_speculative_generate` — greedy self-speculation that is **lossless (== AR)**
by construction, with the accept-path exercised on a constructed predictable model.
27 oracles in `test_stdlib_hybrid.py`.

**L3.1 — the `gated_deltanet` shipped-numerics decision (made):** added
`erase=False` (default = current gated *linear* attention, **backward-compatible**)
to `gated_deltanet`/`kimi_delta`/`modified_delta`; `erase=True` opts into the
genuine DeltaNet rule (≡ `stdlib.delta_rule`). The default is **deliberately not
flipped** — that would silently change every caller's numerics; a future major
version may. The ODS `erase` attr is deferred until the graph→kernel path honors
it (a no-op attr would just be new drift). Guards in `test_stdlib_delta_rule.py`;
no regression in the existing delta/attention suites.

### Landed 2026-06-15 — L2.2 + erase routing + named models

**L2.2 (cooperative chunk kernel):** the L2.1 chunked kernel gained a `coop` mode
(default on) — the within-chunk `(I+Ã)⁻¹` solve parallelizes its independent d_v
column-chains across threads **barrier-free**, and the state carry parallelizes
over cells. Measured (`benchmark_delta_rule.py`, high occupancy): **2.3× over L2.1
lane-0, 2.9–3.1× over recurrent**, both modes bit-equal to numpy. The earlier
"deferred" call was overturned by sharper data (the lane-0 cost only shows at
high occupancy). **graph→Metal `erase` routing:** `gated_deltanet(erase=True)` on
`@jit(target="apple_gpu")` now runs the genuine rule on Metal (the L1.1 kernel),
not the composed linear form — verified e2e (`test_apple_gpu_delta_erase_routing.py`).
**Named models:** `models/{qwen3_6,nemotron3,lfm2_5}.py` wire the full-block stack
into named `config()`/`scaled_config()` factories (shapes match published configs;
scaled instances run decode≡recompute).

**Still open:** the ODS `erase` attr (deferred until lowering honors it), LatentMoE
(distinct from standard MoE) for a weight-faithful Nemotron, and per-layer-type
head dims / short-conv for a weight-faithful Qwen3.6.

Sequencing: **L1 unblocks L2** (decode state is the chunk carry); **L2 is the
keystone** (only the chunked GEMM form is tensor-core-viable for prefill — the
papers are unanimous); L3 is independent of L1/L2; L4 reuses L2; L5 is parallel.

### The one hard kernel detail

The only non-GEMM step in the chunked form is the within-chunk `T=(I−A)⁻¹` where
`A` is strictly-lower-triangular C×C — solved by forward substitution, **not** a
GEMM, and it is the throughput bottleneck on accelerators. It wants a dedicated
tile primitive (C=64 → a 64×64 unit-lower-triangular solve). Everything else
(WY factors `W=TβK`, `U=TβV`; cross-chunk `S_new=S_prev+(U−W S_prevᵀ)ᵀK`; intra-
chunk `O=QS_prevᵀ+(QKᵀ⊙M)(U−W S_prevᵀ)`) is tensor-core GEMM. **State accumulates
in fp32** regardless of bf16 storage — the `(I−A)⁻¹` and rank-updates are
numerically sensitive.

## Sources

Primary sources (verified at depth) live inline in the per-model table above:
Cosmos 3 [arXiv:2606.02800 + NVIDIA tech report], Nemotron-3 Super
[arXiv:2604.12374], Qwen3.6 [HF config.json] + Gated Delta Networks
[arXiv:2412.06464], Mellum2 [arXiv:2605.31268], LFM2.5 [Liquid blog + LFM2
report arXiv:2511.23404], GLM-5 [arXiv:2602.15763], MiniMax-M2 [arXiv:2605.26494],
DeepSeek-V4 [official report, HF deepseek-ai/DeepSeek-V4-Pro], DiffusionGemma
[Google ai.google.dev docs], MemTrace [arXiv:2605.28732], YOLO26
[arXiv:2606.03748 + Ultralytics docs], Holo3.1 [hcompany.ai — existence only].
