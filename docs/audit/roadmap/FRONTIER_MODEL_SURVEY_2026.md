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

### Doc-drift correction (fix before building)

The audit pass surfaced a CLAUDE.md claim that a "dedicated Mamba2 Graph IR op
landed (2026-05-18)". **It did not** — there is no `selective_ssm` op in
`src/compiler/ir/TesseraOps.td`; the registry only flips `graph_ir_lowering =
registered` (intent). Conversely, an earlier internal audit claimed
`gated_deltanet` has *no* ODS op — **also wrong**: `Tessera_GatedDeltaNetOp`
exists at `TesseraOps.td:1071` with a rich operand set. Both errors are the same
hazard (Decision #25/#26): registry/prose intent ≠ materialized compiler surface.
Every status in this doc is grounded to a `file:line`.

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
| **L0** | Doc-drift fix + contract lock | Remove the false "Mamba2 op landed" claim; lock the `gated_deltanet` operand contract; add a failing scaled gate | n/a |
| **L1** | **Wire β/decay/state through the recurrent kernel** | `linear_attn_f32` becomes a true gated-delta-rule step: `S_t = α_t·S_{t-1}(I − β_t k kᵀ) + β_t v kᵀ`, fp32 state accum; decode path correct | metamorphic: decay=1,β=1 ≡ existing linear_attn; vs numpy delta-rule ref |
| **L2** | **Chunked UT-transform prefill (the keystone)** | New lowering: chunk C=64, per-chunk `A=tril(−diag(β)KKᵀ,−1)`, `T=(I−A)⁻¹` as a dedicated C×C triangular-solve tile primitive, WY factors via GEMM, cross-chunk state carry; α-decay folded in | **chunk ≡ recurrent** (DESIL cross-path) — the make-or-break proof |
| **L3** | **Hybrid layer schedule** as a first-class attribute | Lower `layer_types` literally (`(i+1)%period==0 → full_attention`) into the layer schedule; dual KV/recurrent-state cache coexist per layer | full-config artifact lit (Qwen3.6 dims) |
| **L4** | `selective_ssm` (Mamba2) ODS op + chunk-scan lowering | Materialize the op the docs already claim; reuse the L2 chunk machinery | chunk-scan ≡ sequential-scan |
| **L5** | LFM2.5 LIV mixer variant | `Linear→(B⊙x̃)→depthwise-causal-conv(k=3)→(C⊙z)→Linear_out` as a fused mixer over the existing `depthwise_conv1d` | vs numpy LIV ref; scaled exec |

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
