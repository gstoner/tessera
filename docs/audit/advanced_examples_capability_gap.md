---
status: Informative
classification: Audit
authority: Tracking — links each `examples/advanced/` subdir to compiler-capability gaps
last_updated: 2026-05-09
---

# Advanced Examples — Compiler Capability Gap Audit

> **Tier 1.1 landed (2026-05-09):** Stateful `nn.Module` + `Parameter` + the
> 7 layer wrappers (`Linear`, `RMSNorm`, `LayerNorm`, `Embedding`, `Dropout`,
> `MLP`, `MultiHeadAttention`) and the 3 containers (`Sequential`,
> `ModuleList`, `ModuleDict`) ship with 71 unit tests. See
> `python/tessera/nn/{module,layers,functional}.py` and
> `tests/unit/test_nn_module.py`.
>
> **Tier 2 v1 first slice landed (2026-05-09):** `tessera.autodiff` —
> tape-based reverse-mode at the numpy-reference op layer.
> `tessera.autodiff.{tape, reverse, custom_rule}` + 22 built-in VJPs (gemm,
> add, mul, transpose, cast, relu, sigmoid, tanh, silu, gelu, softmax,
> layer_norm, rmsnorm, rmsnorm_safe, reduce, sum, dropout, flash_attn,
> fft/ifft/rfft/irfft). Unit tests
> covering numerical-Jacobian per op + end-to-end MLP one-step SGD loss
> decrease. See `docs/spec/AUTODIFF_SPEC.md`,
> `python/tessera/autodiff/{tape,vjp,__init__}.py`,
> `tests/unit/test_autodiff.py` and `tests/unit/test_phase_e_f.py`. **Tier 2
> still defers:** higher-order derivatives and JAX-style `vmap`/`jacrev`;
> `moe` still needs custom VJP registration.
>
> **Theme 1 + Theme 2 below are mostly closed** — the tables mark unblocked
> phantoms with ✅. Remaining notable gaps are GPU execution, optimizer/distribution
> utilities, FP8-family lowering, and a few example-specific ops such as
> `gather`, `einsum`, and `masked_fill`.

## Summary

`examples/advanced/` contains **11 subdirectories totaling ~5,400 LOC**, written
ahead of compiler capability:

- **3 are scaffolds** referencing phantom APIs (`Diffusion_LLM`, `Jet_nemotron`, `power_retention`)
- **3 are compiler-smoke tests** building Graph IR directly to bypass the missing Pythonic layer (`Fast_dLLM_v2`, `MLA`, `Nemotron_Nano_12B_v2`)
- **4 are pure-Python planning utilities** with zero Tessera op usage (`speculative_decoding`, `kv_cache_serving`, `long_context_attention`, `rlvr_reasoning_suite`)
- **1 is an integration framework skeleton** (`Tessera_Empirical_Software_Agent`)

Conclusion: examples encode an ambitious vision of where the compiler is going.
This document tracks the capability gaps that block them from running today.

---

## Per-example status

| Example | LOC | Runs today? | Forward? | Backward? | Primary blockers |
|---------|-----|-------------|----------|-----------|------------------|
| Diffusion_LLM | 5,326 | ❌ | 🟡 partial | ❌ | nn.Module/Parameter, autodiff, autocast, gather/clip/einsum/masked_fill |
| Fast_dLLM_v2 | 272 | ✅ smoke | ✅ Graph IR | N/A | KV-cache state machine, speculative branching |
| Jet_nemotron | 1,105 | 🟡 partial | 🟡 partial | ❌ | nn.Module, depthwise conv 1d (streaming), autodiff, fp8 lowering |
| Nemotron_Nano_12B_v2 | 506 | ✅ smoke | ✅ Graph IR | N/A | Mamba2 selective SSM (placeholder reference) |
| Empirical_Software_Agent | 537 | ✅ | ✅ no ops | N/A | LLM provider hooks, autotune feedback loop |
| kv_cache_serving | 87 | ✅ | ✅ no ops | N/A | KV-cache compression ops |
| long_context_attention | 81 | ✅ | ✅ no ops | N/A | Sliding-window + sink attention masks |
| mla | 259 | ✅ smoke | ✅ Graph IR | N/A | Latent projection ops, paged latent KV cache |
| power_retention | 2 | ❌ | ❌ stub | N/A | Retention/power attention op (CUDA kernel sketch only) |
| rlvr_reasoning_suite | 195 | ✅ | ✅ no ops | N/A | Distributed rollout + GRPO accounting |
| speculative_decoding | 67 | ✅ | ✅ no ops | N/A | Tree expansion / batch verify ops |

"Smoke" = builds Graph IR via internal compiler hooks rather than `@tessera.jit`,
intended as a compile-path test rather than an end-to-end demo.

---

## Phantom APIs referenced (must error or be implemented)

Names used in advanced examples that **do not exist** in `python/tessera/`:

| Phantom name | Used in | Resolution |
|--------------|---------|------------|
| `tessera.nn.Module` | Diffusion_LLM, Jet_nemotron | ✅ landed Tier 1.1 |
| `tessera.nn.Parameter` | Diffusion_LLM | ✅ landed Tier 1.1 |
| `tessera.nn.Linear` (stateful) | Diffusion_LLM, Jet_nemotron | ✅ landed Tier 1.1 |
| `tessera.nn.Sequential` / `ModuleList` / `ModuleDict` | Diffusion_LLM | ✅ landed Tier 1.1 |
| `tessera.nn.RMSNorm` (stateful) | Diffusion_LLM | ✅ landed Tier 1.1 |
| `tessera.nn.LayerNorm`, `MLP`, `MultiHeadAttention` (stateful) | Diffusion_LLM, Jet_nemotron | ✅ landed Tier 1.1 |
| `tessera.nn.Dropout` | Diffusion_LLM | ✅ landed Tier 1.1 |
| `tessera.nn.Embedding` | Diffusion_LLM | ✅ landed Tier 1.1 |
| `tessera.nn.SiLU`, `Sigmoid`, `GELU`, `ReLU`, `Tanh`, `Identity` | several | ✅ landed Phase A4 (2026-05-09) |
| `tessera.nn.MultiHeadCrossAttention` | several | ✅ landed Phase A4 |
| `tessera.nn.RotaryEmbedding` | several | ✅ landed Phase A4 |
| `tessera.nn.CastedLinear` / `CastedEmbedding` | Jet_nemotron | ✅ landed Phase A4 |
| `tessera.nn.CrossEntropyLoss` | several | ✅ landed Phase A4 |
| `tessera.nn.utils.clip_grad_norm_` | training paths | ✅ landed Phase A4 |
| `tessera.nn.functional` (submodule) | HF_transformer | ✅ shipped as self-alias of `tessera.nn` |
| `tessera.nn.flash_attention` | flash_attention_demo | ✅ shipped as alias for `tessera.ops.flash_attn` |
| `tessera.nn.DynamicDepthwiseConv1d` | Jet_nemotron | 🔲 still phantom — needs `ops.depthwise_conv1d` (Theme 3) |
| `@tessera.function`, `@ts.compile(backend=...)` | Diffusion_LLM | Replaced — use `@tessera.jit(target=...)` |
| `@tessera.autodiff`, `tessera.autodiff.*` | Jet_nemotron, Diffusion_LLM | Phase 5 — see Programming Guide Ch.7 |
| `tessera.autocast`, `tessera.checkpoint` (context) | Diffusion_LLM, Jet_nemotron | Implement (Tier 3 / Phase 5) |
| `tessera.optimizers.AdamW` | Diffusion_LLM | Composite of `ops.adam` + Module — Phase 5 |
| `tessera.distributions.{Normal,Beta}` | Diffusion_LLM | Out of scope today |
| `tessera.arange`, `tessera.gather`, `tessera.clip`, `tessera.einsum`, `tessera.masked_fill` | Diffusion_LLM | Add as `ops.*` (Tier 3) |
| `fp8_e4m3` / `fp6` / `fp4` runtime dtypes | Jet_nemotron, Nemotron_Nano | Tier 3 — autocast + lowering rules |

---

## Cross-cutting capability themes

Each theme groups missing capabilities; numbers indicate examples blocked.

### Theme 1 — Stateful `nn.Module` surface (blocks 2) — **almost fully closed**
Storage + lifecycle for trainable parameters. ✅ **Shipped (Tier 1.1 + A4 + B1 + C1 + D4):**
`Module`, `Parameter`, **`Buffer`**, `Sequential`, `ModuleList`, `ModuleDict`,
`Linear`, `RMSNorm`, `LayerNorm`, **`BatchNorm1d`**, `Embedding`, `Dropout`,
`MLP`, `MultiHeadAttention`, `MultiHeadCrossAttention`, `RotaryEmbedding`,
`CastedLinear`, `CastedEmbedding`, all activation Modules,
`CrossEntropyLoss`, `nn.utils.clip_grad_norm_`, **`KVCache`**,
**`DynamicDepthwiseConv1d`**. State_dict round-trip (incl. persistent
buffers), `train`/`eval`, `Module.to(dtype)`, `parameters()` /
`named_parameters()` / `buffers()` / `named_buffers()`, `register_buffer`.
🔲 **Still deferred:** `Conv2d` Module (Phase H1 — NHWC layout decision
locked), `LSTM` (Phase H2 deferred — RNN cells need their own design).
**Roadmap:** Theme 1 effectively closed except Conv2d + LSTM.

### Theme 2 — Reverse-mode autodiff (blocks 2) — **v1 + F1/F2/F3 landed; F4 ODS landed**
✅ **Shipped (Tier 2 v1):** `tessera.autodiff.tape()`, `tessera.autodiff.reverse(fn)`,
`tessera.autodiff.custom_rule(name)`, `Parameter.grad` accumulation, 17
built-in VJPs covering linear algebra / activations / norms / reductions /
dropout. ~600 LOC code + ~470 LOC tests = **~1,070 LOC**, well below the
original 1,200–1,800 LOC estimate because we explicitly deferred Graph/Tile IR
adjoint work. See `docs/spec/AUTODIFF_SPEC.md`. 🔲 **Still deferred (separate
follow-ups):** Graph/Tile IR adjoint ops, effect-aware adjoint collective
insertion, activation checkpointing / `rematerialize`, mixed-precision
gradient master-copy + loss scaling, higher-order derivatives,
`jax.vmap`-style transforms, fused-kernel adjoints (`flash_attn`, `moe`,
`spmm_*` — register via `custom_rule` until kernels exist).
**Roadmap:** Tier 2 v1 lands the surface. Each deferred item is its own
follow-up sized in `docs/spec/AUTODIFF_SPEC.md`.

### Theme 3 — Streaming kernels + dynamic shapes (blocks 2) — **closed (forward); D3 VJP follow-up**
✅ **Shipped (Phase D1/D2/D3/D4, 2026-05-09):** `ops.depthwise_conv1d` (causal +
streaming state, with VJP), `ops.online_softmax` + `ops.online_softmax_state`
(numerically stable, streaming via explicit state helper, with VJP for
single-chunk path), `nn.DynamicDepthwiseConv1d` Module wrapper (state via
non-persistent buffer from B1), **`ops.selective_ssm`** (Mamba2: A/B/C/Δ
projections, chunked scan, optional gate + initial state, forward only).
🔲 **Follow-up:** `selective_ssm` VJP (Mamba2 backward — analytical
gradient is derivable but tedious; ship as a `custom_rule` registration in
a focused PR). Also depthwise_conv2d and shape-polymorphic tiling (no
active demand).

### Theme 4 — KV-cache abstraction + block quantization (blocks 3) — **Phase E landed 2026-05-09**
✅ **Shipped:** `KVCacheHandle` (Phase B2, paged buffer + max_seq enforcement),
`quantize_kv` / `dequantize_kv` (Phase E1, 2/4/8-bit per-token symmetric
quantization), `KVCacheHandle(quantize_bits=...)` mode (Phase E2 — int8
storage + transparent dequant on read), `auto_evict=True` rolling-window
sliding window (Phase E3), explicit `evict_oldest(n)` method.
🔲 **Still deferred:** confidence-aware retention (paper-grade
implementations like H2O / DuoAttention require a confidence scorer that
isn't in scope this cycle).
**Roadmap:** Phase E closed for the v1 surface.

### Theme 5 — Multi-Latent Attention primitives (blocks 1)
Latent Q/KV projections, confidence softmax, paged latent KV cache, RoPE
split/merge for latent dim, target kernels (FlashMLA for SM_90/SM_100).
**Scope:** medium (~300–500 LOC + target kernels).
**Roadmap:** Design doc at `examples/advanced/mla/flashmla_tessera.md`; no ops yet.

### Theme 6 — Speculative decoding scheduler (blocks 1)
Tree expansion, batch verification, acceptance mask, KV-cache advance with
path selection. Requires Graph IR control flow.
**Scope:** medium (~400–700 LOC).
**Roadmap:** Not documented; downstream of control-flow work in Phase 4.

### Theme 7 — Mamba2 / selective SSM ops (blocks 1)
Selective SSM, depthwise causal conv, chunked scan, output gate. Per-backend
target kernels.
**Scope:** large (~500–800 LOC + 3K+ per-backend).
**Roadmap:** Not documented.

### Theme 8 — Distributed training plumbing (blocks 2)
Backward-pass collective insertion (`reduce_scatter` / `all_gather` on
gradients), gradient accumulation, parameter shard layout. Largely infrastructural
once autodiff exists.
**Scope:** small once Theme 2 lands.
**Roadmap:** `GPUCollectiveInsertionPass` (Phase 4) handles forward; needs adjoint extension.

### Theme 9 — Utility tensor ops (blocks 1)
`arange`, `gather`, `clip`, `einsum` (full), `masked_fill`, parametric softmax.
**Scope:** small (~150–250 LOC).
**Roadmap:** Not documented.

### Theme 10 — Mixed-precision + autocast (blocks 2)
Runtime dtype parameters (`fp8_e4m3`, `fp6`, `fp4`), `@autocast` context, dequant/requant ops, hardware-specific fp8 rules.
**Scope:** medium (~200–350 LOC + lowering rules).
**Roadmap:** Programming Guide Ch.3 outlines strategy; no lowering today.

---

## Tracking plan — prioritized backlog

Ordered by `unblocked_examples × (1 / scope)`. File new tracking issues using
the headings below as titles.

### Tier 1 — critical path

1. **`nn.Module` + `Parameter` + stateful layer wrappers** (Theme 1)
   - File new: `python/tessera/nn/module.py`
   - Unblocks: `Diffusion_LLM` (forward, partial), `Jet_nemotron` (forward)
   - Tests: `tests/unit/test_nn_module.py`, parameter lifecycle, state_dict round-trip
   - Update `tessera.nn` `__all__` to add `Module`, `Parameter`, stateful `Linear`/`RMSNorm`/`Sequential`/`Embedding`/`Dropout`/`SiLU`

2. **Streaming kernels — depthwise conv 1d + online softmax** (Theme 3)
   - File new: `python/tessera/ops/streaming.py`, register in registry
   - Unblocks: `Jet_nemotron` JetBlock forward, partial `Nemotron_Nano_12B_v2`
   - Tile IR: streaming state threading; per-backend kernels follow
   - Tests: `tests/unit/test_depthwise_conv1d.py`, `test_online_softmax.py`

3. **KV-cache abstraction + block quantization** (Theme 4)
   - File new: `python/tessera/cache/`
   - Unblocks: `kv_cache_serving` (real ops), `Fast_dLLM_v2` (real KV state), `mla` (paged latent cache)
   - Add `KVCacheHandle` value type to Graph IR
   - Tests: `test_kv_cache_quant.py`, `test_kv_cache_rolling.py`

### Tier 2 — Phase 5 epic

4. **Reverse-mode autodiff (Theme 2)**
   - File new: `python/tessera/autodiff/{__init__.py, reverse.py, custom_rule.py, checkpoint.py}`
   - Unblocks: training paths in `Diffusion_LLM`, `Jet_nemotron`
   - Graph IR: adjoint ops; Schedule IR: recompute scheduling; effect-aware adjoint collective insertion (extends `GPUCollectiveInsertionPass`)
   - **Open as a separate epic** — this is the largest single chunk and gates 3+ other items
   - Already documented as planned in Programming Guide Ch.7

### Tier 3 — high-impact unblocks (1 example each)

5. **Speculative decoding ops** (Theme 6) → unblocks `speculative_decoding`
6. **MLA primitives** (Theme 5) → unblocks `mla` (full compilation)
7. **Mamba2 selective SSM** (Theme 7) → unblocks `Nemotron_Nano_12B_v2`
8. **Utility ops** (Theme 9) → unblocks `Diffusion_LLM` schedule extraction
9. **Mixed-precision / autocast** (Theme 10) → unblocks fp8 paths in `Jet_nemotron`, `Nemotron_Nano_12B_v2`

### Tier 4 — documentation

10. **Porting guide** — `docs/porting_advanced_examples.md`
    - For each phantom API, show the equivalent today's-API rewrite
    - For each example, mark "runnable today" / "blocked on X" / "future"

---

## Recommended first commit after this audit

Two small commits that immediately reduce the phantom-API copy-paste hazard:

1. **Make every phantom `tessera.nn.X` raise `NotImplementedError`** with a
   message like `"tessera.nn.Module is Tier 1 backlog (see docs/audit/advanced_examples_capability_gap.md); use the functional surface tessera.nn.linear/rms_norm/swiglu/multi_head_attention until then."` Prevents the silent-stub footgun for users browsing examples.

2. **Add a top-level `examples/advanced/README.md`** that is *honest*: lists which examples run today (4 utilities + 3 smoke), which don't (3 scaffolds + 1 stub), and which Tier in this audit each blocked example depends on.

Both are <100 LOC, don't touch the compiler, and immediately fix the
"copy-paste from `examples/advanced/X` and hit AttributeError" experience that
sent us down this audit in the first place.

---

## Cross-references

- **`docs/audit/execution_roadmap.md`** — sequenced execution plan. Theme 1
  follow-ups → Phase A4/B1/C; Theme 3 → Phase D; Theme 4 → Phase E; Theme 2
  follow-ups → Phase F. Pick tasks from there.
- CANONICAL_API.md — the actual surface today
- Programming Guide Ch.2 (Programming Model) — current canonical patterns
- Programming Guide Ch.5 (Kernel Programming) — `@tessera.kernel` + `index_launch`
- Programming Guide Ch.7 (Autodiff) — Phase 5 planned API
- Programming Guide Ch.8 (Layouts & Data Movement) — KV-cache + sharding
- CLAUDE.md Architecture Decision #22 — doc surface ≠ implementation surface
