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
> `tests/unit/test_autodiff.py` and `tests/unit/test_phase_e_f.py`. Tier 2
> follow-ups now ship reference higher-order derivatives, JAX-style
> `vmap`/`jacrev`/`jacfwd`, and `moe` VJP registration; backend fused/IR
> lowering remains tracked separately.
>
> **Theme 1 + Theme 2 below are fully closed** (Conv2d / LSTM landed in
> Phase H1/H2; F4+F5 verified end-to-end on MLIR 21). **Themes 3 / 4 / 5 /
> 6 / 9 / 10 also fully closed at the Python op surface (2026-05-09)** —
> the only remaining gaps are GPU execution kernels (Phase G long pole),
> optimizer Modules (`AdamW`), and the deliberately-out-of-scope
> `tessera.distributions.*` / `power_retention` items.
>
> **Theme 9 (utility ops)** — `arange`, `gather`, `clip`, `masked_fill`
> shipped 2026-05-09. `einsum` was already real.
>
> **Theme 10 (fp8)** — `quantize_fp8` / `dequantize_fp8` ops + autocast
> extension (`fp8_e4m3` / `fp8_e5m2`) shipped 2026-05-09. Per-backend
> Hopper tcgen05 fp8 mma + ROCm OCP fp8 deferred to Phase G.
>
> **Theme 5 (MLA)** — `latent_kv_compress` / `latent_kv_expand_k` /
> `latent_kv_expand_v` / `rope_split` / `rope_merge` ops + the
> `tessera.cache.LatentKVCacheHandle` paged latent storage shipped
> 2026-05-09. FlashMLA absorb-K target kernel deferred to Phase G.
>
> **Theme 6 (speculative decoding)** — `tessera.speculative.{expand_tree,
> acceptance_probabilities, batch_verify, advance_kv, SpeculativeStep}`
> shipped 2026-05-09. The scheduler primitives are pure Python today;
> Graph IR control-flow ops for end-to-end JIT'd speculative loops are a
> Phase 4 follow-up.
>
> **moe VJP** — registered 2026-05-09 (Theme 2 follow-up F3-moe).
> Theme 2 v1 + F1 + F2 + F3 (full) + F4 + F5 are all complete now.

## Summary

`examples/advanced/` contains **11 subdirectories totaling ~5,400 LOC**, written
ahead of compiler capability:

- **3 are scaffolds** referencing phantom APIs (`Diffusion_LLM`, `Jet_nemotron`, `power_retention`)
- **3 are compiler-smoke tests** building Graph IR directly to bypass the missing Pythonic layer (`Fast_dLLM_v2`, `MLA`, `Nemotron_Nano_12B_v2`)
- **3 are pure-Python planning utilities** with zero Tessera op usage (`long_context_attention`, `rlvr_reasoning_suite`, and the archived `speculative_decoding` toy; `kv_cache_serving` and `gumiho` now execute on the Apple backend)
- **1 is an integration framework skeleton** (`Tessera_Empirical_Software_Agent`)

Conclusion: examples encode an ambitious vision of where the compiler is going.
This document tracks the capability gaps that block them from running today.

---

## Per-example status

| Example | LOC | Runs today? | Forward? | Backward? | Primary blockers |
|---------|-----|-------------|----------|-----------|------------------|
| Diffusion_LLM | 5,326 | 🟡 partial | ✅ | 🟡 (autocast ✅, F1/F2 ✅) | `tessera.distributions.{Normal, Beta}` (out of scope); manual AdamW step until Module ships |
| Fast_dLLM_v2 | 272 | ✅ smoke | ✅ Graph IR | N/A | Real KV-cache + speculative-step ops now ship; native KV-cache lowering still a Phase G item |
| Jet_nemotron | 1,105 | 🟡 partial | ✅ | ✅ (CPU ref) | Hopper tcgen05 fp8 mma (Phase G); Python fp8 autocast surface ready |
| Nemotron_Nano_12B_v2 | 506 | ✅ smoke | ✅ | ✅ (CPU ref) | Per-backend Mamba2 chunked-scan kernels (Phase G); selective_ssm forward + VJP both ship |
| Empirical_Software_Agent | 537 | ✅ | ✅ no ops | N/A | LLM provider hooks (out of scope) |
| kv_cache_serving | 87 | ✅ | ✅ | N/A | Closed — `quantize_kv` / `KVCacheHandle(quantize_bits=…)` / `auto_evict=True` |
| long_context_attention | 81 | ✅ | ✅ no ops | N/A | Sliding-window & sink masks land via `ops.masked_fill` (Theme 9, 2026-05-09) |
| mla | 259 | ✅ smoke | ✅ | N/A | Theme 5 ops + `LatentKVCacheHandle` ship today; FlashMLA absorb-K kernel = Phase G |
| power_retention | 2 | ❌ | ❌ stub | N/A | Out of scope — folder is a CUDA kernel sketch only |
| rlvr_reasoning_suite | 195 | ✅ | ✅ no ops | N/A | — |
| gumiho | ~600 | ✅ | ✅ Apple GPU/CPU | numpy-validated | Gumiho (ICML'25) hybrid speculative decoding; draft+FTA-verify run on the Apple backend. The `speculative_decoding` string toy was archived (`examples/archive/`); `tessera.speculative.*` scheduler stays. |

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
| `tessera.nn.DynamicDepthwiseConv1d` | Jet_nemotron | ✅ landed Phase D4 on top of `ops.depthwise_conv1d` |
| `@tessera.function`, `@ts.compile(backend=...)` | Diffusion_LLM | Replaced — use `@tessera.jit(target=...)` |
| `@tessera.autodiff`, `tessera.autodiff.*` | Jet_nemotron, Diffusion_LLM | `tessera.autodiff.*` ✅ shipped; bare `@tessera.autodiff` remains an intentional error because `autodiff` is a module namespace |
| `tessera.autocast`, `tessera.checkpoint` (context) | Diffusion_LLM, Jet_nemotron | ✅ top-level `autocast`; callable `tessera.checkpoint` delegates to activation checkpointing while retaining save/load helpers |
| `tessera.optimizers.AdamW` | Diffusion_LLM | ✅ stateful wrapper over `tessera.optim.adamw`; `tessera.optimizers.Adam` also available |
| `tessera.distributions.{Normal,Beta}` | Diffusion_LLM | ✅ `Normal`, `Beta`, and `kl_divergence` shipped |
| `tessera.arange`, `tessera.gather`, `tessera.clip`, `tessera.einsum`, `tessera.masked_fill` | Diffusion_LLM | ✅ top-level aliases for `ops.*` |
| `fp8_e4m3` / `fp6` / `fp4` runtime dtypes | Jet_nemotron, Nemotron_Nano | ✅ annotation shorthands + autocast/quantization reference support; backend lowering remains Phase G |

---

## Cross-cutting capability themes

Each theme groups missing capabilities; numbers indicate examples blocked.

### Theme 1 — Stateful `nn.Module` surface (blocks 2) — **fully closed**
Storage + lifecycle for trainable parameters. ✅ **Shipped (Tier 1.1 + A4 + B1 + C1 + D4 + H1 + H2):**
`Module`, `Parameter`, **`Buffer`**, `Sequential`, `ModuleList`, `ModuleDict`,
`Linear`, `RMSNorm`, `LayerNorm`, **`BatchNorm1d`**, `Embedding`, `Dropout`,
`MLP`, `MultiHeadAttention`, `MultiHeadCrossAttention`, `RotaryEmbedding`,
`CastedLinear`, `CastedEmbedding`, all activation Modules,
`CrossEntropyLoss`, `nn.utils.clip_grad_norm_`, **`KVCache`**,
**`DynamicDepthwiseConv1d`**, **`Conv2d`** (NHWC) + **`Conv2dNCHW`** shim,
**`LSTMCell`** + **`LSTM`** (with state-propagation primitive
`ops.lstm_cell` + autodiff through BPTT). State_dict round-trip (incl.
persistent buffers), `train`/`eval`, `Module.to(dtype)`, `parameters()` /
`named_parameters()` / `buffers()` / `named_buffers()`, `register_buffer`.
**Roadmap:** Theme 1 fully closed.

### Theme 2 — Reverse-mode autodiff (blocks 2) — **F4+F5 landed (verified on MLIR 21)**
✅ **Shipped (Tier 2 v1 + F1 + F2 + F3 + F4 + F5):**
- **v1 surface:** `tessera.autodiff.tape()`, `tessera.autodiff.reverse(fn)`,
  `tessera.autodiff.custom_rule(name)`, `Parameter.grad` accumulation,
  built-in VJPs covering linear algebra, activations, norms, reductions,
  dropout, FFT family, gather, clip, masked_fill, **moe**, silu_mul,
  selective_ssm, flash_attn.
- **F1 mixed-precision:** `tessera.autodiff.autocast("fp16" | "bf16" | "fp32" | "fp64" | "fp8_e4m3" | "fp8_e5m2")`
  + `GradScaler`.
- **F2 activation checkpointing:** `tessera.autodiff.rematerialize()` /
  `tessera.autodiff.checkpoint(fn)`.
- **F3 custom-kernel adjoints:** `flash_attn` ✅, `fft`/`ifft`/`rfft`/`irfft` ✅, **`moe`** ✅ (registered 2026-05-09 — Theme 9 / Theme 6 round).
- **F4 Graph IR adjoints:** end-to-end verified on MLIR 21 — `--tessera-autodiff` builds clean against LLVM/MLIR 21, lit fixture passes (`tests/tessera-ir/phase_f4/autodiff_pass_smoke.mlir`).
- **F5 effect-aware adjoint collective insertion:** real `tessera.collective.{reduce_scatter, all_gather, all_reduce}` emission on cotangent SSA values; pipeline alias `tessera-autodiff-pipeline` runs F4+F5 together.

✅ **No longer deferred:** higher-order derivatives (`grad`, `hvp`,
`elementwise_grad`) and JAX-style `vmap` / `jacrev` / `jacfwd` shipped as
reference transforms. Remaining work is performance/IR maturation.
**Roadmap:** Theme 2 closed for the v1 + follow-ups surface.

### Theme 3 — Streaming kernels + dynamic shapes (blocks 2) — **closed**
✅ **Shipped (Phase D1/D2/D3/D4, 2026-05-09):** `ops.depthwise_conv1d` (causal +
streaming state, with VJP), `ops.online_softmax` + `ops.online_softmax_state`
(numerically stable, streaming via explicit state helper, with VJP for
single-chunk path), `nn.DynamicDepthwiseConv1d` Module wrapper (state via
non-persistent buffer from B1), **`ops.selective_ssm`** (Mamba2: A/B/C/Δ
projections, chunked scan, optional gate + initial state) — **VJP shipped
as well** (registered `@_vjp("selective_ssm")` in
`python/tessera/autodiff/vjp.py`).
🔲 **Out-of-scope follow-ups (no active demand):** `depthwise_conv2d`,
shape-polymorphic tiling.

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

### Theme 5 — Multi-Latent Attention primitives (blocks 1) — **Python op surface closed (2026-05-09)**
✅ **Shipped:** `tessera.ops.latent_kv_compress`, `latent_kv_expand_k`,
`latent_kv_expand_v`, `rope_split`, `rope_merge` + the
`tessera.cache.LatentKVCacheHandle` paged latent storage (append, read,
evict_oldest, sliding-window via `auto_evict=True`). Each projection op
has a distinct op_name so a Phase G FlashMLA target pass can match the
chain end-to-end (compress → cache → expand → absorbed-attention) and
emit a fused absorbed-K kernel on Hopper / Blackwell.
🔲 **Still deferred (Phase G):** FlashMLA target kernel (absorb-K
fusion); confidence-aware retention scoring (paper-grade).
**Roadmap:** Theme 5 closed for the Python surface.

### Theme 6 — Speculative decoding scheduler (blocks 1) — **Python primitives closed (2026-05-09)**
✅ **Shipped:** `tessera.speculative.expand_tree` (balanced draft tree
with parents + paths), `acceptance_probabilities` (Leviathan eq.1 ratio
clipped to [0,1]), `batch_verify` (per-path acceptance with
longest-prefix tie-break), `advance_kv` (cache trim by accepted-prefix
length; works on `KVCacheHandle` and `LatentKVCacheHandle`), and
`SpeculativeStep` (orchestration helper). 19 unit tests.
🔲 **Still deferred:** Graph IR control-flow ops so a `@tessera.jit` of
the entire speculative-decoding loop can lower to a single dispatched
kernel (Phase 4 follow-up).
**Roadmap:** Theme 6 closed for the Python surface.

### Theme 7 — Mamba2 / selective SSM ops (blocks 1) — **Python op surface closed**
✅ **Shipped (overlaps with Theme 3):** `ops.selective_ssm` (forward + VJP),
`ops.depthwise_conv1d` (causal + streaming), `ops.online_softmax` (chunked
scan numerics). The Python primitives unblock Mamba2 forward + backward on
the CPU reference path.
🔲 **Still deferred:** per-backend Mamba2 chunked-scan target kernels
(NVIDIA / ROCm — Phase G).

### Theme 8 — Distributed training plumbing (blocks 2) — **v1 closed (2026-05-09)**
✅ **Shipped:**
- **Phase F5 — effect-aware adjoint collective insertion:** the existing
  `GPUCollectiveInsertionPass` was extended to insert
  `reduce_scatter`/`all_gather`/`all_reduce` ops on cotangent SSA values
  (extends Phase 4's forward-side machinery to gradients).
- **Phase I1 — `tessera.distributed.DDP(module, mesh_axis="dp")`:**
  all-reduce on adjoint path; backward triggers gradient sync.
- **Phase I2 — `tessera.distributed.FSDP(module, mesh_axis="dp")`:** v1 —
  per-rank Module instances, sharded leading-dim, mock_collective tested.

Both wrappers run against the in-process
`tessera.testing.mock_collective.MockRankGroup`; real NCCL/RCCL bindings
land alongside Phase G's NVIDIA execution path.
🔲 **Still deferred:** real NCCL/RCCL backends (Phase G); ZeRO stage 3
parameter sharding (out of scope).

### Theme 9 — Utility tensor ops (blocks 1) — **closed (2026-05-09)**
✅ **Shipped:** `ops.arange`, `ops.gather` (with `axis` + scatter-via-`np.add.at`
VJP for repeated indices), `ops.clip` (strict-bounds straight-through gradient
matching PyTorch / numerical Jacobian), `ops.masked_fill` (broadcast-aware,
attention-mask use case). `ops.einsum` was already real. 12 unit tests.
**Roadmap:** Theme 9 closed.

### Theme 10 — Mixed-precision + autocast (blocks 2) — **Python op surface closed (2026-05-09)**
✅ **Shipped:**
- **`ops.quantize_fp8(x, format, scale)`** + **`ops.dequantize_fp8`** —
  per-tensor symmetric fp8 with `format="e4m3"` (max 448) or
  `format="e5m2"` (max 57344). Native cast via `ml_dtypes` when
  installed; pure-numpy mantissa-snap fallback otherwise. Saturation
  before cast handles values past max_normal.
- **`tessera.autodiff.autocast("fp8_e4m3" | "fp8_e5m2")`** — extends
  Phase F1 autocast to route per-op input casts through `quantize_fp8`
  on the boundary. fp16/bf16/fp32/fp64 paths unchanged.
- 9 unit tests covering roundtrip error bounds, saturation, autocast
  acceptance, and the e4m3-vs-e5m2 precision relationship.

🔲 **Still deferred (Phase G):** Hopper `tcgen05.fp8.mma` lowering, ROCm
OCP fp8 mfma rules, fp6/fp4 dtype tags (recognized in `graph_ir.py` but
not yet surfaced through ops/autocast).

---

## Tracking plan — closure scoreboard

Most of the original tier 1–4 backlog has shipped. The remaining items
are infrastructural (Phase G GPU execution) or out-of-scope deliberately.

### Closed (sized in original audit, now real)

| # | Theme | Status |
|---|-------|--------|
| 1 | `nn.Module` + `Parameter` + stateful layers (Theme 1) | ✅ Tier 1.1 + A4 + B1 + C1 + D4 + H1 + H2 |
| 2 | Streaming kernels — depthwise conv1d + online softmax + selective_ssm (Theme 3 + 7) | ✅ Phase D1/D2/D3/D4 + VJP for selective_ssm |
| 3 | KV-cache abstraction + block quantization (Theme 4) | ✅ Phase B2 + E1/E2/E3 |
| 4 | Reverse-mode autodiff (Theme 2) | ✅ v1 + F1 + F2 + F3 (full incl. moe) + F4 + F5 |
| 5 | Speculative decoding ops (Theme 6) | ✅ `tessera.speculative.*` (2026-05-09) |
| 6 | MLA primitives (Theme 5) | ✅ Python op surface + `LatentKVCacheHandle` (2026-05-09) |
| 7 | Mamba2 selective SSM (Theme 7) | ✅ forward + VJP; per-backend kernels = Phase G |
| 8 | Utility ops (Theme 9) | ✅ `arange` / `gather` / `clip` / `masked_fill` (2026-05-09) |
| 9 | Mixed-precision / autocast (Theme 10) | ✅ `quantize_fp8` / `dequantize_fp8` + `autocast("fp8_*")`; per-backend GPU rules = Phase G |
| 10 | Porting guide | ✅ [`docs/porting_advanced_examples.md`](../porting_advanced_examples.md) (2026-05-09) |

### Still open (Phase G long pole)

- Hopper `tcgen05.fp8.mma` lowering (Theme 10).
- ROCm OCP fp8 mfma rules (Theme 10).
- FlashMLA absorb-K target kernel (Theme 5).
- Mamba2 chunked-scan target kernels for NVIDIA / ROCm (Theme 7).
- Real NCCL/RCCL bindings for DDP/FSDP (Theme 8).
- Graph IR control-flow ops so `@tessera.jit` of a speculative-decoding
  loop can lower to a single dispatched kernel (Theme 6 follow-up).

### Deferred-items plan landings (2026-05-09) + remaining

Items 1–5 from [`docs/audit/deferred_items_plan.md`](deferred_items_plan.md)
all landed in a single sprint:

- ✅ Item 1 — `tessera.distributions.{Normal, Beta, kl_divergence}` (17 tests).
- ✅ Item 2 — fp6 / fp4 / nvfp4 quantize ops + autocast extension (16 tests).
- ✅ Item 3 — ZeRO stage 3 unification: `FSDP(stage=3)` + `ZeRO3` alias (7 tests).
- ✅ Item 4 — Higher-order autodiff: `grad`, `hvp`, `elementwise_grad`, re-runnable tape (15 tests).
- ✅ Item 5 — JAX-style transforms: `vmap`, `jacrev`, `jacfwd` + forward-mode JVP engine (15 tests).

🔲 **Still deferred (no active demand):**
- Item 6 — `power_retention` op. Folder is a CUDA kernel sketch only;
  start when a real retention research project lands. Numpy-reference
  op is ~200 LOC; full per-backend kernels are Phase G material.
- Phase G long pole — Hopper `tcgen05.fp4`/`fp6`/`fp8.mma`, ROCm OCP
  fp4/fp6/fp8 mfma rules, native FlashMLA absorb-K kernel, Mamba2
  chunked-scan target kernels, real NCCL bindings for DDP/FSDP/ZeRO3.
  Phase G remains the only unfinished frontier.

---

## Historical: original "first commits" recommendation

The original audit recommended two small commits to reduce the
phantom-API copy-paste hazard. Both shipped:

1. ✅ **Make every phantom `tessera.nn.X` raise `NotImplementedError`** —
   moot now. Every Tier 1.1 + A4 phantom is real, so there are no
   `nn.X` phantoms left to gate.
2. ✅ **Honest `examples/advanced/README.md`** — see
   [`examples/advanced/README.md`](../../examples/advanced/README.md).

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
