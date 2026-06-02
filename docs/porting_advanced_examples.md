---
status: Informative
classification: Guide
authority: Tier 4 deliverable from `docs/audit/coverage/COVERAGE_AUDIT.md`
last_updated: 2026-05-09
---

# Porting Advanced Examples — Phantom API → Today's API

This guide is the canonical reference for translating code in
`examples/advanced/` (and similar copy-paste templates floating around)
to the surface that ships in Tessera today.

It lives here because the `examples/advanced/` subtree was written
ahead of compiler capability and references APIs Tessera doesn't
expose. Most of those phantom APIs are now real (Tier 1.1, Tier 2 v1,
Themes 3/4/5/6/9/10), so this guide has shrunk to the set of small
remaining gaps plus the patterns that *changed shape* between the
example sketches and the canonical surface.

If you're new to Tessera, read
[`docs/CANONICAL_API.md`](CANONICAL_API.md) first. This guide assumes you
already know how `@tessera.jit`, `tessera.ops.*`, `tessera.nn.*`, and
`tessera.autodiff.*` fit together.

## Per-example status

| Example | Runs today? | What you can run | What still needs work |
|---------|-------------|-------------------|------------------------|
| `Diffusion_LLM` | 🟡 partial | Forward pass with `tessera.nn.{Linear, MLP, MultiHeadAttention, RMSNorm, LayerNorm, Embedding, Dropout, RotaryEmbedding}`; autocast via `tessera.autodiff.autocast("fp16")`; gather/clip/masked_fill/arange now real ops. | Distributions (`Normal`, `Beta`) — replace with `numpy.random` for now. |
| `Fast_dLLM_v2` | ✅ smoke | The Graph-IR-builder smoke path. KV-cache state machine via `tessera.cache.KVCacheHandle` + speculative-step orchestration via `tessera.speculative.SpeculativeStep`. | Native KV-cache lowering (Phase G). |
| `Jet_nemotron` | 🟡 partial | Full forward — JetBlock streaming via `ops.depthwise_conv1d` + `nn.DynamicDepthwiseConv1d` (state buffer); fp8 paths via `tessera.autodiff.autocast("fp8_e4m3")`. | Hopper tcgen05 fp8 mma (Phase G). |
| `Nemotron_Nano_12B_v2` | ✅ smoke | Selective SSM forward + VJP; the 12B reference path. | Per-backend Mamba2 chunked-scan kernel (Phase G). |
| `Empirical_Software_Agent` | ✅ | Tree-search agent skeleton; LLM-provider hooks are stubs. | Wire real LLM providers in your fork. |
| `kv_cache_serving` | ✅ | Real ops via `tessera.ops.{quantize_kv, dequantize_kv}` + `tessera.cache.KVCacheHandle(quantize_bits=...)` + `auto_evict=True`. | — |
| `long_context_attention` | ✅ | Sliding-window + retrieval-head pure-Python. | Optional: GPU sliding-window attention kernel (Phase G). |
| `mla` | ✅ smoke | Latent compress/expand + RoPE split/merge + `tessera.cache.LatentKVCacheHandle`. | FlashMLA absorb-K kernel (Phase G). |
| `power_retention` | ❌ stub | — | Retention/power attention op (CUDA kernel sketch only). |
| `rlvr_reasoning_suite` | ✅ | GRPO/RLVR rollout batching + reward accounting. | — |
| `gumiho` | ✅ | Hybrid speculative decoding (Gumiho, ICML'25): serial 2-layer Transformer + 5 parallel MLP heads + Full Tree Attention; draft+verify dense math on `@jit(target="apple_gpu"/"apple_cpu")`, acceptance/advance via `tessera.speculative`. | Single-kernel `@jit` of the whole loop (Phase G). |

## Naming changes (decoration / compilation)

| Phantom (in examples) | Today's API | Notes |
|------------------------|-------------|-------|
| `@tessera.function` | `@tessera.jit` | The single decorator name. Optional `target=...` selects backend. |
| `@ts.compile(backend="...")` | `@tessera.jit(target="...")` | Valid string targets: `"x86"`, `"rocm"`, `"metalium"`, `"apple_cpu"`, `"apple_gpu"`. Use `GPUTargetProfile` for SM-specific config. |
| `tessera.autodiff` (decorator form) | `tessera.autodiff.reverse(fn)` | `reverse` is the canonical wrapper. See `docs/spec/AUTODIFF_SPEC.md`. |
| `tessera.autocast` (top-level) | `tessera.autodiff.autocast("fp16")` | Lives under autodiff. Accepts `fp16`, `bf16`, `fp32`, `fp64`, `fp8_e4m3`, `fp8_e5m2`. |
| `tessera.checkpoint` (top-level context) | `tessera.autodiff.rematerialize()` or `tessera.autodiff.checkpoint(fn)` | Phase F2 — activation checkpointing. |

```python
# Phantom (examples sketch):
@tessera.function
@ts.compile(backend="cuda")
def step(x):
    with tessera.autocast("fp16"):
        return model(x)

# Today's API:
@tessera.jit(target="apple_gpu")  # or "x86" / GPUTargetProfile(isa=ISA.SM_90)
def step(x):
    with tessera.autodiff.autocast("fp16"):
        return model(x)
```

## `tessera.nn.*` — stateful layers

Tier 1.1 + Phase A4 + Phase B1/C1/D4/H1/H2 closed Theme 1. Every phantom
layer the examples reference is now real.

| Phantom | Today's API | Notes |
|---------|-------------|-------|
| `tessera.nn.Module` | `tessera.nn.Module` | Real base class. `register_buffer(name, value, persistent=True)` for non-trainable named tensors. |
| `tessera.nn.Parameter` | `tessera.nn.Parameter` | Wraps a `DistributedArray`. `.grad` is populated by the autodiff tape when on the gradient path. |
| `tessera.nn.Buffer` | `tessera.nn.Buffer` (Phase B1) | Non-trainable named tensor; rides alongside parameters in `state_dict()`. |
| `tessera.nn.Sequential / ModuleList / ModuleDict` | Same | Real containers. |
| `tessera.nn.Linear / MLP / MultiHeadAttention / MultiHeadCrossAttention / RMSNorm / LayerNorm / Embedding / Dropout / RotaryEmbedding / CastedLinear / CastedEmbedding / SiLU / Sigmoid / GELU / ReLU / Tanh / Identity / CrossEntropyLoss / BatchNorm1d / DynamicDepthwiseConv1d / Conv2d (NHWC) / Conv2dNCHW / LSTMCell / LSTM / KVCache` | Same | All real. State_dict round-trip + `train`/`eval` + `Module.to(dtype)` + `parameters()` / `named_parameters()` / `buffers()` / `named_buffers()` all work. |
| `tessera.nn.utils.clip_grad_norm_` | Same | Real. `tessera.nn.utils.clip_grad_norm_(module.parameters(), max_norm)`. |
| `tessera.nn.functional` | `tessera.nn` (functional surface) | `tessera.nn.functional` is a self-alias of `tessera.nn` so functional callsites (`F.linear`, `F.swiglu`, `F.multi_head_attention`) resolve. |
| `tessera.nn.flash_attention` | `tessera.ops.flash_attn` (or `tessera.nn.flash_attention` alias) | Same op; the `nn.flash_attention` form is a backward-compat alias. |

## `tessera.autodiff.*` — reverse-mode autodiff

```python
import tessera as ts

with ts.autodiff.tape() as t:
    y = model(x)
    loss = ts.ops.reduce((y - target) ** 2, op="sum")
    t.backward(loss)
# `Parameter.grad` is now populated for every Parameter on the path.
```

Or via the convenience wrapper (returns `(loss, grads_dict)`):

```python
loss_fn = ts.autodiff.reverse(lambda model, x: ts.ops.reduce((model(x) - target) ** 2, op="sum"))
loss, grads = loss_fn(model, x)
```

22 + 4 (Theme 9) + 1 (moe Theme 2 follow-up) built-in VJPs cover linear
algebra, activations, norms, reductions, dropout, FFT family,
flash_attn, selective_ssm, gather/clip/masked_fill, and moe. Custom
ops register their VJP via the `custom_rule` decorator:

```python
@ts.autodiff.custom_rule("my_op")
def vjp_my_op(dout, *forward_inputs, **kwargs):
    ...
    return (dx_for_each_input,)
```

Mixed-precision: `tessera.autodiff.autocast("fp16" | "bf16" | "fp8_e4m3" | "fp8_e5m2")` plus `tessera.autodiff.GradScaler` for the standard fp32-master-copy pattern. Activation checkpointing: `with ts.autodiff.rematerialize():` or `ts.autodiff.checkpoint(fn)`.

See `docs/spec/AUTODIFF_SPEC.md` and Programming Guide Ch.7 for the
full design.

## Ops added in Themes 9 / 10 / 5 / 6 / Tier 2 follow-ups

These ops weren't in the v1 op catalog but are now real, so any
example that used them as phantoms can switch to `tessera.ops.*`
directly.

### Theme 9 — utility tensor ops

| Op | Use case |
|----|----------|
| `tessera.ops.arange(start, stop=None, step=1, dtype="fp32")` | 1-D range; mirrors `numpy.arange` |
| `tessera.ops.gather(x, indices, axis=0)` | `numpy.take`. VJP scatters via `np.add.at` (correct under repeated indices) |
| `tessera.ops.clip(x, min_val=None, max_val=None)` | Element-wise clamp. Either bound may be `None`. Straight-through gradient |
| `tessera.ops.masked_fill(x, mask, value=...)` | Replace `x` where `mask` is True with `value`. Used by attention masks, softmax `-inf` fill |

`min_val`/`max_val`/`value` are keyword-only because the autodiff tape
records non-tensor positionals as `_NON_ARRAY` (which would drop them
from the VJP signature). Always pass them by name.

### Theme 10 — fp8 quantize/dequantize + autocast extension

| Op | Use case |
|----|----------|
| `tessera.ops.quantize_fp8(x, format="e4m3", scale=None)` | Per-tensor symmetric fp8 quantization. Returns `(x_q_as_fp32, scale)`. `format` is `"e4m3"` (max 448) or `"e5m2"` (max 57344). Native cast via `ml_dtypes` when installed; pure-numpy mantissa-snap fallback otherwise. |
| `tessera.ops.dequantize_fp8(x_q, scale, format="e4m3")` | Pair-wise op so the IR layer can intercept (quantize → dequantize) for fusion / cancellation. |
| `tessera.autodiff.autocast("fp8_e4m3" \| "fp8_e5m2")` | Routes per-op input casts through `quantize_fp8`. |

Per-backend GPU lowering rules (Hopper tcgen05 fp8 mma, ROCm OCP fp8)
are deferred to Phase G — the Python op surface is the unblock today.

### Theme 5 — Multi-Latent Attention primitives

| Op | Use case |
|----|----------|
| `tessera.ops.latent_kv_compress(x, w_dkv)` | `c = x @ W_dkv`. Distinct op_name anchors the FlashMLA fusion. |
| `tessera.ops.latent_kv_expand_k(c, w_uk)` / `latent_kv_expand_v(c, w_uv)` | Expand cached latent back to K / V. Will be absorbed by Phase G FlashMLA pass. |
| `tessera.ops.rope_split(x, rope_dim=...)` | Split last dim into `(rope_part, no_rope_part)`. MLA's decoupled-RoPE pattern. |
| `tessera.ops.rope_merge(rope_part, no_rope_part)` | Inverse of `rope_split`. |
| `tessera.cache.LatentKVCacheHandle(latent_dim=..., max_seq=...)` | Paged latent storage. Same `append`/`read`/`evict_oldest`/`auto_evict` surface as `KVCacheHandle`, but stores `(seq, latent_dim)` instead of `(seq, num_heads, head_dim)`. |

### Theme 6 — Speculative decoding scheduler

| API | Use case |
|------|----------|
| `tessera.speculative.expand_tree(root_token, draft_tokens, branching, depth)` | Build a balanced draft tree. Returns `DraftTree(tokens, parents, paths, branching, depth)`. |
| `tessera.speculative.acceptance_probabilities(target_lp, draft_lp)` | Leviathan eq.1 ratio clipped to [0, 1]. |
| `tessera.speculative.batch_verify(target_log_probs, draft_log_probs, paths, rng=None)` | Returns `VerificationResult(acceptance_mask, accepted_path_idx, accepted_prefix_length, accepted_prefix)`. |
| `tessera.speculative.advance_kv(cache, accepted_prefix_length)` | Trim the KV cache to the accepted prefix. Works on both `KVCacheHandle` and `LatentKVCacheHandle`. |
| `tessera.speculative.SpeculativeStep(branching, depth)` | Convenience orchestrator that drives all four primitives in order. |

## Distributed training

| Phantom | Today's API |
|---------|-------------|
| Manual `all_reduce(grads)` | `tessera.distributed.DDP(module, mesh_axis="dp")` (Phase I1) |
| Manual sharded params | `tessera.distributed.FSDP(module, mesh_axis="dp", stage=2 or 3)` (Phase I2 v1 + deferred-items Item 3) |
| ZeRO-3 explicit naming | `tessera.distributed.ZeRO3(module, mesh_axis="dp")` — alias for `FSDP(stage=3)` |
| `tessera.optimizers.AdamW` | ✅ now real — stateful wrapper over `tessera.optim.adamw`; `tessera.optimizers.Adam` also available. |
| `tessera.distributions.{Normal, Beta}` | ✅ now real — `tessera.distributions.{Normal, Beta, kl_divergence}`. |

DDP/FSDP today run against `tessera.testing.mock_collective.MockRankGroup` (in-process simulator); real NCCL/RCCL bindings wait on Phase G's NVIDIA execution path.

## Common rewrite patterns

### Phantom: `tessera.nn.Module` with `tessera.optimizers.AdamW`

```python
# Phantom:
class MyModel(tessera.nn.Module): ...
model = MyModel()
optim = tessera.optimizers.AdamW(model.parameters(), lr=3e-4)

for batch in data:
    with tessera.autocast("fp16"):
        loss = compute_loss(model, batch)
    optim.zero_grad()
    loss.backward()
    optim.step()
```

```python
# Today's API:
import tessera as ts

class MyModel(ts.nn.Module): ...
model = MyModel()

for batch in data:
    with ts.autodiff.tape() as t:
        with ts.autodiff.autocast("fp16"):
            loss = compute_loss(model, batch)
        t.backward(loss)
    # Manual Adam step until AdamW Module ships:
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        # Maintain (m, v, step) per parameter externally for now.
        new_p, new_m, new_v = ts.ops.adam(
            p._data._data, p.grad.numpy(), m_state[name], v_state[name],
            lr=3e-4, step=step,
        )
        p._data._data[:] = new_p
        m_state[name], v_state[name] = new_m, new_v
        p.grad = None  # zero grad
    step += 1
```

### Phantom: `tessera.gather(x, indices)` / `tessera.clip(x, lo, hi)`

```python
# Phantom (top-level functions):
y = tessera.gather(x, indices)
z = tessera.clip(x, -1.0, 1.0)
m = tessera.masked_fill(x, mask, -float("inf"))

# Today's API (Theme 9 — under tessera.ops):
y = ts.ops.gather(x, indices, axis=0)
z = ts.ops.clip(x, min_val=-1.0, max_val=1.0)
m = ts.ops.masked_fill(x, mask, value=-float("inf"))
```

### Phantom: `tessera.einsum(...)`

```python
# Phantom: arbitrary einsum
y = tessera.einsum("bij,bjk->bik", a, b)

# Today's API: ts.ops.einsum supports the full numpy einsum spec
y = ts.ops.einsum("bij,bjk->bik", a, b)
```

### Pattern: MLA-style attention

```python
import tessera as ts

# Compress hidden state to latent, cache only the latent
c = ts.ops.latent_kv_compress(hidden, W_dkv)            # (B, S, latent_dim)
cache.append(c.reshape(-1, latent_dim))                  # LatentKVCacheHandle

# At attention time: expand back to K/V (will be absorb-fused on Hopper)
c_full = cache.read(0, current_seq)
K = ts.ops.latent_kv_expand_k(c_full, W_uk)              # (current_seq, num_heads * head_dim)
V = ts.ops.latent_kv_expand_v(c_full, W_uv)

# Decoupled-RoPE pattern: positional encoding only on a slice
rope_part, no_rope_part = ts.ops.rope_split(query, rope_dim=64)
rope_part = ts.ops.rope(rope_part, theta)
query = ts.ops.rope_merge(rope_part, no_rope_part)
```

### Pattern: speculative decoding loop

```python
import tessera as ts

step = ts.speculative.SpeculativeStep(branching=2, depth=3)
rng = np.random.default_rng(seed)

while not done:
    # 1. Get draft proposals (one row per level, branching entries each).
    draft_tokens, draft_log_probs = draft_model.generate_tree(prefix, branching=2, depth=3)

    # 2. Run target on the full tree (one batched forward).
    target_log_probs = target_model.batch_eval(prefix, draft_tokens)

    # 3. Build tree → verify → trim cache (one step).
    result = step.run(
        root_token=prefix[-1],
        draft_tokens=draft_tokens,
        target_log_probs=target_log_probs,
        draft_log_probs=draft_log_probs,
        cache=cache,
        cache_pre_seq=cache_pre_seq,
        rng=rng,
    )

    # 4. Append the accepted prefix to the running output.
    output.extend(result.accepted_prefix.tolist())
    cache_pre_seq = cache.current_seq
    if result.accepted_prefix_length == 0:
        done = stop_criterion(output)
```

## Deferred-items plan landings (2026-05-09)

The following surfaces, previously tracked as "Deliberately out of
scope" in `docs/audit/coverage/COVERAGE_AUDIT.md`, are now
real and exercise-able from `examples/advanced/`. See
`docs/audit/roadmap/ROADMAP_AUDIT.md` for design rationale.

### `tessera.distributions.*`

```python
import tessera as ts
p = ts.distributions.Normal(loc=0.0, scale=1.0)
q = ts.distributions.Beta(alpha=2.0, beta=5.0)
samples = p.sample((4, 8), seed=42)        # reproducible
log_prob = p.log_prob(samples)
kl = ts.distributions.kl_divergence(p, q)  # closed form / MC fallback
```

### fp6 / fp4 / nvfp4 + autocast

```python
ops.quantize_fp6(x, format="e3m2")     # max ±28; range-favored
ops.quantize_fp4(x, format="e2m1")     # Blackwell hardware grid
ops.quantize_nvfp4(x, block_size=16)   # block-scaled fp4

with ts.autodiff.autocast("fp6_e3m2"):
    y = ts.ops.matmul(A, B)            # boundary cast through quantize_fp6
```

### ZeRO stage 3

```python
fsdp = ts.distributed.FSDP(model, stage=3)  # parameter-sharding flag
zero3 = ts.distributed.ZeRO3(model)         # DeepSpeed-style alias
assert fsdp.zero_config.partition_parameters
```

### Higher-order autodiff + JAX-style transforms

```python
g = ts.autodiff.grad(loss_fn)              # JAX-style
hvp_v = ts.autodiff.hvp(loss_fn, x, v)     # FD-based HVP

per_sample = ts.autodiff.vmap(g)           # batched gradients
J_rev = ts.autodiff.jacrev(fn)(x)          # reverse-mode Jacobian
J_fwd = ts.autodiff.jacfwd(fn)(x)          # forward-mode Jacobian
```

The `vmap(grad(fn))` per-sample-gradient pattern is the canonical
"differential privacy / individual influence functions" idiom from
JAX — it works as written.

## What's still phantom (small, tracked)

| Item | Status |
|------|--------|
| Hopper `tcgen05.fp8.mma` / `cvt.fp4` / `cvt.fp6` lowering | Phase G — needs real H100/Blackwell hardware. |
| ROCm OCP fp6/fp4/fp8 mfma rules | Phase G. |
| Native FlashMLA target kernel (Hopper / Blackwell absorb-K) | Phase G. |
| Mamba2 chunked-scan target kernels (NVIDIA / ROCm) | Phase G — Python op + VJP ship today. |
| Real NCCL all-gather of parameters in `FSDP(stage=3)` / `ZeRO3` | Phase G. |
| `power_retention` op (retention attention) | Out of scope. The example folder is a CUDA kernel sketch with no current users. |

## Cross-references

- [`docs/CANONICAL_API.md`](CANONICAL_API.md) — current public surface
- [`docs/audit/coverage/COVERAGE_AUDIT.md`](audit/advanced_examples_capability_gap.md) — what's already shipped vs. what's still open, per theme
- [`docs/audit/roadmap/ROADMAP_AUDIT.md`](audit/execution_roadmap.md) — sequenced execution plan; Phase G is the long pole
- [`docs/spec/AUTODIFF_SPEC.md`](spec/AUTODIFF_SPEC.md) — reverse-mode autodiff design
- [`docs/spec/PYTHON_API_SPEC.md`](spec/PYTHON_API_SPEC.md) — full op-by-op spec (the row table this guide pulls from)
- [`examples/advanced/README.md`](../examples/advanced/README.md) — honest per-example status
