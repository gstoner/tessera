---
classification: Architecture / Workload
authority: DFlash workload design and implementation guide
last_updated: 2026-07-13
---

# DFlash — block-diffusion speculative decoding

DFlash ([z-lab/dflash](https://github.com/z-lab/dflash), arXiv:2602.06036) is a
lightweight **block-diffusion draft model** for speculative decoding: instead of
an autoregressive drafter that emits one token per forward, it predicts a whole
block of `block_size − 1` tokens in a **single parallel forward** using mask
tokens and block-structured attention, conditioned on hidden features injected
from the target model. The target then verifies the block in one forward.

This page documents Tessera's implementation: the `attn_bias` compiler substrate
it rides on, the draft model, and the serving surface. Everything is grounded in
an independent numpy port of the DFlash MLX reference (`dflash/model_mlx.py`) —
the design was pinned from that source, not paraphrased.

> **Status.** Python reference; the attention core runs on the Apple GPU
> `metal_runtime` lane. The gold-standard correctness invariant — *greedy
> speculative-decode output equals greedy autoregressive decode* — is proven
> against the MLX reference. Two gates are external: numerical parity vs a
> downloaded `z-lab/*-DFlash` checkpoint (network), and a single fully-jitted GPU
> draft artifact (needs a GPU gather/embedding op).

---

## 1. The `attn_bias` substrate (compiler)

DFlash's structured masks (and general additive attention masks) are expressed
through an optional additive **`attn_bias`** operand on `flash_attn`:

```
O = softmax(scale · Q·Kᵀ + attn_bias) · V
```

`attn_bias` has shape `(B, Sq, Sk)`, broadcastable from `(1, Sq, Sk)`. The path
is end-to-end:

| Layer | What landed |
|---|---|
| **Graph IR** | `FlashAttnOp` gains `Optional<TensorType>:$attn_bias` + `AttrSizedOperandSegments`; verifier pins rank-3 / `Sq` / broadcastable batch. |
| **Apple GPU** | Tile→Apple lowering routes the bias operand to MPSGraph symbols `tessera_apple_gpu_flash_attn_bias_{f32,f16,bf16}` (transpose K → matmul → scale → +bias → softmax → PV) + a non-Darwin reference. Causal + bias applies both masks. Broadcast bias falls back to the reference. |
| **Python** | `ops.flash_attn(Q, K, V, attn_bias=…)` (eager + `@jit(target="apple_gpu")` → `metal_runtime`); CPU reference; VJP returns `dbias` when the bias is a positional (recorded) input. |

Validated on Metal vs numpy at ≈3e-7 (fp32). It's a general feature: any
structured additive mask now flows through `flash_attn`, not just DFlash.

---

## 2. Modules

| Module | Surface |
|---|---|
| `tessera.nn.functional.block_diffusion_attention` | One DFlash attention layer: QK-norm, KV injection (`concat([context_KV, proposal_KV])`), GQA, rope offsets, full (bidirectional) or sliding-window-via-`attn_bias`. Heads fold into batch → the rank-3 `flash_attn` lane. `attention_fn=` routes the core onto a backend. |
| `tessera.dflash` | The draft: config + weights, `target_feature_projection` (multi-layer tap `fc`), decoder layer, `dflash_draft_forward`(`_cached`), `DraftKVCache` / `RotatingDraftKVCache`, samplers, verification (`dflash_linear_verify`, `dflash_speculative_verify`), `dflash_step` / `dflash_generate` / `dflash_generate_cached`, the position-weighted training loss, `HiddenStateTap`, `DFlashDraft` (`nn.Module`), and the `{apple_gpu,rocm}_attention_fn` backend seams. |
| `tessera.dflash_reference` | `ReferenceDecoderLM` — a numpy causal target (MHA + SwiGLU, rope, multi-layer hidden tap) with a stateless `forward` (greedy-AR ground truth) and a stateful KV cache with `step` / `rollback`. |
| `tessera.dflash_io` | Dependency-free safetensors reader/writer + HF state-dict ↔ `DFlashWeights` mapping; `load_dflash_weights` reads a `z-lab/*-DFlash` draft checkpoint. |
| `tessera.dflash_serve` | `dflash_generate_text` (string-in/out via any tokenizer) and `DFlashScheduler`. |

---

## 3. Quick start (numpy reference)

```python
import numpy as np
from tessera import dflash as D
from tessera import dflash_reference as R
from tessera.dflash_serve import DFlashScheduler

# A target model + a DFlash draft (random weights here; load real ones with
# tessera.dflash_io.load_dflash_weights).
lm_cfg = R.DecoderLMConfig(vocab_size=256, hidden_size=64, num_layers=4,
                           num_heads=8, head_dim=8, intermediate_size=128,
                           target_layer_ids=(0, 1, 2))
target = R.random_decoder_lm(lm_cfg, np.random.default_rng(0))
cfg, weights = ...  # DFlashConfig + DFlashWeights matching the target

sched = DFlashScheduler(weights, cfg, target)
tokens = sched.generate([3, 1, 4, 1, 5], max_new_tokens=64)            # greedy
tokens = sched.generate([3, 1, 4, 1, 5], max_new_tokens=64,
                        temperature=0.8, top_p=0.9,
                        rng=np.random.default_rng(7))                  # sampled
```

Run the draft's attention on the Apple GPU by passing the seam through:

```python
logits = D.dflash_draft_forward(block, target_hidden, weights, cfg,
                                rope_fn=D.make_rope(cfg.head_dim),
                                attention_fn=D.apple_gpu_attention_fn)
```

---

## 4. What's proven

| Property | Test |
|---|---|
| `attn_bias` on Metal == numpy (incl. causal+bias, broadcast fallback) | `test_flash_attn_bias_semantics.py`, `test_dflash_apple_gpu_e2e.py` |
| Per-layer attention vs MLX reference (full / GQA / sliding / rope / cache) | `test_block_diffusion_attention.py` |
| Draft forward vs MLX reference | `test_dflash_draft.py` |
| Cached draft == non-cached(full context); cache accumulates | `test_dflash_cached_sampling.py` |
| Rejection sampling marginal == target distribution (Monte Carlo) | `test_dflash_cached_sampling.py` |
| Stateful KV cache == stateless forward; rollback exact | `test_dflash_reference_target.py` |
| **Greedy spec-decode == greedy autoregressive decode** | `test_dflash_draft.py`, `test_dflash_reference_target.py` |
| Training loss gradient vs finite differences | `test_dflash_train_io.py` |
| Checkpoint round-trip → identical logits | `test_dflash_train_io.py` |
| `nn.Module` / rotating cache / tokenizer / scheduler | `test_dflash_module_serve.py` |

The spec-decode invariant is the key guarantee: speculation changes only speed,
never the output. It holds regardless of draft quality because the target
verification corrects every divergence.

---

## 5. Backend plan (ROCm / CUDA / x86)

DFlash rides `flash_attn` + `attn_bias`, so its non-Apple backend seams are
sequenced with the rest of the attention family in
[`attention-family.md`](attention-family.md). DFlash
*always* passes an `attn_bias`, so each backend seam needs its flash lane to
accept the bias operand first.

- **ROCm** — ✅ **`tessera.dflash.rocm_attention_fn`** runs the draft attention on
  the gfx1151 WMMA flash lane (after the #328 additive-bias landing). f16 storage
  (f32 softmax/accumulate); falls back to the numpy reference off-silicon or when
  `head_dim` isn't a multiple of 16. Proven on real gfx1151: block-attention
  parity to f16 and the whole-draft greedy tokens identical to the numpy path.
- **x86** — ✅ **`tessera.dflash.x86_attention_fn`** runs the draft attention on
  the AVX-512 flash lane. f32-native, no head-dim constraint, so it matches the
  numpy reference to f32 epsilon (block-attention parity + whole-draft greedy
  tokens identical); falls back when the x86 elementwise lib isn't built.
- **CUDA** — pending the emit/ FA bias operand (Phase 1 CUDA).

## 6. External gates (not yet closed)

- **Real-checkpoint numerical parity** — the safetensors loader is built and
  round-trip-proven; comparing against a downloaded `z-lab/*-DFlash` checkpoint
  needs network access to Hugging Face.
- **Fully-jitted GPU draft** — the attention core runs on Metal via the
  `attention_fn` seam; folding the whole draft (embedding gather, `fc`, MLP, LM
  head) into one `@jit(target="apple_gpu")` artifact needs a GPU gather/embedding
  op. The matmul-heavy pieces are GPU-eligible today via `ops.*`.

Audit trail: [`docs/audit/MASTER_AUDIT.md`](../../audit/MASTER_AUDIT.md) (Domain
Tracks) and [`docs/audit/domain/DOMAIN_AUDIT.md`](../../audit/domain/DOMAIN_AUDIT.md).
