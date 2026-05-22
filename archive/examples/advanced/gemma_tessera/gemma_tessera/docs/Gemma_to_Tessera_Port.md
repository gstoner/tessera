<<<MERGE BEGIN: GEMMA_TO_TESSERA_PORT >>>
# Gemma → Tessera Port Plan (Starter)

This document outlines a pragmatic path to port **Gemma** (Gemma 1/2/3-class models) to the **Tessera Programming Model**.

## Scope
- Decoder-only Transformer with RoPE, RMSNorm, SwiGLU.
- GQA/MQA attention (num_kv_heads ≤ num_heads).
- KV-cache, sampling (top-k/p, temperature), and minimal tokenizer glue.

## Mapping (Gemma → Tessera)
| Component | Gemma (JAX/PyTorch) | Tessera Primitive |
|---|---|---|
| Token Embedding | `nn.Embedding` | `tessera.tensor.lookup` |
| RoPE | rotary on q/k | `tessera.rope.apply` |
| Attention (GQA) | SDPA / FlashAttn | `tessera.attention.flash` (tile=64x64x16) |
| Norm | RMSNorm | `tessera.norm.rms` |
| MLP | SwiGLU (2*ffn → split) | `tessera.mlp.swi_glu` |
| Output proj | Linear | `tessera.mma.proj` |

## IR & Lowering
1. **High-level**: `tessera.region` per decoder block; attach tile shapes + dtypes.
2. **Mid-level**: fuse QK^T + softmax + PV into `tessera.attention.flash` with causal mask.
3. **Target IR**: lower to GPU backends (Tile IR/NVVM/ROCm) with appropriate TMA/async copies.

## Incremental Milestones
- M0: Torch baseline passes tests (done in this starter).
- M1: Replace SDPA with Tessera FlashAttention kernel (TSR path).
- M2: Add KV-cache + paged attention.
- M3: Export/Import checkpoints via converter.
- M4: Full Target IR path + microbench.

## Notes
- Weights must be fetched via official gates (HF/Kaggle) under Gemma terms.
- Keep configs aligned to released checkpoints (2B/7B/…).
<<<MERGE END: GEMMA_TO_TESSERA_PORT >>>


## Update: Native Tessera Attention + Autotune Stub
- Added `tessera_gemma/kernels/native_attention_tessera.py` (tile-blocked, online-softmax).
- `TesseraAttention` autotunes block size on first use and caches per (S,D,device,dtype).
- Forward paths now include shape-typed checks via `utils/shapes.py`.


## Update v3: KV Cache + Paged Attention + Microbench
- Added `kernels/kv_cache_tessera.py` (paged KV cache).
- Added `kernels/native_attention_paged_tessera.py` using streaming online-softmax over KV pages.
- `TesseraAttention.forward(..., kv_cache, use_cache=True)` enables paged path.
- Microbench: `scripts/bench_attention_blocks.py` prints timing across block sizes and head dims.


## Update v4: PEFT (LoRA) for PyTorch
- Added `tessera_gemma/peft/lora.py` with LoRA wrappers for Linear + helpers (apply/save/load/merge/unmerge).
- Example training stub: `scripts/train_lora_stub.py`.
- Basic test: `tests/test_peft_lora.py`.


## Update v5: Targeted PEFT, QLoRA-sim, Paged Decode, Trainer Glue
- **PEFT**: regex-targeted adapters with per-module rank/alpha/dropout; multi-adapter composition with enable/disable and merge order.
- **QLoRA-sim**: Int4 (Q4_0) forward simulation wrappers (`QLinearSim`) + `apply_qlora_sim`.
- **KV-cache decode**: `utils/decoding.py::greedy_generate_paged` uses the paged attention path.
- **Trainer glue**: `scripts/train_lora_regex_qlora.py` with NVTX & CSV logging. Falls back to random data if HF datasets unavailable.


## Update v6: LR multipliers, Freezing, Eval PPL, Target-IR stubs
- **Per-adapter LR multipliers**: `param_groups_with_adapter_lrmult(base_lr, {name:mult})`.
- **Freezing policies**: `freeze_by_regex(model, patterns=('embed','norm'))`.
- **Eval harness**: `scripts/eval_ppl_local.py --text file.txt [--tokenizer_id google/gemma-2-2b-it]` logs PPL to CSV.
- **Target-IR lowering**: `mlir/gemma_target_ir_lowering.mlir` and `mlir/GemmaLoweringPass.cpp` stubs to guide a real pass.
