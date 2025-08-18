# Tessera Examples & Integration Guide

This document provides **informative worked examples**, lowering flows, numerics checklists, and integration paths for Tessera.

---

## Chapter 6: Worked Operator Examples

(FlashAttention, Spectral Denoising, MoR, MoE, Test-Time Compute, SFT + RLHF)

Each example includes:
- **Graph IR**
- **Tile IR lowering**
- **ABI mapping**
- **Numerics/determinism checklist**

---

## Appendix E: Integration Examples

### E.1 GPT-OSS-120B in Tessera

- End-to-end operator graph for inference
- KVCache, attention, rotary embeddings
- Speculative decoding

### E.2 Hugging Face Transformers Importer

- Mapping from `transformers` objects → Tessera operators
- Auto-import tooling design

### E.3 Speculative Decoding Loop

- Draft/Verify/Refine operators in Tessera

### E.4 SFT + RLHF Training Example

- Cross-entropy SFT loss
- PPO-based RLHF reward loop

### E.5 Minimal C ABI Dispatch

- Tessera ABI calls for model run

### E.6 Hugging Face Baseline (PyTorch/Transformers)

- Inference + generation code
- Speculative decoding code
- SFT + RLHF training sketch
- HF ↔ Tessera mapping cheatsheet

---

## Numerics & Determinism Checklists

- **FlashAttention**: stable softmax, FP16/BF16 accumulators, FP8 scaling rules.
- **Spectral Denoising**: truncation thresholds, reproducible SVD ordering.
- **MoR**: bounded recursion depth, deterministic bucket assignment.
- **MoE**: reproducible top-k gating, consistent random seeds.
- **Test-time compute**: rollback reproducibility, KVCache consistency.
- **SFT/RLHF**: CE loss equality, PPO reward reproducibility.

---

*This Examples Doc is informative, not normative, and complements the Base Spec.*
