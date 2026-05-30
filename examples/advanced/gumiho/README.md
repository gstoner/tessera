# Gumiho — Hybrid Speculative Decoding (Apple backend)

A Tessera port of **Gumiho: A Hybrid Architecture to Prioritize Early Tokens in
Speculative Decoding** (ICML'25, [arXiv:2503.10135](https://arxiv.org/abs/2503.10135),
[AMD-AGI/Gumiho](https://github.com/AMD-AGI/Gumiho)).

Gumiho's insight: **early draft tokens matter more** — a rejection early in a
draft path discards everything after it, so accuracy is worth more up front and
speed is worth more later. It therefore uses a **hybrid** draft:

- a **serial 2-layer Transformer head** (EAGLE-style) for the first 2 tokens,
  generated autoregressively from `concat(target hidden, token embedding)`;
- **5 parallel MLP heads** (`FC → ReLU → FC`, Medusa-style) for the next 5
  tokens, predicted independently;
- **Full Tree Attention (FTA)** — because the parallel tokens are independent,
  any combination forms a candidate path; the top-8 scoring paths are folded
  into a prefix trie and the target verifies all of them in **one tree-masked
  forward pass**.

Unlike the (now-archived) `speculative_decoding` scheduler toy, this example
runs the real draft + verification **dense math on the Apple compiler backend**
(`@tessera.jit(target="apple_gpu" | "apple_cpu")` composing
`matmul / linear_general / rmsnorm / silu_mul / relu / softmax`), then reuses
`tessera.speculative` for the Leviathan acceptance check and the KV advance.

## What runs where

| Stage | Compute | Backend |
|---|---|---|
| Target model (RMSNorm + MHA + SwiGLU + LM head) | `linear_general`, `matmul`, `rmsnorm`, `silu_mul`, `softmax` | Apple GPU/CPU |
| Serial head (2-layer Transformer, autoregressive) | same decoder-layer ops | Apple GPU/CPU |
| Parallel heads (5× `FC → ReLU → FC`) | `linear_general`, `relu` | Apple GPU/CPU |
| FTA tree-attention verify (1 pass, additive ancestor mask) | batched `matmul` + masked `softmax` | Apple GPU/CPU |
| Acceptance (Leviathan rule) + KV advance | `tessera.speculative.batch_verify` / `advance_kv` | host |

Everything degrades to a float64 numpy backend off Apple Silicon, and the demo
**runs both paths and cross-checks them**, so it proves the backend executes
Gumiho correctly (not just that it prints a schedule).

## Quick Start

```bash
# Real Apple GPU path (Metal); falls back to numpy off Darwin:
python3 examples/advanced/gumiho/demo.py --target apple_gpu

# Accelerate (CPU) path, or the pure numpy reference:
python3 examples/advanced/gumiho/demo.py --target apple_cpu
python3 examples/advanced/gumiho/demo.py --target numpy
```

Sample output:

```
backend=metal | draft=2serial+5parallel=7 | FTA paths=8 tree_nodes=27 |
accepted=1 [31] | kv 4->5 | match_ref=True (max_logp_err=1.60e-06) validated=True
```

## Notes

- **Weights are tiny seeded synthetics** — the example validates the
  *architecture* and its *backend execution*, not pretrained acceptance quality.
  With untrained weights the accepted prefix is short (often 1 token); a trained
  Gumiho draft accepts several. The win this example proves is structural and
  numerical, not throughput.
- The single-kernel `@tessera.jit` lowering of the *whole* speculative loop
  (control flow into one dispatched kernel) remains a Phase-G item; today the
  loop is host-orchestrated and each dense op lowers to the Apple backend.

## Tessera Mapping

- Graph IR: draft branches as a bounded prefix trie; ancestor mask as an
  additive attention bias.
- Schedule IR: batch the 8 candidate paths into one tree-masked target pass.
- Tile IR: compact the accepted prefix and roll KV pages forward.
- Runtime: tune serial/parallel split, `top-paths`, and acceptance threshold per
  model latency and acceptance rate.
