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
# Multi-step decode: distill the draft, then measure acceptance + speedup:
python3 examples/advanced/gumiho/demo.py --mode decode --target apple_gpu

# One validated speculative step (backend vs numpy cross-check):
python3 examples/advanced/gumiho/demo.py --mode step --target apple_gpu

# Accelerate (CPU) path, or the pure numpy reference:
python3 examples/advanced/gumiho/demo.py --mode decode --target apple_cpu
python3 examples/advanced/gumiho/demo.py --mode decode --target numpy
```

`--mode step` output (backend cross-checked against numpy):

```
backend=metal | draft=2serial+5parallel=7 | FTA paths=8 tree_nodes=27 |
accepted=1 [31] | kv 4->5 | match_ref=True (max_logp_err=1.60e-06) validated=True
```

`--mode decode` output (distillation → measured speedup):

```
[untrained] backend=metal ... mean_accepted=0.86 tokens/step=1.86 speedup=1.86x
[trained]   backend=metal ... mean_accepted=3.62 tokens/step=4.62 speedup=4.62x
distillation lifted tokens/target-pass 1.86 -> 4.62 (2.49x), vanilla = 1.00
```

## Distillation + the speculative-decoding win

Out of the box the draft heads are random, so the target accepts ~1 token/step.
`--mode decode` first **distills** the heads against the target (the
target is deterministic, so its continuation is a clean supervised signal) and
then runs a **multi-step decode**, reporting **mean accepted length** and
**tokens per target pass** (= speedup vs. vanilla autoregressive decode, which
commits 1 token/pass).

Training uses **Tessera's own autograd + optimizer**: the draft forward is
written in `tessera.ops`, `tessera.autodiff.grad` differentiates it, and
`tessera.optim.adam` applies the updates (the S10/S11 standalone-compiler
training surface). Two details keep it exact and tractable:

- The serial head runs at **T=1**, so its self-attention is degenerate
  (`softmax` of one score = 1) and reduces to a value projection — we train the
  value slice of `Wqkv` and write it back, leaving inference untouched.
- The distillation target is the target's **distribution** (soft labels), not
  its argmax. Matching the distribution makes the Leviathan acceptance ratio
  `p_target/p_draft ≈ 1`, so correct tokens are actually accepted — argmax-only
  distillation over-rejects and barely moves the accepted length.

## What this example is

A complete, validated speculative-decoding pipeline on Apple Silicon: a faithful
Gumiho draft (serial Transformer + parallel MLP heads + Full Tree Attention), its
draft and tree-verification dense math executing on the Tessera Apple GPU/CPU
backend, a distillation loop driven by Tessera's own autograd and optimizer, and
a multi-step decode that **measures** the acceptance length and per-target-pass
speedup. Every dense kernel is cross-checked against a float64 numpy reference,
and the decode loop is genuine speculative decoding (Leviathan acceptance +
`advance_kv`). It exercises a real slice of the stack end to end —
embeddings, projections, attention with a tree mask, SwiGLU, `tessera.autodiff`,
`tessera.optim`, `tessera.speculative`, and the Apple runtime.

**Scope.** The models are small seeded synthetics and the draft is distilled on
the target's own generation trajectories, then measured on the same prompts, so
the reported speedup demonstrates the *mechanism* (distillation → acceptance↑ →
speedup) rather than generalization to unseen text. A random target has no
learnable structure to generalize across contexts; a production Gumiho trains a
real draft on a real corpus. Swap in a trained target + corpus and the same code
path measures real-workload acceptance.

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
