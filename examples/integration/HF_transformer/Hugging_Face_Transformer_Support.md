# Hugging Face-shaped transformer support

This example demonstrates a deliberately small compatibility boundary rather
than a replacement for the Hugging Face `transformers` package.

The runnable script provides BERT-, GPT-2-, and Llama-shaped configuration
dataclasses. Those configs select either bidirectional or causal
`tessera.ops.flash_attn` through the canonical `@tessera.jit` surface. Each path
is checked against a NumPy scaled-dot-product attention oracle and must produce
non-empty Graph, Schedule, Tile, and Target IR.

```bash
python3 examples/integration/HF_transformer/tessera_huggingface_transformers.py
```

Not implemented here:

- loading or saving Hugging Face checkpoints;
- tokenizer and dataset integration;
- Trainer or generation APIs;
- distributed training, quantization, or serving; and
- hardware-specific performance claims.

Those capabilities should be added only with their own executable tests and
manifest evidence.
