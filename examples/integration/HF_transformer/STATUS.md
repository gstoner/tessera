# Status: `runnable`

Tracked by `python/tessera/compiler/examples_manifest.py`.

The maintained entry point is a small, runnable Hugging Face-shaped adapter.

## What it proves

`tessera_huggingface_transformers.py` now:

* models BERT-, GPT-2-, and Llama-shaped configuration objects;
* selects encoder or causal decoder attention through `@tessera.jit`;
* checks the result against a NumPy scaled-dot-product attention oracle; and
* requires non-empty Graph, Schedule, Tile, and Target IR artifacts.

Run it from the repository root:

```
python examples/integration/HF_transformer/tessera_huggingface_transformers.py
```

## Boundary

This is configuration and compiler-surface interoperability, not a drop-in
replacement for the external `transformers` package. It does not claim
`from_pretrained`, tokenizer, checkpoint-conversion, or device-performance
coverage.
