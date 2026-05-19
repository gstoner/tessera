# Status: `scaffold`

Tracked by `python/tessera/compiler/examples_manifest.py`.

This directory is a **research sketch**, not a runnable example today.

## Why

`tessera_huggingface_transformers.py` references several APIs that do
not exist on the canonical Tessera surface:

* `from tessera import function, Module, compile, kernel` — only
  `tessera.kernel` and `tessera.nn.Module` exist; `function` and the
  module-level `compile` were removed when `@tessera.jit` became
  canonical (see `docs/CANONICAL_API.md`).

Importing the file therefore fails immediately:

```
ImportError: cannot import name 'function' from 'tessera'
```

## Path forward

Rewrite against the canonical surface:

* `@tessera.jit` instead of `@tessera.function` / `compile(mode=...)`.
* `tessera.nn.Module` for layers.
* `tessera.nn.functional` (`F.scaled_dot_product_attention`, etc.) for
  the building blocks.

The Hugging Face *interface* (`PretrainedConfig`, `from_pretrained`,
tokenizer integration) is reasonable to keep; only the Tessera
bindings need updating.

Until that work lands, this scaffold ships unchanged.
