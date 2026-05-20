# Status: `scaffold`

Tracked by `python/tessera/compiler/benchmarks_manifest.py`.

This directory is a **research sketch**, not a runnable benchmark today.

## Why

`tessera_deepscholar_model.py` imports several non-existent symbols:

* `from tessera.models import HierarchicalReasoningModel` — there is
  no `tessera.models` submodule on the canonical surface.
* `from tessera.attention import FlashMLA, CitationAwareAttention` —
  no `tessera.attention` submodule; the equivalent live API is
  `tessera.nn.MultiHeadAttention` + the FlashMLA Tile IR path.
* `@ts.application` decorator — does not exist; the closest live API
  is `@tessera.jit`.

Importing the file therefore fails immediately:

```
ModuleNotFoundError: No module named 'tessera.models'
```

The DeepScholar evaluation harness itself (LOTUS framework integration)
is a reasonable target; only the Tessera bindings need updating.

## Path forward

Either:

1. Rewrite against the canonical surface (`@tessera.jit`, 
   `tessera.nn.MultiHeadAttention`, the existing FlashMLA Tile IR
   path) and promote the row to `runnable` once a smoke command exists.
2. Move to `benchmarks/archive/` with a deprecation note.

Until that decision lands, this scaffold ships unchanged.  The
manifest's drift gate keeps the README honest.
