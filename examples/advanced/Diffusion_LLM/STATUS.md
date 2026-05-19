# Status: `scaffold`

Tracked by `python/tessera/compiler/examples_manifest.py`.

This directory is a **research sketch**, not a runnable example today.

## Why

`tessera_diffusion_llm.py` references several non-existent Tessera APIs:

* `ts.compile(mode="training")` — Tessera's compile decorator is
  `@tessera.jit`, not `ts.compile`, and it does not take a `mode`
  argument.
* `ts.randint(...)` — the canonical RNG surface is `tessera.rng.randint`
  off an `RNGKey`, not a module-level `randint`.
* `Tensor[]` — empty-bracket return type annotation is invalid Python
  syntax (`SyntaxError` at line 488).

The supporting package `tessera_diffusion_llm/` also `import torch` in
`models/transformer.py` and several kernel modules. Per Architecture
Decision #23, Tessera is torch-free at runtime; this sketch would need
to be rewritten against the canonical Tessera surface (`@tessera.jit`,
`tessera.nn.Module`, `tessera.rng.*`) before it can be classified as
`runnable`.

## Path forward

Either:

1. Rewrite `tessera_diffusion_llm.py` against the canonical Tessera
   surface and drop the torch dependency from the supporting package
   (large effort, ~weeks of focused work).
2. Or delete this directory and move it under `examples/archive/` with
   a banner pointing at the equivalent S11/S15 reference primitives in
   `python/tessera/`.

Until that decision lands, this scaffold ships unchanged. The
manifest's drift gate will flag any README that claims it is runnable.
