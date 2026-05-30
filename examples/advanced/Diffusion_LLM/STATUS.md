# Status: `scaffold` (+ runnable GPU demo)

Tracked by `python/tessera/compiler/examples_manifest.py`.

The original `tessera_diffusion_llm/` package + `tessera_diffusion_llm.py` remain
a **research sketch** (torch-dependent, references non-existent APIs — see below).

**New:** `gpu_denoise.py` is a standalone, **torch-free, runnable** MDLM
(masked-diffusion LM) denoising demo on the canonical Tessera surface. It drives
the Apple GPU work that landed in the runtime — the bidirectional backbone
(RMSNorm + attention via device `bmm` → softmax → `bmm` + MLP through the device
`rowop`/`bmm` kernels) and **per-step token sampling via the GPU Gumbel-max
sampler** (`runtime._apple_gpu_gumbel_sample`) — through a full iterative
unmasking loop, cross-checked against numpy and deterministic by seed:

```bash
PYTHONPATH=python python examples/advanced/Diffusion_LLM/gpu_denoise.py
# OK diffusion mdlm: metal steps 6 all_unmasked True backbone==np True sampler==np True deterministic True
```

Diffusion LMs are the workload where on-device sampling matters most — every
denoising step samples, so the Gumbel sampler is on the critical path. Covered by
`tests/unit/test_example_diffusion_mdlm.py`.

---

## The original sketch (still torch-dependent)

This part is **not** runnable today.

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
2. Or delete this directory and move it under `archive/examples/` with
   a banner pointing at the equivalent S11/S15 reference primitives in
   `python/tessera/`.

Until that decision lands, this scaffold ships unchanged. The
manifest's drift gate will flag any README that claims it is runnable.
