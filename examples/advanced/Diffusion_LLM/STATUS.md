# Status: `runnable`

Tracked by `python/tessera/compiler/examples_manifest.py`.

`gpu_denoise.py` is the maintained, **torch-free, runnable** MDLM
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

Diffusion LMs are a workload where on-device sampling matters — every
denoising step samples, so the Gumbel sampler is on the critical path. Covered by
`tests/unit/test_example_diffusion_mdlm.py`.

---

## Retired and optional material

The obsolete top-level `tessera_diffusion_llm.py` monolith was removed because
it used deleted decorators and invalid annotations. The supporting
torch-dependent package, tests, and MLIR sketches are preserved under
`archive/examples/advanced/Diffusion_LLM/`; they are not runtime evidence for
Tessera.
