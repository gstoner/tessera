# Diffusion LLM example

The maintained entry point is `gpu_denoise.py`, a small masked-diffusion
language-model denoising loop. It is torch-free, validates its transformer
backbone and Gumbel sampler against NumPy, and checks deterministic iterative
unmasking.

```bash
PYTHONPATH=python python3 examples/advanced/Diffusion_LLM/gpu_denoise.py
```

On Apple Silicon, the demo uses the available Metal `bmm`, row-operation, and
Gumbel-sampling runtime paths. When Metal is unavailable it executes the NumPy
path, so that fallback is correctness evidence only and not Apple device
evidence.

## Optional research package

`archive/examples/advanced/Diffusion_LLM/` preserves the larger,
torch-dependent research package, its tests, and its MLIR sketches. It is not
part of Tessera's torch-free runtime contract.

The former top-level monolith was removed because it depended on deleted
Tessera decorators and invalid annotations. New executable coverage should be
added to `gpu_denoise.py` or as another explicit manifest entry rather than
reviving those APIs.
