# FlashMLA / Multi-Latent Attention -> Tessera

This example contains a current-compiler FlashMLA smoke path plus the original
design note for a full Hopper/Blackwell-style Multi-Latent Attention
implementation.

Contents:
- `mla/` — dependency-light NumPy MLA reference and Graph IR compiler smoke.
- `ir/flash_mla_tiny.mlir` — parser-valid current-dialect Graph IR tensor skeleton.
- `tests/smoke_random.py` — NumPy shape/cache smoke plus Graph -> Schedule -> Tile -> Apple Target IR artifact check.
- `flashmla_tessera.md` — full design note for native MLA kernels, paged latent KV cache, RoPE split, and weight absorption.

## Quick Start

From the repository root:

```bash
PYTHONPATH=python /Users/gregorystoner/venv/bin/python \
  examples/advanced/mla/tests/smoke_random.py

PATH="$PWD/build/tools/tessera-opt:/opt/homebrew/Cellar/llvm@21/21.1.8/bin:$PATH" \
  tessera-opt examples/advanced/mla/ir/flash_mla_tiny.mlir >/tmp/flash_mla_tiny.mlir
```

Expected smoke output:

```text
OK mla tiny: (2, 8, 64) kv_reduction 0.75 apple_cpu cpu_accelerate
```

## Current Compiler Contract

The current smoke intentionally separates two concerns:

- `mla.compiler_smoke` builds Graph IR with the Python object model and lowers it
  through Graph IR -> Schedule IR -> Tile IR -> Apple Target IR artifacts.
- `ir/flash_mla_tiny.mlir` uses quoted registered `tessera.*` ops so `tessera-opt`
  can parse and verify the checked-in textual fixture.

Today this is represented as a straight-line tensor skeleton:
Q down/up projection, KV down projection, latent RMSNorm, absorbed K/V
projection, confidence softmax, context matmul, output projection, final
RMSNorm. Native paged latent cache handles, RoPE split/merge, online softmax,
and target FlashMLA kernels remain the roadmap captured in
`flashmla_tessera.md`.
