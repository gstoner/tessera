# FlashMLA / Multi-Latent Attention -> Tessera

This example contains a current-compiler FlashMLA smoke path plus the original
design note for a full Hopper/Blackwell-style Multi-Latent Attention
implementation.

Contents:
- `mla/` — dependency-light NumPy MLA reference, Graph IR compiler smoke, **and
  `gpu_decode.py` driving the shipped Apple GPU MLA decode surfaces**.
- `ir/flash_mla_tiny.mlir` — parser-valid current-dialect Graph IR tensor skeleton.
- `tests/smoke_random.py` — NumPy shape/cache smoke, Graph -> Schedule -> Tile ->
  Apple Target IR artifact check, **plus the GPU-decode demo validated vs numpy**.
- `flashmla_tessera.md` — full design note for native MLA kernels, paged latent KV cache, RoPE split, and weight absorption.

## Apple GPU MLA decode (shipped)

`mla.run_gpu_decode_demo(cfg)` exercises the MLA decode work now in the Tessera
runtime, driven from this example's config and cross-checked against a numpy
reference (it runs on the GPU on Apple Silicon, and falls back to numpy
elsewhere):

- **Weight absorption** (`runtime._apple_gpu_mla_absorb_decode`) — attention runs
  directly against the cached latent; verified **numerically identical** to the
  explicit decoupled-RoPE path.
- **Paged single-sequence decode** (`tessera.cache.MLAPagedDecoder`).
- **GPU-resident multi-step decode loop** (`tessera.cache.ResidentMLADecoder`) —
  weights uploaded once, each step in one command buffer, only the token id reads
  back.
- **Concurrent block-paged serving** (`tessera.cache.MLABlockPagedCache`).

Expected GPU-decode line:

```text
OK mla gpu-decode: metal absorbed==explicit True paged==ref True block_paged==ref True resident_tokens 4 kv_cache_ratio 7.2x
```

(The `kv_cache_ratio` is the per-token cache footprint of the compressed latent +
shared rope key vs. explicit per-head K/V.)

## Quick Start

From the repository root:

```bash
PYTHONPATH=python /Users/gregorystoner/venv/bin/python \
  examples/advanced/mla/tests/smoke_random.py

PATH="$PWD/build/tools/tessera-opt:/opt/homebrew/opt/llvm@23/bin:$PATH" \
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
