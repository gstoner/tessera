# FlashMLA / Multi-Latent Attention -> Tessera

This example contains a current-compiler FlashMLA smoke path plus the original
design note for a full Hopper/Blackwell-style Multi-Latent Attention
implementation.

The checked-in Graph IR is a projection/softmax skeleton, not a fused
FlashMLA kernel. The Apple and ROCm decode demonstrations below are separate
runtime paths; neither is produced by lowering that skeleton end to end.

Contents:
- `mla/` — dependency-light NumPy MLA reference, Graph IR compiler smoke, **and
  `gpu_decode.py` driving the shipped Apple GPU MLA decode surfaces**;
  `rocm_decode.py` exercises the separate compiled ROCm decode-step lane.
- `ir/flash_mla_tiny.mlir` — parser-valid current-dialect Graph IR tensor skeleton.
- `tests/smoke_random.py` — NumPy shape/cache smoke, Graph -> Schedule -> Tile ->
  Apple Target IR artifact check, **plus the GPU-decode demo validated vs numpy**.
- `tests/smoke_rocm.py` — exact-selected-device ROCm absorbed-latent decode
  and cache-side-effect check against `tessera.stdlib.attention`.
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

## ROCm MLA decode step (opt-in)

On a ROCm host with a built `tessera-opt`, select the actual HIP device's
architecture explicitly and run from the repository root:

```bash
TESSERA_ROCM_CHIP=gfx1201 \
TESSERA_OPT="$PWD/build/tools/tessera-opt/tessera-opt" \
PYTHONPATH=python python3 examples/advanced/mla/tests/smoke_rocm.py
```

Use `gfx1151` on an actual gfx1151 host; do not carry a result from one device
to the other. The smoke refuses a missing/mismatched chip pin or a
`reference_cpu` fallback. It checks native output and latent/RoPE cache
mutation against the same seeded stdlib reference, and prints the live
architecture, execution kind, compiler path, maximum absolute error, and
cache length. Tajasarus (RX 9070 XT, gfx1201) passed this bounded f32
two-token decode step on 2026-09-23; Princess Luna (Radeon 8060S, gfx1151)
passed the same check independently. This is the existing
`rocm_exotic_attn_compiled` runtime path,
not a claim of native FlashMLA, paged-cache scheduling, model-scale serving,
or complete Graph→Target lowering.

## Quick Start

From the repository root:

```bash
PYTHONPATH=python python3 \
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
