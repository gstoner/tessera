
# Tessera Spectral Guide (v1.0)

## Overview
Tessera Spectral provides tile‑first FFT/iFFT with mixed‑precision (FP4/FP8/FP16/BF16/FP32) and optional f32 accumulation, plan caching, distributed 2D/3D decomposition, and hooks for NVIDIA/AMD/CPU backends.

## Key Concepts
- **Plan**: holds axes, radix sequence (autotuned by default), element and accumulation precision, scaling policy (BlockFP per stage), and normalization.
- **Mixed Precision**: use FP8/FP4 storage with FP32 accumulators; per‑tile amax → exponent metadata.
- **Distributed**: pencil/slab decomposition emits overlapped all‑to‑alls.

## Quick Start
```mlir
%plan = "tessera_spectral.plan"() {axes=[0,1], elem_precision="fp8_e4m3",
                                   acc_precision="f32", scaling="blockfp_per_stage",
                                   inplace=false, is_real_input=false, norm_policy="backward"} : () -> !any
"tessera_spectral.fft"(%plan, %src, %dst) : (!any, memref<?xcomplex<f32>>, memref<?xcomplex<f32>>) -> ()
```

## Policies
- `elem_precision`: `fp4_e2m1 | fp8_e4m3 | fp8_e5m2 | fp16 | bf16 | f32`
- `acc_precision`: `f16 | bf16 | f32` (recommend `f32` for long transforms)
- `scaling`: `none | blockfp_per_stage | dynamic_per_tile`
- `norm_policy`: `none | ortho | backward`

## Examples
- **FFT‑based convolution**: see `examples/fft_conv_example.mlir`
- **Spectral normalization**: see `examples/spectral_norm_example.mlir`

## Roadmap
v1.1 adds Bluestein and improved FP4 twiddle packing; v1.2 explores sFFT and NTT.
