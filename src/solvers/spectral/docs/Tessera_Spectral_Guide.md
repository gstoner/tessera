
# Tessera Spectral Guide (v1.0)

## Overview
Tessera Spectral provides content-addressed one-dimensional FFT/iFFT packages,
mixed-radix and Bluestein planning, and compound DCT, spectral-convolution,
STFT, ISTFT, and spectral-filter programs. Native typed consumers are currently
validated on Zen 5 AVX-512 and ROCm gfx1151. NVIDIA and Apple physical
consumption remains architecture-owned follow-up work.

## Key Concepts
- **Plan**: binds the transform axis, radix sequence, algorithm, workspace,
  residency, normalization, and exact target architecture to a Schedule→Tile
  digest.
- **Precision**: FFT arithmetic is complex64/f32. Compound real transforms can
  accept fp16 or bf16 storage through explicit native-package conversion and
  still accumulate in f32. This is not an FP16/BF16 arithmetic FFT claim.
- **Scope**: the promoted physical contract is one-dimensional. Distributed
  pencil/slab decomposition and native 2D/3D execution are not implemented.

## Quick Start
```mlir
%plan = "tessera_spectral.plan"() {axes=[-1], elem_precision="f32",
                                   acc_precision="f32", scaling="none",
                                   inplace=false, is_real_input=false,
                                   norm_policy="backward"} : () -> !any
"tessera_spectral.fft"(%plan, %src, %dst) : (!any, memref<?xcomplex<f32>>, memref<?xcomplex<f32>>) -> ()
```

## Policies
- FFT element/accumulation precision: complex64/f32.
- Compound real-storage policy: `fp16 | bf16 | fp32`, with f32 arithmetic.
- Normalization: `backward | forward | ortho` for compound physical packages;
  the core FFT artifact currently uses backward normalization.
- DCT: type II only. Types I, III, and IV fail closed until their identities,
  adjoints, and native packages are implemented.

## Examples
- **FFT‑based convolution**: see `examples/fft_conv_example.mlir`
- **Spectral normalization**: see `examples/spectral_norm_example.mlir`

## Roadmap

Even-length RFFT/IRFFT use a content-addressed N/2 complex transform with
architecture-owned Hermitian pre/post processing on x86 and gfx1151.
`tessera.scheduled_spectral.v5` carries that boundary through one compound
artifact for RFFT→multiply→IRFFT spectral convolution, framed/windowed RFFT
STFT, and packed IRFFT→deterministic overlap-add ISTFT. Intermediate real and
Hermitian slabs remain in native package workspace. Odd lengths retain an
explicitly hashed full-complex fallback. The next physical work is folding the
gfx1151 Hermitian step into its fused-LDS launch, production STFT/ISTFT padding
and streaming policies, and measured multidimensional/distributed transforms.
Reduced-arithmetic FP16/BF16, FP8/FP4, sFFT, and NTT remain research or planned
work rather than shipped capabilities.
