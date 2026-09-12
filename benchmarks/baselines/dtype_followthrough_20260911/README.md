# FP8, packed formats and numerical boundary follow-through

Owner: NUMPOL-CARRIER-1 / LAYOUT-ALG-1; sync `DTYPE-CODEGEN-2026-09-11`.
These packets supplement, rather than replace, the earlier dtype arithmetic
inventory. They do not promote public frontend dtypes or performance candidates.

## Conversion contract

`tessera-expand-lowp-conversions` consumes byte-backed scalar/fixed-vector
`arith.bitcast` + `arith.extf` / `arith.truncf` boundaries for E4M3FN and E5M2.
It emits integer bit operations and explicit f32 arithmetic. Rounding is nearest
even; signed zero and subnormals are preserved, E4M3FN overflow becomes NaN,
and E5M2 overflow becomes infinity. Other rounding modes remain unlowered.
No native FP8 scalar arithmetic or matrix instruction is implied. The pass is
consumed by `build_native_gpu_storage` before LLVM conversion.

`record_dtype_arithmetic.py` checks all 65,536 byte pairs for add/subtract/
multiply/divide in each FP8 format and scalar/vector2 case. Bool probes check
OR/XOR/AND/equality, with byte storage normalized to i1. Complex probes check
finite, bounded interleaved component arithmetic with scalar/unrolled lanes;
they do not prove overflow-safe general complex division. GPU images have
compiler, image and disassembly fingerprints.

## Packed physical consumers

`nvidia/record_packed_dtype_correctness.py` uses the existing compiler-owned
`tile.packed_load` route. Ten cases cover INT4, FP4, NVFP4 and both FP6 formats,
both packing axes, ragged extents, offsets, padding, nonzero signed values and
varying power-of-two scales. Independent `ml_dtypes` decoding supplies the
floating oracle. These are generic decode consumers, not Tensor Core promotion.

The ROCm INT4 producer regression exercises generator-to-ROCDL conversion in
one compiler process with assertions enabled. Legacy `gpu.kernel` metadata
must not duplicate the inherent kernel property. ROCm serializer linkage is
independent of building a HIP/CUDA host runtime. The refreshed gfx1151 packet
executes scaled dequant-GEMM, packed ReLU, sparse gather and cache append;
its three timing repetitions are diagnostic, not selector-grade calibration.

## Apple numerical boundary

`record_apple_dtype_arithmetic.py` exports compiler-owned MSL and executes a
fresh status-returning bridge on resident M1 Max buffers, with fast math off.
Bounded complex64 component arithmetic passes. The strict fp32 probe fails
three comparisons: `tiny * .5`, `smallest_subnormal * 1`, and
`smallest_subnormal / 1` return zero. Add/subtract pass. The packet retains the
exact inputs and expected/actual values; this is not gradual-underflow proof.
A policy-aware software path or explicit flush-to-zero admission remains open.
Apple FP8, packed/scaled and general bool/vector storage remain separate work.

## Evidence limits

CUDA/HIP execution belongs only to SM120/gfx1151 respectively. WSL timings are
diagnostic; no clean bare-metal performance admission is claimed. x86 inherits
host contract tests only; its earlier mixed-integer matmul proof is unchanged.
General numerical error budgets, layout envelopes, matrix/systolic promotion,
and broader dtype/backend combinations remain open.
