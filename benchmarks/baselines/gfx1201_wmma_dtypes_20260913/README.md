# GFX1201 dense WMMA datatype proof

Owner: NUMPOL-CARRIER-1 / ROCM-2. Sync: `GFX1201-WMMA-DTYPES-2026-09-13`.
Measured on Tajasarus, RX 9070 XT (`gfx1201`), Ubuntu 26.04 WSL2,
ROCm 10.0, assertions-enabled LLVM/MLIR 23.1.1. Source identities are in
`source-identity.json`; `device.json` binds the compiler and each emitted image.

## ISA and implemented boundary

The RDNA4 archive has eleven dense WMMA instruction signatures and eleven
sparse SWMMAC signatures. RDNA3.5 has six dense signatures. The consumed
`wmma_dtype_forms` catalog is checked against both ISA JSON inventories.
This catalog describes hardware forms, not public package admission.

| A / B input | Accumulator | K | Dense signatures |
|---|---|---|---:|
| f16 / f16 | f32 or f16 | 16 | 2 |
| bf16 / bf16 | f32 or bf16 | 16 | 2 |
| E4M3 / E4M3, E4M3 / E5M2, E5M2 / E4M3, E5M2 / E5M2 | f32 | 16 | 4 |
| IU8 / IU8 | i32 | 16 | 1 |
| IU4 / IU4 | i32 | 16 or 32 | 2 |

Fixed mixed FP8/BF8 operand typing and intrinsic selection, INT4 K16 packing,
independent integer A/B signedness, and native f16/bf16 accumulator storage.
Malformed BF16 operands now fail instead of being reinterpreted. Integer
signedness is an instruction modifier over raw containers; no new public
uint4 dtype or unsigned Graph admission is claimed.

## Validation and numerical interpretation

`compiler-tests.txt`: **427 passed**, including ISA totality, compiler
serialization, RDNA3.5 refusal boundaries, dtype/diagnostic/pass registries.
`audit-tests.txt`: **25 passed**. `build-lint.txt`: both the final ROCm tool
link and ruff/zero-error mypy ratchet pass.

`device.json`: **76 passed** across all eleven dense signatures, all four
integer sign combinations, full and externally padded ragged tiles, finite
range/subnormal probes, and NaN/infinity classification. These are one-tile
executions, not arbitrary-size GEMM or attention packages.
Every emitted code object is disassembled and checked for its exact datatype
instruction signature; all eleven are present. This is static instruction
evidence, not runtime profiler attribution.

The f32/integer and range/classification probes use exact comparisons.
Low-precision dyadic accumulation uses the explicit bound
`gamma_k * (abs(A) @ abs(B))`, where `gamma_k = k*u/(1-k*u)` and
`u = 2^-11` for f16 or `2^-8` for bf16. Native BF16 accumulation differs from
f32 accumulation rounded once (up to 0.25 in these samples). The bound is a
finite, non-overflowing test envelope, not a general numerical-policy proof.
Signed-zero bit preservation and exhaustive rounding cases are not certified.

## Remaining work

- Sparse SWMMAC: LLVM has intrinsics, but Tessera needs a compiler-owned 2:4
  sparsity-index producer, packing/validation, and executable artifact ABI.
- General Graph/Schedule matmul and attention admission still needs dtype-aware
  descriptors, native consumers and exact-device differential proof.
- Scaled/packed MX formats require scale and dequantization consumers. FP4,
  FP6, TF32, f32/f64, bool and complex are not RDNA4 WMMA input signatures.
- Integrate low-accumulator bounds with numerical-policy consumers before
  permitting precision-changing rewrites.
- No timings, overlap, hardware-counter attribution or performance promotion.
  WSL lacks `/dev/kfd`; clean owning-device profiler evidence remains required.
  No gfx1200 or sibling-backend execution evidence transfers from this packet.

Reproduce on the owning host after sourcing `~/.config/tessera/env.sh`:

```sh
python -m benchmarks.rocm.record_gfx1201_wmma_types --output /tmp/wmma-types.json
```
