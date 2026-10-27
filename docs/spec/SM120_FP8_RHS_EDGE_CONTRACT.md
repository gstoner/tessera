---
last_updated: 2026-10-26
audit_role: reference
---

# SM120 scaled FP8 RHS edge contract

Synchronization key: SM120-W11-FP8-RHS-20261026.
Owner: E2E-REAL-6 / W1.1 (NVIDIA producer migration).

The resident RMSNorm/LayerNorm RHS edge (`nvidia_tensor_rhs.py`) feeds a
normalized `[K, N]` tensor to the row-major B of a matmul. This contract adds
an FP8 edge. Both of its ends are native Schedule→Tile products; Python only
partitions the verified Graph and packages the two native images.

## Public form

The edge is a composition of four Graph operations, not a new signature:

```python
codes = ts.ops.quantize_scaled(ts.ops.rmsnorm(source), scale, format="fp8_e4m3")
out = ts.ops.matmul(lhs, ts.ops.dequantize_scaled(codes, scale), output_dtype="fp32")
```

- `source` and `lhs` are f16 for the native edge. A bf16 FP8 edge runs the
  eager reference composition (see the consumer's overflow condition below).
- `scale` is a `(1, 1)` f32 per-tensor scale. Both sides bind the same argument.
- `format` is `fp8_e4m3` or `fp8_e5m2` and is required; it is never defaulted.

## `tessera.quantize_scaled`

A new Graph op: the inverse storage map of `dequantize_scaled`.

1. Each value widens exactly to f32.
2. It is divided by its broadcast f32 scale with IEEE round-to-nearest-even.
3. The quotient rounds to `format` with ties to even and finite saturation:
   - out-of-range values and infinities become the signed largest finite code
     (448 or 57344);
   - every NaN becomes the canonical code `0x7F`.

`format` must agree with the result element type, and the verifier checks it.
The eager reference reproduces `cvt.rn.satfinite` bit for bit, including the
NaN code: ml_dtypes alone would pick `0x7E` for E5M2.

## Producer: fused FP8 norm store

`quantize_scaled(norm(x), s)` lowers to one `tile.norm_kernel` whenever the
norm's only use is the quantize and `s` is a per-tensor f32 entry argument.

- **Contract:** `schedule.norm` and the Tile kernel carry
  `output_storage = "fp8_e4m3|fp8_e5m2"` and `scale = "per_tensor_f32"`. Both
  enter the schedule digest only when present, so every unfused norm digest is
  unchanged.
- **ABI:** x, scale, codes, Rows, Columns. The ABI id is
  `tessera.nvidia.norm.x_scale_o_rows_columns.<storage>_to_<fp8>_f32acc.v1`.
- **Entry:** `tessera_tile_norm_<kind>_<storage>_to_<fp8>[_cooperative_128]_<digest>`.
- **Store:** the f32 norm value first rounds to the Graph's declared norm
  storage, because the unfused composition reads that rounded tensor. It then
  widens exactly and divides by the scale. Finally it converts with
  `nvvm.convert.f32x2.to.f8x2` (`rn`, `satfinite`) and stores one byte.

The codes are bit-identical to `quantize_scaled` applied to the unfused native
norm's output.

## Consumer: FP8 row-major B decode

`matmul(A, dequantize_scaled(Bq, s))` lowers to the typed m16n8k16 producer
under these conditions: A is f16 `[M, K]`, `Bq` is row-major FP8
`[K, N]`, `s` is per-tensor f32, the shapes are static, and the output is
unfused f32.

- **Schedule:** `schedule.matmul` carries `storage_b = "e4m3|e5m2"` and
  `b_scale = "per_tensor_f32"`. `b_scale` enters the digest only when present.
  The scalar-FMA route and the multi-op tensor-program outliner both step aside
  for this form.
- **Tile:** the MMA descriptor stays in A's half storage. The B
  `tile.fragment_pack` carries `decode = "fp8_<fmt>"`, and `tile.store` carries
  `tile.output_scale` with one trailing scale pointer. Both are verifier-checked
  and refuse composition with a fused epilogue.
- **Gather:** the row-major-B gather loads one code byte per element and
  decodes it exactly with `cvt.rn.f16x2.{e4m3,e5m2}x2`. Every FP8 value is an
  f16 value.
- **Scale:** the f32 accumulator is multiplied by `s` once, before the store.
- **ABI:** A, B codes, scale, D, M, N, K. The ABI id is
  `tessera.nvidia.matmul.a_row_bcodes_scale_d_m_n_k.<storage>_<fp8>.v1` and the
  entry ends `_rhs_fp8_<fmt>_scaled_kernel`.

**This is a declared reassociation.** The Graph computes
`sum_k a[m,k] * f32(code[k,n] * s)`. The kernel computes
`f32(sum_k a[m,k] * code[k,n]) * s`. That factors the scale out of the K sum
and stays within the f32 accumulation tolerance, but it is not bit-identical to
the Graph. It is bit-identical to the native half matmul of the decoded codes
multiplied by `s`, and the device tests hold it to that.

**Why f16 A only.** Factoring the scale out is safe only if the unscaled f32
accumulator cannot overflow where the Graph's result is finite. For f16 A,
every term satisfies |a · code| ≤ 65504 · 57344 ≈ 3.76·10⁹, so even K = 2³¹
keeps the sum below 8.1·10¹⁸, far under the f32 maximum. bf16 spans the f32
range: A = 2¹²⁰ against E4M3 code 448 overflows before a 2⁻¹⁰ scale could
bring it back, even though the Graph result 7·2¹²⁰ is finite. The Schedule,
the Python lowering and `@jit` therefore admit only f16 A. A bf16 FP8 edge
runs the eager composition. (Codex review on #1017.)

## Edge program

`NvidiaNormRhsProgram` gains `edge_storage` and `scale_name`. The FP8 edge is
an FP8 `[K, N]` device buffer that is never copied to the host, and one scale
upload is bound by both kernels. Supporting changes:

- the traced partition accepts the four-op Graph;
- public `@jit` admits three arrays: two half `[*, *]` and one `(1, 1)` f32;
- the portable manifest is `tessera.nvidia.norm_rhs_program.v3`.

The resident launch admits the four-buffer matmul. A CUDA buffer whose declared
dtype produces its own typestr is described by that dtype; this is needed for
ml_dtypes E5M2, whose `<f1` NumPy cannot parse.

## Not admitted

- bf16 A in the fused consumer (see above);
- per-row, per-column or block scales;
- dynamic or bounded shapes;
- fused bias, activation or residual, and f16 output;
- column-major B;
- an FP8 LHS (the `mma.sync` e4m3/e5m2 form);
- standalone `quantize_scaled` / `dequantize_scaled` kernels;
- MXFP8 and MXFP4 (the remaining W1.1 gates);
- AD;
- any non-SM120 target.
