# SM120 resident RHS edge: scaled FP8 storage

Owner: E2E-REAL-6 / W1.1 (NVIDIA producer migration).
Sync: SM120-W11-FP8-RHS-20261026.
Contract: `docs/spec/SM120_FP8_RHS_EDGE_CONTRACT.md`; `docs/spec/PYTHON_API_SPEC.md` §quantize_scaled.
Recorded by `benchmarks/nvidia/diagnose_fp8_rhs_edge_pairing.py`.

## What it is

The W1.1 RMSNorm→matmul RHS edge now has an FP8 edge, alongside the half one:

```python
codes = ts.ops.quantize_scaled(ts.ops.rmsnorm(source), scale, format="fp8_e4m3")  # or fp8_e5m2
out = ts.ops.matmul(lhs, ts.ops.dequantize_scaled(codes, scale), output_dtype="fp32")
```

- **Producer:** the native RMSNorm/LayerNorm kernel stores
  `satfinite(rn(round_to_storage(y) / s))` as FP8 codes, using the sm_120
  f32x2→f8x2 convert.
- **Consumer:** the native row-major-B matmul decodes each code exactly to A's
  f16/bf16 in its B gather, and multiplies the per-tensor scale once into the
  f32 accumulator.
- **Wiring:** both packages share one FP8 device edge and one scale upload, and
  run through the ordinary `@jit` call and the portable v3 manifest.

`tessera.quantize_scaled` is a new Graph op, the inverse of `dequantize_scaled`.

## Host

- Device: The-Super-Bear, RTX 5070 (sm_120), WSL2, CUDA 13.4, `build-nvidia-cuda`.
- Branch `claude/w11-fp8-rhs-edge`, clean tree.
- Contract tests on Tajasarus with the assertions-ON LLVM/MLIR 23.1.1.

## Correctness

- `tests/device/nvidia/test_scaled_fp8_norm.py`: **122 passed.** Fused codes are
  bit-identical to `quantize_scaled` of the unfused native norm. Grid: f16, bf16
  and f32 sources × E4M3/E5M2 × RMSNorm/LayerNorm × serial/cooperative schedule
  × five scales (incl. saturating ones), with guard bytes around the codes.
  Saturation and NaN rows are checked too.
- `tests/device/nvidia/test_scaled_fp8_rhs_matmul.py`: **50 passed.** The fused
  matmul is bit-identical to f32(native half matmul of the decoded codes × s)
  for f16/bf16 A × E4M3/E5M2 × four shapes (incl. ragged 37×40×29 and
  1×2048×256) × three scales. It is within 1e-5·(|A|·|B|) of the fp64 Graph
  composition. With A = I, every finite code of both formats decodes exactly.
- `tests/device/nvidia/test_rhs_fp8_jit_dispatch.py`: **13 passed.** These run
  the ordinary `@jit` call over both formats, both storages, three shapes and
  permuted frontend argument order. They check native receipts for both
  packages; the result against the program's own edge codes (1e-5 bound);
  ≥ 99.5% code agreement with the eager composition (rounding ties may move
  with the norm's f32 reduction order); portable manifest replay; and a warm
  call with the eager body and recompilation forbidden.
- Tajasarus, assertions-ON 23.1.1: the FP8 RHS contract, quantize_scaled, RHS
  program and scheduled-kernel unit tests pass. The two exceptions are x86
  executable tests, because that build tree has no x86 backend; they fail
  identically on `main`.

- Full `tests/device/nvidia -n 8` sweep: it found one regression from this
  branch, fixed in `e983582f4`. A new Schedule verifier branch refused every
  SM120 half matmul with a distinct B storage, which also caught the legitimate
  f16/f32 scalar-FMA contract: 58 `test_scalar_matmul_package` /
  `test_public_scalar_matmul` cases. The branch now refuses only an FP8 B
  without `b_scale`.
  - After the fix, 66 of the 67 entries new against the previous sweep pass;
    the 67th fails in both main baselines.
  - The setup errors in that sweep are the harness's own `nvidia-smi` probe
    (`tests/conftest.py`) timing out after 5 s under load, before any test
    body runs. The 42 affected files were rerun at `-n 4` on this branch and
    on `main`: the branch had 135 errors and `main` had 280.
  - The only branch-only *failures* in that A/B were six
    `test_public_jit_resident_roots_order_pending_writes` /
    `test_single_sided_public_resident_chain_and_raw_rhs` cases. They fail
    serially on current `main` (`53b0a677f`) too, with variants flipping
    between runs.
  - Lit on this host: `tests/tessera-ir` 441 passed / 124 unsupported / 0 failed;
    `check-tessera-nvidia` 72 passed.

## Timing (diagnostic only)

`paired-interleaved.json`. Each arm is the resident edge with a correctness gate
before timing. Session CUDA events wrap 20 stream launches after 2 warmup,
12 alternating rounds, median µs per launch; host enqueue gaps are included.
CUDA events alone are not a promotion witness on sm_120.

| Shape (M, K, N) | Arm | Producer | Consumer | Edge (both) |
|---|---|---:|---:|---:|
| 16, 4096, 4096 | f16 edge | 192 µs | 208 µs | 409 µs |
| | fp8_e4m3 edge | 207 µs | 227 µs | 432 µs |
| | fp8_e5m2 edge | 205 µs | 222 µs | 440 µs |
| 256, 4096, 4096 | f16 edge | 203 µs | 1250 µs | 1349 µs |
| | fp8_e4m3 edge | 219 µs | 1247 µs | 1326 µs |
| | fp8_e5m2 edge | 220 µs | 1247 µs | 1326 µs |

**No latency win.** The FP8 edge halves the edge's storage (32 MiB to 16 MiB
at K = N = 4096). Its edge time is 1.06–1.07× the f16 edge at M = 16 and
0.98× at M = 256.

- The producer pays about 7% for the extra divide and convert.
- The consumer is the typed m16n8k16 row-B kernel, which is bound by its
  per-lane gathers rather than by B bytes. Decoding a byte costs about what
  loading a half costs.

This is the W1.1 FP8 evaluation result: capacity, not speed, on this consumer.
It promotes no strategy or default. MXFP8 and MXFP4 remain the open gates.

## Not covered

Per-row/column/block scales; dynamic shapes; fused epilogues; an FP8 LHS
(`mma.sync` e4m3); standalone quantize/dequantize kernels; AD; any non-SM120
device.

The public *half* RHS dispatch was already failing on `main`: those 18
`test_rhs_jit_dispatch` cases select the generic tensor-program route. This
slice doesn't change it, and it is recorded in the NVIDIA queue.
