# Qwen3-8B perplexity with the native MXFP4 MLP on gfx1201

Owner ROCM-NVFP4-INGEST-1. Synchronization key ROCM-MXFP4-NATIVE-MLP-20261102.
Host: Tajasarus (RX 9070 XT, gfx1201), source `d7ad606b2`, run under
`flock /tmp/tessera-gfx1201-device.lock`. **gfx1201 evidence only; gfx1151
and sm_120 inherit nothing from it.**

## Acceptance rule (fixed before measuring)

Committed in `d7ad606b2` (the plan's ROCM-NVFP4-INGEST-1 entry) before this
run started. The `native_mxfp4` arm executes exactly the `mxfp4_direct`
weights. With Δ(arm) = ppl(arm) − ppl(BF16), it passes only if
Δ(mxfp4_direct) > 0 and Δ(native_mxfp4) ≤ 1.5 × Δ(mxfp4_direct), with every
native output finite and the staged certificate passing on every layer.
Δ(native_mxfp4)/Δ(shipped) is reported, not gated.

The reference is the arm's own weight-only MXFP4 counterpart, not the shipped
NVFP4 weights. The gate measures what native W4A8 execution adds to the MXFP4
weights. It does not re-test the MXFP4 format, which already failed against
shipped NVFP4 in `../qwen3_mxfp4_perplexity_20261102/`.

## What was measured

Everything matches the earlier packets: the pinned `Qwen/Qwen3-8B`
(`b968826d`), the same sixteen 512-token wikitext-2 test windows (same
token-id digest; 8,176 scored predictions), and every arm with its own
residual stream, with only the MLPs substituted. The arms are:

- `bf16`: the upstream Torch MLP;
- `shipped`: the decoded `nvidia/Qwen3-8B-NVFP4` weights, weight-only, FP64;
- `mxfp4_direct`: each BF16 original quantized once to MXFP4 (E2M1 codes, one
  E8M0 scale per 32), weight-only, FP64, under the stage 1 `layer_reference`
  policy;
- `native_mxfp4`: the same MXFP4 codes and E8M0 bytes fed as roots to the
  public seven-member native MLP (`canonical_rocm_quantized_mlp_program`) on
  the GPU. The FP4 expansion decodes each E8M0 byte to 2^(e−127), multiplies
  by the projection global (1.0 here), and writes the f32 result to both K16
  groups. Everything downstream is the accepted NVFP4 program: BF16
  activations quantized per 16 to E4M3 with f32 scales, scaled FP8 products,
  SiLU-gated BF16 parent, and the down projection.

The worker builds both MXFP4 arms from one cached quantization per
projection, so the codes and exponents are identical
(`test_native_mxfp4_roots_carry_exactly_the_direct_weights`).

**The baseline reproduces the earlier packets exactly.** The `bf16`,
`shipped` and `mxfp4_direct` per-token NLL arrays equal those of
`../qwen3_mxfp4_perplexity_20261102/` bitwise.

## Result

| Arm | Perplexity | Δ vs BF16 | Δ / Δ(mxfp4_direct) | Δ / Δ(shipped) |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 12.3763 | — | — | — |
| Shipped NVFP4 (weight-only) | 12.6881 | +0.3118 | — | 1 |
| MXFP4 direct (weight-only) | 12.9338 | +0.5575 | 1 | 1.788 |
| **Native MXFP4 (gfx1201, W4A8)** | **12.9319** | **+0.5555** | **0.997** | 1.782 |

**The pre-registered gate passes:** 0.997 ≤ 1.5, Δ(mxfp4_direct) > 0. All
2,304 native chunks were finite, and the staged certificate passed in all 36
layers (staged and resident output bitwise equal, propagated bound held).

For comparison, the native NVFP4 MLP on gfx1201 gave 12.691, a ratio of 1.008
against its own weight-only reference
(`../qwen3_native_mlp_perplexity_20261101/`). Native W4A8 execution adds
nothing measurable to either format's weight-only cost.

### `model_quality_acceptance` is false, and why

`perplexity.json` records `model_quality_acceptance: false`. The recorder
ANDs every verdict it computed, and it also computes the earlier rule for
`mxfp4_direct` against shipped (ratio 1.788, fails), which is the known
format failure from the earlier packet. That verdict is not part of this
packet's pre-registered rule. The recorded JSON is left as produced; the rule
above is the one applied. Whether a reference arm should be excluded from
the aggregate when it is only present as another arm's reference is left to a
follow-up, not changed after measuring.

### Noise

Window-bootstrap 95% intervals use 10,000 resamples of the 16 windows, seed
20261102.

| Quantity | Point | 95% interval | Resamples ≤ 1.5 |
| --- | ---: | --- | ---: |
| Δ(native_mxfp4) / Δ(mxfp4_direct) | 0.997 | 0.93–1.07 | 100% |
| Δ(native_mxfp4) / Δ(shipped) | 1.78 | 1.09–3.23 | 23% |
| Δ(mxfp4_direct) / Δ(shipped) | 1.79 | 1.07–3.27 | 24% |

Paired mean token-NLL differences, with standard error over windows:

- native_mxfp4 − mxfp4_direct: **−0.0002 ± 0.0014**. No resolvable cost from
  W4A8 execution. Per token the two arms do differ (98.5% of tokens, mean
  |ΔNLL| 0.057), as activation quantization should, but the differences
  cancel.
- native_mxfp4 − shipped: +0.0190 ± 0.0089, the same as mxfp4_direct −
  shipped (+0.0192 ± 0.0093). The cost against NVFP4 is the format's.

### Depth

MLP-input drift from the BF16 arm at layer 18 (the peak in the earlier
packet): shipped 21.8%, mxfp4_direct 27.0%, native_mxfp4 27.4%. At layer 35:
15.8%, 19.3%, 19.5%. The native arm tracks its weight-only reference within
half a point at every layer, with no growth over depth.

Per-window perplexity (`per_window_perplexity`):

| Window | BF16 | Shipped | Direct | Native MXFP4 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 8.79 | 8.97 | 9.04 | 9.06 |
| 1 | 17.19 | 17.56 | 18.47 | 18.35 |
| 2 | 14.50 | 14.90 | 16.10 | 16.07 |
| 3 | 26.46 | 27.07 | 26.48 | 26.48 |
| 4 | 6.68 | 6.91 | 6.81 | 6.85 |
| 5 | 10.81 | 11.36 | 11.81 | 11.80 |
| 6 | 8.83 | 9.33 | 10.21 | 10.18 |
| 7 | 15.69 | 16.34 | 16.74 | 16.67 |
| 8 | 14.19 | 13.91 | 14.81 | 14.96 |
| 9 | 10.92 | 10.81 | 10.94 | 10.89 |
| 10 | 35.13 | 36.09 | 35.44 | 35.61 |
| 11 | 8.55 | 8.57 | 8.72 | 8.72 |
| 12 | 8.25 | 8.73 | 8.67 | 8.68 |
| 13 | 15.72 | 16.31 | 15.80 | 15.88 |
| 14 | 10.69 | 11.26 | 11.07 | 11.10 |
| 15 | 8.10 | 7.95 | 8.26 | 8.15 |

## What this means for the program

- The native gfx1201 program executes MXFP4 weights at their weight-only
  quality. The compiler change (an MXFP4 source for the FP4 expansion) is
  correct at model level, not only at the kernel and certificate level.
- MXFP4 weights still have **no model-quality acceptance against NVFP4** for
  Qwen3-8B: the format costs about 1.8× NVFP4's perplexity increase on this
  corpus, native or not. The native NVFP4 route remains the accepted one.
- The program can now serve MXFP4 checkpoints on gfx1201 without a lossy
  NVFP4 requantization. Whether that is worth it depends on a model whose
  weights ship as MXFP4, which this packet does not test.

## Does not close

- **Hardware:** gfx1201 only. gfx1151 (Princess-Luna) and sm_120
  (Super-Bear) are not measured; the MXFP4 source is refused on bounded
  images and on any target but the static gfx1201 path.
- **Scope:** one corpus, one model, MLP substitution only; attention, norms
  and `lm_head` stay BF16 Torch.
- **Not measured:** a checkpoint that ships MXFP4 weights; W4A4; downstream
  tasks; performance (no timing claim is made).

## Files

- `perplexity.json`: identities (source revision, driver/worker/compiler/
  provider digests, device UUID), per-arm and per-window perplexity, the
  verdicts, per-layer certificates, weight error, and drift.
- `token_nll.npz`: per-token NLL for each arm, shape 16 × 511, plus the token
  ids.
- `run.txt`: the progress log and `time -v` output (1 h 56 min wall,
  17.0 GiB peak RSS). The two `overflow encountered in exp` warnings come
  from the host certificate's SiLU on large negative gate values
  (exp(−x) → inf, x/inf → 0); they are not device outputs.
- `tests.txt`: the gfx1201 device tests, host/packaging tests and harness
  tests on Tajasarus at `d7ad606b2` (30 passed, none skipped).

Scripts: `benchmarks/rocm/record_qwen3_native_mlp_perplexity.py --arms
native_mxfp4,mxfp4_direct` and `benchmarks/rocm/qwen3_native_mlp_worker.py`.
