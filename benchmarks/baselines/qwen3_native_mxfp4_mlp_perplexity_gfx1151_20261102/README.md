# Qwen3-8B perplexity with the native MXFP4 MLP on gfx1151

Owner ROCM-NVFP4-INGEST-1. Synchronization key
ROCM-MXFP4-NATIVE-MLP-GFX1151-20261102. Host: Princess-Luna (Radeon 8060S,
gfx1151), source `dbc469236`, run under
`flock /tmp/tessera-gfx1151-device.lock`. **gfx1151 evidence only; gfx1201
and sm_120 inherit nothing from it, and the gfx1201 packet
(`../qwen3_native_mxfp4_mlp_perplexity_20261102/`) counts for nothing here.**

## Acceptance rule (fixed before measuring)

Committed in `5028c314a` (the plan's ROCM-NVFP4-INGEST-1 entry) before any
gfx1151 measurement. On Princess-Luna the `native_mxfp4` arm executes exactly
the `mxfp4_direct` weights. With Δ(arm) = ppl(arm) − ppl(BF16), it passes only
if all of these hold:

- Δ(mxfp4_direct) > 0;
- Δ(native_mxfp4) ≤ 1.5 × Δ(mxfp4_direct);
- every native output is finite;
- the staged certificate passes on every layer;
- the gfx1151 MXFP4 device tests (public MLP certified at three shapes,
  expansion image bitwise equal to the reference, code 255 refused) pass on
  that box with none skipped.

The gfx1151 − gfx1201 paired token-NLL difference and Δ(native_mxfp4)/Δ(shipped)
are reported, not gated.

## What changed to run it

No compiler change: the expansion emitter already accepted gfx1151, where the
E4M3 products decode to F16 WMMA. Two refusals by name stood in the way:

- `5028c314a`: the perplexity worker and the device tests refused any arch but
  gfx1201 for the MXFP4 arm. They now follow the live owning architecture,
  as the NVFP4 path does.
- `dbc469236`: the shared FP4 storage launcher (`_execute_arrays` in
  `rocm_nvfp4_ingest_native.py`) required a live gfx1201 whatever the image
  was compiled for. It now requires the live device to equal the image's own
  architecture. Ingest and MXFP4-storage packages still validate gfx1201
  images when packaged, so they stay gfx1201-only. A unit test builds a
  gfx1201 expansion and checks it is refused on a gfx1151 live device.

At `5028c314a` the gfx1151 device tests failed 2 of 30 on that launcher guard;
`tests.txt` here is the rerun at `dbc469236`.

## What was measured

Identical to the gfx1201 packet: the pinned `Qwen/Qwen3-8B` (`b968826d`), the
same sixteen 512-token wikitext-2 test windows (same token-id digest; 8,176
scored predictions), every arm with its own residual stream, only the MLPs
substituted. Arms `bf16`, `shipped`, `mxfp4_direct` (host references) and
`native_mxfp4` (the public seven-member native MLP on the gfx1151 device). See
the gfx1201 packet for the arm definitions.

**The baseline reproduces the earlier packets exactly.** The `bf16`,
`shipped` and `mxfp4_direct` per-token NLL arrays equal those of the gfx1201
packet (and so of `../qwen3_mxfp4_perplexity_20261102/`) bitwise.

## Result

| Arm | Perplexity | Δ vs BF16 | Δ / Δ(mxfp4_direct) | Δ / Δ(shipped) |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 12.3763 | — | — | — |
| Shipped NVFP4 (weight-only) | 12.6881 | +0.3118 | — | 1 |
| MXFP4 direct (weight-only) | 12.9338 | +0.5575 | 1 | 1.788 |
| **Native MXFP4 (gfx1151, W4A8)** | **12.9324** | **+0.5561** | **0.997** | 1.783 |

**The pre-registered gate passes:**

- 0.997 ≤ 1.5, and Δ(mxfp4_direct) > 0.
- All 2,304 native chunks were finite. The worker raises on any non-finite
  chunk, and it closed with `ok: true`.
- The staged certificate passed in all 36 layers: staged and resident output
  bitwise equal, and the propagated bound held.
- `tests.txt`: 31 passed, none skipped. That is the gfx1151 device tests,
  host/packaging tests and harness tests.

### `model_quality_acceptance` is false, and why

As in the gfx1201 packet, `perplexity.json` records
`model_quality_acceptance: false` because the recorder also ANDs the earlier
`mxfp4_direct`-vs-shipped verdict (ratio 1.788, the known format failure),
which is not part of this rule. The JSON is left as produced.

### Noise

Window-bootstrap 95% intervals use 10,000 resamples of the 16 windows, seed
20261102.

| Quantity | Point | 95% interval | Resamples ≤ 1.5 |
| --- | ---: | --- | ---: |
| Δ(native_mxfp4) / Δ(mxfp4_direct) | 0.997 | 0.94–1.06 | 100% |
| Δ(native_mxfp4) / Δ(shipped) | 1.78 | 1.10–3.16 | 23% |
| Δ(mxfp4_direct) / Δ(shipped) | 1.79 | 1.07–3.27 | 24% |

Paired mean token-NLL differences, with standard error over windows:

- native_mxfp4 − mxfp4_direct: **−0.0001 ± 0.0013**. No resolvable cost from
  W4A8 execution; per token the arms differ (98.5% of tokens, mean |ΔNLL|
  0.056) and the differences cancel.
- native_mxfp4 − shipped: +0.0191 ± 0.0089, the same as mxfp4_direct −
  shipped (+0.0192 ± 0.0093).
- **gfx1151 − gfx1201 native_mxfp4: +0.00004 ± 0.0012** (reported, not gated).
  The two architectures' native outputs differ on 98.7% of tokens (mean
  |ΔNLL| 0.068: F16-decode WMMA against FP8 WMMA) and cannot be told apart on
  this corpus. Neither stands for the other.

### Depth

MLP-input drift from the BF16 arm at layer 18: shipped 21.8%, mxfp4_direct
27.0%, native_mxfp4 27.3%. At layer 35: 15.8%, 19.3%, 19.4%. The native arm
tracks its weight-only reference within half a point at every layer
(`mlp_input_drift`), with no growth over depth.

Per-window perplexity (`per_window_perplexity`):

| Window | BF16 | Shipped | Direct | Native MXFP4 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 8.79 | 8.97 | 9.04 | 9.05 |
| 1 | 17.19 | 17.56 | 18.47 | 18.37 |
| 2 | 14.50 | 14.90 | 16.10 | 15.99 |
| 3 | 26.46 | 27.07 | 26.48 | 26.44 |
| 4 | 6.68 | 6.91 | 6.81 | 6.85 |
| 5 | 10.81 | 11.36 | 11.81 | 11.79 |
| 6 | 8.83 | 9.33 | 10.21 | 10.27 |
| 7 | 15.69 | 16.34 | 16.74 | 16.62 |
| 8 | 14.19 | 13.91 | 14.81 | 14.83 |
| 9 | 10.92 | 10.81 | 10.94 | 10.91 |
| 10 | 35.13 | 36.09 | 35.44 | 35.41 |
| 11 | 8.55 | 8.57 | 8.72 | 8.69 |
| 12 | 8.25 | 8.73 | 8.67 | 8.71 |
| 13 | 15.72 | 16.31 | 15.80 | 15.91 |
| 14 | 10.69 | 11.26 | 11.07 | 11.17 |
| 15 | 8.10 | 7.95 | 8.26 | 8.21 |

## What this means for the program

- The native MXFP4 source now holds at model level on both RDNA parts, each
  on its own proof: native W4A8 execution adds nothing measurable to MXFP4
  weights on gfx1151 either.
- MXFP4 weights still have **no model-quality acceptance against NVFP4** for
  Qwen3-8B (about 1.8× NVFP4's perplexity increase on this corpus). The native
  NVFP4 route remains the accepted one.

## Does not close

- **Hardware:** gfx1151 only. sm_120 has no `nvfp4_expand` lowering; bounded
  and runtime-extent images refuse an MXFP4 source on both RDNA parts.
- **Scope:** one corpus, one model, MLP substitution only; attention, norms
  and `lm_head` stay BF16 Torch.
- **Not measured:** a checkpoint that ships MXFP4 weights; W4A4; downstream
  tasks; performance (no timing claim is made).

## Files

- `perplexity.json`: identities (source revision `dbc469236`, driver/worker/
  compiler/provider digests, architecture `gfx1151`), per-arm and per-window
  perplexity, the verdicts, per-layer certificates, weight error, and drift.
- `token_nll.npz`: per-token NLL for each arm, shape 16 × 511, plus the token
  ids.
- `run.txt`: the progress log and `time -v` output (2 h 0 min wall,
  17.0 GiB peak RSS). The `overflow encountered in exp` warnings come from the
  host certificate's SiLU on large negative gate values; they are not device
  outputs.
- `tests.txt`: the gfx1151 device tests, host/packaging tests and harness
  tests on Princess-Luna at `dbc469236` (31 passed, none skipped).

Scripts: `benchmarks/rocm/record_qwen3_native_mlp_perplexity.py --arms
native_mxfp4,mxfp4_direct` and `benchmarks/rocm/qwen3_native_mlp_worker.py`.
