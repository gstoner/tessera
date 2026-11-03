# Qwen3-8B perplexity with MXFP4 weights in every MLP (stage 2, MXFP4 arms)

Owner ROCM-NVFP4-INGEST-1. Synchronization key QWEN3-MXFP4-PERPLEXITY-20261102.
Host: Tajasarus (CPU only), source `3a96c891d`. **Host references: no device
was opened and no device result is claimed.**

## Acceptance rule (fixed before measuring)

Committed in `3a96c891d` (the plan's ROCM-NVFP4-INGEST-1 entry and
`record_qwen3_native_mlp_perplexity.THRESHOLD_RATIO`) before this run started:
the stage 2 rule, applied to each MXFP4 arm on its own. Let Δ(arm) =
ppl(arm) − ppl(BF16). An arm passes only if Δ(shipped) > 0 and
Δ(arm) ≤ 1.5 × Δ(shipped). Δ(ingested)/Δ(direct) is reported, not gated.

## What was measured

Everything matches stage 2 (`../qwen3_native_mlp_perplexity_20261101/`): the
pinned `Qwen/Qwen3-8B` (`b968826d`) on CPU in the separate Torch reference
venv, the same sixteen 512-token wikitext-2 test windows (same token-id
digest; 8,176 scored predictions), and every arm with its own residual
stream, with only the MLPs substituted. The arms are:

- `bf16`: the upstream Torch MLP;
- `shipped`: the decoded `nvidia/Qwen3-8B-NVFP4` (`ccd10a89`) weights;
- `mxfp4_direct`: each BF16 original quantized once to MXFP4 (E2M1 codes,
  one E8M0 scale per 32) by the shared joint code/scale search
  (`benchmark_rocm_nvfp4_checkpoint._direct_bf16_to_mxfp4`);
- `mxfp4_ingested`: each shipped NVFP4 projection requantized to MXFP4 by
  `rocm_nvfp4_ingest.ingest_nvfp4_projections`, the program's one declared
  lossy step. Each projection converts on its own, keeping its own global
  scale.

The three quantized arms share one policy, the stage 1 `layer_reference`
policy:
- the weights are decoded exactly;
- products and the down projection are FP64;
- the gate/up parent is stored in BF16, as the native program does;
- activations are not quantized.

These are weight-only references. They are **not** native MXFP4 execution,
which has no full-MLP program yet. They are not the W4A8 activation recipe
the native MXFP4 projections use.

**The baseline reproduces stage 2 exactly.** The `bf16` and `shipped`
per-token NLL arrays equal the stage 2 packet's bitwise. That holds across
two runs and two source revisions on the same host, so Δ(shipped) here is
the stage 2 Δ(shipped).

## Result

| Arm | Perplexity | Δ vs BF16 | Δ / Δ(shipped) | Gate (≤ 1.5) |
| --- | ---: | ---: | ---: | --- |
| BF16 | 12.3763 | — | — | |
| Shipped NVFP4 | 12.6881 | +0.3118 | 1 | |
| MXFP4 direct | 12.9338 | +0.5575 | 1.788 | **fails** |
| MXFP4 ingested | 12.9982 | +0.6218 | 1.994 | **fails** |

**Neither MXFP4 arm passes.** Δ(ingested)/Δ(direct) = 1.115.

### Noise: what 8,176 tokens can and cannot resolve

The window-bootstrap 95% intervals use 10,000 resamples of the 16 windows,
seed 20261102.

| Quantity | Point | 95% interval |
| --- | ---: | --- |
| Δ(direct) / Δ(shipped) | 1.79 | 1.07–3.27 |
| Δ(ingested) / Δ(shipped) | 1.99 | 1.25–3.31 |
| Δ(ingested) / Δ(direct) | 1.12 | 0.71–1.75 |

In 24% of resamples the direct ratio is ≤ 1.5; for the ingested ratio, 11%.
The paired mean token-NLL differences, with standard error over windows:

- direct − shipped: **+0.0192 ± 0.0093**, about two standard errors. The MXFP4
  format with E8M0 scales per 32 costs more than NVFP4 with E4M3 scales per
  16, even with no double rounding.
- ingested − direct: **+0.0050 ± 0.0103**. This corpus cannot separate it from
  zero. The extra cost of requantizing NVFP4, compared with quantizing the
  BF16 original once, is not resolved at model level here.

### Weight error and depth

Mean weight relative RMS against the BF16 originals, over all 108
projections:
- shipped 9.48%;
- direct 11.34%;
- ingested 14.66%.

These match the per-tensor study. By that measure, ingest adds the most
error of any step. Yet at the model level it adds the least of the three
steps measured here: about 12% more Δ than direct, against 79% for the
format change itself.

MLP-input drift from the BF16 arm peaks at layer 18:
- shipped 21.8%;
- direct 27.0%;
- ingested 33.7%.

At layer 35 the three are 15.8%, 19.3% and 23.8%.

Per-window perplexity (`per_window_perplexity`):

| Window | BF16 | Shipped | Direct | Ingested |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 8.79 | 8.97 | 9.04 | 9.60 |
| 1 | 17.19 | 17.56 | 18.47 | 17.84 |
| 2 | 14.50 | 14.90 | 16.10 | 15.97 |
| 3 | 26.46 | 27.07 | 26.48 | 27.48 |
| 4 | 6.68 | 6.91 | 6.81 | 6.86 |
| 5 | 10.81 | 11.36 | 11.81 | 12.90 |
| 6 | 8.83 | 9.33 | 10.21 | 9.80 |
| 7 | 15.69 | 16.34 | 16.74 | 15.84 |
| 8 | 14.19 | 13.91 | 14.81 | 14.16 |
| 9 | 10.92 | 10.81 | 10.94 | 11.17 |
| 10 | 35.13 | 36.09 | 35.44 | 37.39 |
| 11 | 8.55 | 8.57 | 8.72 | 8.79 |
| 12 | 8.25 | 8.73 | 8.67 | 8.72 |
| 13 | 15.72 | 16.31 | 15.80 | 16.07 |
| 14 | 10.69 | 11.26 | 11.07 | 11.11 |
| 15 | 8.10 | 7.95 | 8.26 | 7.95 |

## What this means for the program

Both MXFP4 arms fail the pre-registered gate, so MXFP4 weights have **no
model-quality acceptance** for Qwen3-8B. The fault is the MXFP4 format, not
the ingest step:
- Most of the MXFP4 cost is already present when the BF16 original is
  quantized once.
- The double rounding the gate names as the one lossy step is too small to
  resolve on this corpus.

Two consequences:
- The native NVFP4 MLP passes on gfx1201, executing the source scales through
  K16 expansion. It remains the only route with model-quality acceptance.
- Any MXFP4 route, ingested or direct, would need a larger corpus to estimate
  its cost precisely, and a different acceptance argument to pass.

## Does not close

- **Scope:** one corpus (wikitext-2, 8,176 tokens) and one model, with only
  the MLP substituted.
- **Not measured:**
  - native MXFP4 execution or its W4A8 activation recipe;
  - downstream task accuracy;
  - quantized attention or `lm_head`;
  - the checkpoint's W4A4 deployment recipe.
- **No device evidence:** these are CPU references, so no architecture
  inherits anything from them.

## Files

- `perplexity.json`: identities, per-arm and per-window perplexity, the
  verdicts, per-layer weight error, and drift.
- `token_nll.npz`: per-token NLL for each arm, shape 16 × 511, plus the token
  ids.
- `run.txt`: the progress log and `time -v` output (58 min wall, 17.0 GiB
  peak RSS).
- `harness-tests.txt`: the host-free harness tests on Tajasarus.

Scripts: `benchmarks/rocm/record_qwen3_native_mlp_perplexity.py --arms
mxfp4_direct,mxfp4_ingested` and `benchmarks/rocm/qwen3_native_mlp_worker.py`.
