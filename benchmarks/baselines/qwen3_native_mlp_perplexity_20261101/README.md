# Qwen3-8B perplexity with the native quantized MLP in every layer (stage 2)

Owner ROCM-NVFP4-INGEST-1. Synchronization key QWEN3-NATIVE-MLP-PERPLEXITY-20261101.
Host: Tajasarus, RX 9070 XT (gfx1201), source `39b048f27`. **gfx1201 evidence only.**

## Acceptance rule (fixed before measuring)

Committed in `39b048f27` (the plan's ROCM-NVFP4-INGEST-1 entry and
`record_qwen3_native_mlp_perplexity.THRESHOLD_RATIO`) before the full run
started. Let Δ(arm) = ppl(arm) − ppl(BF16). The gate passes only if
Δ(shipped) > 0 and Δ(native) ≤ 1.5 × Δ(shipped), every native output is
finite, and the staged certificate passes on each layer's first native chunk.

## What was measured

- **Model and tokens.** The pinned `Qwen/Qwen3-8B` (`b968826d`), upstream
  transformers 4.57.1 on CPU (Torch 2.14.1+cpu, separate reference venv).
  Sixteen 512-token windows evenly spaced across the wikitext-2-raw-v1 test
  split (`Salesforce/wikitext` `b08601e0`), 8,176 scored next-token
  predictions.
- **Three arms, each with its own residual stream.** Every one of the 36
  layers' MLP is served by:
  - `bf16`: the upstream Torch MLP;
  - `shipped`: the decoded `nvidia/Qwen3-8B-NVFP4` (`ccd10a89`) weights, FP64
    products, FP32-then-BF16 stores, no activation quantization (the format's
    own loss; the checkpoint's W4A4 deployment recipe is *not* reproduced);
  - `native`: the public seven-member native quantized MLP
    (`canonical_rocm_quantized_mlp_program`) on gfx1201, 128-row chunks.
  Attention, norms, embeddings and `lm_head` stay BF16 Torch in all arms.
- **Process split (Decision #23).** The native and shipped arms run in a
  Tessera worker (`qwen3_native_mlp_worker.py`) that the Torch driver talks to
  over a framed pipe; the driver verifies at exit that it never imported
  Tessera.
- **Split loop checked against the stock model.** The driver runs layer-major
  (every window through layer i before layer i+1, so the worker holds one
  layer's weights at a time). The `bf16` arm's logits equal
  `model(input_ids)` **bitwise** for all 16 windows.

## Correctness

2,304 native chunk calls and 2,304 shipped chunk calls, all finite. The staged
tolerance certificate ran and passed on the first native chunk of **every**
layer (36 of 2,304 chunks; about 80 s each). The inputs it certified are the
native arm's own drifted MLP inputs, not stage 1's teacher-forced ones.

## Result

| Arm | Perplexity | Δ vs BF16 |
| --- | ---: | ---: |
| BF16 | 12.3763 | — |
| Shipped NVFP4 | 12.6881 | +0.3118 |
| Native | 12.6906 | +0.3143 |

**Δ(native) / Δ(shipped) = 1.008 ≤ 1.5: the gate passes.**

- **Noise.** The native − shipped mean token NLL difference is +0.0002
  (standard error 0.0017 over the 16 windows). A window bootstrap of the
  ratio gives a 95% interval of 0.86–1.15, entirely below 1.5. Δ(shipped) is
  clearly positive (mean ΔNLL 0.0249, window SE 0.0063).
- **Error growth through depth.** Each arm's MLP input drifts from the BF16
  arm's as errors accumulate (`mlp_input_drift`), peaking at layer 18
  (shipped 21.8%, native 22.3%) and ending at layer 35 at 15.8% and 16.2%.
  The native arm's drift is 1.004–1.029× the shipped arm's at every layer
  after the first.

Per-window perplexity (`per_window_perplexity`):

| Window | Start token | BF16 | Shipped | Native | Native − shipped |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 8.79 | 8.97 | 8.98 | +0.01 |
| 1 | 18,683 | 17.19 | 17.56 | 17.62 | +0.06 |
| 2 | 37,366 | 14.50 | 14.90 | 14.81 | −0.09 |
| 3 | 56,049 | 26.46 | 27.07 | 27.06 | −0.00 |
| 4 | 74,732 | 6.68 | 6.91 | 6.99 | +0.08 |
| 5 | 93,415 | 10.81 | 11.36 | 11.32 | −0.04 |
| 6 | 112,098 | 8.83 | 9.33 | 9.38 | +0.04 |
| 7 | 130,781 | 15.69 | 16.34 | 16.50 | +0.16 |
| 8 | 149,464 | 14.19 | 13.91 | 13.90 | −0.01 |
| 9 | 168,147 | 10.92 | 10.81 | 10.88 | +0.07 |
| 10 | 186,830 | 35.13 | 36.09 | 35.93 | −0.15 |
| 11 | 205,513 | 8.55 | 8.57 | 8.58 | +0.01 |
| 12 | 224,196 | 8.25 | 8.73 | 8.74 | +0.00 |
| 13 | 242,879 | 15.72 | 16.31 | 16.34 | +0.03 |
| 14 | 261,562 | 10.69 | 11.26 | 11.10 | −0.17 |
| 15 | 280,245 | 8.10 | 7.95 | 7.89 | −0.06 |

## Does not close

This is one corpus (wikitext-2, 8,176 tokens) and one model, with the MLP
substituted and everything else BF16. It does not cover downstream task
accuracy, quantized attention or `lm_head`, the checkpoint's W4A4 deployment
recipe, the direct BF16→MXFP4 and NVFP4→MXFP4-ingest arms, gfx1151 (no
evidence transfers), or any default or selector promotion.

## Files

`perplexity.json` (identities, per-arm and per-window perplexity, drift,
all 36 certificates), `token_nll.npz` (per-token NLL for each arm, shape
16 × 511, plus the token ids), `run.txt` (progress log, `time -v`; repeated
overflow warnings from the FP64 SiLU references removed), `harness-tests.txt`.
Scripts: `benchmarks/rocm/record_qwen3_native_mlp_perplexity.py`,
`benchmarks/rocm/qwen3_native_mlp_worker.py`.
