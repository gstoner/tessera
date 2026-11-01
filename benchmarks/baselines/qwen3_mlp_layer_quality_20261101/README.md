# Qwen3-8B quantized-MLP quality, every layer (stage 1)

Owner ROCM-NVFP4-INGEST-1. Synchronization key QWEN3-MLP-LAYER-QUALITY-20261101.
Host: Tajasarus, RX 9070 XT (gfx1201), source `8a4e68429`. **gfx1201 evidence only.**

## What was measured

Stage 1 of the model-quality gate: each decoder layer's MLP in isolation, on
real model inputs. It is **not** model accuracy; `model_quality_acceptance`
stays false and no acceptance threshold applies to these numbers.

- **Inputs** (`capture.json`). The teacher-forced BF16 MLP input of all 36
  layers of the pinned `Qwen/Qwen3-8B` (`b968826d`), run with upstream
  transformers 4.57.1 on CPU (Torch 2.14.1+cpu, a separate reference venv;
  Torch never executes a Tessera package). Tokens are four 512-token windows,
  evenly spaced across the wikitext-2-raw-v1 test split
  (`Salesforce/wikitext` `b08601e0`; 298,938 tokens; starts 0, 74734,
  149468, 224202), so 2,048 rows per layer. Two full runs captured
  bitwise-identical inputs. The 1.2 GB of captured activations are not
  stored here; `capture.json` records every layer's hash.
- **Weights.** The pinned `nvidia/Qwen3-8B-NVFP4` (`ccd10a89`) and BF16
  checkpoints, read from the local cache. Every shard was hashed against its
  content-addressed blob name before use.
- **Native arm.** The public seven-member quantized MLP
  (`canonical_rocm_quantized_mlp_program`): activation E4M3 quantization,
  source-preserving NVFP4 expansion, exact per-block FP32 products, BF16
  stores. It ran at 128 rows × 12288 × 4096 × 4096, 16 chunks per layer.
- **References.** FP64 products with FP32-then-BF16 stores
  (`benchmark_checkpoint_quantized_mlp.layer_reference`): the BF16 weights,
  and the decoded shipped NVFP4 weights, both without activation
  quantization.

## Correctness

Every one of the 576 outputs is finite. The staged tolerance certificate
(`assert_compiled_chain`) ran on the **first chunk of each layer only** (36
certified chunks; 540 scored without it), because it costs about 80 s per
chunk. The native call and both references together take about 1–3 s. Within a
layer every chunk runs the same image on the same weights. All 36 certificates
passed (per-stage rtol 2⁻⁶, atol 2e-5, plus the propagated bound).

## Results

Relative RMS error over all 2,048 rows of each layer (`quality.json`):

| Layer | Shipped NVFP4 vs BF16 | Native vs BF16 | Native vs shipped NVFP4 | Native / shipped loss |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 7.00% | 7.09% | 1.08% | 1.012 |
| 1 | 10.61% | 10.62% | 1.02% | 1.002 |
| 2 | 14.26% | 14.32% | 1.25% | 1.004 |
| 3 | 13.86% | 13.92% | 1.31% | 1.004 |
| 4 | 12.54% | 12.61% | 1.49% | 1.006 |
| 5 | 12.31% | 12.47% | 2.05% | 1.013 |
| 6 | 3.75% | 3.75% | 0.09% | 1.000 |
| 7 | 14.36% | 14.60% | 2.68% | 1.016 |
| 8 | 12.85% | 13.13% | 2.75% | 1.022 |
| 9 | 12.60% | 12.88% | 2.71% | 1.022 |
| 10 | 12.40% | 12.68% | 2.69% | 1.023 |
| 11 | 12.66% | 12.95% | 2.75% | 1.022 |
| 12 | 12.79% | 13.07% | 2.77% | 1.022 |
| 13 | 13.10% | 13.40% | 2.86% | 1.023 |
| 14 | 13.10% | 13.39% | 2.85% | 1.022 |
| 15 | 13.19% | 13.48% | 2.87% | 1.023 |
| 16 | 3.17% | 3.25% | 0.35% | 1.025 |
| 17 | 13.36% | 13.66% | 2.90% | 1.022 |
| 18 | 13.10% | 13.38% | 2.79% | 1.021 |
| 19 | 12.91% | 13.18% | 2.73% | 1.021 |
| 20 | 13.42% | 13.70% | 2.81% | 1.021 |
| 21 | 13.34% | 13.60% | 2.74% | 1.020 |
| 22 | 12.60% | 12.82% | 2.46% | 1.018 |
| 23 | 12.16% | 12.36% | 2.28% | 1.017 |
| 24 | 11.64% | 11.81% | 2.09% | 1.015 |
| 25 | 12.16% | 12.34% | 2.17% | 1.015 |
| 26 | 12.19% | 12.37% | 2.20% | 1.015 |
| 27 | 12.46% | 12.65% | 2.20% | 1.015 |
| 28 | 12.44% | 12.64% | 2.26% | 1.015 |
| 29 | 11.69% | 11.85% | 1.99% | 1.013 |
| 30 | 12.06% | 12.24% | 2.16% | 1.015 |
| 31 | 12.83% | 13.02% | 2.32% | 1.015 |
| 32 | 13.50% | 13.72% | 2.53% | 1.017 |
| 33 | 13.12% | 13.34% | 2.46% | 1.016 |
| 34 | 5.99% | 6.08% | 1.09% | 1.016 |
| 35 | 12.73% | 12.78% | 1.11% | 1.004 |


- **The format dominates.** The shipped NVFP4 weights alone are 3.17–14.36%
  from BF16 (median 12.63%). The native MLP is 3.25–14.60% (median 12.85%).
- **Tessera's added loss is small and flat.** Native error divided by the
  format's own error is 1.000–1.025 (median 1.016) in every layer; it does
  not grow with depth. Against the shipped NVFP4 weights the native MLP is
  0.09–2.90% off (median 2.30%), which is the cost of quantizing the
  activations to E4M3 per 16, the one approximation the native route adds.
- **The 33% layer-35 smoke result did not hold.** It came from the first 16
  tokens of one sequence; over 2,048 rows layer 35 is 12.73% (format) and
  12.78% (native).
- Layers 6, 16 and 34 have much lower error under both arms (3–6%). The
  format arm shows it with no Tessera arithmetic, so it is a property of
  those layers' weights and inputs, not of the native route. Why is not
  established here.

## Does not close

End-to-end accuracy (stage 2: perplexity with the native MLP substituted in
every layer), the direct BF16→MXFP4 and NVFP4→MXFP4-ingest arms (the native
MLP consumes NVFP4 source-preserving, not MXFP4), gfx1151 (no evidence
transfers), and any default or selector promotion.

## Files

`capture.json` (input identities and hashes), `quality.json` (per-layer
arms, certificates, checkpoint tensor hashes), `run.txt`, `harness-tests.txt`
(host-free harness tests on the Mac). Scripts:
`benchmarks/rocm/qwen3_local_checkpoint.py`,
`benchmarks/rocm/capture_qwen3_all_layer_mlp_inputs.py`,
`benchmarks/rocm/record_qwen3_mlp_layer_quality.py`.
