# Full pinned Qwen3 layer-0 MLP through native quantized Graph

Owner: ROCM-NVFP4-INGEST-1 / ROCM-FP8-BLOCKSCALE-1.
Sync: ROCM-CHECKPOINT-MLP-QUALITY-20261017. Follows PR972 and pending PR973.
Source: a1df1af73f; exact RX 9070 XT/gfx1201 identity, compiler/provider and
recorder hashes are recorded per row. Compiler and all three native program
source bytes match the retained build manifest despite checkout mtime warnings.

The ordinary public JIT takes the original pinned gate/up/down NVFP4 weights
and actual captured upstream BF16 model inputs. All seven members run through
Graph -> Schedule -> Tile -> ROCm Target -> LLVM/HSACO under one checked HIP
owner. H=12288, K=4096, N=4096. No Python numerical implementation executes
inside the production path. Reference and staged diagnostic work precedes timing.

| M | RMS vs BF16 weights | RMS vs shipped NVFP4 weights | Public ms | Prepared invoke/read ms | Ordinary device window ms |
|---|---:|---:|---:|---:|---:|
| 1 | 4.6787% | 0.4542% | 100.518 | 2.761 | 2.418 |
| 16 | 6.5237% | 0.9967% | 106.129 | 3.135 | 2.038 |
| 128 | 6.9206% | 1.0378% | 106.853 | 3.772 | 2.442 |

Both quality references use FP64 products, FP32-to-BF16 parent/down stores,
FP32 SiLU and a BF16 activation store. They omit activation quantization;
therefore their differences include activation quantization and arithmetic,
and are not an isolated weight-error measurement or an upstream-model replay.
The pinned BF16 and NVFP4 revisions, original byte-range hashes, independent
projection globals and captured-input provenance are included in each record.
No checkpoint shard or duplicate activation archive is committed.

Resident and diagnostic staged outputs match bitwise in every case. Existing
per-stage and propagated bounds pass without tolerance changes. The flat
independent reference has 0/43/187 out-of-gate elements at M1/16/128; these
remain visible and are not claimed as flat-reference passes. Explicit native
BF16 parent-store differences can cross quantizer bins downstream.

Captured member windows, ordinary event windows, prepared invoke/read, public
warm calls and cold preparation are separate scopes. Captured dispatch windows
are not pure instruction latency; no default selector or speedup promotion is
made. Public/owner overhead and import-once weight reuse need further work.

Validation: 28 checkpoint/reference tests, 39 combined documentation/reference
tests, Ruff and zero-error mypy for the two recorders. Native full-layer proof
uses the unchanged compiled program from PR972. Each portable artifact is
stored once as lossless .json.xz; its decompressed SHA matches the row. This
packet does not add compiler-free replay proof for these specific model roots.

## Reproduce

On Tajasaurus, source the matching public compiler/runtime environment and run:

```sh
python -m benchmarks.rocm.benchmark_checkpoint_quantized_mlp \
  --activations benchmarks/baselines/nvfp4_model_activations_20261015/inputs.npz \
  --rows 1 16 128 --output-dir /home/angstorms/scratch/checkpoint-mlp-quality/device
```

## Remaining

This is one full layer on captured inputs, not corpus/perplexity or full-model
quality acceptance. Bounded/wider/differentiable composition and original
NVIDIA W1.1, saved-LSE forward/backward and ROCm performance obligations remain
open. Apple/NVIDIA/x86 physical execution receives no evidence from this packet.
