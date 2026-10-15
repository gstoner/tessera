# NVFP4 ingest on captured Qwen3 model inputs

Owner ROCM-NVFP4-INGEST-1 / E2E-REAL-6.
Synchronization key: NVFP4-MODEL-ACTIVATIONS-20261015.

## What was measured

`inputs.npz` holds 256x4096 actual layer-0 MLP input values from the pinned
BF16 Qwen3-8B reference (two authored 128-token text sequences). The upstream
[Qwen3 decoder implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py)
runs on CPU in BF16. All executed attention/norm parameters and the selected
embedding-row windows come from revision b968826d9c46dd6066d109eabc6255188de91218.
The capture stops at the MLP entry hook before its deliberately unmaterialized
weights execute. Two captures are bitwise identical. `inputs.json` records
parameter/range hashes, tokenizer/config identity, module/dependency versions,
input hashes and the exact boundary. Torch is a reference dependency only;
the native benchmark ran in the separate compiler environment without Torch.

The pinned gate/up source tensors and their independent globals match the
historical checkpoint-format-controls packet exactly. No model weight shard
is stored here. `quality-formats.json` records the real input identity and
native ingest plus 18 correctness-gated RX9070XT/gfx1201 format arms. They
use native Graph/Schedule/Tile/Target/LLVM/HSACO packages. Ingest uses checked
host readback before consumer upload in this recorder; this is not the fully
resident whole-program transport benchmark.

## Projection quality

These are **diagnostic f64 products of decoded weights with identical captured
BF16-model inputs**, before activation quantization, folding or output rounding.
They measure projection error against the selected BF16 checkpoint, not model
accuracy or deployed NVIDIA NVFP4 execution.

| Rows | Shipped NVFP4 decoded weights | Direct BF16->MXFP4 joint SSE | NVFP4->MXFP4 ingest | Ingest versus shipped NVFP4 |
| --- | ---: | ---: | ---: | ---: |
| 128 | 6.0851% | 7.4334% | 9.7480% | 7.4131% |
| 256 | 6.0802% | 7.4189% | 9.7367% | 7.4077% |

At 256 rows, the source projection SQNR is 24.32 dB shipped, 22.59 dB direct,
and 20.23 dB after ingest. This retains a material requantization penalty on
actual model inputs. It does not establish an application acceptance threshold.

Native-output RMS error additionally includes each arm's activation
quantization, declared folded-weight approximation and BF16 output store.
Across the two shapes it is 2.35-2.36% for FP8 K128/N128, 2.45-2.46% for
MXFP8, 7.62-7.64% for direct joint-SSE MXFP4, and 9.89-9.90% after NVFP4 ingest.
The K32/N1 FP8 control and native-packed MXFP4 controls remain in the packet;
physical schedules differ, so this is not an isolated dtype-cost comparison.

## Correctness and timing

The native converter matches packed codes/exponents bitwise and bounded f64
loss statistics against the independent reference. Every native format arm
passes the elementwise f32-accumulation plus BF16-store error bound. Balanced
native event/dispatch windows and completed host calls are separate. Converter
median is 17.59 ms in its resident event region and 134.23 ms for checked host
execution. These are characterization numbers, not an optimization A/B or
selector promotion. Full model quality, representative-corpus evaluation,
wider layouts/composition and the W8A8/MXFP4 attribution obligations remain open.

The measurement source epoch is eeaebfff6d1431952379ee2aeb2e892ea8f4554d.
Later cleanup adds only local annotations/identity typing casts; the normalized
executable AST check and exact source hashes are in annotation-equivalence.json.
Compiler/runtime arithmetic and images are unchanged. All 45 host input guards
pass, including four new wrong/missing Content-Range cases that failed before
hardening. A same-sized response from a different byte range can no longer be
accepted as checkpoint evidence.

## Reproduce and inspect

The optional reference environment is pinned by reference-packages.txt; the
Torch CPU wheel comes from https://download.pytorch.org/whl/cpu. It is only
needed to regenerate inputs with capture_qwen3_layer0_activations.py. The
provided NPZ/JSON lets the native comparison run without that environment.
Run benchmark_gfx1201_checkpoint_formats.py with `--activations inputs.npz`,
`--m 128 256 --native-ingest --include-native-packed --windows 3`, and matching
LLVM/MLIR23.1.1 tools on gfx1201.

To avoid duplicate artifacts, 18 raw disassemblies are stored as seven unique
compressed native-image disassemblies. isa-index.json retains original headers
and hashes. `python verify_isa.py` reconstructs every original hash exactly;
only the temporary object-file pathname header is normalized. No instruction
bytes or mnemonic text is discarded. The raw packet's isa_file names resolve
through this index.
