# Checkpoint-preserving K16 native FP8 consumer control

Owner ROCM-NVFP4-INGEST-1 / ROCM-FP8-BLOCKSCALE-1.
Synchronization key: NVFP4-CHECKPOINT-K16-20261015.

## Scope and identities

Two full layer-0 gate/up projections from pinned nvidia/Qwen3-8B-NVFP4 and the
same original BF16 checkpoint are evaluated on the captured BF16 model inputs
from nvfp4_model_activations_20261015. Both checkpoint records and both M128/
M256 activation hashes match that earlier packet exactly (comparison.json).
HTTP Content-Range checks remain active. No model shard or duplicate activation
NPZ is stored here. Torch is not used by this native measurement process.

Host reference expansion retains every E2M1 code value as an exactly
representable FP8 value and each original K16 E4M3 block/global product.
Products are rounded once to FP32 scales, with overflow/underflow rejected;
the recorded source-weight relative rounding error is 2.68e-8. Independent
nibble decoding bounds that error before any native package is launched.

The consumer follows the existing native Graph -> Schedule -> Tile -> ROCm
Target -> LLVM -> HSACO route. Both cases select the LDS 128x128 eight-wave
body and pass the independent elementwise f32-accumulation/BF16-store bound.
The two M sizes use one byte-identical native image, with separate descriptors.
This is not yet a native NVFP4 expansion producer or packed matmul transport.

## Quality and cost

| M,N,K | Shipped-weight projection RMS | Native RMS vs BF16 projection | Device execution/dispatch median | Completed host median |
| --- | ---: | ---: | ---: | ---: |
| 128,24576,4096 | 6.0851% | 6.2349% | 0.807 ms | 18.513 ms |
| 256,24576,4096 | 6.0802% | 6.2307% | 1.332 ms | 19.896 ms |

Native error against the shipped-weight projection itself is about 1.36%.
The expanded representation changes the unquantized-input projection by only
1.76e-8 relative RMS at M256. Its physical input size is approximately 127 MB;
expanded FP8 bytes and FP32 K16 scales cost more storage than packed MXFP4.

The earlier same-input MXFP4-ingest arm has about 9.89% native RMS at M256;
its unquantized-input weight projection is 9.74%, versus 6.08% for shipped
weights. These are source/input-matched quality characterizations across
different compiler epochs, representations and activation quantization.
They are not a controlled performance A/B or isolated dtype-cost comparison.
All FP8/MXFP8/MXFP4 arms remain available in the earlier packet.

Timing separates resident HIP graph execution/dispatch, completed host calls
and compile/prepare (384 ms then 49 ms, including Graph/package work). Graph
dispatch overhead is included; no pure-kernel latency claim is made. Raw
samples and deduplicated image disassembly are retained.

## Architecture consequence and remaining gate

A native source-preserving producer is justified for evaluation. Its checked
contract must consume packed [N,K/2] bytes, E4M3 [N,K/16] bytes and independent
projection globals; return ordinary FP8 [N,K] values and FP32 [K/16,N] scales;
preserve original row boundaries and signed zeros; and explicitly declare the
FP32 scale-product rounding. Invalid scales/globals and unrepresentable products
must reject. The current reference helper admits native K16 geometry rather
than inheriting the lossy MXFP4 converter's K32 requirement.

Production integration must derive that producer through native Graph,
Schedule, Tile and Target lowering with checked disjoint output ownership.
It must then compose with the existing K16 consumer without a Python numerical
rewrite or host readback/reupload. A fused packed consumer is a further
performance/storage option, not proved by this expanded control.

This packet does not establish full-model accuracy/perplexity acceptance,
general AD/layout closure, or a default strategy promotion. Those gates remain
open. Apple/x86/NVIDIA receive no physical evidence from this ROCm experiment.

## Reproduce

Use matching LLVM/MLIR23.1.1 tools and the gfx1201 runtime, then run
benchmarks/rocm/benchmark_nvfp4_source_preserving_control.py with --compiler,
--llvm-bin, --activations pointing to the earlier packet inputs.npz, --output,
--m 128 256 and --windows 3. The source epoch is 6511429a7. The measured source
hashes, compiler/image/ISA identities and exact device are in checkpoint-k16.json.
The reference-only input helper is covered by 18 focused control cases,
including all codes, signed-zero values/scales, distinct globals, K16/K48/K80
geometry, malformed storage and unrepresentable scale products. The combined
control/capture/checkpoint-input gate passes 51 cases.


## Native rounding boundary probe

Before choosing the producer conversion, a separate native GPU-IR microprobe
multiplies decoded scale/global doubles and narrows to FP32. All 64 lanes match
NumPy reference output bits across 18 signed-zero, subnormal, ties-to-even,
minimum-normal and overflow boundary scenarios. This includes deliberate
invalid-for-import globals only to characterize hardware conversion. The
registered producer will still need its own input validation and complete
Graph/Schedule/Tile/Target proof.

rounding-probe.mlir and rounding_probe.py reproduce the test on gfx1201; the
script uses the existing native GPU pipeline with zero shared memory. It is
explicitly a diagnostic leaf, not a production Graph route. The RDNA4 archive
confirms V_MUL_F64 and V_CVT_F32_F64 before emission. Ordinary native F64 multiply
and FP32 truncation are therefore a measured candidate for the producer, with
no need to assume that this target flushes representable FP32 scale products.
