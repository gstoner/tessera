# Prepared native NVFP4 import-once expansion/product

Owner ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1.
Synchronization key: NVFP4-NATIVE-MULTI-OUTPUT-20261015.

Normal Python frontend capture produces the native two-stage SSA program.
Both member graphs lower through native Schedule/Tile/ROCm/LLVM into checked
images. A prepared multi-result HIP owner holds both expansion outputs privately.
One producer execution imports weights; subsequent activation-only updates and
consumer-only invocations reuse those values. No numerical backend code or
launch loop is authored in Python.

## Matched synthetic measurements

| MxNxK | Full update + all stages + read (ms) | Activation update + consumer + read (ms) |
| --- | ---: | ---: |
| 200x2048x1536 | 1.298 | 0.609 |
| 256x4096x2048 | 2.737 | 0.834 |

Three alternating-order matched windows use the same package/images and exact
input values. Every result is bitwise equal to an independent FP32 oracle
built from nibble decoding and dyadic scales. The host difference includes
avoiding weight boundary validation, weight upload and expansion; it is not
a faster matmul kernel or model-serving throughput claim. The checkpoint
input is synthetic, and no model-quality or physical-selector promotion is
established.

Packets retain initial package compilation/packaging, native owner setup/upload, initial
import host/event timings, three per-member captured HIP graph windows and
completed update/invoke/read samples. Native invocation events include enqueue
gaps. Member graph windows exclude capture/instantiate but include graph dispatch.
These clock domains remain distinct. Per-call upload byte counts describe
the admitted host frame, not hardware memory counters.

## Correctness and lifecycle

The prepared fixture passes 22 exact-gfx1201 cases: FP32/BF16 stores, permuted
roots, K16/K48/K80 ragged extents, ordinary frontend capture with eager numerical
oracles forbidden, invalid import, corrupted member binding, transactional
invalid partial updates, weight-driven transitive invalidation a forged launch geometry with a recomputed binding digest, a forged launch geometry with a recomputed binding digest, and a verified
allocation-cache hit that cannot reuse the previous session's initialized data.

One final frozen-source gate passes 800 cases, including all 22 prepared
cases, native multi-result admission, original single-result host packing,
readback/event/projection contracts, existing public primal/JVP routes,
audit lifecycle and diagnostic/pass metadata. The earlier 435/363/72-case
gates overlap this certificate and are retained as intermediate evidence,
not additive unique-test totals. Strict imported-contract mypy passes both
changed production modules; Ruff is clean.

## Reproduce and limits

In host WSL on RX9070XT/gfx1201, source the matching LLVM/MLIR23.1.1 environment.
Build the native owner provider from the recorded source, then run
benchmarks/rocm/benchmark_nvfp4_prepared_expansion.py with --provider, --output,
--m, --n and --k. The benchmark records exact device, compiler/provider/image,
native plan, checked binding and relevant source identities.

Ordinary composed public JIT, compiler-free serialized replay, original
checkpoint composed validation, bounded source-preserving frames and full-model
quality acceptance remain open. This pure SSA ownership protocol does not
establish transactional recurrent-state behavior for ReplaySSM. No Apple,
NVIDIA or gfx1151 physical proof is inferred.
