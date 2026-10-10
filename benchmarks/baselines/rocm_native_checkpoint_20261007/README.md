# Native real-checkpoint NVFP4 ingest

Owner: ROCM-NVFP4-INGEST-1.
Synchronization key: ROCM-NATIVE-CHECKPOINT-2026-10-07.
Exact device: Tajasaurus RX 9070 XT / gfx1201, GPU-28d9e7efbf2ef716.
Implementation and packets remain in the unpublished aggregate.

## Native route and correctness

The pinned Qwen3-8B q_proj and merged gate/up source tensors now enter
a typed Graph ingest package through native Schedule, Tile, ROCm Target and
LLVM lowering to HSACO. The recorded conversion executes the native image;
the host joint-search converter supplies only the independent diagnostic oracle.
Native packed codes and E8M0 exponents match that oracle bitwise.
Independent bounded f64 decoded-weight reductions verify every native block's
signal and error before and after timing. Fault-injection tests reject wrong
codes, exponents or loss statistics before creating conversion timing samples.

The separately packaged native matmul consumes the native conversion outputs
and matches the decoded-weight oracle with zero output error for both groups.
Independent globals and projection row boundaries are preserved. These two
real gate/up globals happen to be equal; unequal-global unit coverage remains
necessary. Source revisions/byte hashes, IR/image hashes, actual compiler,
runtime and source hashes and the live GPU probe are retained.

## Separate timings

| Projection | M/N/K | Native ingest event ms | Consumer event ms | Consumer end-to-end ms | Ingested weights vs BF16 RMS |
| --- | --- | ---: | ---: | ---: | ---: |
| q-proj | [16, 4096, 4096] | 3.225 | 0.076 | 6.339 | 14.99% |
| gate-up | [16, 24576, 4096] | 19.215 | 0.180 | 9.813 | 14.97% |

Conversion samples each cover ten resident native launches with HIP events;
dispatch/enqueue gaps are included. Consumer event samples are independent.
Consumer wall time includes its runtime/module/copy lifecycle. Conversion
checked-host time also includes independent correctness reductions and must
not be interpreted as pure launch overhead or compared with kernel time.
Host reference conversion, checkpoint fetch and compiler packaging are separate.
This packet establishes no comparative speedup or selector promotion.

## Validation and reproduction

host-tests.txt: 359 focused ingest/recorder/operator/diagnostic/pass tests pass,
seven gated cases skip, on Super-Bear WSL. Ruff and focused whitespace checks
pass. q-proj.log and gate-up.log retain exact-device recorder completion.
identity.json binds the actual gfx1201 compiler/runtime/source and GPU.
Schema v2 distinguishes host_reference_ingest_ms from native_ingest timing.

On Tajasaurus, from the scratch checkout:

    source .build-gfx1201-current/validation-env.sh
    export PYTHONPATH=$PWD/python:$PWD TESSERA_ROCM_CHIP=gfx1201
    .venv-movement-capture/bin/python benchmarks/rocm/benchmark_rocm_nvfp4_checkpoint.py --projection-group q_proj --m 16 --repeats 3 --iterations 10 --output q-proj.json
    .venv-movement-capture/bin/python benchmarks/rocm/benchmark_rocm_nvfp4_checkpoint.py --projection-group gate_up --m 16 --repeats 3 --iterations 10 --output gate-up.json

## Remaining scope

BF16-relative loss is unchanged from the earlier host conversion packets;
native integration does not repair information lost by source quantization
or K16/E4M3 to K32/E8M0 conversion. Whole-model/source-activation quality,
model-derived FP8/MXFP8 comparisons, dynamic/layout/AD and other packing variants
remain open. This recorder copies native outputs to host between separately
packaged stages; the existing ordinary JIT resident program has its own narrower
proof and this packet does not establish resident composition for model shapes.
gfx1151 lacks this RDNA4 physical route. Apple, x86 and NVIDIA receive no new
physical execution or schedule promotion from this recorder change.
