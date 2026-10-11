# Bounded resident tensor frontend — SM120

Owner: W1.1 / E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Synchronization key: SM120-BOUNDED-RESIDENT-FRONTEND-20261009.
This extends the ordinary resident frontend in PR 927.

## Architectural contract

A shape-bounded public JIT call accepts all-resident compact CUDA roots.
Its specialization key uses validated CUDA metadata rather than host coercion,
and substitutes declared capacities only for bounded axes. Contraction and
bias/residual extents are checked against the active frame. Unbounded axes
and storage dtypes remain real specialization boundaries.

The original tracer Graph is retained. The native MLIR export records its
original Graph witness, dynamic axes, capacities and actual member Graphs;
Schedule/Tile/native image generation owns the dynamic lowering. No Python
kernel generation or backend arithmetic is introduced.

The existing prepared C++ owner allocates bounded intermediate/result capacity,
orders explicit producer streams and returns independent host results.
One owner/program can serve changing active shapes and transitions between
resident and host inputs for this same row-major two-sided producer DAG.

## Numerical and lifecycle proof

Super-Bear: NVIDIA GeForce RTX 5070, SM120, CUDA 13.3,
matching LLVM/MLIR 23.1.1 compiler and native provider.

- 534 focused host tests pass, one skips. These cover bounded native projection,
  metadata specialization, frontend authority and ABI/registry drift.
- 39 new owning device tests pass: FP16/BF16; ordinary, reordered, deeper and
  fused consumers; same/different producer streams; positional/keyword calls;
  independent combinations of bounded M/N/K; host/resident transitions;
  malformed warm frames and recovery.
- 114 adjacent device regressions pass for older bounded host-array producer
  routes and the static public resident frontend.
- Pending-write tests prime every private allocation before queuing a delayed
  producer upload. Warm compiler/tracer/eager entry is forbidden.
- Active test frames include singleton, ragged and full capacities. Returned
  arrays survive source owner closure; scratch accounting and package identity
  stay fixed. Invalid frames reject before compiler/native context entry.

## Benchmark packets

Two isolated fresh processes each record four programs and 20 active frames.
Each program retains one contract digest across frames. Native plans and image
digests are stored once per program; rows reference their contract digest.

Declared MNK capacities are (129,65,513). Frames are
(17,19,35), (129,65,513), (1,1,1), (63,31,255), then (17,19,35).
Each has seven counterbalanced public-call samples and native CUDA-event
measurements using 128 repeats. Producer/consumer grouped event samples are
recorded separately. Numerical checks use an independent FP64 stage oracle;
host and resident public outputs agree bitwise.

Across both packets:

| Timing domain | Median range |
| --- | --- |
| Native whole-program CUDA events | 25.6–63.5 microseconds |
| Public resident completed wall time | 0.663–1.375 milliseconds |
| Public host-input completed wall time | 0.616–1.314 milliseconds |

Resident/host wall-time ratios span 0.982–1.167. This is functional dynamic
integration evidence, with no general resident speedup promotion.
Public wall time includes input waits, binding and completed output download;
native event timing is a distinct domain. Worst recorded absolute error against
the stage oracle is 0.000359.

Both packets record actual device UUID, current source hashes, compiler/provider
hashes, native dynamic plans, images and scratch accounting. The compiler and
CUDA provider are unchanged from PR 927.

## Remaining obligations

Mixed host/device roots, pitched/column-major resident layouts, single-sided or
generic tensor producers and resident AD integration remain open. This proof
does not close W1.1 or the broader five-slice program. Quantized schedules and
saved-LSE attention paths are unchanged. Apple, ROCm and x86 execution require
their own architecture-specific proof.

Recorder: `benchmarks/nvidia/record_bounded_resident_tensor_frontend.py`.
Device: `tests/device/nvidia/test_bounded_resident_tensor_frontend.py`.
Host: `tests/unit/test_bounded_resident_tensor_frontend.py`.

Run on the owning WSL host with matching TESSERA_OPT, TESSERA_NVIDIA_OPT,
TESSERA_NVIDIA_PTX_LAUNCH_LIB, CUDA_HOME and project PYTHONPATH:

```bash
python benchmarks/nvidia/record_bounded_resident_tensor_frontend.py --output /scratch/run.json
```
