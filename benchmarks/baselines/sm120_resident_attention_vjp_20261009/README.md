# SM120 resident public attention backward

Synchronization key: SM120-RESIDENT-ATTENTION-VJP-20261009.
Owners: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.

Public native_backward consumes compact rank-four FP32 CUDA Q/K/V, optional
broadcast bias and output cotangent. Metadata-only capture emits typed Graph
IR; native MLIR AD and Schedule/Tile/Target/LLVM-PTX supply the unchanged
saved-LSE forward/backward images. The frontend structural certificate records
zero concrete executions and seals the exact target-annotated Graph used by
native backward. The execution certificate records prohibited source
reexecution and exact SM120 physical attestation.

A shared native dependency validator checks context, allocation extent,
alignment, device memory type, private-arena aliasing and producer streams.
Events order both primal and cotangent writes before private snapshots.
Forward output/LSE stays private through backward consumption; requested
gradients are independently downloaded before return. Borrowed roots must
remain live for the call. Concurrent future writes are not automatically
ordered by this synchronous envelope.

## Validation

379 focused host/frontend/runtime/diagnostic/pass tests pass; Ruff and mypy
ratchet pass with zero errors. The forward/reverse regression suite passes
44 owning RTX5070 tests. After exact Graph identity and scalar finite-
difference checks were added, all 19 owning resident reverse tests pass.
Coverage includes all six Q/K/V permutations, K=5/129, causal GQA, requested
gradient order, broadcast bias, pending K/cotangent writes on different
streams, held gradient outputs, allocation/alignment/stream-count rejection,
recovery, exact Graph pins and independent scalar finite differences.

Two isolated six-program packets compare requested gradients to an independent
float64 analytical oracle before timing. Maximum absolute error: 1.692e-08.
Packets query the actual RTX5070 UUID, SM120 and driver, and pin current source,
compiler and native provider. Every row retains its native compiler receipt.

## Timing

Each packet alternates nine resident/host rounds for six programs.
Resident completed public calls span 1.349–1.838 ms, host calls 0.970–1.422 ms,
with resident/host ratios 1.274–1.423. Resident forward device medians span
10.91–55.14 us; backward medians span 4.54–47.33 us.
Events exclude producer waits/snapshots. Public calls include certificate and
binding overhead, private snapshots and host gradient downloads.
This is execution proof, not a resident speedup or physical-schedule promotion.

Reproduce with the matching SM120 compiler and native provider:
python -m benchmarks.nvidia.record_resident_attention_vjp --output /scratch/run.json

## Remaining work

Resident ordinary tuple forward, dynamic/pitched/mixed frames, composed or
nested attention AD, GPU gradient output ownership, automatic external future-
write tracking and frontend binding performance remain open. No exact-device
claim is transferred to ROCm, Apple or x86. The complete five-slice objective
remains active.
