# SM120 native producer-to-matmul ownership

Owner W1.1 / FRONTEND-IR-MEDIUM-1.
Synchronization key NVIDIA-PREPARED-LHS-2026-10-06.

## Architectural increment

The named static FP16/BF16 RMSNorm, LayerNorm and last-axis softmax -> matmul
chain retains verified Graph -> Schedule -> Tile -> NVIDIA Target ->
NVVM/LLVM/PTX component packages. Native C++ now owns both CUDA modules,
one ordered stream, private intermediate device storage, and context-shared
device scratch plus pinned host staging. Python binds host arrays and the
sealed compiler contract; it constructs no shader, Tile arithmetic or
intermediate numerical result.

Uploads, producer, consumer and readback all queue on the owner stream.
The intermediate never escapes to host or another owner. One native mutex
leases staging/scratch through stream completion and final host output copy.
All inputs are captured before caller output writes, so host aliases retain
the synchronous snapshot contract. Distinct outputs remain caller-owned arrays.
Native guards cover shape, dtype, physical pitches, process, context, closed
handles and one-time producer attachment. Failure after submission drains
the stream before scratch retirement. Public close_native_storage retires
both modules and permits lazy rebinding of the sealed packages.

Static C/F RHS layouts, plain FP32 output and bias/ReLU/residual with final
FP16 output are proved. Existing cooperative_128 producer policies are carried
from the native descriptor, not invented by this host owner.
Compile reports bind the retained full Graph, the executed product artifact,
and ordered producer/consumer fingerprints for Schedule/Tile/Target.
They do not claim that two modules are one monolithic lowered IR. Failed
calls clear prior success receipts rather than reporting a previous product.

## Correctness failure found and fixed

The first benchmark exposed zeros on the prepared 128x1024x64 case while the
canonical route passed. Explicit pinned staging and owner-stream transfers
fixed first-frame behavior. Large-K first-call and changed-value checks now
cover all three producers, both storage dtypes and both RHS orders.
The shared single-matmul owner has additional large first-frame checks.
This increment supersedes the earlier prepared runtime transfer mechanism;
historical standalone-matmul packets keep their original runtime hashes.

## Exact-device proof

RTX 5070 UUID, driver and compute capability 12.0 are queried in every packet.
921 focused checks pass, 67 skip, including 182 owning-device cases.
The new owner suite covers 37 cases: numeric changed inputs/output lifetime,
shared-scratch concurrency, immutable producer contract, context recovery,
public close/rebind, truthful compile reports and large single-matmul frames.
A private-C-ABI fault fixture deliberately rejects a consumer after a real
compiled producer queues; native stream readiness proves retirement, caller
output remains untouched, and subsequent correct execution succeeds.
Existing portable/compiler-disabled replay and single-matmul lifecycle tests
remain included. See regression-tests.txt and native-build.txt.

## Matched benchmark

Eight independent processes run control/prepared then reverse process order
for each C/F RHS layout. Each packet has 24 correctness-gated cases:
FP16/BF16, all three producers, plain/fused output, M,K,N = 17,35,19 and
128,1024,64. Control uses the existing checked descriptor/resident route.
Prepared uses the native C++ owner. Every matched row has identical component
images, consumer ABI, semantic contract, compiler, runtime and source hashes.
The analyzer rejects identity/envelope/binding mismatches and stale sources.

Median per-case prepared/control warm public-wall ratios:

| RHS | Forward | Reverse |
|---|---:|---:|
| row-major | 0.104156 | 0.107967 |
| column-major | 0.105785 | 0.112359 |

This is about 89% lower warm public-call overhead. Cold wall time includes
tracing and compilation. Public/replay wall windows include relevant host
work, transfers and completion. Separate producer/consumer CUDA-event windows
use 20 warmups and 100 repetitions, three samples each, and include driver
dispatch gaps. This is not a kernel speedup or universal compile-latency claim.
FP8/MXFP8/MXFP4 physical strategy selection is not changed.

## Reproduction and remaining scope

On Super-Bear WSL, source .build-sm120-w1-1/validation-env.sh.
Run benchmarks/nvidia/benchmark_lhs_jit_dispatch.py with
TESSERA_NVIDIA_PREPARED_LHS=0 or 1, TESSERA_LHS_RHS_ORDER=C or F and
TESSERA_LHS_PACKET set to the desired JSON path. Run this directory's
analyze.py from the repository root. Use the host device test module
tests/device/nvidia/test_prepared_lhs_owner.py and the captured regression lane.

Common portable replay retains its existing checked route. General producer
composition, AD, dynamic row-RHS, general A layouts, asynchronous/resident
native ownership, wider formats and sibling physical consumers remain open.
The full five-slice objective remains active.
