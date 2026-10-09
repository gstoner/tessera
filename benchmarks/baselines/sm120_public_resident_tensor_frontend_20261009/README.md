# Public resident tensor frontend — SM120

Owning items: W1.1 / E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 /
AD-RESIDUAL-EVAL-1. Synchronization key:
SM120-PUBLIC-RESIDENT-TENSOR-FRONTEND-20261009.

## Contract and implementation

Ordinary `@jit` now traces all-resident compact row-major CUDA inputs using
validated array-interface metadata and abstract tracer specifications.
The named primal contract is two-sided RMSNorm/softmax producer DAGs feeding
matmul, with FP16/BF16 roots, reordered arguments, one or two stages per
operand, and checked FP32 bias/residual epilogues. No input host coercion or
eager arithmetic supplies the compiled result.

The existing typed Graph -> Schedule -> Tile -> NVIDIA Target -> LLVM/PTX
packages execute through a prepared native C++ owner. The owner orders
explicit producer streams with events, retains intermediate and output
allocations, validates capacity/context/aliasing, and downloads the independent
host result under the invocation mutex. The new resident-to-host ABI avoids
creating a Python CUDA session and GPU output on every public call.

## Evidence

Super-Bear: NVIDIA GeForce RTX 5070, SM120, CUDA 13.3, LLVM/MLIR 23.1.1.
Both packets record the actual device UUID, source hashes, compiler/provider
hashes, Graph IR, program contract digest and native image digests.

- 106 owning device tests pass, including the separately enabled saved-LSE
  attention JVP regression. Four static-envelope cases skip checks applicable
  only to bounded-capacity fixtures.
- The new public frontend has 38 device cases: FP16/BF16, ordinary/reordered/
  deeper/fused graphs, same/different producer streams, positional/keyword
  roots, changed asynchronous inputs, native descriptor corruption, cached
  compiler/body refusal and result lifetime after source owners close.
- 456 focused matching-compiler host gates pass, with one skip; these cover
  frontend authority, resident metadata, DAG contracts, runtime ABI and
  operation/diagnostic/pass registries.
- Two isolated fresh-process benchmark packets cover eight arms each:
  MNK=(17,19,35)/(129,65,513), FP16/BF16 and one/two stages per operand.
  Independent FP64 stage oracles pass; host and resident results agree
  bitwise.

## Timing and limits

Seven counterbalanced completed-call windows and 128 repeated native event
launches per window are recorded. Native whole-program event medians span
27.0–61.0 microseconds. Public resident completed-call medians span
0.729–1.136 milliseconds; matching host-input public calls span
0.655–1.215 milliseconds. Grouped member samples are a separate event domain.
Public calls include input-stream waits, binding checks and output allocation/
download; native events do not establish total application latency.

Resident input does not imply a speedup over the matching host-input path.
Use each recorded row and its ratio for comparison; no general performance
promotion follows from these packets.

Mixed host/device roots, pitched/general layouts, bounded dynamic public
resident entry, generic producer/accumulator graphs and resident AD integration
remain open. FP8/MXFP8/MXFP4/NVFP4 physical schedules are unchanged. No Apple,
ROCm or x86 exact-device parity is established by CUDA execution.

Recorder: `benchmarks/nvidia/record_public_resident_tensor_frontend.py`.
Device tests: `tests/device/nvidia/test_public_resident_tensor_frontend.py`.
Unit tests: `tests/unit/test_public_resident_tensor_frontend.py`.
