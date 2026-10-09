# SM120 two-sided normalized tensor DAG

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Synchronization key: SM120-TWO-SIDED-PRODUCER-DAG-20261009.

## Executable boundary

Python frontend capture feeds native MLIR SSA export, actual outlined members,
Schedule/Tile views and fragments, NVIDIA Target IR, LLVM NVPTX and checked
native images. The C++ owner executes both operand chains and allocates all
intermediate/ping-pong buffers. Python allocates roots and returned output only.

Named envelopes: fp16/bf16 RMSNorm/softmax on LHS, softmax/layer-norm on RHS,
one or two stages per operand, static or bounded M/N/K capacities.
Public JIT also proves reordered roots and fp16 bias/ReLU/residual epilogues.
Portable replay retains native ancestry. The resident result owns its output
until close; reading a closed buffer is rejected before invoking CUDA.

## Validation

Super-Bear RTX 5070 sm_120, LLVM/MLIR 23.1.1 and CUDA 13.3:
146 device cases passed across the new DAG and legacy native partition,
bounded-chain and resident routes. Tests compare independent oracles, changed
inputs and retained results; reject stale profiling leases and forged manifests;
and forbid recompilation during warm replay. 368 native-manifest/diagnostic/
pass-metadata/audit gates passed. Ruff passes; mypy reports zero errors.

## Timing

Recorder: benchmarks/nvidia/record_two_sided_tensor_dag.py.
Run with --output benchmarks/baselines/sm120_two_sided_tensor_dag_20261009/sm120.json.
The packet queries model/UUID, hashes source and all three compiler/runtime
tools, and records native manifest, Schedule/Tile and image witnesses.
Eight arms pass independent oracles before and after timing.
CUDA event program windows include ordered stream submission gaps.
Grouped stage repeats isolate producer/consumer windows and are not additive
program time. Prepared host and warm public JIT measurements include host work,
copies and synchronization. Cold compilation is separate.
Timing covers prepared host-input execution. Resident output has numerical/
lifetime proof; no resident performance advantage is claimed.

## Remaining

External CUDA roots, general producer/layout/dtype/AD closure and broader
attention/ROCm obligations remain open. FP8/MXFP8/MXFP4 evaluation is required
before strategy/default promotion. No speedup, universal compiler closure or
sibling physical claim. Generic scaled batching/transpose failures remain open.
