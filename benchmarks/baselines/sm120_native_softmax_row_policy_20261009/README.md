# Native SM120 softmax row policy

Owners: W1.1 / E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Synchronization: SM120-SOFTMAX-ROW-POLICY-20261009.
Recorder: `benchmarks/nvidia/record_native_softmax_row_policy_ab.py`.
Packets: [run1.json](run1.json), [run2.json](run2.json).
Both bind compiler binaries, runtime, source and recorder hashes to the active
RTX5070 identity. Supplemental imported-helper and backend-tool witnesses are
in [source-witnesses.json](source-witnesses.json).

## Architecture

Graph-to-Schedule now chooses cooperative_128 for SM120 softmax with a compiled
column extent >=256. Short static rows remain serial. Explicit serial/cooperative
requests override the default. Bounded packages select using compiled capacity
and run checked active frames through the same native image.
The Python adapter reads that Schedule decision and exported entry; there is
no Python shape selector or Tile constructor.

Ordinary frontend Graph → native Schedule/Tile views and typed fragments →
NVIDIA Target → LLVM NVPTX/PTX → checked C++ ordered resident DAG.
Existing max-subtracted, NaN-propagating, approximate-exp2/f32 accumulation and
storage-rounding policies remain in force. Cooperative sum grouping can change
rounding; no bitwise serial-versus-cooperative equivalence is claimed.

## Evidence and timing

Two fresh runs alternate seven windows per arm, 128 native launches per window.
Identical traced Graphs and root values are compiled by the frozen serial-control
and candidate binaries. Each result agrees with an independent f64 oracle before
and after timing. Portable replay and unchanged consumer Target IR are checked.
DAG windows measure device execution after incoming waits. Grouped stage windows
are separate and nonadditive. Prepared-invoke wall includes metadata and native
completion; compilation, uploads and output downloads are outside that interval.
Standalone f32 uses native C++ CUDA-event timing with staging outside the event
window. Its results are checked at rtol=3e-5, atol=2e-6.

Named long-row DAG device-program gains range 3.04–4.60x in the two fresh runs.
At the 256-column boundary the whole-chain gain is 1.34–1.41x. Short static rows
keep the same serial Tile witnesses; small differences are timing variation.
Bounded short frames are retained as controls, rather than omitted.

## Validation

- 451 compiler/registry tests pass, 17 skip; explicit overrides, 255/256 boundary,
  caller Graph immutability, native replay and sibling serial policies are covered.
- 181 exact-device tests pass, four static-envelope skips: f16/bf16/f32, finite,
  NaN/+Inf/-Inf inputs, explicit schedules, host/resident launches and DAG ownership.
- 18 ordinary public JIT softmax/safe alias cases pass: typed storage, nested
  rows, serialized artifact ancestry and compiler/eager-forbidden warm replay.
- Two additional f16/bf16 long-capacity lifetime tests pass: short→long→singleton→short,
  delayed producer writes, constant private capacity, host-native/resident bit parity
  under the same schedule, and output survival after external roots close.
- Python Ruff and zero-error CI mypy ratchet pass.
- All 32 generated documents are in sync; claim lint, compiler-plan and 15
  recorder/packet citation and audit-document gates pass.
- Broader CPU unit selection: 24,641 passed, 5,905 skipped, two failures. Both
  are the existing scaled-matmul batching/transpose closure tests; the full
  receipt is retained in full-unit.txt.

Generic scaled-matmul batching/transpose closure remains open. This packet does
not claim a green full unit lane, arbitrary producer composition, attention/AD
closure, sibling execution, or FP8/MXFP8/MXFP4 performance. Those quantized matmul
schedules and semantics are unchanged by this floating softmax policy.

## Reproduction

On the owning RTX5070 WSL host with matching LLVM/MLIR23.1.1 tools and runtime:

```sh
python -m pytest tests/unit/test_native_softmax_row_policy.py -q
python -m pytest tests/device/nvidia/test_cooperative_softmax.py \
  tests/device/nvidia/test_native_softmax_row_policy_lifetime.py -q
python benchmarks/nvidia/record_native_softmax_row_policy_ab.py \
  --control-opt /path/to/frozen-pr924-tessera-opt --output /path/to/run.json
```

Control source commit: 8d1c9c800c796261e1fa9309dd14f8e7e80fc65e.
The recorder obtains the candidate from TESSERA_OPT; do not run timing concurrently
with pytest or Graphify. Source witnesses are checked against the final packet.

## DAG device-program comparison

| Storage | Bounded | M,N,K | Run1 gain | Run2 gain |
| --- | --- | --- | ---: | ---: |
| fp16 | False | [17, 19, 35] | 1.068x | 1.011x |
| fp16 | False | [17, 65, 255] | 1.001x | 1.002x |
| fp16 | False | [17, 65, 256] | 1.341x | 1.344x |
| fp16 | False | [129, 65, 513] | 3.236x | 3.047x |
| fp16 | False | [17, 257, 1024] | 4.596x | 4.379x |
| fp16 | True | [17, 19, 35] | 1.034x | 1.061x |
| fp16 | True | [129, 65, 513] | 3.091x | 3.042x |
| bf16 | False | [17, 19, 35] | 1.024x | 0.999x |
| bf16 | False | [17, 65, 255] | 1.000x | 1.003x |
| bf16 | False | [17, 65, 256] | 1.349x | 1.415x |
| bf16 | False | [129, 65, 513] | 3.130x | 3.049x |
| bf16 | False | [17, 257, 1024] | 4.541x | 4.447x |
| bf16 | True | [17, 19, 35] | 1.007x | 1.077x |
| bf16 | True | [129, 65, 513] | 3.059x | 3.058x |

## Standalone f32 device comparison

| Rows,columns | Run1 gain | Run2 gain |
| --- | ---: | ---: |
| [3, 35] | 0.998x | 1.112x |
| [3, 255] | 0.998x | 0.978x |
| [3, 256] | 2.822x | 2.821x |
| [129, 513] | 15.673x | 15.421x |
| [17, 1024] | 15.284x | 14.827x |
| [3, 4097] | 40.154x | 40.284x |
