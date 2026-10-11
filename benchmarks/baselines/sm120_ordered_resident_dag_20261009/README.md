# Ordered resident tensor DAG — RTX 5070

Synchronization key: SM120-ORDERED-RESIDENT-ROOTS-20261009.
Owners: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.

Recorder: `benchmarks/nvidia/record_ordered_resident_tensor_dag.py`.
Packet: [sm120.json](sm120.json). It records the active device identity,
source/tool hashes, member image/descriptor witnesses and actual member policies.

## Contract and scope

Python frontend → Graph MLIR → native SSA export → Schedule/Tile views and
typed fragments → NVIDIA Target → LLVM NVPTX/PTX → checked C++ ownership.
Only root metadata projection occurs in Python; intermediate allocation,
producer ordering, launches and completion belong to C++.

The two ordered resident ABI entry points accept borrowed f16/bf16 roots.
C++ checks allocation context, capacity, shape and aliasing, then records
producer events and waits on the launch stream. Roots and producer streams
must remain valid and immutable until synchronous completion. Output and private
intermediates have separate ownership. Static and bounded frames, one/two-stage
chains on both operands and fused bias/residual/activation consumers are tested.
Primitive CUDA array-interface typestr carries the storage type; opaque BF16
requires an explicit compatible dtype hint.

Tests exercise pending producer writes, shared/different/legacy/per-thread streams,
changed bounded frames, portable replay, invalid capacity/context, retained outputs
and closure of external root owners. Resident outputs are bitwise equal to the
host-staged native route and agree with an independent f64 oracle.
CuPy/PyTorch are not installed on this host; those providers are not execution claims.

## Timing

Each native window contains 128 launches; seven changed-input windows are retained.
Program CUDA events start after incoming waits. Grouped stage windows measure
each stage separately and are not additive. Resident package wall intervals
include metadata/seal checks, owner/session/output allocation, native execution
and synchronous completion, with pre-existing device roots. Root uploads,
output downloads and compilation are outside those intervals.
These are route characterization measurements, with no speedup or selector promotion.

## Validation

- Final ordered exact-device suite: 64 passed, four static-envelope skips.
- Adjacent ordered/legacy ownership suites: 159 passed, four static-envelope skips.
- Focused metadata, compiler, ABI, dtype and diagnostic gates: 525 passed.
- Targeted mypy and the CI mypy ratchet: no issues; Python Ruff passes.
- All 32 derived documents are in sync; public claim lint and compiler-plan gates pass.
- Post-repair recorder/packet citation and audit-document gates: 15 passed.
- Full CPU selection: 24,638 passed, 5,905 skipped, five failures.
  Three missing recorder/packet documentation failures are repaired by this manifest
  and the four backend queue citations, with focused post-repair gates retained.
  Two existing scaled-matmul batching/transpose closure failures remain open.
  This packet does not establish a green full unit lane.

Public package resident execution is covered. Ordinary public JIT tracing of
CUDA roots, mixed host/device roots, padded layouts, arbitrary producer composition
and general AD remain follow-ups. CUDA proof does not establish sibling execution.

## Reproduction

Use the owning RTX 5070 WSL host, matched compiler tools, CUDA runtime libraries
and validation environment. Run tests before the recorder, with no concurrent
pytest or Graphify job:

```sh
python -m pytest tests/device/nvidia/test_ordered_resident_tensor_dag.py -q
python benchmarks/nvidia/record_ordered_resident_tensor_dag.py \
  --output benchmarks/baselines/sm120_ordered_resident_dag_20261009/sm120.json
```

## Measured medians

| Storage | Depth | M,N,K | Program event µs | Resident package wall ms | Max abs error |
| --- | --- | --- | ---: | ---: | ---: |
| fp16 | 1 | [17, 19, 35] | 28.39 | 3.514 | 8.6763874e-05 |
| fp16 | 1 | [129, 65, 513] | 41.02 | 3.681 | 5.4813885e-05 |
| fp16 | 2 | [17, 19, 35] | 47.31 | 5.571 | 1.5719794e-05 |
| fp16 | 2 | [129, 65, 513] | 192.49 | 6.455 | 9.1565704e-05 |
| bf16 | 1 | [17, 19, 35] | 28.09 | 3.888 | 6.6589564e-08 |
| bf16 | 1 | [129, 65, 513] | 39.63 | 3.275 | 0.00021265261 |
| bf16 | 2 | [17, 19, 35] | 45.78 | 5.556 | 1.6391277e-07 |
| bf16 | 2 | [129, 65, 513] | 192.43 | 6.169 | 0.0005836785 |
