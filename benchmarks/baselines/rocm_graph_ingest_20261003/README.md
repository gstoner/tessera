# Native NVFP4 Graph ownership checkpoint

Owner: ROCM-NVFP4-INGEST-1. Synchronization key: ROCM-GRAPH-INGEST-2026-10-05.

The three-result semantic Graph operation retains packed codes, group-major
exponents, and independent signal/error statistics. A content-addressed Schedule
record binds the numeric policy, projection boundaries, target, operand names,
storage layout and private-output ownership. Native passes lower that record
through Tile and ROCm Target IR to GPU MLIR/LLVM/HSACO.

## Current evidence

- Super-Bear registry-resumed.txt: 315 registry/diagnostic/pass tests passed.
- Tajasaurus graph-device-resumed.txt: 27 Graph/Target/device tests passed.
- Tajasaurus graph-execution.txt: one Graph-originated conversion executed on
  the RX 9070 XT / gfx1201; packed codes/exponents matched bitwise, independent
  f64 signal/error statistics passed after three resident HIP-event windows.
- Super-Bear RTX 5070 UUID and toolchain availability were revalidated after
  maintenance. No NVIDIA conversion parity is inferred.

The physical execution test uses a raw HIP harness with six separately allocated
buffers. It proves the Graph/Schedule/Tile-originated image computes the conversion;
it does not prove public checked package ABI integration or ordinary JIT dispatch.
Pinned-checkpoint native conversion, checked ABI, public backend/AD/conformance
registration and ingest-plus-consumer timing remain open.

FP8, MXFP8 and MXFP4 remain independent correctness, quality and performance gates.
No selector or default route is promoted.

## Checked package increment, 2026-10-05

The explicit package API now retains the Graph, Schedule, Tile, native image and
six-buffer descriptor. Validation binds every IR digest, exact storage/shape,
projection boundary, fixed launch geometry and private output ownership. The
synchronous host call performs policy-value preflight, allocates distinct device
outputs, bit-preserves scale storage, waits before readback and frees allocations
after completion. Ordinary JIT/native-executor integration remains open.

Tajasaurus package.txt: 12 package tests passed, including serialization replay,
Graph/Tile/ABI/geometry mutations, numerical conversion and unchanged caller
inputs. Invalid dtype, shape, non-finite globals and negative scales are rejected
before HIP access. Super-Bear package-drift.txt: 322 focused registry tests passed.

Three synthetic rows in gfx1201-package.json pass correctness before and after
timing on the RX 9070 XT:

| N, K | Resident HIP-event median ms | Checked host-call median ms |
| --- | ---: | ---: |
| 67, 256 | 0.165606 | 3.635355 |
| 513, 1024 | 0.389722 | 6.804982 |
| 4097, 4096 | 7.032511 | 46.526162 |

Resident event windows cover repeated resident launches. Host-call windows include
validation, module load, allocation, transfers, conversion, synchronization and
cleanup; compilation and oracle work are excluded. These scopes are not compared
as a kernel speedup. This is synthetic conversion evidence, not pinned-checkpoint
conversion-plus-consumer timing. The explicit f64-statistics contract rejects
effective scales that would overflow block statistics.

No format default is promoted. FP8, MXFP8 and MXFP4 remain mandatory independent
correctness, model-quality and performance gates. Graphify is unavailable on the
authoritative scratch host; no graph refresh is claimed.
