# Actual SM120 tensor Graph partition

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Synchronization key: SM120-NATIVE-TENSOR-PROGRAM-2026-10-08.
Recorded 2026-10-08 on Super-Bear, RTX 5070 SM120,
GPU-cba12639-821a-7a10-4cd3-f918f9c0a545.

## Proof

The native exporter outlines the actual Graph operations, preserves SSA
producer/consumer connection and argument order, and records allocation sizes
and read/write lifetimes. Native member projection feeds Graph-to-Schedule,
Schedule-to-Tile and NVIDIA Target/image packaging. The raw adapter constructs
artifact records, never Python semantic Graph operations.

The owning-device test receipt records 32 passing checks. Twelve executions
forbid GraphIRFunction and IROp construction after frontend MLIR serialization.
They compare both the resident intermediate and matmul output with an independent
float64 oracle, including rounding the producer output to FP16/BF16 storage.

The benchmark packet contains 24 correctness-gated profiles: three producers,
two storage dtypes, two RHS storage orders, and shapes M/K/N = 17/32/11 and
128/1024/64. It includes native program digests, image digests, source hashes,
compiler/runtime hashes, three event windows per stage and three end-to-end
wall-time samples. Maximum output error is 0.0010767364.
End-to-end medians span 2.580791–6.734496 ms.
Event windows include device dispatch; no isolated-kernel or speedup claim.

Compiler SHA256:
08165455e4bf6a5babce1680cabf0fdabe006a767ace49ebce6ef12a6b9f620d.

## Reproduce

From the Super-Bear repository root with the matching compiler/runtime environment:

    pytest -q tests/unit/test_native_sm120_tensor_partition.py
    TESSERA_NATIVE_TENSOR_PACKET=/path/to/new-packet.json python benchmarks/nvidia/benchmark_native_sm120_tensor_partition.py

The recorder verifies live GPU compute capability before executing. It does not
substitute host execution or another architecture.

## Still open

The raw-adapter snapshot originally left public JIT and portable witness integration open; the public integration below supersedes that gate. Remaining work includes
bounded dynamic projection, wider composition/accumulator/asynchronous lifetime
proof and focused PR delivery. Bounded dynamic public JIT still retains its older Python partitioning path. Other backend parity requires owning-device evidence.
No general W1.1 closure or backend performance promotion is claimed.


## Static public JIT and portable v2 integration

The public static route now preserves the actual frontend Graph and uses native
outlining for both packages. Prepared execution initializes bindings from the
checked descriptor without authoring a Python Graph. Native manifest v2 binds
member Graph hashes and the whole native plan hash to both packages. Portable
replay recomputes byte capacities and ownership/read-write lifetimes without
compiler calls. V1 remains readable for legacy/dynamic packages.

public-integration-tests.txt: 155 checks across native partitioning, existing
resident compatibility and public/device JIT. native-portable-drift-repaired-tests.txt:
358 checks across native/portable, audit documents and registry gates. The
no-Graph tests cover reordered external arguments and bias/ReLU/residual FP16
stores; compiler calls are forbidden during restoration and prepared execution.
Seven corruption cases modify Graphs, capacities or lifetime metadata while
recomputing the outer manifest digest and must be refused before compiler/CUDA.

public-jit-benchmark.json and public-jit-row-rhs-benchmark.json contain 24
profiles each (column/row RHS), three producer families, FP16/BF16, fused/plain,
and M/K/N = 17/35/19 or 128/1024/64. Separate producer/consumer CUDA event-dispatch
windows, cold/warm JIT wall times and portable replay wall times are recorded.
Numerics are checked before and after timing. Source and compiler/runtime
hashes bind each packet to its snapshot. benchmark.json remains the earlier
raw-adapter snapshot and does not claim the later public implementation.

Native dynamic-capacity projection, generic composition, accumulator and
asynchronous lifetime envelopes, sibling parity and focused PR delivery remain
open. No speedup or universal W1.1 closure is claimed.
