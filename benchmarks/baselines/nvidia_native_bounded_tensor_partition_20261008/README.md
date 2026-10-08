# Native bounded tensor partition: RTX 5070

## SM120 native bounded whole-Graph projection — owning checks

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync SM120-NATIVE-BOUNDED-TENSOR-PROGRAM-2026-10-08.
C++ MLIR outlining now projects bounded M/N/K from the actual frontend Graph;
the public bounded path no longer constructs replacement Python Graphs.
Original caller IR is preserved. Native v2 witnesses bind active shapes,
capacities, member Graphs and private buffer read/write lifetimes.
All seven nonempty dynamic-axis subsets are covered for RMSNorm/LayerNorm/
softmax and FP16/BF16, including compiler-free portable replay and stable
scratch ownership: 130 focused native/host checks pass on Super-Bear.
The wider device lane reports 291 passes and one legacy-name fixture failure.
After replacing that fixture's reconstructed Graph with actual descriptor
bindings, the entire prepared-owner file passes 100 checks. The static-kernel
dynamic-ABI refusal and numerical checks are retained.
Both RHS layouts complete 48 correctness-gated profiles each (96 total).
Maximum absolute output error is 0.015625. Warm public-call medians span
0.420867–0.837734 ms for column RHS and 0.411667–0.838203 ms for row RHS. Producer/consumer CUDA-event dispatch and
public wall timing are separate; no speedup or kernel-only claim is made.
Compiler SHA256:
65126f8c651e5ecc0473239e311e9e1e81a812b5a59b348b0b75d1c9d982e4b2.
Evidence: benchmarks/baselines/nvidia_native_bounded_tensor_partition_20261008/README.md.
Broader composition/asynchronous ownership, sibling owning-device proof,
broader performance evaluation and focused native publication remain open.

Validation logs retain the historical failing device lane and the complete 100-check owner repair lane separately. column-rhs-benchmark.json and row-rhs-benchmark.json record the 96 completed profiles, with matching compiler SHA256 and numerical checks before timing. Timing scopes are event-dispatch and public wall time.

Final native partition/diagnostic/pass/operator/audit gate: 423 passed in 54.75 seconds. The layout packets share compiler, runtime and production-source identities; the owner refusal test changed between runs. layout-snapshot-comparison.json records both hashes. This is characterization, with no default-route performance promotion.
