# Native whole-Graph NVFP4 public integration

Owner ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync NVFP4-NATIVE-PROGRAM-2026-10-08.

Recorded by benchmarks/rocm/benchmark_jit_nvfp4_program.py.

## Current proof

The native exporter clones the actual conversion/storage/scaled-matmul Graph
operations, preserving original SSA roles and eleven buffer capacity/lifetime
records. It serializes the exact projected member Graphs. Public packaging
consumes those compiler records rather than constructing replacement Python
Graph operations. Portable v2 replay retains the source/member witness and
validates capacities and read/write lifetimes without a compiler invocation.
Existing v1 artifacts retain their previous validation path.

The matching LLVM/MLIR 23.1.1 compiler SHA256 is
ebb6722b5d8ce9baf2384e51779e39210ff2cc7438096f09b6c6c044ca3638b7.

On Tajasaurus, live HIP reports gfx1201 and the benchmark queries
AMD Radeon RX 9070 XT. The final lane passes 68 mixed contract/device checks
(one pytest timeout-plugin configuration warning). Three public package cases
forbid Graph constructors and legacy author helpers, execute independent
numerical comparisons, restore with subprocesses forbidden, check reordered
roles and reject a consumer launch before successful ingest.

Six ordinary JIT/portable profiles cover M/N/K 128/32/256, 257/80/1024 and
256/64/64, each in two frontend argument orders. All report zero maximum
absolute output error against the independent conversion/storage/folded
float64 oracle after BF16 output rounding. Three timing windows retain raw
converter/storage/consumer/combined device-event samples, graph-dispatch
windows and checked end-to-end measurements. These are characterization
receipts; no speedup or selector promotion is claimed.

Final source fingerprints in final-native-public-benchmark.json match this
tree. WSL final portable contract checks: 41 passed, 26 hardware skips.
Shared native export checks: 97 passed, three hardware skips.
Owning RTX 5070 producer/saved-LSE sibling lane: 72 passed.
Earlier shared registry/public adapter lane: 332 passed, 26 hardware skips;
its source precedes the final metadata capacity/lifetime validation.

## Remaining work

Model-quality acceptance, broader dynamic/layout/storage-AD envelopes,
generic producer composition, performance optimization and focused PR
delivery remain open. Apple/x86 gain no physical execution claim.
The separate cache PR #894 is still draft and has no full-suite green receipt.

## Historical receipts

reproduction.json and schedule.txt record the original whole-Graph refusal.
native-export.json, partition-tests.txt and shared-native-tests.txt describe
the earlier artifact-only gate. Final public integration above supersedes
that state; original receipts are retained rather than rewritten.
public-tests.txt and native-public-benchmark.json precede the final portable
capacity/lifetime checks; final-* files are authoritative for current source.
