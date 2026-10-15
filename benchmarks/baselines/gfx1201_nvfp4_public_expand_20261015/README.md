# Public native gfx1201 NVFP4 expansion and portable replay

Owner: ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1.
Synchronization: NVFP4-PUBLIC-EXPAND-20261015.

Ordinary @jit captures the semantic two-result operation and follows native
Graph -> Schedule -> Tile -> ROCm Target IR -> LLVM -> HSACO. A five-buffer checked
ABI returns exact E4M3 code bits (including signed zero) and the declared
FP64-product-to-FP32 K16 scale plane. Python allocates/marshals storage and
validates imports; it does not generate Tile IR or execute numerical lowering.

## Proof boundary

Cold JIT, warm positional/keyword calls, JSON artifact replay, and native
resident timing are compared bitwise with the reference for both outputs.
The eager oracle is forbidden throughout compiled execution.
Permanent coverage adds fresh-process replay with both compiler paths
unavailable and recompilation/oracle calls forbidden.
Invalid scale/global values, incompatible dtype, aliased output and altered
artifact lineage are rejected before HIP access. Image and descriptor
roundtrip, native compiler-stage lineage, and legacy ingest/storage remain
covered by the referenced test logs.

The initial public failures exposed stale artifact-only link/numerical registry
states. Native numerical proof already existed. Linked/ready state was reconciled,
then public/replay proof was recorded before promoting the exact gfx1201 manifest.
No canonical gate was weakened.

## Pinned checkpoint

gfx1201.json records original Qwen3 layer-0 gate/up packed weights and E4M3
scales from the pinned NVFP4 revision, with per-projection original globals.
Both weight/source hashes and actual GPU identity are retained.
These are weight-expansion checks, not full-model accuracy or perplexity results.

For N=24576, K=4096:
- device graph median: 0.695356 ms;
- cold first public call (compile + execute): 394.650 ms;
- completed warm public median: 54.938 ms;
- completed serialized replay median: 49.562 ms.

## Synthetic characterization

All times below are milliseconds. These are seeded inputs, not model weights.

| N x K | device graph | warm public | completed replay |
| --- | ---: | ---: | ---: |
| 7 x 48 | 0.003685 | 3.368 | 3.052 |
| 128 x 4096 | 0.003738 | 6.084 | 5.547 |
| 24576 x 4096 | 0.695726 | 59.022 | 51.660 |

Device windows exceed 20 ms and agree with HIP events within 5%; they include
GPU graph dispatch and exclude validation, allocation, transfers, compilation
and capture/instantiation. Inputs are warm/repeated. Host calls include
validation, staging, allocation, module handling, copies and completion.
This is not an isolated memory-bandwidth measurement or a speedup comparison.
Both outputs are checked before and after timing.

## Remaining work

The expansion is a standalone public native route. Producer-to-matmul residency,
native ownership of the whole launch sequence, import-once/update lifetimes,
numerical consumer parity and representative model-quality acceptance remain
open. There is no default-route promotion or sibling execution claim.
The generic HIP program owner currently assumes one output per step; the next
integration must represent both producer outputs as independent SSA buffers
with checked write/read lifetimes, rather than treating one as a fake input.

## Final delivery gates

Separate overlapping gates pass: 393 public/native/registry cases, 524 delivery
cases and 122 final W8A8/K16 owning-device/default cases. All 33 generated docs
are in sync. Ruff, typing, compiler-plan ownership, claim lint and Graphify
update pass. Exact command scopes and logs are described by validation.json
and the permanent test files. These counts must not be summed as unique tests.
