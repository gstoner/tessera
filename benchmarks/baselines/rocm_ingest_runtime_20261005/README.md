# Canonical gfx1201 NVFP4 checkpoint JIT and common runtime

Owner ROCM-NVFP4-INGEST-1; sync ROCM-INGEST-RUNTIME-2026-10-05.

The Python frontend traces the three-result operation using catalog shape/type
inference. It does not execute the host converter to infer outputs. Literal
projection boundaries captured by a closure remain attributes; tensor values
remain operands. Each function result receives its own inferred tensor type.

Canonical compilation projects a copy of the typed Graph, adds checked binding
ownership, and lowers Graph MLIR -> Schedule MLIR -> Tile MLIR -> ROCm Target IR
-> LLVM -> HSACO. The canonical seven gates remain unchanged. The exact gfx1201
manifest now describes the measured compiler-generated package, rather than a
planned family placeholder; no gfx1151/CDNA/NVIDIA/Apple/x86 proof is inherited.

Ordinary static @jit calls use the complete six-buffer descriptor through the
common runtime. Warm calls reuse the native image. Outputs are private; input
and output aliases, invalid scales/globals/storage and IR lineage changes are
rejected before HIP access. Serialized RuntimeArtifact replay runs the same
image and checked ABI. Tests forbid host conversion during tracing and native
execution and compare packed codes/exponents bitwise and f64 statistics at
rtol=1e-13. The manifest names canonical logical NVFP4/E4M3/f64 payloads; the checked
descriptor names physical byte containers. General uint8 remains planned/gated
outside this named contract.

## Exact-device measurements

AMD Radeon RX 9070 XT / live gfx1201; compiler, source, image hashes, row
boundaries and numerical policy are in gfx1201.json.

| N x K | Cold compile + call wall ms | Warm JIT wall median ms | Resident HIP-event median ms |
| --- | ---: | ---: | ---: |
| 7 x 64 | 3550.567 | 4.348 | 0.304 |
| 67 x 256 | 3604.563 | 4.743 | 0.426 |
| 513 x 1024 | 3681.431 | 8.300 | 0.393 |

Cold time includes trace, compile and execution. Warm wall includes guards,
allocation, upload, module load/launch, completion and readback. Resident
events use the same JIT image and exclude module/allocation/transfer. They
include any gaps between host submissions, so they are resident dispatch
windows rather than isolated kernel-only attribution. Cross-session timing
variation was observed; this packet makes no optimization speedup claim. These
timing scopes must not be combined or represented as isolated consumer timing.
All rows pass correctness before timing.

## Validation and remaining scope

jit-tests.txt records 27 focused package/frontend/device cases on gfx1201.
frontend-regression.txt records 60 passed / 31 skipped host WSL cases.
drift-regression.txt records 630 passing cases plus the initial physical-byte
manifest registration failure; drift-final.txt records 123 passing final
manifest/fixture/shape/audit gates after that registration fix. The broader
run includes the adjacent RTX 5070 RHS regression. The post-maintenance NVIDIA packet is separate from this
AMD proof.

This closes ordinary static converter JIT/common-runtime dispatch and portable
replay. Resident converter-to-consumer lifetime/storage integration, measured
combined execution, general packing/dynamic envelopes, differentiated conversion
semantics and model-quality acceptance remain open. The pinned checkpoint
format packet retains its independent evidence and host-mediated consumer edge.
FP8, MXFP8 and MXFP4 remain mandatory correctness/quality/performance gates;
no default format, numerical policy or schedule is promoted.

Apple and x86 need implementation/exact-device follow-ups. NVIDIA validates the
shared frontend/runtime regression on RTX 5070; its NVFP4 conversion operation
remains unsupported. Physical conversion proof is gfx1201 only.
