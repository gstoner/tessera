# Pinned native NVFP4 conversion and format consumers

Owner ROCM-NVFP4-INGEST-1; sync ROCM-CHECKPOINT-NATIVE-INGEST-2026-10-05.

RX 9070 XT / gfx1201 evidence uses pinned Qwen3-8B layer-0 gate/up tensors,
merged N=24576 and K=4096. Source revisions, tensor hashes, independent
projection globals, row boundaries, compiler/image hashes and numeric policy
are recorded in gfx1201.json.

The frontend builds a typed GraphIRModule and uses the catalog's three-result
inference. Native Graph/Schedule/Tile/Target/LLVM passes own conversion.
The checked six-buffer package returns native-produced packed codes, group-major
exponents and f64 loss statistics. Codes/exponents match the host oracle bitwise;
bounded independent checkpoint-domain signal/error checks also pass.

The byte tensor spelling is now ui8. uint8 remains planned/gated, with a named
gfx1201 checkpoint capability. Other architectures explicitly declare this
operation unsupported until their physical implementation/evidence exists;
ordinary gfx1201 JIT was artifact_only when this packet was recorded.
The subsequent [canonical JIT/runtime packet](../rocm_ingest_runtime_20261005/README.md)
proves static converter dispatch without relabeling these historical timings.

## Conversion timing

- Resident HIP-event median: 41.391 ms.
- Checked synchronous conversion host-call median: 157.385 ms.
- Cold packaging/compile: 3606.145 ms, excluded from both timing scopes.

Resident windows cover ten resident launches. Host calls include validation,
module load, private allocations, transfers, conversion, synchronization and
cleanup. Conversion is a checkpoint-loading step, not charged to each consumer
invocation.

## Consumer format gates

All eighteen arms (nine at each M=128/256) pass independent arithmetic checks.
Seeded synthetic activations are used; this is not whole-model quality evidence.

Representative M=256 rows:

| Arm | Resident consumer us | Checked consumer wall ms | Output RMS error vs BF16 |
| --- | ---: | ---: | ---: |
| FP8 K128/N128 | 548.948 | 18.474 | 3.686% |
| MXFP8 K32/N1 | 823.655 | 16.835 | 3.751% |
| NVFP4-ingested MXFP4 expanded | 389.009 | 99.729 | 15.177% |
| NVFP4-ingested MXFP4 native packed | 392.171 | 56.084 | 15.177% |
| Direct BF16-to-MXFP4 native packed | 392.089 | 56.313 | 11.584% |

The ingested consumers receive the native converter's output. The current edge
reads those values back to the host and prepares/uploads the consumer layout.
Resident conversion-to-consumer storage/lifetime integration and a directly
measured combined host-call window remain open; these separate windows must not
be added and presented as a measured combined result.

The folded consumer retains its explicit approximate E4M3 expansion policy.
No selector/default/dtype promotion is made. FP8, MXFP8 and MXFP4 remain mandatory
independent correctness, model-quality and performance gates.

## Validation and remaining work

package.txt: 15 package/frontend/storage-capability tests passed on gfx1201.
The host dtype/registry lane passed 458 tests, and capability lane passed 34.
All four backend plans are assessed. Ordinary JIT, semantic AD/conformance
registration, resident edge and broader route/performance obligations remain
open. Graphify is unavailable on the authoritative scratch host.

Reproduce with benchmark_gfx1201_checkpoint_formats.py --native-ingest
--include-native-packed --m 128 256 --windows 3 and the owning gfx1201 compiler
and LLVM paths. Exact argv appears in the recorded host process/checkpoint log.
