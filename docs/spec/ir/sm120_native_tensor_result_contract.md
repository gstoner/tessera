---
last_updated: 2026-10-18
---

# Native SM120 paired tensor result ownership

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization key: SM120-NATIVE-JVP-RESIDENT-RESULTS-20261017.

## Compiler and execution boundary

The existing source-active and RHS-linear paired producer/matmul Graph
programs retain native AD, SSA outlining, Schedule, Tile, NVIDIA Target IR
and LLVM/PTX images. The execution option
`native_jvp(..., result_placement="cuda")` returns two read-only FP32
`NativeTensorResult` objects. The default placement is `host`.
This option does not construct numerical arithmetic or Tile IR in Python.

Admission is static SM120, the existing FP16/BF16 paired tensor programs,
three host roots or three compact resident roots, and the original checked
shape/storage contracts. Unsupported derivative families and sequence bounds
do not acquire device-result admission. Explicit native SM120 normalization
members accept the matching sole accumulator declaration (fp32/f32); additional
policy fields require their own consumers. Sibling admission is unchanged.

## Additive native ABI

The native provider exports:

- `tessera_nvidia_tensor_program_reserve_result`: allocate a private,
  unmaterialized two-slot result generation for one checked program.
- `tessera_nvidia_tensor_program_submit_result`: bind three explicit roots,
  order resident producer streams or upload host roots, execute the original
  stages on the generation stream, and complete before exposing results.
- `tessera_nvidia_tensor_result_query`: return one materialized FP32 view,
  stream and context identity; clear output metadata on failure.
- `tessera_nvidia_tensor_result_read`: explicit completed host copy.
- `tessera_nvidia_tensor_result_record_consumer`: retain an event after an
  external consumer has enqueued its last read.
- `tessera_nvidia_tensor_result_close`: release one native generation.

Reservation precedes pending producer writes in the stream-ordering diagnostic.
A completed generation cannot be resubmitted. Result allocations are separate
from private scratch, other generations and caller roots. Prepared-owner
closure/cache eviction cannot reclaim returned storage. No result pool or
allocation-free invocation claim is made.

## Lifetime and streams

The two Python slots share a native generation lease. Closing one slot leaves
the other usable. Closing the final slot releases the generation after its
registered consumers complete. CUDA Array Interface metadata is read-only;
implicit NumPy conversion is refused, and `to_host()` is explicit.

All operations reject inherited objects after fork before acquiring Python
locks or touching CUDA. Consumer registration refuses non-sentinel stream tokens below 4096 before
calling the driver; CUDA default-stream sentinels 1/2 remain valid. Isolated
child regressions prove native and Python refusal and retained-result recovery.
Reads reject a foreign current context. A missing
current context is restored to the recorded owner context. Finalization pushes
the recorded context, frees its own storage and restores the caller context.

Consumers launched through `runtime.launch` traverse accepted `NativeBufferValue`
wrappers, arbitrary `Mapping` implementations and tuple/list subclasses. Cycle
guards and generation deduplication keep this metadata-only traversal bounded.
They pin native result inputs through
submission and register an explicit consumer stream before releasing those
pins. The descriptor route retains its existing same-stream requirement;
outputs/inout bindings require writable CUDA Array Interface storage, so raw
or wrapped read-only generations cannot be overwritten. Different streams need
explicit producer ordering and truthful destination
metadata. External asynchronous consumers call `record_consumer(stream)`
after enqueueing their last read. Host downloads must complete the actual
writer stream, including when allocation and writer streams differ.

If event creation/recording/retention fails after enqueue, the native API drains
the consumer stream before returning failure. Injected CUDA event-create and
event-record failures test this observation before any result free or download
can implicitly synchronize it.

## Portable artifacts and inspection

The parent RuntimeArtifact records `result_placement="cuda"` in hashed
execution metadata. Existing semantic program schemas and member images are
unchanged. Replay validates the manifest before binding a native owner and
returns independently owned results even when its temporary owner closes.
Fresh processes execute serialized images with compiler invocation forbidden.
Public artifact inspection deep-copies metadata. Placement-specific immutable
artifacts are cached on the prepared owner to avoid repeated serialization and
hashing; returned generations remain separate allocations.

## Evidence and remaining scope

Packet: benchmarks/baselines/sm120_native_tensor_results_20261017/README.md.

The current-source device tests cover source/RHS activity, FP16/BF16, root
permutations, host/resident inputs, retained generations, owner closure, cache
eviction, fresh-process replay, context restoration/refusal, fork-lock refusal,
registered asynchronous consumers and event failures. The compiled FP32
RMSNorm consumer has serial/cooperative tests through textual frontend,
Graph/Schedule/Tile/Target/PTX with independent FP64 numerical comparison.

The recorder `benchmarks/nvidia/benchmark_native_tensor_results.py` compares
completed public calls with resident roots, counterbalanced orders, common
repeat counts and minimum window admission. Validation/readback is outside
timed windows. CUDA timing includes prior-generation release; host timing
includes host outputs. Native event controls are a separate measurement scope.
This is an output-placement comparison, not a kernel-speedup claim.

FP32 standalone ReLU, general layouts/dynamic shapes, broader derivative
families, quantized results, sibling backends and aggregate five-slice closure
remain separate obligations. Registry/documentation and broad validation gates
are required before publication.
