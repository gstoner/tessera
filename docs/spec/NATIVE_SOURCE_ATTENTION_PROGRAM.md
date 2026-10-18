---
last_updated: 2026-10-18
audit_role: reference
---

# Native source attention program

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Synchronization key: NATIVE-COMPOSED-ATTENTION-PREFIX-20261018.

## Source and compiler boundary

The initial owning profile is static homogeneous FP16 or BF16
RMSNorm(Q), with explicit FP32 accumulation, followed by unbiased,
zero-dropout saved-LSE attention on NVIDIA SM120. The original typed Graph
contains both operations. The native tessera-outline-source-program pass
outlines the original SSA operations into private typed func.call members.
It does not generate replacement mathematical expressions.

The tessera.native.source_program.v1 export records the original Graph,
its SHA256, original argument roles, result order, standalone member Graphs
and logical buffer lifetimes. The export is artifact_only: its presence alone
does not certify executable images or device execution. The native primal
outliner rejects pending differentiation intent on the module or root function;
it cannot discard a forward/reverse request while cloning primal members.

NativeSourceProgram verifies its compiler identity and replays the original
source before consuming members. Each retained member goes through existing
Graph-to-Schedule and Schedule-to-Tile passes. Norm metadata is projected from
the native Tile carrier; saved-LSE metadata uses the existing checkpoint
decoder. Existing NVIDIA package builders produce the Target/NVVM/PTX images.
The Python transport does not reconstruct either numerical Graph.

## Prepared ownership

NativeSourceProgram.prepare() compiles both member packages and binds a native
prepared frame. The additive tessera_nvidia_norm_attention_prepare C API reuses
the existing forward owner, checked static dimensions, pinned host staging,
context identity and process/fork checks. Native dimensions currently have
the existing per-axis limit of 65536.

The owner retains source Q/K/V, a separate private normalized-Q allocation,
half O and FP32 LSE. Normalization reads the source snapshot and writes the
private allocation. Attention reads that allocation on the same private
nonblocking stream. No intermediate is downloaded, uploaded, exposed as a
CUDA Array Interface object or allocated by Python between the kernels.
Serial and cooperative_128 normalization use the compiler-selected schedule.

Resident roots retain the existing ordered snapshot contract: each producer
stream records an event, the owner stream waits, and device-to-device copies
snapshot the three roots. Host roots use pinned staging. Completion waits on
the owner stream. O and LSE are returned as independent host arrays in the
original source result order; they remain valid after frame reuse and close.

Close is idempotent through the Python transport. Native cleanup drains the
owner stream, frees the private allocation and staging, and unloads retained
modules under the recorded CUDA context. A partial prefix registration
failure erases its unpublished owner and releases acquired resources.

## Numerical and measurement evidence

The owning RTX5070 suite covers both storage formats, serial and width-257
cooperative normalization, permuted original arguments, swapped O/LSE return
order, unchanged roots, retained outputs and closed-frame rejection.
Pending resident writes are tested after private allocation is primed, with
separate producer streams and CUDA legacy/per-thread stream handles.

benchmark_source_attention_prefix.py checks independent reference O and LSE
before and after timing. It records normalization, attention and composed
CUDA-event dispatch windows, completed host-call time, cold initialization
and member compilation separately. Event windows can include enqueue gaps;
they are not pure-kernel measurements or evidence of a performance promotion.

## Public execution and portable programs

Ordinary @jit(target="nvidia_sm120") selects the named homogeneous half
RMSNorm -> two-result saved-LSE attention source. Native outlining and member
packaging remain the authority. The frontend retains up to 24 static prepared
specializations and restores the matching original Graph when revisiting a
cached shape. Warm calls do not trace or compile. Native source witnesses
and inspection-artifact caches are reset at each public call boundary; leaving
the half profile cannot reuse its previous native receipt or executable program.
The selector does not claim output-only, FP32 or other operation sequences.

The tessera.nvidia.source_attention_program.v1 executable manifest contains
the native source plan, original argument names, both complete member stages,
native images and checked launch descriptors. Its integrity digest binds the
whole payload. Restoration checks allocation lifetimes, root/result roles,
shape guards, physical bindings, launch geometry, numeric policy and image
provenance before provider acquisition. Content hashes detect corruption;
they do not authenticate arbitrary supplied native code.

RuntimeArtifact JSON uses nvidia_sm120_source_attention_compiled and the same
native owner. No pointers, streams, live handles or compiler installation are
needed in the persisted manifest. The owning suite proves fresh-process launch
with both compiler paths unavailable and external tool invocation forbidden.
Completed host outputs remain independent of the native frame and its close.

## Open integration

The forward proof does not establish original-source Q/K/V reverse integration,
higher derivatives, dynamic shapes, bias, dropout, mixed storage or wider
output policy. Existing isolated attention backward proof is separate. Apple,
x86 and ROCm require their own physical consumers and exact-device evidence;
no SM120 schedule transfers to those targets. Final broad validation and
publication gates are still pending. All five aggregate slices remain open.

## Original-source reverse candidate: prepared owner

Synchronization key: NATIVE-ATTENTION-SOURCE-REVERSE-20261018.

The unit marker tessera.native.source_attention_reverse_request requests an
original-source export inside tessera-autodiff-paired. Its schema
tessera.native.source_attention_reverse.v1 contains original, paired and
outlined program witnesses, local member Graphs, root and cotangent roles,
requested gradient order and checked logical allocation lifetimes. The export
remains artifact_only; it is not itself an execution receipt.

The native transform clones the actual pure normalization pullback from paired
AD, cutting at the checkpoint dQ SSA value. The root has four typed func.call
members. Static homogeneous f16/bf16 roots, FP32 accumulation and two saved
O/LSE outputs are admitted; either result order and original argument
permutations are preserved. Dynamic shapes, bias, dropout, encoded layouts and
other source programs are outside this named profile.

PreparedSourceAttentionReverse validates native replay, lowers each original
member through Schedule/Tile/NVIDIA/NVVM and compiles four images. It performs
no tensor mathematics. The additive source_attention_reverse_prepare/invoke/
last_times/close C APIs own private normalized Q, O, FP32 LSE, checkpoint dQ/K/V
and source dQ. All selected outputs are independent completed host arrays.
The current implementation materializes all three checkpoint gradients even
for a requested subset; subset return order does not imply pruned computation.

Inputs are the original three arguments followed by cotangents in original
result order. O seeds retain the half storage; LSE seeds are FP32. Host inputs
require exact native-endian storage/shape. Resident roots require one complete
CUDA placement, compact version-three metadata and explicit producer streams.
The binding snapshots metadata once, and native event waits establish all
producer dependencies before private snapshots and member launches. The owner
checks static byte extents, context identity/generation, process lifetime and
kernel parameter counts. It drains its stream before releasing buffers/modules.

An execution receipt is published only after native completion and timing
collection. Four event windows describe normalization, saved forward, backward
and source pullback. The program window spans these launches, excludes root
copies and includes possible dispatch gaps. Compile, native preparation,
completed native-host and Python end-to-end timing remain distinct.

The prepared candidate has independent-oracle SM120 numerical evidence.
The public integration below adds ordinary native_backward, public execution
certificates, sealed restoration and compiler-free RuntimeArtifact replay
for this named static original-source profile. No sibling backend
inherits SM120 physical proof, and none of the five aggregate slices is closed.


## Public source reverse and portable execution

The source_normalization_attention composite declaration owns only the named
SM120 source profile. The existing primitive zero-dropout effect policy is
checked before concrete frontend certification; unsupported stateful requests
cannot reexecute source through that parity gate. Ordinary native_backward
certifies the original frontend
Graph, projects only module intent/target attributes, and invokes native paired
AD outlining. It never reconstructs derivative or Tile expressions in Python.
The prepared cache holds at most eight static source products. Warm calls use
the already-validated native owner directly; no compiler or manifest rehash
occurs on that dispatch path. Execution certificates follow actual completion.

native_backward_runtime_artifact materializes an independent portable wrapper
on request. Its compiled-program schema is
tessera.nvidia.source_attention_reverse_program.v1. Each member carries Graph,
Schedule, Tile, Target, native LLVM and PTX witnesses, content hashes and a
tessera.nvidia.source_attention_reverse_step.v1 launch ABI. The sealed plan
records all twelve logical buffers and their exact read/write/return lifetimes.
Typed root calls must agree with standalone member signatures and local names.
Normalization epsilon, original attention scale/causal/zero-dropout policy,
storage, saved O/LSE and both seed roles must agree across source and members.
Static launch dimensions, parameter widths, entry symbols and image target are
checked before provider acquisition. Restoration uses no production decoder
that replays compiler passes. Content digests establish integrity, not native
code authenticity; these checks do not prove arbitrary supplied PTX mathematics.

The common runtime path nvidia_sm120_source_attention_reverse_compiled uses
the same retained C++ owner after offline validation. The five input bindings
are primal_0, primal_1, primal_2, cotangent_0, cotangent_1, preserving original
argument/result order. Fresh-process replay needs no compiler and returns the
requested original-root gradient order. Invalid calls clear public receipts
and pending artifact factories. A previously exported mutable wrapper cannot
alter the immutable prepared owner or a subsequent independent export.

The ordinary execution certificate binds the sealed compiled-program digest.
A RuntimeArtifact wrapper has its own content identity, attested separately
when launched through the common runtime. Device event windows and completed
host/public timing stay separate. The measured public dispatch improvement
uses identical images and operands; it is not a physical kernel improvement.

Native program executors validate their own actual device/provider. Generic
C-runtime backend discovery is deferred to the fallback branch that consumes
it, allowing cold compiler-free replay without external library discovery.
Shared mock dispatch tests cover Apple, x86 and both AMD targets; those tests
are not exact-device numerical proof for sibling architectures.

The public SM120 evidence is a named increment. Full dynamic/layout/AD and
remaining attention/ROCm/NVFP4/performance programs, broad validation and
publication are still open. None of the five aggregate slices is closed here.
