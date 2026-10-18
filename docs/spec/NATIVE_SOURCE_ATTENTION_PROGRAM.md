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
