# Native storage integration: contracts, concurrency and measurement

Owner: W2.4a / CAKE / SO-2. Synchronization: `IR-NATIVE-FOUNDATION-1`.

| Area | This increment | Remaining boundary |
|---|---|---|
| Tensor descriptors | Frontend/compiler producer writes one `tessera.native_tensor_contract` manifest into MLIR. The compiler preserves it in the content-addressed package; `bind_native_storage(package)` generates the tensor ABI and launch geometry. | Pointer-only IR cannot infer shapes. Existing Schedule producers still need to emit their manifests; attaching one is not a Python/kernel equivalence proof. |
| Arbiter | Generated Tier-2 candidates enter the existing arbiter with a numerical oracle. A callback that never executes the candidate cannot attest it. Retiring a candidate removes only that registry instance. | No default-route promotion, device timing hook, or inferred operation-specific oracle. |
| Paired AD | A consumer binds an existing `NativeJVPArtifact` to pinned native child packages and preflights the complete primal/tangent ABI before any writes. Backend, child lineage and output ordering are checked. | Host contract tests only: the production AD planner must emit storage children from its actual differentiated program. Reverse AD, automatic derivative production and exact-device paired-AD proof remain open. |
| Streams | CUDA/HIP submissions use caller-owned streams, producer events and completion events. Tensor owners survive until completion; conflicting submissions through one binding are event-ordered. Error cleanup retains owners if completion cannot be proved. | External writers must publish their producer stream. Overlap across independent submissions is possible, not measured here. Forgotten wait/close retains ownership rather than freeing in-flight storage. |
| Token generations | Loop-external generation replacement is accepted only if the loop is proven nonempty or the zero-trip seed also matches. Fresh loop-issued generations cannot masquerade as an invariant token. | Rotating ring generations and inter-iteration allocation reuse still require a separate proof. |
| Apple | One typed slot contract produces the existing MSL `threadgroup(0)` declaration and validates its native tiled ABI, including alignment and static reduction scratch. | WSL validates contracts, not Metal. Generic GPU-arena-to-MSL lowering and an Apple-native sizing companion are not implemented. |
| x86 | The native host sizing companions run on the WSL x86 hosts. | This supplies no evidence about x86 compute-kernel performance. |

## Owning-device evidence

The [RTX 5070 packet](baselines/native_generated_stream_nvidia.json) and
[gfx1151 packet](baselines/native_generated_stream_rocm.json) each cover four
runtime widths/iteration counts. The recorder compiles a manifest-bearing native
recipe, generates its JIT descriptor, runs the arbiter's numerical oracle, then
submits through two native streams with completion handles. All outputs match
exactly; the Python body raises if invoked. NVIDIA also exercises loop-external
token replacement through the native compiler. These packets make no concurrency
speedup or paired-AD claim.

Reproduce with `record_native_tensor_producer.py --generated --arbiter --streams`
and the owning compiler/backend arguments. Add `--replacement-tokens` on NVIDIA.
Always wait or call `close_native_storage()` before freeing caller-owned buffers.
Closing an arbiter candidate retires it; re-enable arbitration with a live oracle
before another arbitrated call. Direct bindings can reload their package lazily.

## ROCm wait ablation

`measure_rocm_prefetch.py` builds the register-prefetch recipe and a matched
control with an explicit `s_waitcnt vmcnt(0)` after the next-generation load.
Both execute through MLIR → LLVM/ROCDL → gfx1151 code objects. Input/output copies,
compilation and module loading are outside resident HIP-event samples. Seven
alternating samples contain twenty launches each; all three shapes pass the
exact numerical oracle.

| Blocks × threads × generations | Prefetch median (ms) | Immediate wait (ms) |
|---|---:|---:|
| 32 × 64 × 7 | 0.003889 | 0.004072 |
| 256 × 256 × 33 | 0.017481 | 0.018159 |
| 256 × 256 × 65 | 0.031215 | 0.031927 |

The [packet](baselines/rocm_prefetch_ablation.json),
[prefetch ISA](baselines/rocm_prefetch_ablation.disasm) and
[control ISA](baselines/rocm_prefetch_immediate_wait.disasm) retain compiler,
source and image identities. The control is approximately 2–4.5% slower in this
run. Additional wait instructions and scheduling perturbations can contribute;
this is not sufficient evidence of useful load/compute overlap. Independent
runs, resource accounting and architecture-specific profiling remain necessary
before promotion. gfx1250 direct global-to-LDS operations are not used on gfx1151.
