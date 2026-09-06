# Native tensor bindings and architecture-owned producers

Owner: W2.4a / CAKE / SO-2. Synchronization: `IR-NATIVE-FOUNDATION-1`.

The explicit `JitFn.bind_native_storage` adapter binds a compiled native package
and tensor/index descriptors to a Python signature. Calls validate dtype, shape,
contiguity, writable outputs, nonaliasing, allocation bounds and device ownership;
then invoke the native sizing companion and CUDA/HIP module. Device allocations
and outputs remain caller-owned. `close_native_storage()` releases the loaded
module; a later call lazily reloads the same configured package. Calls synchronize producer work and completion.
No Python body, Graph IR regeneration, automatic selection, differentiation or
inferred equivalence to arbitrary Python math is involved. Array-interface
wrappers are used for owning-device proof; other tensor protocols are not implied.

`record_native_tensor_producer.py` validates four runtime widths/iteration counts
per device. Its body raises if traced/executed, and all eight cases return the
exact NumPy oracle from native kernels. The evidence packets are
[CUDA](baselines/native_tensor_producer_nvidia.json) and
[ROCm](baselines/native_tensor_producer_rocm.json). They contain source, compiler,
package and binding identities. These are correctness packets without timings.

The shared lifetime proof follows NVGPU tokens through all arms of `scf.if`
and identity `scf.for` carries, including zero-trip initialization. Changing
backedges and an unrelated token on either arm remain rejected. This is not a
proof for arbitrary rotating generations. NVIDIA's production SM120 macro-GEMM
now constructs typed NVGPU copy/commit/wait tokens, including its prefetched
loop-carried group, before NVGPU-to-NVVM conversion. Its architecture-owned
static double buffer remains distinct from generic dynamic-arena coalescing.

The [macro packet](baselines/tokenized_macro_gemm_nvidia.json) validates six shapes
for both deferred-wait and immediate-wait images on RTX 5070. Three alternating
CUDA-event samples of five calls are exploratory: the 1024-cube medians are
0.1270 versus 0.1310 ms, and 2048-cube 0.9361 versus 0.9567 ms. Smaller cases are
noisy. This rerun does not promote a selector, ledger, or cross-run winner.

The gfx1151 producer prefetches the next input plane into registers while
consuming the current plane through dynamic LDS scratch. Its generated
[ISA](baselines/native_tensor_producer_rocm.disasm) contains `global_load_b32`,
LDS operations, `s_waitcnt` and barriers. Inspection found immediate VMEM drains
at several release barriers; useful load/compute overlap is **not established**.
LLVM's direct asynchronous global-to-LDS operation requires gfx1250 and is not a
gfx1151 implementation option. Follow-up: an architecture-owned scheduling and
barrier strategy plus a controlled HIP-event ablation; retain this packet as
correctness evidence, not a performance result.

Remaining integration: automatic tensor descriptor/arbiter generation, paired AD,
stream-aware concurrency, generation-sensitive generic lifetime proofs, and
ROCm overlap measurement. Apple needs its own MSL dynamic threadgroup binding;
x86 supplies the host companion and does not inherit CUDA/HIP execution proof.
