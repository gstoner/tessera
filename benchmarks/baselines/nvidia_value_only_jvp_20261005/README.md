# Native saved-LSE value-only attention JVP

Owner AD-RESIDUAL-EVAL-1; sibling FRONTEND-IR-MEDIUM-1.
Synchronization key NVIDIA-VALUE-JVP-2026-10-05.

## Architectural change

The general TangentInterface already emits the linear V-only derivative checkpoint_forward(Q,K,dV). That general recipe remains unchanged. The optional isolated native JVP export now constructs an explicit paired checkpoint_forward(Q,K,V) and checkpoint_jvp(Q,K,V,O,LSE,0,0,dV) from actual generated SSA. Native Graph verification and Schedule role/policy hashes bind the same forward generation.

Schedule selects cooperative_saved_lse_value_linear_v1 for inactive Q/K tangents. Native C++ GPU/Tile lowering does not load primal V or O and does not allocate/reduce the unused score moment. Removing those dependencies prevents 0*Inf/NaN contamination in a V-only linear product. Native scratch sizing returns 512 bytes rather than 1024; nine synchronization barriers preserve scratch lifetime. The nine-pointer tensor ABI, private capture, no-alias checks and CUDA context ownership remain unchanged.

The public JIT compile_native_attention_jvp API now admits active V alone. Python drives a typed frontend/thin API; there is no Python GPU body or derivative reconstruction. Flow: native TangentInterface → verified paired Graph → Schedule → GPU/Tile arena → NVVM/LLVM → image plus checked ABI → resident execution.

## Exact-device evidence

RTX 5070 / sm_120, GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, driver 610.88, matching LLVM 23.1.1 assertions compiler and CUDA 13.3 host.

- 341 focused Graph/JVP/public frontend/pass metadata/diagnostic tests pass.
- Twelve ordinary JIT cases pass float64 forward and finite-difference oracles, including reordered Q/K tangents, QKV and newly admitted V-only activity, Sk=5/129.
- Twenty-four V-only cases prove fixed-QK linearity for finite/Inf/NaN primal V, causal/noncausal and Sq/Sk = 3/5, 5/3, 4/4, 3/129. Maximum absolute error 3.105e-8 at unchanged 3e-5 tolerance. The source-V values are deliberately irrelevant to this partial derivative.
- The same package accepts repeated directions and preserves scalar linearity. Native arena/image snapshots, source/tool hashes and device identity are retained.
- V-only Sk5/129 preloaded-kernel event dispatch medians: 0.01065/0.01107 ms. Checked allocating/synchronizing JVP host wall medians: 0.25216/0.31751 ms. These windows include dispatch/host gaps; no isolated-kernel speedup claim.
- Actual block: 128 threads; shared bytes 512; registers 29/38; reported local bytes zero; occupancy queried with actual block/shared size, 12 active blocks/SM.

## Sibling assessment and remaining work

Apple/x86/HIP physical value-only schedules are not implemented by this SM120 change. General target-neutral V-only AD remains the existing linear recipe. The first Tajasaurus shared test run exposed stale Python GPU JVP construction and older paired-AD sources. A partial source sync then exposed a mismatched NVIDIA lowering signature at link. The full compiler source was synchronized with fresh source timestamps. The matching gfx1201 compiler rebuild completed; 40 shared native AD/JVP tests and 18 existing norm/epilogue exact-device regressions pass on RX 9070 XT / gfx1201. The retained value-jvp-full-source-build-20261005.txt, value-jvp-matched-shared-20261005.txt and value-jvp-matched-gfx1201-20261005.txt logs support these distinct claims. Those initial failures were integration drift, not accepted parity; no HIP JVP execution is claimed.

General composed/dynamic/bias/dropout/higher AD and wider producer/backend routes remain open. FP8/MXFP8/MXFP4 numerical/quality/performance gates remain independent. No universal compiler completion is claimed. Graphify CLI is unavailable in the authoritative scratch checkout; no refreshed graph is claimed.
