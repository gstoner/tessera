# Native saved-LSE attention JVP Schedule migration

Synchronization key: NVIDIA-JVP-NATIVE-SCHEDULE-2026-10-05.
Owners: FRONTEND-IR-MEDIUM-1 / AD residual integration, sibling W1.1.

## Route and bounded contract

Python typed Graph frontend or native TangentInterface export → verified paired checkpoint Graph → C++ content-addressed Schedule → native GPU/Tile arithmetic and shared buffers → checked native arena/sizing companion → NVVM/LLVM → native image plus checked tensor ABI → resident CUDA execution.

This replaces the Python GPU body constructor and the Python inactive-load rewrite. It uses existing registered Graph operations and passes. Shapes are positive static unencoded f32, each extent ≤65536, compatible GQA, representable row grid and byte extents; policy is finite positive f32 scale and end-aligned optional causal masking. The selected product is isolated; original native AD primal companions retain matching argument/policy lineage. The hash includes argument roles and active tangent slots. Unrelated functions cannot be silently erased. O/LSE must come from the same verified forward generation.

## Current evidence

- Super-Bear RTX 5070 / sm_120, UUID GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, driver 610.88, CUDA 13.3, matching LLVM 23.1.1 assertions.
- 298 focused JVP/AD/frontend/pass metadata tests pass on the matching WSL compiler build; Schedule policy/role tampering and unrelated-function rejection are included.
- Ten ordinary JIT cases pass finite-difference and float64 forward oracles at atol/rtol 3e-5: active Q, K, QK, reversed KQ, QKV; Sk=5 and ragged 129.
- Each case retains a native arena IR snapshot and serialized device image. The packet records compiler/implementation hashes, device identity, resources and correctness before timing.
- Five 64-launch preloaded-kernel CUDA-event dispatch windows per case: medians 0.0095–0.0133 ms. These windows include dispatch gaps; they are not a claim of isolated hardware kernel latency.
- Checked allocating/synchronizing JVP wall medians: 0.251–0.361 ms, measured separately.
- Launch is 128 threads, 1024 native-sized dynamic shared bytes, 37–40 registers, zero reported local bytes; occupancy uses the actual launch block and dynamic shared size.

Both direct and automatic native AD paths pass eight saved-state cases each, covering 32 direction modes per path, causal/noncausal and Sq greater/less than Sk. Private captured inputs survive caller mutation and closed results refuse access. The separate runtime/arena/stream/diagnostic suite passes 98 tests. Eleven audit tests pass and generated documents are regenerated. Matching shared-pass rebuild on RX 9070 XT / gfx1201 passes 18 native norm/epilogue regressions (80 unrelated cases deselected). This is regression proof for unchanged ROCm routes, not a ROCm JVP execution claim.

## Remaining obligations

General composed/dynamic AD, bias/dropout and value-only JVP integration remain open. FP8, MXFP8 and MXFP4 retain independent numerical, quality and performance gates. ROCm/Apple/x86 JVP migration requires architecture-owned lowering and exact-device proof. Graphify is not installed in this authoritative scratch checkout; no refreshed graph is claimed.

The early unit attempt during the compiler relink is infrastructure noise and was rerun after the build completed. Only the completed matching-tool results above support the claim.
