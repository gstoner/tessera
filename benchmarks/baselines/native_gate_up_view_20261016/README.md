# Native gfx1201 gate/up parent-view candidate

Owner: ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1 / LAYOUT-ALG-1 / E2E-REAL-6.
Sync: ROCM-NATIVE-GATE-UP-CONSUMER-20261016.

## Proved boundary

The explicit candidate consumes the actual one-parent Graph, two complementary
slice producers and SiLU. Native C++ owns Graph -> Schedule -> typed Tile ->
ROCm Target -> GPU/ROCDL/LLVM -> HSACO and exports byte spans, lifetimes,
separate returned output, scalar count and launch geometry. The existing checked
C++ HIP program owner executes the exported frame. No slices are uploaded as
separate compact tensors and no Python kernel/Tile constructor is introduced.
SM120 row schema 1 retains its meaning; this gfx1201 profile is row schema 2.
These are distinct from attention/JVP schema namespaces.

Five RX 9070 XT / gfx1201 cases pass, including swapped gate/up roles, changed
inputs, malformed-shape refusal without losing the previous result, retained
host outputs and unchanged parents. Compiler subprocesses and eager exp are
forbidden during preparation and execution. The independent f64-then-f32 oracle
covers sampled finite inputs in [-5,5] and uses identical f32 values; rtol=3e-6, atol=2e-7. Maximum absolute error is
3.814697265625e-6. M3/I17 and M16/I17 bind byte-identical images with separate
static source contracts; this does not authorize cross-shape package replay.
Actual device name, architecture, UUID, runtime bytes and image hashes are recorded.

The 17 compiler cases pass with no skips. The shared compiler/registry selection
passes 1,206 tests. Twelve existing SM120 source-active JVP device cases also
pass; they prove sibling regression parity only. Standalone ROCm parsing initially exposed a missing Math
dialect registration; the backend tool now declares and links it. A first
Schedule experiment also exposed premature Tile construction before dialect
loading; source validation is now separate from Tile construction.

## Timing interpretation

Each row has five correctness-gated HIP event windows, 8,192 repetitions each,
with at least 10 ms per window. Device windows include GPU dispatch. Completed
host samples include input upload, invoke, completion and output readback;
preparation and compilation are excluded. At M256/I12288, medians are 34.419 us
for the activation device window and 7.047 ms for the completed host call.
These are candidate characterization, not a matched speedup or default-route
promotion. The original producer matmul is absent from this diagnostic. Static image
resource metadata is in `static-resources.json`; it is not occupancy or runtime
counter evidence.

Admission follow-through rejects unknown structured-CFG execution policy and
non-v1 Graph IR before lowering. The 17-case native gate passes and all five
previously measured image hashes remain identical.

## Still required

Canonical family/plugin packaging and public JIT selection, a compiler-owned
resident producer/activation chain, low-storage and explicit rounding envelopes,
activation K16 quantization/down projection, original-model quality, matched
performance comparisons and the full original five-slice completion audit remain
open. The new profile has no gfx1151, NVIDIA, Apple or x86 execution promotion.
Direct registered MLIR finishing passes built these diagnostic images; the final
public packager must use the canonical backend pipeline rather than assemble it
in Python.

## Reproduction

Frozen owning-host diagnostics: `compile_native_gate_up_candidate.py` and
`check_native_gate_up_candidate.py`. They retain original scratch/compiler and
runtime-provider paths; run the compiler diagnostic from the source root with
the matching LLVM/MLIR 23.1.1 environment, copy its packets to the owning gfx1201
host, and run the checker after sourcing its HIP environment. These scripts are
engineering diagnostics, not the final public packaging API.
