# Non-leading typed scaled maps — 2026-10-08

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization SCALED-MAP-AXIS-INTEGRATION-20261008.

## Implementation and proof

Typed gfx1201 E4M3 FP8 and E8M0 MXFP8 public maps now admit non-leading and
negative input-axis indices, including nested and Cartesian level policies.
Scalar constraints stay on logical dimensions. Frontend projection only
permutes alias views and inserts singleton axes. Native C++ byte packing
materializes checked positive-stride storage before strict compact launch
admission; MLIR Graph/Schedule/Tile/Target/LLVM owns all arithmetic.

FP32 scale JVP seeds preserve those axes. Native VJP unbroadcast reductions
are followed by inverse view projection to original caller axes. Encoded scale
gradients are not admitted. Kernel images, launch geometry and numerical
policy are unchanged by host packing; the runtime host pack API is additive.

Host gates: 553 passed, 38 hardware/tool skips. This includes native production
packer compilation and adversarial span/capacity/alias/overflow checks, existing
leading NVIDIA NVFP4 maps, nested/composed typed maps and shared registry gates.
Additive ABI/dashboard gates: 127 passed, 3 tool/environment skips. Audit
lifecycle tests: 11 passed; all 32 generated documents match the final source.
The host packer fixture compiles the exact production function in isolation;
it proves byte movement and refusal, not a GPU backend. Mypy has zero errors.

Owning gfx1201 gates: 118 passed, no hardware skips, including 24 new cases.
Primal/JVP/VJP compare independent float64 scalar-map oracles and coordinate
finite differences. Changed source/tangent/cotangent, original-axis adjoints,
retained outputs, independent scalar owners and compiler/reference-forbidden
warm replay are checked. The native HIP runtime is freshly built; twelve
source hashes match delivery. These are RX 9070 XT proofs, not gfx1151 evidence.

## Timing scope

device.json contains twelve FP8/MXFP8 × KN/NK × single/nested/Cartesian rows.
Each records seven windows. Public completed-call medians span 0.94–1.13 ms
and include native packing, transfer, native execution and readback. Resident
native-program HIP-event medians span 0.011–0.029 ms and include native enqueue
gaps; every repetition window exceeds 20 ms. They exclude packing/transfers
and are not a kernel-only/public speedup comparison. No selector promotion.

Run on the owning host from a matching source/tool/runtime environment:
python benchmarks/rocm/benchmark_scaled_map_axes.py --output packet.json
TESSERA_ROCM_NATIVE_MOVEMENT_LIB must point to the freshly built runtime.
TESSERA_ROCM_PROGRAM_CACHE and TESSERA_ROCM_PROGRAM_PINNED used native defaults.

## Remaining work

Generic primitive batching/transpose status stays partial/planned and the
zero-open CI assertions stay unchanged. This packet does not fix those two
remaining full-unit failures. Dynamic maps, nonzero output axes, broader
storage AD, packed NVIDIA non-leading axes and sibling native parity remain
open. Existing JVP input preparation still compacts its host frame in Python;
this slice moves strided primal/VJP program preparation into native C++.
The wider five-slice compiler goal is not closed.
