# Native seeded attention bridge

Owner NVIDIA-LSE-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Synchronization key NVIDIA-LSE-COTANGENT-BRIDGE-2026-10-07.

The C++ host, resident and CUDA-event benchmark bridges account explicitly
for the saved row-LSE cotangent before output pointers. Bias arity inference
excludes the seed. Allocation and kernel argument capacity support the extra
pointer; malformed counts, null pointers and unsupported seeded storage are
rejected. Compact activity retains its explicit integration gate.

RTX 5070 exact-device validation: 81 tests pass, including four new native
bridge cases (causal/full, plain/score-bias). Each case checks host, resident
and event-timed output against independent FP64 Q/K/V derivatives. Timed
outputs are checked after readback. Existing physical leaf and generated
Graph/Schedule cases remain in the same device lane. Counts overlap earlier
packets. This is native bridge integration, not serialized descriptor proof.
Recorder/test: tests/device/nvidia/test_lse_cotangent_native.py.

The runtime is rebuilt against the matching WSL CUDA toolchain. Sources and
compiler/runtime hashes are recorded in source.json. No performance promotion
is claimed from the short correctness-gated event windows.

Remaining: seeded package descriptor/provenance, portable/common runtime,
compact activity and bias-gradient bridge envelopes, multi-result AD and
private residual ownership, complete extent/alias/stream checks. NVIDIA
symbol arity is not sufficient to validate a caller allocation capacity.
Apple, x86 and ROCm receive no new native admission or device parity claim.
