# Complete static folded panels: measured-negative optimization

Owner ROCM-MXFP4-W4A8-1; sync ROCM-FOLDED-NATIVE-OPT-2026-10-02.

This experiment added complete-panel metadata to Target request setup.
Closer inspection showed the shared emitter already derives the same static
proofs (M divisible by 256, N by 64) and selects unclamped LDS copies and an
unmasked typed store. Mixed/ragged panels retain their independent guards.
The redundant request setup and metadata were removed; mixed-panel store
contract tests remain.
The image is still static-shape-specific; future runtime image projection must
preserve these classes and checked launch geometry.

Compiler/mixed-panel/owning-device numerical gate: 58 passed (device-tests.txt).
M256/K5120 N4096/8192/16384 matched HIP timing is in gfx1201.json.

All three native selected-symbol instruction-stream hashes are identical to
the prior packet rocm_folded_native_package_20261002. VGPR count remains 177.
The emitter already eliminated conservative masks in these static complete shapes.
The 1.0389–1.1068 native/HIP ratios are separate timing observations, not a
before/after performance regression attributable to an identical instruction
stream. This change clarifies the IR bounds contract but provides no measured
kernel improvement and does not explain the native-versus-HIP gap.

Next experiment: constrain K16 operand read/MMA issue boundaries, retain all
ragged numerical gates, inspect register/ISA changes and rerun matched timing.
No profiler counters, Radiance comparison or performance promotion.
