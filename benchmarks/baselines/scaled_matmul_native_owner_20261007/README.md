# Native HIP scaled-product program owner

Sync: SCALED-MATMUL-NATIVE-OWNER-2026-10-07.
Owner: FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Owning architecture: gfx1201; live HIP architecture guard and rocminfo.txt.
Device: AMD Radeon RX 9070 XT.
Scope: static M17/N19/K256 exact-per-block FP8 scale-only paired JVP.

## Implementation and proof

The C++ owner retains four compiler-produced image leases, snapshots six
readonly input buffers, allocates intermediate and returned buffers, verifies
SSA prefix edges and exact read/write lifetimes, and runs all four kernels
on its private stream. Repetitions execute inside C++, with no Python member
launch loop. Completion precedes output publication/readback and release.
Returned generations reject stale reads; private scratch cannot be exported.
Failed preparation retains a closeable owner; failed asynchronous work is
poisoned and retained until safe cleanup. Full native architecture names are
checked, including target suffix separation.

The repository CMake tessera_rocm_native_movement target rebuilt on Tajasaurus.
The recorder validates primal/tangent numerics, independent scale finite
differences, changed-input reuse, invalidation after upload, future SSA edges,
forged lifetimes, wrong byte sizes, wrong architecture, missing symbol cleanup,
scratch reads, stale generations, rejected updates and fork identity.

Maximum errors after changed tangent input reuse:
primal 1.9878801e-06,
tangent 1.7086885e-06,
finite differences 1.6137993e-06.

## Measurements

Eleven samples, 100 complete native sequences per HIP event window:
median 0.028854229 ms per sequence.
The device window includes native C++ enqueue gaps across four kernels;
it is not isolated arithmetic kernel time or a performance improvement claim.
Warm invoke plus both readbacks: median
0.41870913 ms.
All-input update plus invoke and both readbacks: median
0.47937501 ms.
Host intervals exclude compilation/preparation. All samples are retained.

## Reproduction and evidence

Recorder: benchmarks/rocm/record_native_scaled_program.py.
Use --images with the four images/MLIR from the preceding native_members packet,
--runtime with the owning CMake libtessera_rocm_native_movement.so,
and --output for the JSON receipt.
native-owner-cmake.json records complete buffer/step contracts and image/runtime
hashes. native-owner-inputs.sha256 binds owning source and both runtime libraries;
runtime-cmake-toolchain.txt records the actual CMake host compiler/HIP configuration; runtime-toolchain.txt records the separate direct-Clang toolchain. native-owner-cmake-build.log binds the final owning build.
native-owner-cmake.log retains the complete run, including its fork warning.

## Remaining integration

The driver supplies a diagnostic C ABI plan. Automatic compiler/package
projection of the exported SSA witness into this checked native owner is not
integrated. This is native program execution proof, not ordinary public JIT AD
package closure. Generic batching/transpose and the full-unit gate remain open.
This gfx1201 receipt does not establish gfx1151 FP8 WMMA, SM120, Apple or x86
physical program parity.

Focused host WSL audit, citation, plan routing, image lifetime, NVFP4 lifetime
and prepared-program regression gates pass 34 tests; receipt:
native-owner-regressions.log. This does not replace the red full-unit gate.
