# Native scaled-product program members — 2026-10-07

Owner: FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Synchronization key: SCALED-MATMUL-NATIVE-MEMBERS-2026-10-07.

The native AD pass select-scaled-member=N option projects an actual outlined
member after export-scaled-program=true. A full serialized MLIR program witness
and the selected member's typed input/output buffer IDs survive projection.
The isolated member retains its exact semantic operation and argument metadata;
Python does not reconstruct Graph or Tile operations.

A sum member on ROCm receives compiler-owned launch aliases and reaches the
existing native math add Schedule/Tile/image recipe. The three product members
reach the existing gfx1201 W8A8 exact-per-block FP8 WMMA recipe. All four compile
to real HSACO images through native Graph/Schedule/Tile and backend lowering.
Image command lines, compiler output, errors and byte digests are recorded in
images.json and member-*.mlir/hsaco/err. Compilation is cross-target artifact
proof on Super-Bear WSL, not AMD physical execution. Images are not a runnable
whole-program package yet. No timing or AD device result is claimed.

The native fixture checks exact scale-seed IDs (0,1,4,3 -> 7), sum IDs
(7,8 -> 9), preserved block policy, native sum materialization and invalid
member selection. Combined core/backend lanes pass 644, with 66 unsupported
(710 discovered). Focused native AD/registry tests pass 347, with 16 skips.
The initial build API error is retained; corrected native tools rebuild.

Next: checked image/ABI role projection and native whole-program ownership
must allocate/preserve private intermediate buffers, order all launches and
retain returned arrays through completion. Prove the paired output numerically
on gfx1201 and separate kernel/program and public-call timing before widening
the profile. Dynamic/layout/composed/general batching and transpose closure
are still open. The prior program-export packet binds its own earlier source
and existing RTX 5070 regression snapshot; it is not new AD execution proof.


## Exact-device member numerical diagnostic

The four recorded gfx1201 HSACO images executed on Tajasaurus, AMD Radeon RX
9070 XT. The accompanying rocminfo.txt identifies gfx1201 and the owning GPU.
The static ragged M17/N19/K256 scale-only JVP passed an independent float64
block-product oracle for all three products and the native sum; maximum
primal absolute error was 1.99e-6. The tangent also passed a central scale
finite-difference oracle with maximum absolute error 9.52e-7. Entry symbols,
launch geometry, and image hashes are in numerics.json.

validate_members.py is a diagnostic member-launch loop, not the production
program runtime. No whole-program ownership, public JIT AD execution, native
timing, general batching/transpose, or performance closure is claimed.
