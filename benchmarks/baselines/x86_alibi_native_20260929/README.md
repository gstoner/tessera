# Native x86 ALiBi package, 2026-09-29

Owner E2E-REAL-6; sync E2E-REAL-6-ALIBI-2026-09-29.

The explicit-slopes Graph envelope now descends through verified Graph,
content-addressed Schedule and Tile IR to the existing AVX-512 C ABI. The
descriptor is projected from replayed native IR. The retired Graph-owned
constructor remains only in the differential test oracle. This packet was
recorded on Princess-Luna's Zen 5 CPU under WSL2, using exact LLVM/MLIR
23.1.1 and the x86-enabled Tessera compiler.

Run from the repository root with the production tessera-opt and the owning
AVX-512 shared image selected:

    python benchmarks/x86/record_alibi_native_package.py --samples 31       --output benchmarks/baselines/x86_alibi_native_20260929/princess_luna.json

The recorder refuses a missing compiler or image, checks each package image
digest against its launch receipt, compares every output with an independent
NumPy ALiBi formula, and records compiler/library SHA-256. Four shapes cover
a unit case, a small ragged sequence, and two larger sequences. The 1,088
x86 differential/cohort/position tests additionally compare retired and
compiled descriptor, Tile operation, Target call and exact CPU output.

Timing is synchronized host wall under WSL2. Cold package includes first
compiler/cache initialization. Warm package and full launch medians are
diagnostic overhead, not kernel time. The packet is not selector eligible
or a performance promotion claim. Apple, ROCm and NVIDIA explicit-slopes
consumers require their own backend proof.

[Packet](princess_luna.json).
