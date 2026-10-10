# Three-format native staging attribution

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Live device AMD Radeon RX 9070 XT / gfx1201; rocminfo identity, actual image hashes and source/compiler/runtime fingerprints are retained.

Existing compiled FP8, MXFP8 and approximate folded MXFP4 packages run through a diagnostic one-member native ownership plan. Original launcher parity is bitwise BF16. Independent decoded-float64 forward error bounds and changed-scale updates pass before timing. MXFP8 half-scale updates decrement E8M0 encoded exponents; folded MXFP4 retains its explicit approximate semantics. This is staging compatibility proof, not public JIT/AD program projection for MXFP8/MXFP4.

| Shape M/N/K | Format | Pageable host ms | Pinned host ms |
| --- | --- | --- | --- |
| [200, 256, 256] | fp8 | 1.17648 | 0.61606 |
| [200, 256, 256] | mxfp8 | 1.50141 | 0.69957 |
| [200, 256, 256] | mxfp4_folded | 1.50107 | 0.59411 |
| [256, 256, 1536] | fp8 | 3.60134 | 0.78108 |
| [256, 256, 1536] | mxfp8 | 3.12761 | 0.66591 |
| [256, 256, 1536] | mxfp4_folded | 3.52979 | 0.65752 |

Host intervals include diagnostic ABI marshaling, native prepare/update, one image launch, one output readback and close with native idle reuse. Separate native event windows include enqueue gaps; no isolated kernel speedup claim. Candidate remains opt-in. Native compiler program projection and public JIT integration for the wider formats, generic batching/transpose and sibling physical proof remain open. Recorder: benchmarks/rocm/benchmark_native_program_format_staging.py.
