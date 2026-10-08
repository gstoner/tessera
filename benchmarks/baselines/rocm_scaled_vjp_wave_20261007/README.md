# Native scale-VJP wave reduction candidate

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization: SCALED-TRANSPOSE-WAVE-2026-10-07.

The explicit scale-transpose-wave=true option uses one 32-thread workgroup per gradient element. Native proof requires zero initialization and exactly one accumulator use through additive or nested reduction links. Lanes partition only the outer loop; original K-group dot products and scale/cotangent multiplication remain intact. Five XOR shuffle/add stages combine sums, with lane zero storing. No LDS barrier is introduced. Outer FP32 addition order changes; independent numerical bounds govern parity.

Algorithm and width bind Schedule identity, sealed Tile/Target body and codegen-owned launch ABI. Python marshals packages and constructs no numerical IR or launch sequence. Serial remains the production default.

372 native export/package/serial/wave/diagnostic/pass gates pass. 24 paired gfx1201 rows at [2,3,7,19,256] cover three rank-four batching policies, KN/NK storage and four gradient requests/orders. Float64 oracle checks pass before/after timing and changed-cotangent replay; maximum absolute error is 2.682e-5.

Six alternating windows per arm, ten native repetitions each, show serial/wave ratios of 1.752–18.379, median 7.597. Serial event medians span 0.173–9.778 ms; wave medians span 0.011–5.032 ms. Windows include native submission work, exclude Python enqueue loops and host transfer/readback, and use different allocations per arm. They are not isolated ISA clocks, public-call speedups or selector promotion.

Both arms use the same compiler snapshot; identity.json binds compiler, sources and manifests. Images were cross-compiled on Super-Bear and replayed on Tajasaurus with the existing checked HIP owner. The initial direct-add-only proof failure is retained.

Open: wider short/long/ragged regimes, repeated controls, public candidate integration/timing, native selector policy, dynamic/composed/storage AD, sibling physical execution and generic/full-unit/publication closure. MXFP8/MXFP4 discrete scale AD is not implied by this FP32-scale FP8 candidate; format-wide primal selector evaluation remains separate.
