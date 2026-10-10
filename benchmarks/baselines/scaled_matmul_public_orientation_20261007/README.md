# Public FP8 scale-JVP RHS orientation and scale-block envelope

Owning gfx1201 public JIT scale JVP executes physical B[N,K] storage with
transposeB=true for M/N/K=(17,129,256), (200,129,1536), (33,257,2048).
The widened device lane passes 13 tests, including independent float64 block
primal/tangent oracles and scale finite differences. This covers ragged rows/
columns, multiple N-scale blocks and twelve/sixteen K-scale groups.

Correctness-gated recorder:
benchmarks/rocm/benchmark_public_scaled_jvp.py --transposed-rhs.
Public warm medians are 1.8281, 6.0960 and 4.1788 ms respectively; native
four-member event windows are 0.05487, 0.25107 and 0.31281 ms. Prepared
update/invoke/two-readback medians are 0.6744, 5.2736 and 3.4854 ms.
Native windows include enqueue gaps, not isolated kernel timing.
Longer shapes expose host transfer/readback costs despite native allocation
reuse; pinned native staging remains an attribution/optimization follow-up.

This is scale-JVP and physical RHS orientation proof, not a generic linear
transpose AD rule, transpose-left support, batched/dynamic/nested AD, sibling
physical proof or a green full suite. Generic batching/transpose closure
assertions and coverage states remain unchanged.
