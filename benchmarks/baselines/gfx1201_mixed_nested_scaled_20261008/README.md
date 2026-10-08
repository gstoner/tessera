# Mixed nested scaled maps on gfx1201

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync GFX1201-MIXED-NESTED-SCALED-2026-10-08.

Each native vmap owner retains its explicit map-level policies. Missing levels
become singleton frontend views with the same address and shared storage;
scalar M/N/K constraints remain attached to the raw argument axes. The traced
Graph uses the existing broadcast semantic contract. Native verified
Graph/AD/Schedule/Tile/ROCm/LLVM lowering owns arithmetic and program dispatch.
There is no Python execution loop in the production route.

The named profiles use compact host ndarray storage. Adding singleton axes
does not compact, replicate or numerically transform an input. General
strided-resident bindings, nonleading maps, dynamic maps and arbitrary Graph
composition remain open.

Scale JVP seeds are validated against original caller shapes, then projected
through the same checked views. Scale VJP computes broadcast reductions in
the native program and returns gradients reshaped to original caller storage
shapes. FP8/MXFP8 matrix storage derivatives are not established here; E8M0
scales remain discrete and these AD rows use f32 scales.

## Owning proof

Tajasarus WSL, AMD Radeon RX 9070 XT, live gfx1201,
UUID bytes 32386439653765666266326566373136.
Compiler 736644c0ecc6d36ad3530aa617c8c66a3c2d0401dab5b6a1b674d2e39688dbf0.

22 exact-device tests pass:
eight FP8/MXFP8 mixed-map primals, four scale JVPs, four scale VJPs,
and six existing K129 partial-group nested primal regressions.
They cover an inner RHS map plus outer all-input or Cartesian LHS maps,
two leading levels (2,3), both physical RHS orientations, and compiler-free
warm execution with changed scales or derivative seeds. VJPs compare every
scale coordinate with independent float64 finite differences and verify
original gradient ranks and extents.

75 WSL frontend/map regression tests pass. The shared JIT/dtype/op lane and
owning RTX 5070 attention/producer lifetime regressions pass 151 tests.
The six formerly negative K129 frontend assertions were stale after the
earlier partial-group primal implementation; their positive replacements
are backed by the six exact-device K129 rows above. The global generic
batching/transpose closure assertions and registry states remain unchanged.

## Timing

mixed-map-benchmark-20261008.json contains 16 profiles:
eight primals plus four JVPs and four VJPs. All outputs match before and after
timing; maximum absolute error is 3.3125287153268346e-6.
The packet binds compiler/runtime and frontend/AD/ABI source hashes. Its
device-test source hash is the 16-row fixture before the six additional K129
regressions were appended; the final 22-row log is separate.

Cold/warm public wall time includes checked package handling and host
transfers. Prepared HIP event samples average 32 native program invocations,
including member kernels and program dispatch; they are not isolated-kernel
or end-to-end timings. Three samples per profile are retained. Warm public
medians range 0.879167–2.716652 ms; prepared program event medians range
0.0226785–0.8934560 ms. These separate costs are characterization, not an
A/B speedup or selector promotion.

## Remaining

Generic/dynamic/nonleading/composed batching and AD, quantized matrix
derivatives, wider resident physical layouts, sibling execution support and
focused native PR delivery remain open. This packet does not establish
Apple, x86, gfx1151 or NVIDIA scaled-map parity. Existing NVIDIA attention
and bounded producer JIT regressions pass on their owning RTX 5070; that is
shared JIT regression evidence, not an NVIDIA mixed-map migration.
