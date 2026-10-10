# Compiler-projected native primal owner, gfx1201

Original typed FP8/MXFP8 Graph SSA is outlined by the native compiler with an
explicit primal program kind, one returned buffer and read/write lifetimes.
Member images, symbols, scalars and geometry are compiler-derived. Ordinary
JIT checked descriptor submission executes the native HIP owner. Python
marshals ABI inputs/readback; it creates no device launch sequence or IR math.

Matching owning RX 9070 XT build and 21 combined primal/JVP device tests pass.
Manifest assertions require this actual owner path; numerics and changed-scale
warm calls run with compiler subprocesses disabled. Full core native fixtures:
492 passed, 66 unsupported. Shared package/frontend/registry gates: 483 passed.
Paired JVP retains its distinct two-output invariant; relabeling is rejected.

FP8 and MXFP8 packets retain separate public-call and native member event
windows and actual owner image digests. Event windows include enqueue gaps.
No general speedup is claimed. Large MXFP8 NK native event timing is about
0.51 ms versus about 0.05 ms in the earlier diagnostic shape-projected image.
These are different compiled physical paths, not an isolated owner-overhead
A/B. Native member packaging must preserve/retune the prior physical image
projection and be compared with identical-image controls before performance
closure. Pinned staging remains opt-in; full-unit generic closure remains red.
