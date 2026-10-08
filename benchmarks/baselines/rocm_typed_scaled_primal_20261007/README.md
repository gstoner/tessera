# Canonical typed FP8 primal JIT

Ordinary @jit(target="rocm_gfx1201") carries the original traced Graph through native Schedule/Tile/Target and checked native image ABI. No replacement Graph, Tile arithmetic or Python numerical backend is constructed. Argument/guard names follow actual SSA operand roles. Four primal device cases pass independent float64 block numerics and changed-scale warm reuse with compiler subprocesses forbidden. Seventeen combined primal/AD/cache device cases pass, and 337 focused semantic/registry unit checks pass.

| M/N/K | RHS storage | Public warm ms | Same-image native event ms |
| --- | --- | --- | --- |
| [17, 19, 256] | KN | 1.04636 | 0.0193 |
| [17, 19, 256] | NK | 1.08646 | 0.04117 |
| [200, 129, 1536] | KN | 4.48031 | 0.52618 |
| [200, 129, 1536] | NK | 4.00437 | 0.08178 |

Public timing includes checked descriptor staging/readback. Native timing uses a diagnostic one-member owner of the same image and includes enqueue gaps; it is not isolated kernel or public native-program proof. Compiler IR/image hashes and live rocminfo identity accompany the packet. MXFP8 unsigned-byte frontend integration, broader batching/transpose-left/composed AD, native primal owner projection and full-unit closure remain open. Recorder: benchmarks/rocm/benchmark_public_typed_scaled_primal.py.
