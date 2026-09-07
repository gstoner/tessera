# Mixed storage and selected checkpoint execution

Owner: F3 / AD-RESIDUAL-EVAL-1 / W2.4a / IR-NATIVE-FOUNDATION-1.

The [recorder](../../record_tape_checkpoint_execution.py) compiles fresh split
products and runs them on RTX 5070 (SM120) and Radeon 8060S (gfx1151). Reports
bind compiler, recorder, source and both native product identities. They are
correctness and retained-allocation evidence, not latency or overlap packets.
No performance candidate is promoted.

| Case | Captured frame bytes | Residual bytes |
|---|---:|---:|
| Mixed f32/f64 independent squares | 96 | 0 |
| Nested SAVE (three outer × three inner steps) | 80 | 32 |
| Nested HYBRID (one interior checkpoint per loop) | 64 | 16 |
| Nested recompute-all | 48 | 0 |
| Counted while SAVE (three steps) | 80 | 32 |

Both hosts produce these counts. Captured frame bytes include input snapshots,
primal outputs and persistent residuals, measured before backward allocates its
returned gradients. They exclude device allocator granularity and private
kernel temporaries. Each case checks two cotangent scales and unchanged saved
values after repeated backward. fp64 results use 1e-12 relative/absolute
tolerances; f32 uses 1e-5/1e-6. Each frame and caller allocation is released.

The while case is the proven zero-origin/unit-step constant-bound normalization,
not data-dependent termination. These five cases do not establish general
mixed scalar/predicate storage, asynchronous retirement, automatic checkpoint
selection, native GPU ANN execution, or Apple/x86 tape execution.
