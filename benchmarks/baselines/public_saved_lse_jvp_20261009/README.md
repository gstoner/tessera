# Public native saved-LSE JVP — RTX5070

Owning items: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Synchronization key: PUBLIC-NATIVE-JVP-20261009.

Two fresh processes record six profiles each: K=5/129, grouped causal FP32
attention with broadcast score bias, and active Q/K/V, V-only, or bias-only
directions. Public `tessera.autodiff.jvp` projects frontend intent into native
Graph AD, hashed Schedule, Tile/native GPU MLIR and LLVM-PTX packages.
The paired outputs are O/LSE/dO/dLSE, with distinct native allocations and
checked host spans. V-only activity emits an exact zero LSE direction.

The independent FP64 attention oracle checks all four outputs before timing
and after every sample. Nine alternating host/resident rounds report native
forward/JVP CUDA event samples separately from completed public-call time.
Cold tracing, compilation and preparation are excluded; public invocation,
binding, snapshot, launch, completion and independent host outputs are included.

Recorder: `benchmarks/nvidia/record_public_saved_lse_jvp.py`.
Packets: `run1.json`, `run2.json`. Each records the live GPU/UUID/driver,
source hashes, both compiler hashes, native provider hash and native receipts.

Maximum absolute oracle error: 6.16759608e-07.

Resident/host completed-call ratios span 1.187–1.300; resident calls are slower for these small profiles. No speedup or performance promotion is claimed.

| Active roles | K | Run 1 resident/host | Run 2 resident/host |
| --- | ---: | ---: | ---: |
| v, q, k | 5 | 1.300 | 1.278 |
| v | 5 | 1.246 | 1.213 |
| bias | 5 | 1.245 | 1.209 |
| v, q, k | 129 | 1.238 | 1.255 |
| v | 129 | 1.235 | 1.211 |
| bias | 129 | 1.220 | 1.187 |

This proof is SM120-specific. It does not establish half-storage AD, dynamic/composed/higher-order attention or sibling-backend parity.
