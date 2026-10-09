# gfx1201 continuous scaled-product SSA chain

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / LAYOUT-ALG-1.
Synchronization key: CONTINUOUS-SCALED-SSA-20261009.

The public frontend now accepts computed FP32 operands in continuous primal
scaled-product chains. Native MLIR already outlines actual operations and tracks
private buffer first-write/last-read lifetimes. Forward differentiation also
packages and executes the dependency graph. The source Graph is unchanged.
Reverse residual capture and mapped composed chains remain open. No dtype, op,
pass, diagnostic or coverage state is promoted.

## Validation

Six matching native compiler/admission checks pass, including producer buffer
lifetime and chained forward export. 566 frontend/AD tests pass with 11 explicit
compiler skips. Ruff and zero-error mypy pass. Two public gfx1201 device cases
prove primal/JVP against an independent scalar oracle, changed-input replay,
compiler/eager-free warm execution and retained output ownership.

Owning GPU: RX 9070 XT, UUID in gfx1201.json.
Measured frame (M,K,N,P)=(2,9,5,3), both products with K/N block size 4.
Maximum absolute error: 3.49246e-10.
Seven 128-repeat HIP windows: full native median 0.00393286 ms.
Grouped producer/consumer: 0.00187186/0.00202467 ms, including graph dispatch;
these must not be summed as interleaved program timing.
Public compile-warm median: 1.75026 ms; preparation/allocation is not excluded.
This characterizes the named frame, not a speedup claim.

Reproduce with matching core/ROCm tools and native runtime provider:
python benchmarks/rocm/record_continuous_scaled_ssa_chain.py --output packet.json

No NVIDIA, gfx1151, Apple or x86 physical evidence follows from this packet.
Generic batching/transpose, shared residuals, dynamic shapes and wider DAGs
remain required.
