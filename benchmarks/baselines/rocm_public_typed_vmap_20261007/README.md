# Public typed FP8/MXFP8 vmap integration

Owner: FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Cross-backend synchronization key: ROCM-PUBLIC-TYPED-VMAP-2026-10-07.

Public vmap of a direct scalar typed E4M3 scaled product now creates an independent JIT owner and projects leading batch intent into Graph IR. Native verification, Schedule, Tile, Target, image and checked ABI own the arithmetic and batch launch. No Python per-batch arithmetic/launch loop is used by this route. Scalar Graph and compile-result ownership stay unchanged.

Static shared-RHS, independent-RHS and shared-LHS policies support FP32 scales and E8M0 MXFP8 scales, KN/NK RHS, and leading output axis zero. Explicit planned-gated byte annotations survive projection. Invalid scale shapes and mismatched batch extents are rejected at frontend capture; native map capture errors no longer fall back to an unrelated scalar AST specialization.

Validation: 362 shared frontend/diagnostic/pass/operator/dtype/audit checks pass. 25 host WSL frontend tests (including existing NVFP4 regressions) pass. 12 exact gfx1201 numerical tests pass against an independent float64 block oracle, including changed scales with compiler subprocesses forbidden on warm calls. Device warning: the owning venv lacks the pytest timeout plugin.

The recorder retains separate public-call wall, prepared update/invoke/read wall, and native HIP sequence event windows. Native events include sequence enqueue gaps and are not isolated kernel timing. Timings are characterization; no speedup comparison is claimed.

Remaining: dynamic/nested/nonleading batching, composed AD and linear-transpose closure, larger envelope characterization, sibling physical consumers, full unit suite and PR delivery. This does not close the generic batching registry.

Commands on owning gfx1201 checkout after sourcing its matching validation environment:
python -m pytest tests/device/rocm/test_public_typed_scaled_vmap.py -q
python -m benchmarks.rocm.benchmark_public_typed_scaled_vmap --output benchmarks/baselines/rocm_public_typed_vmap_20261007/timings.json

## Wider public envelope proof

The wider recorder and device suite extend the same public route to
B=2,M=200,N=129,K=1536 and B=2,M=128,N=4096,K=256.
36 owning numerical/warm-call cases pass; timings-wide.json has 36 rows,
each with one native program step. Three recorded FP32 NK rows use actual
LDS staging, including the larger matrix envelope. Independent source/tool
hashes are in source-tools-wide.json; original narrow receipts are retained.

For the M=200 rows, public medians range from 1.03 to 2.00 ms and native
sequence-event medians range from 0.038 to 0.537 ms. The wider LDS envelope
records public medians from 1.27 to 1.46 ms and sequence events from 0.012 to
0.033 ms. These ranges combine different formats, orientations and sharing
policies; they are characterization, not a matched comparative speedup.
