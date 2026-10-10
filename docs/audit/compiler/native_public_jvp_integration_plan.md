---
audit_role: plan
plan_state: landing
owner: Compiler frontend and AD
last_updated: 2026-10-09
---

# Public native JVP integration

Discovery: [compiler map](README.md). Global sequencing defers to
[INTEGRATED_COMPILER_PLAN.md](INTEGRATED_COMPILER_PLAN.md).

Synchronization key: PUBLIC-NATIVE-JVP-20261009.
Owning items: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
This is active work toward the five compiler slices; it does not close them.

## Current implementation and proof

The public `tessera.autodiff.jvp` entry projects explicit native JIT requests
into an independent native Graph AD owner before Python tangent tracing.
Caller Graph IR, differentiation intent and shape constraints are retained.
Inactive operands are explicit `None`; the child owner retains the active
input mapping. Warm calls reuse the child; source or constraint changes
invalidate its witness. Native execution receipts are required.

The already traced `flash_attn(..., bias=...)` spelling now reaches the
reference primal and derivative certificates through the interception wrapper.
`bias` and `attn_bias` cannot both select non-null score operands.
This fixes reference certificate intent, not production GPU arithmetic.

Host WSL: 26 focused tests pass, one Apple hardware test skips, and Ruff passes.
The new independent FP64 oracle covers grouped causal attention with dense
and broadcast bias, output and saved LSE, JVP direction products and projected
VJP products. These are reference certificates, not Apple execution proof.

Exact RTX5070 / SM120 / CUDA13.3: 26 public native JVP cases pass,
covering host/resident roots, plain/biased/saved-LSE forward, K=5/129,
value-only and bias-only activity, retained output storage, original graph
invariants, and warm compiler/frontend refusal. Checks reject output-span
overlap, input/output overlap, incorrect extents and package output tampering;
eight independently pending producer streams are ordered before native reads.
The combined public and adjacent resident attention lane passes 84 cases;
All 24 separately opted-in prepared-owner cases also pass using freshly
generated native compiler fixtures.

Six production-compiler contract cases and 106 adjacent JVP unit cases pass.
Another 312 diagnostic/pass metadata and public-transform gates pass.
The immutable parent full unit suite completed: 25,728 passed, 7,565 skipped,
874 deselected and two failures in the existing scaled_matmul batching and
transpose closure tests. No full CI pass is claimed.

Tests: tests/unit/test_attention_bias_alias_certificate.py,
tests/unit/test_public_native_jvp_transform.py,
tests/unit/test_attention_saved_lse_jvp_ir.py,
tests/device/nvidia/test_public_native_jvp_transform.py.

## Native saved-LSE product implementation

The typed Graph tangent interface admits paired O/LSE products. The registered
checkpoint JVP op carries an optional row-LSE tangent of the forward LSE shape.
The hashed native Schedule contract verifies the primal/return mapping and
carries both derivative outputs through Tile and LLVM/PTX lowering.
dLSE is the native probability-weighted score direction; value-only activity
writes exact zero. Production derivative arithmetic remains in native IR.

Portable schema v3 seals saved-LSE output selection. The retained C++ owner
owns distinct O, LSE, dO and dLSE spans, checks extents and overlap, orders
producer streams and retains private generations. Legacy one-result products
retain their previous portable schemas and 9/11-pointer ABIs; paired products
use separate 10/12-pointer contracts.

Two fresh six-profile packets validate all four outputs before alternating
host/resident completed-call timing and separate forward/JVP CUDA events.
Maximum absolute error is 6.167596078299198e-7 against the independent FP64
direction oracle. Resident/host completed-call ratios span 1.187–1.300.
These small profiles establish correctness and overhead characterization;
they do not establish a performance improvement or default-route promotion.

Evidence: benchmarks/baselines/public_saved_lse_jvp_20261009/README.md,
run1.json and run2.json in that directory.
Matching LLVM/MLIR 23.1.1 compiler tools and the fresh C++ provider were built.

## Remaining integration

Public integration on GFX1201 is now proved for 36 named continuous FP32
scaled-product cases, with two fresh 36-profile event/public-call packets.
Evidence: benchmarks/baselines/public_native_scaled_jvp_20261009/README.md.
This is a different program family from the SM120 attention proof; neither
establishes Apple or x86 parity. GFX1201 public preparation/binding overhead,
mixed-axis and broader composition/quantized AD remain open. Dynamic/composed
products, half-storage AD, nested/higher-order transforms and resident-call
overhead remain open. Generic scaled_matmul batching/transpose closure also
remains open; the existing unit gates have not been weakened.

## Wider scope still open

Nested/higher-order native transforms, bounded tensor products and general
frontend compositions remain active. GFX1201 NVFP4 ingest, W1.1 producer
retirement, general attention forward/backward consumers, and ROCm
route/performance closure retain their existing plans and evidence obligations.

Recorder: `benchmarks/nvidia/record_public_saved_lse_jvp.py`; correctness-gated paired output and tangent checks precede alternating host/resident completed-call and native event timing.

GFX1201 recorder: benchmarks/rocm/record_public_native_scaled_jvp.py.
