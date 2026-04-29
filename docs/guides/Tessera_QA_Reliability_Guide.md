---
status: Informative
classification: Informative
authority: QA and reliability behavior guide
last_updated: 2026-04-28
---

# Tessera QA And Reliability Guide

Tessera’s reliability philosophy is correctness first, deterministic when
requested, and stable across deployment scales: one CPU reference, one GPU, one
node, and multi-node meshes should all be testable with the same behavioral
contracts.

This guide is practical rather than formal. Normative API and IR details remain
in `docs/README.md` and the spec tree. Use this guide to decide what to test,
what failure means, and what evidence a Tessera program should provide before it
is treated as reliable.

For production operations, stress testing, chaos testing, node-scale validation,
and rack-scale/NVL72 validation, continue with
`docs/guides/Tessera_Production_Reliability_And_Chaos_Guide.md`.

## 1. Device-Scale QA Goals

On a single device, validate:

- **Correctness**: operators match a golden CPU reference.
- **Numerical stability**: NaNs, Infs, overflow, cancellation, and mixed
  precision drift are detected.
- **Determinism**: fixed seeds and deterministic mode reproduce results.
- **Fault tolerance**: invalid shapes, out-of-bounds indices, and runaway work
  fail loudly.
- **Performance consistency**: kernels remain within agreed latency,
  throughput, bandwidth, and utilization bands.

At distributed scale, add:

- **Collective reproducibility**: collective ordering is visible through
  future/await or equivalent dependencies.
- **Mesh correctness**: sharding and gathering reconstruct the intended tensor.
- **Schedule reproducibility**: schedule artifacts and hashes are captured with
  deployment bundles.
- **Failure isolation**: a failing rank or device reports enough context to
  reproduce the issue locally.

## 2. Correctness Testing

Every Tessera operator should have a golden CPU reference. Prefer `float64` or
`float32` references, then compare Tessera output with dtype-appropriate
tolerances.

```python
import numpy as np
from tessera.testing import assert_close_to_reference

A = np.random.randn(128, 128).astype(np.float32)
B = np.random.randn(128, 128).astype(np.float32)
ref = A @ B

# Replace this with op.matmul(tensor.from_numpy(A), tensor.from_numpy(B)).numpy()
out = A @ B

assert_close_to_reference(out, ref, rtol=1e-5, atol=1e-6)
```

Best practices:

- Test representative shapes and boundary shapes.
- Include non-multiple dimensions when kernels support masks or bounds checks.
- Compare values and shapes.
- Store tolerances by op, dtype, backend, and schedule when behavior differs.

## 3. Numerical Stability

Critical numerical ops must be tested with random and adversarial inputs:

- zeros, negative zeros, denormals, large magnitudes, tiny magnitudes
- NaN and Inf propagation behavior
- near-cancellation patterns
- extreme logits for softmax/logsumexp
- mixed precision against an FP32 or FP64 baseline

```python
from tessera.testing import assert_finite

Y = softmax_impl(X)
assert_finite(Y, name="softmax output")
```

Tessera’s compiler should preserve `tessera.numeric_policy` through lowering so
test reports can explain which storage dtype, accumulator dtype, rounding mode,
quantization scale, and deterministic setting produced a result.

## 4. Determinism Testing

Determinism is a contract, not a vibe. A deterministic test should fix all
seeds, run the same operation more than once, and compare bitwise unless a
documented tolerance is required.

```python
from tessera.testing import assert_deterministic

def run_once():
    return train_step(seed=42)

assert_deterministic(run_once, runs=3)
```

Best practices:

- Seed RNG streams explicitly.
- Use deterministic execution for collectives and reductions.
- Run determinism checks multiple times in CI.
- Record schedule artifact hashes with test output when autotuning is involved.

Under the current semantic core, deterministic mode governs RNG streams,
dropout masks, movement ordering, collective future/await ordering, reduction
order, and schedule artifact choice.

## 5. Fault Tolerance

Tests should assert expected failures, not merely avoid them.

Examples:

- invalid shapes fail during shape inference
- out-of-bounds gather fails with source context
- invalid mesh axes fail before collective launch
- runaway kernels are stopped by watchdogs
- runtime backend errors include device, stream, kernel, and launch parameters

```python
import pytest

with pytest.raises(RuntimeError):
    gather_impl(np.zeros((10,), dtype=np.float32), indices=[100])
```

Debug builds should enable bounds checks, race detectors where available, and
NaN/Inf sentinels for numerically sensitive regions.

## 6. Performance Consistency

Performance tests should compare against baselines and explain regressions.
Track at least:

- latency and variance
- achieved FLOPs or bandwidth
- occupancy/utilization where available
- compilation and autotuning time
- selected schedule artifact hash

```python
from tessera.testing import PerformanceExpectation

expect = PerformanceExpectation(
    name="matmul_4096_bf16",
    latency_ms_max=2.5,
    tflops_min=150.0,
)
expect.validate({"latency_ms": 2.1, "tflops": 171.0})
```

Use small smoke benchmarks on every PR and larger statistical benchmarks on
scheduled GPU runners.

## 7. Distributed QA

Distributed tests should prove that logical mesh semantics match tensor values:

- `all_reduce` equals the mathematical reduction.
- `reduce_scatter` equals reduction followed by slicing.
- `all_gather` reconstructs the original tensor.
- `all_to_all` preserves token/expert routing invariants.
- sharding annotations and runtime rank layout agree.

For deterministic distributed tests, collectives must expose ordering through
typed futures/awaits or equivalent runtime dependencies. The test should fail if
collective results depend on incidental launch timing.

## 8. Release Checklist

Before a backend, operator, or schedule is promoted:

- Golden reference tests pass.
- Numerical stability tests cover edge cases.
- Determinism tests pass with fixed seeds.
- Expected fault tests produce useful errors.
- Performance baselines are recorded.
- Schedule artifact hashes are stored for tuned kernels.
- Distributed reconstruction tests pass if collectives or sharding are involved.
- Test reports include dtype/numeric policy, backend, target arch, and schedule.

## 9. Existing Test Assets

Use these as implementation anchors:

- `tests/tessera_numerical_validation/`: numerical correctness suite.
- `python/tessera/testing/`: shared testing helpers.
- `tests/unit/test_nccl_adapter.py`: collective behavior tests.
- `tests/unit/test_bayesian_autotuner.py`: schedule artifact tests.
- `tests/unit/`: runtime, diagnostics, shape, and semantic-core tests.
- `docs/TesseraBench/`: benchmark and performance validation material.

## 10. Related Guides

- `docs/guides/Tessera_Error_Handling_And_Diagnostics_Guide.md`: stable error codes, diagnostic fields, environment switches, and debugging workflow.
- `docs/guides/Tessera_Production_Reliability_And_Chaos_Guide.md`: monitoring, replay, stress, chaos, and production recovery validation.
- `docs/operations/Tessera_Standard_Operations.md`: standard operator semantics, deterministic behavior, and operator error codes.
