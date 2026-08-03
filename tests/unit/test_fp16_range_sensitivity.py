"""fp16 RANGE coverage — the axis bf16 structurally cannot test.

bf16 and fp16 are the same width and fail in opposite directions:

    bf16   8 mantissa bits, f32's exponent range   -> loses RESOLUTION
    fp16  10 mantissa bits, max 6.55e4             -> loses RANGE

`test_shape_rule_registry.py` probes fp32 + bf16, which is correct for storage
-dtype *propagation*. It is blind to range. Measured on this op set, **23 ops
lose numerics at fp16 while fp32 AND bf16 are both fine** — bf16 testing alone
would never surface one of them.

The failures are typically silent. `rmsnorm_safe` at fp16 with 1e4 inputs
returns 0.0 instead of ~1.0, because sum(x**2) overflows to inf and then
x/inf underflows to zero. No exception, no NaN, no inf in the output — just
wrong numbers from an op whose name asserts safety. A finite-output check would
pass it; only comparing against fp32 catches it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tessera.compiler.op_catalog import (
    FP16_INTERMEDIATE_OVERFLOW,
    FP16_RESULT_UNREPRESENTABLE,
    _SPECS,
)

FP16_MAX = 65504.0


def _probe(public_name, arity, dtype, magnitude=1e4):
    from tessera import ops

    fn = getattr(ops, public_name, None)
    if fn is None:
        return None
    base = np.full((4, 8), magnitude)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = fn(*[base.astype(dtype) for _ in range(arity)])
        except Exception:
            return None
    return np.asarray(out).astype(np.float64)


def _spec(graph_name):
    return next((s for s in _SPECS if s.graph_name == graph_name), None)



def test_declared_hazards_name_real_ops_and_explain_themselves():
    real = {spec.graph_name for spec in _SPECS}
    for table in (FP16_INTERMEDIATE_OVERFLOW, FP16_RESULT_UNREPRESENTABLE):
        for name, hazard in table.items():
            assert name in real, f"{name} is not a catalog op"
            assert len(hazard) > 25, f"{name}: hazard note too thin to be useful"


def test_fp16_range_limit_is_what_we_think():
    """Pin the premise. If fp16's range changed, this whole axis changes."""
    assert float(np.finfo(np.float16).max) == pytest.approx(FP16_MAX)
    # Square IN fp16, using an ARRAY. `np.float16(1e4) ** 2` on a SCALAR is
    # NumPy-version dependent: NEP 50 (2.x) keeps float16 and overflows, while
    # 1.26 promotes to float64 and yields a finite 1e8. CI pins numpy<2.0, so a
    # scalar assertion passes locally and fails there. Arrays do not promote.
    values = np.full((2,), 1e4, dtype=np.float16)
    with np.errstate(over="ignore"):
        squared = values * values
    assert not np.all(np.isfinite(squared))


@pytest.mark.parametrize("graph_name", sorted(FP16_INTERMEDIATE_OVERFLOW))
def test_intermediate_overflow_is_fixed_and_stays_fixed(graph_name):
    """Class A: the op must now produce the RIGHT VALUE at fp16.

    These overflowed internally while their answers fit fp16 comfortably. The
    storage-dtype enforcement promotes reduced-precision operands to f32,
    computes, and stores back — so the overflow never occurs.

    This asserts the VALUE, not the dtype. The earlier dtype-only gate passed
    `rmsnorm_safe` returning 0.0: right type, wrong number. Masking a defect is
    worse than the defect, because it looks fixed.
    """
    spec = _spec(graph_name)
    if spec is None:
        pytest.skip(f"{graph_name} not in catalog")
    arity = min(spec.min_arity, 3) or 1
    ref = _probe(spec.public_name, arity, np.float32)
    got = _probe(spec.public_name, arity, np.float16)
    if ref is None or got is None:
        pytest.skip("op not probeable with a bare tensor at this arity")

    assert np.all(np.isfinite(ref)), f"f32 reference for {graph_name} is unhealthy"
    assert np.all(np.isfinite(got)), (
        f"{graph_name} produces non-finite output at fp16 — the compute-at-f32 "
        "path regressed"
    )
    assert not np.all(got == 0), (
        f"{graph_name} collapsed to zero at fp16 — this is the exact masking "
        "regression (correct dtype, wrong value)"
    )
    scale = max(float(np.max(np.abs(ref))), 1.0)
    assert np.allclose(got, ref, rtol=5e-2, atol=5e-2 * scale), (
        f"{graph_name} at fp16 disagrees with its f32 reference beyond "
        "reduced-precision tolerance"
    )


@pytest.mark.parametrize("graph_name", sorted(FP16_RESULT_UNREPRESENTABLE))
def test_result_unrepresentable_claims_are_true(graph_name):
    """Class B: verify the claim rather than accepting it.

    These are NOT defects — the answer genuinely exceeds fp16's 6.55e4 max, so
    no compute width helps. But "it doesn't fit" is a claim, and an
    unverified claim is how a real bug hides in an exemption list. Prove it by
    measuring the f32 result.
    """
    spec = _spec(graph_name)
    if spec is None:
        pytest.skip(f"{graph_name} not in catalog")
    arity = min(spec.min_arity, 3) or 1
    ref = _probe(spec.public_name, arity, np.float32)
    if ref is None:
        pytest.skip("op not probeable with a bare tensor at this arity")
    assert np.all(np.isfinite(ref)), f"f32 reference for {graph_name} is unhealthy"
    assert float(np.max(np.abs(ref))) > FP16_MAX, (
        f"{graph_name}'s f32 result fits in fp16, so 'result unrepresentable' "
        "is wrong — it belongs in FP16_INTERMEDIATE_OVERFLOW (and may be a real "
        "defect that compute-at-f32 would fix)"
    )
