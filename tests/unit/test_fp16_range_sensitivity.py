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

from tessera.compiler.op_catalog import FP16_RANGE_SENSITIVE, _SPECS


def test_declared_hazards_name_real_ops_and_explain_themselves():
    real = {spec.graph_name for spec in _SPECS}
    for name, hazard in FP16_RANGE_SENSITIVE.items():
        assert name in real, f"{name} is not a catalog op"
        assert len(hazard) > 25, f"{name}: hazard note too thin to be useful"


def test_fp16_range_limit_is_what_we_think():
    """Pin the premise. If fp16's range changed, this whole axis changes."""
    assert float(np.finfo(np.float16).max) == pytest.approx(65504.0)
    # Square IN fp16 — converting to a Python float first does the arithmetic
    # in double and never overflows, which is the mistake this line originally
    # made and is the same class of error the whole axis is about.
    with np.errstate(over="ignore"):
        squared_in_fp16 = np.float16(1e4) ** 2  # overflow is the point
    assert not np.isfinite(squared_in_fp16), (
        "(1e4)**2 no longer overflows fp16 — re-derive the hazard set"
    )


def test_rmsnorm_safe_is_not_safe_at_fp16():
    """The headline case, pinned.

    This is the one that argues for the whole axis: the output is FINITE, so
    an is-finite assertion passes, while the values are silently zero. Only a
    comparison against fp32 exposes it.
    """
    import ml_dtypes

    from tessera import ops

    x = np.full((4, 8), 1e4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f32 = np.asarray(ops.rmsnorm_safe(x.astype(np.float32))).astype(np.float32)
        f16 = np.asarray(ops.rmsnorm_safe(x.astype(np.float16))).astype(np.float32)
        bf16 = np.asarray(
            ops.rmsnorm_safe(x.astype(ml_dtypes.bfloat16))
        ).astype(np.float32)

    assert np.all(np.isfinite(f16)), "premise changed: fp16 now produces inf/NaN"
    assert float(f32.flat[0]) == pytest.approx(1.0, rel=1e-3)
    assert float(f16.flat[0]) == 0.0, (
        "rmsnorm_safe at fp16 no longer collapses to zero — if it was fixed, "
        "remove it from FP16_RANGE_SENSITIVE and update this test"
    )
    # bf16 is fine, which is exactly why bf16 cannot stand in for fp16 here.
    assert float(bf16.flat[0]) == pytest.approx(1.0, rel=1e-2)


@pytest.mark.parametrize("graph_name", sorted(FP16_RANGE_SENSITIVE))
def test_declared_fp16_hazards_still_reproduce(graph_name):
    """Every declared hazard must still be real.

    A hazard that quietly stops reproducing means either the op was fixed (good
    — remove it and say so) or the probe drifted (bad). Either way the set must
    not rot into folklore.
    """
    import ml_dtypes

    from tessera import ops

    spec = next(s for s in _SPECS if s.graph_name == graph_name)
    fn = getattr(ops, spec.public_name, None)
    if fn is None:
        pytest.skip(f"{spec.public_name} is not exposed on tessera.ops")

    arity = min(spec.min_arity, 3) or 1
    base = np.full((4, 8), 1e4)

    def run(dtype):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                out = fn(*[base.astype(dtype) for _ in range(arity)])
            except Exception:
                return None
        return np.asarray(out).astype(np.float32)

    ref = run(np.float32)
    got = run(np.float16)
    if ref is None or got is None:
        pytest.skip("op not probeable with a bare tensor at this arity")

    healthy = np.all(np.isfinite(ref)) and not np.all(ref == 0)
    degraded = (not np.all(np.isfinite(got))) or np.all(got == 0)
    assert healthy, f"fp32 reference for {graph_name} is itself unhealthy"
    assert degraded, (
        f"{graph_name} no longer degrades at fp16 — if it was fixed, remove it "
        "from FP16_RANGE_SENSITIVE rather than leaving a stale hazard note"
    )
