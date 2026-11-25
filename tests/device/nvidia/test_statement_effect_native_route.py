"""JIT-NATIVE-STATEMENT-EFFECT-2026-11-24 on the RTX 5070 (sm_120).

A body that runs a call only for its side effect must run it on every call.
Before, the native route traced it (running the effect once on a tracer),
declined, and eager ran it again: 4 effects for 3 calls.
"""
import numpy as np
import pytest
import tessera as ts

from tessera import runtime as rt
from tessera.compiler.fallback import FallbackReason

pytestmark = pytest.mark.hardware_nvidia

LOG: list = []


def logged_product(a, b):
    LOG.append(1)
    return ts.ops.matmul(a, b, output_dtype="fp32")


def plain_product(a, b):
    return ts.ops.matmul(a, b, output_dtype="fp32")


def _roots():
    rng = np.random.default_rng(1124)
    return (rng.normal(0, .25, (17, 35)).astype(np.float32),
            rng.normal(0, .25, (35, 19)).astype(np.float16))


def test_statement_effect_runs_once_per_call_and_the_plain_body_stays_native():
    assert rt._nvidia_device_name() == "sm_120"
    a, b = _roots()
    expected = a.astype(np.float64) @ b.astype(np.float64)
    plain = ts.jit(target="nvidia_sm120")(plain_product)
    logged = ts.jit(target="nvidia_sm120")(logged_product)
    del LOG[:]
    for _ in range(3):
        np.testing.assert_allclose(plain(a, b), expected, rtol=2e-3, atol=2e-3)
        assert plain.last_fallback_reason is None
        assert plain._native_descriptor_last_receipt is not None
        np.testing.assert_allclose(np.asarray(logged(a, b)), expected, rtol=2e-3, atol=2e-3)
        assert logged.last_fallback_reason is FallbackReason.FRONTEND_UNLOWERED_CALL
        assert logged._native_descriptor_last_receipt is None
    assert len(LOG) == 3
