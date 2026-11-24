"""JIT-NATIVE-STATEMENT-EFFECT-2026-11-24 on each owning ROCm GPU.

A body that runs a call only for its side effect must run it on every call.
Before, the native math route traced it once and replayed the capture: the
effect ran on 1 of 3 calls on both gfx1151 and gfx1201. Each arch proves
itself; neither result transfers.
"""
import numpy as np
import pytest
import tessera as ts

from tessera import runtime as rt
from tessera.compiler import rocm_native
from tessera.compiler.fallback import FallbackReason

LOG: list = []


def logged_sqrt(a):
    LOG.append(1)
    return ts.ops.sqrt(a)


def plain_sqrt(a):
    return ts.ops.sqrt(a)


@pytest.fixture
def target():
    arch = rt._rocm_live_arch()
    if arch not in {"gfx1151", "gfx1201"} or not rocm_native.native_packaging_available():
        pytest.skip("requires exact owning ROCm GPU and native compiler")
    return "rocm_" + arch


def test_statement_effect_runs_once_per_call_and_the_plain_body_stays_native(target):
    a = np.random.default_rng(1124).uniform(.125, 2, (3, 17)).astype(np.float32)
    plain = ts.jit(target=target)(plain_sqrt)
    logged = ts.jit(target=target)(logged_sqrt)
    del LOG[:]
    for _ in range(3):
        np.testing.assert_allclose(plain(a), np.sqrt(a), rtol=2e-5, atol=2e-5)
        assert plain.last_fallback_reason is None
        assert plain._native_descriptor_last_receipt is not None
        np.testing.assert_allclose(np.asarray(logged(a)), np.sqrt(a), rtol=2e-5, atol=2e-5)
        assert logged.last_fallback_reason is FallbackReason.FRONTEND_UNLOWERED_CALL
        assert logged._native_descriptor_last_receipt is None
    assert len(LOG) == 3
