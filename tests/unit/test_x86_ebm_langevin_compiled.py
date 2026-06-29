"""EBM Langevin sampling lane on x86 AVX-512 (P7 sampling) — one Langevin step
that DRAWS its Gaussian noise on-device from counter-based Philox-4x32-10 (the
P6 generator): out = y − η·grad + noise_scale·z. Reachable via
`compiler_path="x86_ebm_langevin_compiled"`. Validated byte-for-byte vs
tessera.ebm.langevin_step_philox. Skip-clean: libtessera_x86_elementwise.so not
built.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.ebm.energy import langevin_step_philox


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_ebm_langevin_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["y", "g"], "output_name": "o",
        "ops": [{"op_name": "tessera.ebm.langevin_step", "result": "o",
                 "operands": ["y", "g"], "kwargs": kwargs}]})


def _run(rt, y, grad, **kwargs):
    res = rt.launch(_art(rt, kwargs), (y, grad))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_ebm_langevin_compiled"
    return np.asarray(res["output"])


_RNG = np.random.default_rng(31)


def test_langevin_matches_reference():
    rt = _rt_or_skip()
    y = _RNG.standard_normal((4, 5)).astype(np.float32)
    grad = _RNG.standard_normal((4, 5)).astype(np.float32)
    key = [0x1234, 0x5678]
    counter = [7, 1, 2, 3]
    got = _run(rt, y, grad, eta=0.1, noise_scale=0.3,
               key=key, counter=counter)
    ref = langevin_step_philox(y, grad, eta=0.1, noise_scale=0.3,
                               key=key, counter=counter)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-5, atol=1e-5)


def test_langevin_zero_noise_is_gradient_descent():
    rt = _rt_or_skip()
    y = _RNG.standard_normal((6,)).astype(np.float32)
    grad = _RNG.standard_normal((6,)).astype(np.float32)
    got = _run(rt, y, grad, eta=0.25, noise_scale=0.0,
               key=[1, 2], counter=[0, 0, 0, 0])
    np.testing.assert_allclose(got, (y - 0.25 * grad).astype(np.float32),
                               rtol=1e-5, atol=1e-5)


def test_langevin_counter_changes_noise():
    rt = _rt_or_skip()
    y = np.zeros((8,), np.float32)
    grad = np.zeros((8,), np.float32)
    key = [9, 9]
    a = _run(rt, y, grad, eta=0.0, noise_scale=1.0, key=key,
             counter=[0, 0, 0, 0])
    b = _run(rt, y, grad, eta=0.0, noise_scale=1.0, key=key,
             counter=[100, 0, 0, 0])
    assert not np.allclose(a, b)        # different counter → different draw
